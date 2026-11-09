"""Command line interface for chat, agent, and code-task modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .code_task import collect_changes, new_run_dir, prepare_workspace, run_code_task
from .chat import chat_once
from .config import (
    HarnessConfig,
    load_config,
    resolve_config_path,
    resolve_config_relative,
    resolve_data_root,
)
from .providers import create_provider
from .rubrics_scorer import SCORER_SYSTEM_PROMPT
from .runner import DEFAULT_SYSTEM_PROMPT, TurnRunner
from .trace import JsonlTrace
from .types import HarnessError


def _provider_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Harness JSON config; defaults to ./harness.json")
    parser.add_argument("--profile", help="Provider profile name")
    parser.add_argument("--model", help="Override the profile model")
    parser.add_argument("--base-url", help="Override the profile base URL")
    parser.add_argument("--system-prompt", help="Override the system prompt")


def _common(parser: argparse.ArgumentParser) -> None:
    _provider_options(parser)
    parser.add_argument("--max-steps", type=int, help="Maximum provider/tool steps in the one turn")
    parser.add_argument("--tool-timeout", type=float, help="Maximum seconds for each shell tool call")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mixed-pilot", description="Minimal mixed-pilot agent harness")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    chat = subparsers.add_parser("chat", help="Send one chat request without tools or an agent loop")
    _provider_options(chat)
    chat.add_argument("--message")
    chat.add_argument("--task-file", help="Read the user message from a UTF-8 text file")
    chat.add_argument("--result", help="Save the response, model, stop reason, and usage as JSON")

    judge = subparsers.add_parser("per-step-judge", help="Score one collection trace step via a single chat request")
    judge.add_argument("--config", help="Harness JSON config; defaults to ./harness.json")
    judge.add_argument("--profile", required=True, help="Judge provider profile name")
    judge.add_argument("--trace", required=True, type=Path, help="One collection .traj.json file")
    judge.add_argument("--step", required=True, type=int, help="One-based API-step number (1 = step_idx 0)")
    reference = judge.add_mutually_exclusive_group()
    reference.add_argument("--reference-answer", help="Explicit reference answer / desired final state")
    reference.add_argument("--reference-answer-file", type=Path, help="UTF-8 reference answer text")
    judge.add_argument("--dry-run", action="store_true", help="Output the exact prompt without an API call")
    judge.add_argument("--result", type=Path, help="Save prompt, score, raw reply, model and usage as JSON")

    agent = subparsers.add_parser("agent", help="Run one agent turn in an existing workspace")
    _common(agent)
    agent.add_argument("--workspace", required=True)
    agent.add_argument("--message")
    agent.add_argument("--task-file")
    agent.add_argument("--trace")
    agent.add_argument("--result")
    agent.add_argument(
        "--restore-workspace",
        action="store_true",
        help="Run in a disposable clone/copy and remove it after inference",
    )
    agent.add_argument(
        "--run-dir",
        help="Artifact directory for --restore-workspace; defaults under the system temp dir",
    )

    code = subparsers.add_parser(
        "code-task",
        help="Prepare a workspace, run an agent, evaluate, and score a task of any category",
    )
    _common(code)
    source = code.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", help="mixed-pilot task id (any category)")
    source.add_argument("--repo", help="local Git/directory path or Git URL")
    code.add_argument(
        "--dataset-root",
        help="External data root containing tasks/; defaults to MIXED_PILOT_DATA_ROOT or ~/Data/mixed-pilot-data",
    )
    code.add_argument("--message")
    code.add_argument("--task-file")
    code.add_argument("--eval-command")
    code.add_argument(
        "--eval-setup-command",
        help="Prepare isolated evaluator dependencies inside the disposable eval workspace",
    )
    code.add_argument(
        "--eval-test-command",
        help="Run the test ids a native_test criterion declares; use {tests} as the placeholder",
    )
    code.add_argument("--test-patch")
    code.add_argument("--rubrics", help="Rubrics JSON; automatic with --task")
    code.add_argument("--reference-dir", help="Private scorer reference directory")
    code.add_argument("--attempt-root", default="attempts")
    code.add_argument("--eval-timeout", type=float)
    code.add_argument("--shallow", action="store_true")
    code.add_argument("--keep-eval-workspace", action="store_true")
    return parser


def _read_prompt(message: Optional[str], task_file: Optional[str]) -> str:
    if message is not None and task_file is not None:
        raise ValueError("pass only one of --message and --task-file")
    if task_file is not None:
        return Path(task_file).read_text(encoding="utf-8")
    if message is not None:
        return message
    if not sys.stdin.isatty():
        value = sys.stdin.read()
        if value:
            return value
    raise ValueError("provide --message, --task-file, or prompt text on stdin")


def _resolve_profile(args, config: HarnessConfig):
    profile = config.profile(args.profile)
    if args.model:
        profile.model = args.model
    if args.base_url:
        profile.base_url = args.base_url
    if config.agent.get("provider_timeout") is not None:
        profile.timeout = float(config.agent["provider_timeout"])
    return profile


def _profile_and_defaults(args, config: HarnessConfig):
    profile = _resolve_profile(args, config)
    max_steps = args.max_steps or int(config.agent.get("max_steps", 32))
    tool_timeout = args.tool_timeout or float(config.agent.get("tool_timeout", 120))
    return profile, max_steps, tool_timeout


def _make_runner(
    args,
    config: HarnessConfig,
    workspace: Path,
    trace_path: Optional[Path],
    task_id: str = "",
    trace: Optional[JsonlTrace] = None,
    system_prompt: Optional[str] = None,
) -> TurnRunner:
    profile, max_steps, tool_timeout = _profile_and_defaults(args, config)
    provider = create_provider(
        profile,
        config_dir=config.path.parent,
        profiles=config.profiles,
    )
    return TurnRunner(
        provider=provider,
        workspace=workspace,
        max_steps=max_steps,
        tool_timeout=tool_timeout,
        trace=trace or JsonlTrace(trace_path),
        system_prompt=system_prompt or args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        secret_env_name=profile.api_key_env,
        task_id=task_id,
    )


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def _chat(args, config: HarnessConfig) -> int:
    prompt = _read_prompt(args.message, args.task_file)
    response = await chat_once(
        config, prompt, profile_name=args.profile, model=args.model, base_url=args.base_url,
        system=args.system_prompt if args.system_prompt is not None else "You are a helpful assistant.",
        provider_factory=create_provider,
    )
    if args.result:
        _write_json(Path(args.result).resolve(), asdict(response))
    print(response.text)
    return 0


async def _per_step_judge(args, config: HarnessConfig) -> int:
    from .per_step_judge import judge_step, prepare_judgment

    inputs = [args.trace.resolve(), config.path.resolve()]
    if args.reference_answer_file:
        inputs.append(args.reference_answer_file.resolve())
    if args.result and args.result.resolve() in inputs:
        raise ValueError("--result must not overwrite the trace, config or reference answer")
    reference = (args.reference_answer_file.read_text(encoding="utf-8")
                 if args.reference_answer_file else args.reference_answer)
    config.profile(args.profile)  # Validate profile even for a dry run.
    if args.dry_run:
        result = prepare_judgment(args.trace, args.step, reference_answer=reference)
        result["judge_profile"] = args.profile
    else:
        result = await judge_step(args.trace, args.step, config=config, profile=args.profile,
                                  reference_answer=reference)
    if args.result:
        _write_json(args.result.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"prepared", "scored"} else 2


async def _agent(args, config: HarnessConfig) -> int:
    requested_workspace = Path(args.workspace).resolve()
    if not requested_workspace.is_dir():
        raise ValueError("workspace does not exist: %s" % requested_workspace)
    prompt = _read_prompt(args.message, args.task_file)
    if not args.restore_workspace:
        trace_path = Path(args.trace).resolve() if args.trace else None
        result = await _make_runner(args, config, requested_workspace, trace_path).run(prompt)
        if args.result:
            _write_json(Path(args.result).resolve(), result.to_dict())
        print(result.text)
        return 0

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if run_dir.exists():
            raise ValueError("--run-dir already exists: %s" % run_dir)
        if _inside(requested_workspace, run_dir):
            raise ValueError("--run-dir must be outside the workspace being preserved")
        run_dir.mkdir(parents=True)
    else:
        run_root = Path(tempfile.gettempdir()) / "mixed-pilot-agent-runs"
        run_dir = new_run_dir(run_root, "agent")
        run_dir.mkdir(parents=True)
    trace_path = Path(args.trace).resolve() if args.trace else run_dir / "trace" / "events.jsonl"
    result_path = Path(args.result).resolve() if args.result else run_dir / "result.json"
    for label, path in (("--trace", trace_path), ("--result", result_path)):
        if _inside(requested_workspace, path):
            raise ValueError("%s must be outside the workspace being preserved" % label)
    execution_workspace = run_dir / "workspace"
    for label, path in (("--trace", trace_path), ("--result", result_path)):
        if _inside(execution_workspace, path):
            raise ValueError("%s must be outside the disposable execution workspace" % label)
    task_id = uuid.uuid4().hex
    prepare_workspace(str(requested_workspace), execution_workspace)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(
        run_dir / "request.json",
        {
            "task_id": task_id,
            "prompt_file": str(Path(args.task_file).resolve()) if args.task_file else None,
            "requested_workspace": str(requested_workspace),
            "execution_workspace": str(execution_workspace),
            "restore_workspace": True,
        },
    )
    trace = JsonlTrace(trace_path)
    try:
        runner = _make_runner(
            args,
            config,
            execution_workspace,
            trace_path,
            task_id=task_id,
            trace=trace,
        )
        result = await runner.run(prompt)
        patch_path, untracked = collect_changes(execution_workspace, run_dir / "output")
        payload = {
            **result.to_dict(),
            "task_id": task_id,
            "requested_workspace": str(requested_workspace),
            "execution_workspace": str(execution_workspace),
            "workspace_restored": True,
            "patch": patch_path,
            "untracked_files": untracked,
            "workspace_snapshot": (
                str(run_dir / "output" / "workspace")
                if (run_dir / "output" / "workspace").is_dir()
                else ""
            ),
            "untracked_dir": (
                str(run_dir / "output" / "untracked")
                if (run_dir / "output" / "untracked").is_dir()
                else ""
            ),
            "trace_jsonl": str(trace_path),
            "trace_sse": str(trace.sse_path) if trace.sse_path else "",
        }
    finally:
        if execution_workspace.exists():
            shutil.rmtree(execution_workspace)
        trace.emit(
            "workspace_restored",
            node_type="environment",
            node_id="%s:workspace" % task_id,
            parent_id=task_id,
            task_id=task_id,
            requested_workspace=str(requested_workspace),
            removed_execution_workspace=str(execution_workspace),
            workspace_snapshot=(
                str(run_dir / "output" / "workspace")
                if (run_dir / "output" / "workspace").is_dir()
                else ""
            ),
            untracked_dir=(
                str(run_dir / "output" / "untracked")
                if (run_dir / "output" / "untracked").is_dir()
                else ""
            ),
        )
    _write_json(run_dir / "result.json", payload)
    if result_path != run_dir / "result.json":
        _write_json(result_path, payload)
    print(result.text)
    return 0


# Directory names that name a layout slot rather than the task in it, so the run
# directory would read "...-workspace-1a2b3c" for every task.
_GENERIC_REPO_NAMES = {"workspace", "repo", "src", "task", "."}


def _repo_label(source: str) -> str:
    """Name a --repo run after its task directory, not the slot inside it."""
    path = Path(source.rstrip("/"))
    if path.name.lower() in _GENERIC_REPO_NAMES and path.parent.name:
        return path.parent.name
    return path.stem or "code-task"


async def _code_task(args, config: HarnessConfig) -> int:
    dataset_root = resolve_data_root(args.dataset_root)
    public_files = []
    if args.task:
        task_dir = dataset_root / "tasks" / args.task
        if not task_dir.is_dir():
            raise ValueError("unknown mixed-pilot task: %s" % args.task)
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        workspace = task_dir / "workspace"
        if not workspace.is_dir():
            raise ValueError("task workspace does not exist: %s" % workspace)
        source = str(workspace)
        prompt = (task_dir / "prompt.md").read_text(encoding="utf-8")
        label = args.task
        agent_environment = manifest.get("agent_environment") or {"kind": "process"}
        public_files = [task_dir / name for name in ("manifest.json", "prompt.md", "rubrics.json")]
        task_config = config.tasks.get(args.task, {})
        if args.rubrics:
            rubrics_path = Path(args.rubrics).resolve()
        else:
            rubric_candidate = task_dir / "rubrics.json"
            rubrics_path = rubric_candidate if rubric_candidate.is_file() else None
        reference_candidate = (
            Path(args.reference_dir).resolve() if args.reference_dir else task_dir / "reference"
        )
        reference_dir = reference_candidate if args.reference_dir or reference_candidate.is_dir() else None
    else:
        source = args.repo
        prompt = _read_prompt(args.message, args.task_file)
        label = _repo_label(source)
        agent_environment = {"kind": "process"}
        task_config = {}
        rubrics_path = Path(args.rubrics).resolve() if args.rubrics else None
        reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
        if rubrics_path and reference_dir is None:
            candidate = rubrics_path.parent / "reference"
            reference_dir = candidate if candidate.is_dir() else None
    if rubrics_path is not None and not rubrics_path.is_file():
        raise ValueError("rubrics file does not exist: %s" % rubrics_path)
    if reference_dir is not None and not reference_dir.is_dir():
        raise ValueError("reference directory does not exist: %s" % reference_dir)
    eval_command = args.eval_command or str(task_config.get("eval_command") or "")
    if not eval_command:
        raise ValueError("code-task requires --eval-command or a configured task evaluator")
    test_patch_value = args.test_patch or task_config.get("test_patch")
    test_patch = None
    if test_patch_value:
        test_patch = resolve_config_relative(config, str(test_patch_value))
        if not test_patch.is_file() and not Path(str(test_patch_value)).is_absolute():
            test_patch = (dataset_root / str(test_patch_value)).resolve()
        if not test_patch.is_file():
            raise ValueError("test patch does not exist: %s" % test_patch)
    eval_timeout = args.eval_timeout or float(task_config.get("eval_timeout", 600))
    eval_setup_command = args.eval_setup_command or str(
        task_config.get("eval_setup_command") or ""
    )
    eval_test_command = args.eval_test_command or str(task_config.get("eval_test_command") or "")
    attempt_root = Path(args.attempt_root).resolve()

    def runner_factory(workspace: Path, run_dir: Path, tracer) -> TurnRunner:
        return _make_runner(
            args,
            config,
            workspace,
            run_dir / "trace" / "events.jsonl",
            trace=tracer,
        )

    def scorer_runner_factory(workspace: Path, scorer_dir: Path) -> TurnRunner:
        return _make_runner(
            args,
            config,
            workspace,
            scorer_dir / "trace" / "events.jsonl",
            system_prompt=SCORER_SYSTEM_PROMPT,
        )

    result = await run_code_task(
        runner_factory=runner_factory,
        source=source,
        prompt=prompt,
        attempt_root=attempt_root,
        label=label,
        eval_command=eval_command,
        eval_setup_command=eval_setup_command,
        eval_test_command=eval_test_command,
        test_patch=test_patch,
        eval_timeout=eval_timeout,
        shallow=args.shallow,
        keep_eval_workspace=args.keep_eval_workspace,
        public_files=public_files,
        rubrics_path=rubrics_path,
        reference_dir=reference_dir,
        scorer_runner_factory=scorer_runner_factory,
        agent_environment=agent_environment,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    eval_passed = result.evaluation["status"] == "passed"
    rubric_passed = result.rubric["status"] == "not_requested" or (
        result.rubric["status"] == "scored" and result.rubric["passed"] is True
    )
    return 0 if eval_passed and rubric_passed else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_config_path(args.config, Path.cwd())
        config = load_config(config_path)
        if args.mode == "chat":
            return asyncio.run(_chat(args, config))
        if args.mode == "per-step-judge":
            return asyncio.run(_per_step_judge(args, config))
        if args.mode == "agent":
            return asyncio.run(_agent(args, config))
        return asyncio.run(_code_task(args, config))
    except (HarnessError, OSError, ValueError, json.JSONDecodeError) as exc:
        print("mixed-pilot: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
