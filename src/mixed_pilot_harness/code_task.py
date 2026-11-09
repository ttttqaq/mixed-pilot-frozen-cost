"""Single-turn code-task lifecycle.

Adapted from OpenSquilla's fresh-clone code-task workspace pattern, with the
verify-to-fix retry loop deliberately removed. Evaluation runs once in a
throwaway clone so test patches and evaluator side effects never alter the
agent's resulting workspace.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import shlex
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .rubrics_scorer import RubricScore, RubricsScorer
from .runner import TurnRunner
from .runtime import (
    TaskRuntime,
    copy_untracked as _copy_untracked,
    is_git_repo as _is_git_repo,
    prepare_workspace,
    isolated_process_env,
    run_command as _run,
)
from .types import WorkspaceError


@dataclass
class EvalResult:
    status: str
    returncode: Optional[int]
    command: str
    stdout_path: str = ""
    stderr_path: str = ""
    phase: str = "evaluation"
    error: str = ""
    test_patch_status: str = "not_requested"
    setup_command: str = ""
    setup_stdout_path: str = ""
    setup_stderr_path: str = ""
    restored_test_files: List[str] = field(default_factory=list)
    native_checks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeTaskResult:
    run_dir: str
    workspace: str
    turn: Dict[str, Any]
    evaluation: Dict[str, Any]
    rubric: Dict[str, Any]
    patch: str
    untracked_files: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_changes(workspace: Path, output_dir: Path) -> tuple:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not _is_git_repo(workspace):
        snapshot = output_dir / "workspace"
        shutil.copytree(workspace, snapshot, symlinks=True)
        files = sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file() or path.is_symlink()
        )
        (output_dir / "untracked.json").write_text(
            json.dumps(files, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return "", files
    diff = _run(["git", "-C", str(workspace), "diff", "--binary", "HEAD"])
    if diff.returncode != 0:
        raise WorkspaceError("git diff failed: %s" % diff.stderr)
    patch_path = output_dir / "change.patch"
    patch_path.write_text(diff.stdout, encoding="utf-8")
    untracked_result = _run(
        ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard", "-z"]
    )
    untracked = [item for item in untracked_result.stdout.split("\0") if item]
    (output_dir / "untracked.json").write_text(
        json.dumps(untracked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _copy_untracked(workspace, output_dir / "untracked", untracked)
    return str(patch_path), untracked


def _isolated_eval_env() -> Dict[str, str]:
    """Return an environment that does not inherit the agent's Python environment."""
    return isolated_process_env()


def _patch_target_files(patch: Path) -> List[str]:
    """Files a unified diff writes to, as workspace-relative posix paths."""
    files = []
    for line in patch.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ b/"):
            candidate = line[6:].strip()
            if candidate and candidate != "/dev/null":
                files.append(candidate)
    return files


def _test_id_files(test_ids: Sequence[str]) -> List[str]:
    """The file part of pytest node ids such as tests/test_x.py::test_y[a-b]."""
    return [item.split("::", 1)[0] for item in test_ids if "::" in item]


def _is_inside(workspace: Path, relative: str) -> bool:
    try:
        (workspace / relative).resolve().relative_to(workspace.resolve())
        return True
    except (OSError, ValueError):
        return False


def _restore_test_files(workspace: Path, candidates: Sequence[str]) -> List[str]:
    """Discard agent edits to evaluator-owned test files, as SWE-bench does.

    The upstream eval script runs `git checkout <base> -- <test files>` before
    applying the official test patch, so a submission cannot shadow, weaken, or
    delete the tests that grade it.
    """
    wanted = sorted({item for item in candidates if item and _is_inside(workspace, item)})
    restored = []
    for relative in wanted:
        tracked = _run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=workspace)
        if tracked.returncode != 0:
            continue
        checkout = _run(["git", "checkout", "HEAD", "--", relative], cwd=workspace)
        if checkout.returncode != 0:
            raise WorkspaceError(
                "could not restore evaluator test file %s: %s" % (relative, checkout.stderr)
            )
        restored.append(relative)
    return restored


def resolve_native_groups(
    rubrics_path: Optional[Path], reference_dir: Optional[Path]
) -> Dict[str, List[str]]:
    """Read the test id lists that native_test criteria declare via expected_ref.

    An expected_ref looks like "reference/native_record.json#/FAIL_TO_PASS" and is
    resolved relative to the rubrics file. Without this the harness would never
    read the lists it claims to verify.
    """
    if rubrics_path is None or not rubrics_path.is_file():
        return {}
    try:
        rubric = json.loads(rubrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    roots = [rubrics_path.parent]
    if reference_dir is not None:
        roots.append(reference_dir.parent)
    groups: Dict[str, List[str]] = {}
    for criterion in rubric.get("criteria") or []:
        if criterion.get("method") != "native_test":
            continue
        raw = str(criterion.get("expected_ref") or "")
        if "#" not in raw:
            continue
        relative, _, pointer = raw.partition("#")
        key = pointer.lstrip("/").strip()
        if not relative or not key or key in groups:
            continue
        for root in roots:
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            try:
                record = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            values = record.get(key) if isinstance(record, dict) else None
            if isinstance(values, list) and values:
                groups[key] = [str(item) for item in values]
            break
    return groups


def _write_eval_streams(run_dir: Path, stdout: bytes = b"", stderr: bytes = b"") -> tuple:
    stdout_path = run_dir / "eval.stdout.txt"
    stderr_path = run_dir / "eval.stderr.txt"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    return str(stdout_path), str(stderr_path)


async def _run_shell(command: str, cwd: Path, env: Dict[str, str], timeout: float) -> tuple:
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return process.returncode, stdout, stderr
    except asyncio.TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        return None, stdout, stderr


async def _run_native_checks(
    groups: Dict[str, List[str]],
    run_dir: Path,
    workspace: Path,
    env: Dict[str, str],
    timeout: float,
    template: str,
) -> Dict[str, Any]:
    """Run each declared test id list on its own so criteria are graded separately.

    Without this, every native_test criterion collapses onto the single exit code
    of the broad eval command and cannot tell one test group from another.
    """
    if not groups:
        return {}
    if not template or "{tests}" not in template:
        return {
            key: {
                "status": "not_configured",
                "test_count": len(ids),
                "error": "no eval_test_command template with a {tests} placeholder",
            }
            for key, ids in groups.items()
        }
    checks: Dict[str, Any] = {}
    for key in sorted(groups):
        ids = groups[key]
        rendered = template.replace("{tests}", " ".join(shlex.quote(item) for item in ids))
        returncode, stdout, stderr = await _run_shell(rendered, workspace, env, timeout)
        slug = "".join(char if char.isalnum() or char in "-_" else "-" for char in key.lower())
        stdout_file = run_dir / ("eval.native.%s.stdout.txt" % slug)
        stderr_file = run_dir / ("eval.native.%s.stderr.txt" % slug)
        stdout_file.write_bytes(stdout)
        stderr_file.write_bytes(stderr)
        checks[key] = {
            "status": (
                "timeout"
                if returncode is None
                else ("passed" if returncode == 0 else "failed")
            ),
            "returncode": returncode,
            "command": rendered,
            "test_count": len(ids),
            "stdout_path": str(stdout_file),
            "stderr_path": str(stderr_file),
        }
    return checks


async def evaluate_once(
    workspace: Path,
    run_dir: Path,
    command: str,
    test_patch: Optional[Path],
    timeout: float,
    setup_command: str = "",
    native_groups: Optional[Dict[str, List[str]]] = None,
    test_command_template: str = "",
) -> EvalResult:
    if not command:
        return EvalResult(status="skipped", returncode=None, command="")
    eval_workspace = workspace
    phase = "test_patch"
    test_patch_status = "not_requested"
    setup_stdout_path = ""
    setup_stderr_path = ""
    groups = {key: list(value) for key, value in (native_groups or {}).items() if value}
    restored: List[str] = []
    try:
        if test_patch is not None:
            owned = list(_patch_target_files(test_patch))
            for ids in groups.values():
                owned.extend(_test_id_files(ids))
            restored = _restore_test_files(eval_workspace, owned)
            apply_args = [
                "git",
                "apply",
                "--recount",
                "--whitespace=nowarn",
                str(test_patch.resolve()),
            ]
            apply_test = _run(apply_args, cwd=eval_workspace)
            if apply_test.returncode == 0:
                test_patch_status = "applied"
            else:
                reverse_check = _run(
                    [
                        "git",
                        "apply",
                        "--reverse",
                        "--check",
                        "--recount",
                        "--whitespace=nowarn",
                        str(test_patch.resolve()),
                    ],
                    cwd=eval_workspace,
                )
                if reverse_check.returncode == 0:
                    test_patch_status = "already_present"
                else:
                    raise WorkspaceError(
                        "could not apply evaluator test patch: %s" % apply_test.stderr
                    )

        eval_env = _isolated_eval_env()
        if setup_command:
            phase = "setup"
            setup_returncode, setup_stdout, setup_stderr = await _run_shell(
                setup_command, eval_workspace, eval_env, timeout
            )
            setup_stdout_file = run_dir / "eval.setup.stdout.txt"
            setup_stderr_file = run_dir / "eval.setup.stderr.txt"
            setup_stdout_file.write_bytes(setup_stdout)
            setup_stderr_file.write_bytes(setup_stderr)
            setup_stdout_path = str(setup_stdout_file)
            setup_stderr_path = str(setup_stderr_file)
            if setup_returncode is None:
                stdout_path, stderr_path = _write_eval_streams(
                    run_dir, stderr=b"evaluator setup timed out\n"
                )
                return EvalResult(
                    status="error",
                    returncode=None,
                    command=command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    phase=phase,
                    error="evaluator setup timed out",
                    test_patch_status=test_patch_status,
                    setup_command=setup_command,
                    setup_stdout_path=setup_stdout_path,
                    setup_stderr_path=setup_stderr_path,
                )
            if setup_returncode != 0:
                message = "evaluator setup failed with exit code %s" % setup_returncode
                stdout_path, stderr_path = _write_eval_streams(
                    run_dir, stderr=(message + "\n").encode()
                )
                return EvalResult(
                    status="error",
                    returncode=setup_returncode,
                    command=command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    phase=phase,
                    error=message,
                    test_patch_status=test_patch_status,
                    setup_command=setup_command,
                    setup_stdout_path=setup_stdout_path,
                    setup_stderr_path=setup_stderr_path,
                )

        phase = "evaluation"
        returncode, stdout, stderr = await _run_shell(command, eval_workspace, eval_env, timeout)
        status = "timeout" if returncode is None else ("passed" if returncode == 0 else "failed")
        stdout_path, stderr_path = _write_eval_streams(run_dir, stdout, stderr)
        native_checks = await _run_native_checks(
            groups, run_dir, eval_workspace, eval_env, timeout, test_command_template
        )
        return EvalResult(
            status=status,
            returncode=returncode,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            phase=phase,
            test_patch_status=test_patch_status,
            setup_command=setup_command,
            setup_stdout_path=setup_stdout_path,
            setup_stderr_path=setup_stderr_path,
            restored_test_files=restored,
            native_checks=native_checks,
        )
    except (OSError, WorkspaceError) as exc:
        message = str(exc)
        stdout_path, stderr_path = _write_eval_streams(
            run_dir, stderr=(message + "\n").encode("utf-8")
        )
        return EvalResult(
            status="error",
            returncode=None,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            phase=phase,
            error=message,
            test_patch_status=test_patch_status,
            setup_command=setup_command,
            setup_stdout_path=setup_stdout_path,
            setup_stderr_path=setup_stderr_path,
            restored_test_files=restored,
        )


def new_run_dir(root: Path, label: str) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(char if char.isalnum() or char in "-_" else "-" for char in label)
    return (root / ("%s-%s-%s" % (stamp, safe_label, uuid.uuid4().hex[:8]))).resolve()


async def run_code_task(
    runner_factory,
    source: str,
    prompt: str,
    attempt_root: Path,
    label: str,
    eval_command: str,
    test_patch: Optional[Path] = None,
    eval_setup_command: str = "",
    eval_test_command: str = "",
    eval_timeout: float = 600,
    shallow: bool = False,
    keep_eval_workspace: bool = False,
    public_files: Optional[List[Path]] = None,
    rubrics_path: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    scorer_runner_factory=None,
    agent_environment: Optional[Dict[str, str]] = None,
) -> CodeTaskResult:
    run_dir = new_run_dir(attempt_root, label)
    output = run_dir / "output"
    run_dir.mkdir(parents=True)
    output.mkdir()
    runtime = TaskRuntime(run_dir, source, shallow=shallow)
    workspace = runtime.environment("agent").workspace
    for path in public_files or []:
        if path.exists():
            shutil.copy2(path, run_dir / path.name)
    (run_dir / "request.json").write_text(
        json.dumps(
            {
                "label": label,
                "prompt": prompt,
                "workspace": str(workspace),
                "environment": "agent",
                "agent_environment": agent_environment or {"kind": "process"},
                "rubrics": str(rubrics_path) if rubrics_path else None,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        agent_runtime = runtime.prepare_agent(agent_environment)
    except Exception as exc:
        runtime.tracer.runtime_failed("agent_setup", exc)
        failure = {
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "status": "error",
            "phase": "agent_setup",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "turn": {
                "status": "not_run",
                "steps": 0,
                "provider_calls": 0,
                "tool_calls": 0,
            },
            "evaluation": {"status": "not_run"},
            "rubric": {"status": "not_run"},
            "patch": "",
            "untracked_files": [],
        }
        (run_dir / "result.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise
    try:
        runner: TurnRunner = runner_factory(workspace, run_dir, runtime.tracer)
        runner.set_shell_env(agent_runtime.shell_env())
        runtime.tracer.environment_created(agent_runtime)
        turn = await runner.run(prompt)
    except Exception as exc:
        progress = runtime.tracer.progress()
        runtime.tracer.runtime_failed("agent", exc)
        failure = {
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "status": "error",
            "phase": "agent",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "turn": {"status": "error", **progress},
            "evaluation": {"status": "not_run"},
            "rubric": {"status": "not_run"},
            "patch": "",
            "untracked_files": [],
        }
        (run_dir / "result.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise
    patch_path, untracked = collect_changes(workspace, output)
    eval_environment = None
    try:
        eval_environment = runtime.clone_agent_result(
            agent_runtime, "evaluator", patch_path, untracked
        )
        runtime.tracer.environment_created(eval_environment)
        evaluation = await evaluate_once(
            workspace=eval_environment.workspace,
            run_dir=run_dir,
            command=eval_command,
            test_patch=test_patch,
            timeout=eval_timeout,
            setup_command=eval_setup_command,
            native_groups=resolve_native_groups(rubrics_path, reference_dir),
            test_command_template=eval_test_command,
        )
    except (OSError, WorkspaceError) as exc:
        stdout_path, stderr_path = _write_eval_streams(
            run_dir, stderr=(str(exc) + "\n").encode("utf-8")
        )
        evaluation = EvalResult(
            status="error",
            returncode=None,
            command=eval_command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            phase="workspace_prepare",
            error=str(exc),
        )
    finally:
        if eval_environment and not keep_eval_workspace:
            eval_environment.cleanup()

    rubric = RubricScore(status="not_requested").to_dict()
    if rubrics_path is not None:
        scorer_environment = None
        try:
            scorer_environment = runtime.clone_agent_result(
                agent_runtime, "scorer", patch_path, untracked
            )
            runtime.tracer.environment_created(scorer_environment)
            rubric = (
                await RubricsScorer(
                    rubrics_path=rubrics_path,
                    reference_dir=reference_dir,
                    runner_factory=scorer_runner_factory,
                ).score(
                    scorer_environment,
                    run_dir,
                    asdict(evaluation),
                    agent_response=turn.text,
                )
            ).to_dict()
        except Exception as exc:
            rubric = RubricScore(status="error", error=str(exc)).to_dict()
        finally:
            if scorer_environment:
                scorer_environment.cleanup()
        (run_dir / "rubric_result.json").write_text(
            json.dumps(rubric, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    runtime.tracer.runtime_completed(evaluation.status, rubric["status"])
    result = CodeTaskResult(
        run_dir=str(run_dir),
        workspace=str(workspace),
        turn=turn.to_dict(),
        evaluation=asdict(evaluation),
        rubric=rubric,
        patch=patch_path,
        untracked_files=untracked,
    )
    (run_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
