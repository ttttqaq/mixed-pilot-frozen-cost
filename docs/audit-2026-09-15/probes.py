"""Offline review probes. All simulated task files live in TemporaryDirectory."""
import asyncio
import json
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mixed_pilot_harness.cli import build_parser, _profile_and_defaults
from mixed_pilot_harness.code_task import collect_changes, evaluate_once, run_code_task
from mixed_pilot_harness.config import ProviderProfile, load_config
from mixed_pilot_harness.providers import AnthropicProvider, ScriptedProvider
from mixed_pilot_harness.router.config import load_router_config
from mixed_pilot_harness.rubrics_scorer import RubricsScorer
from mixed_pilot_harness.runner import TurnRunner
from mixed_pilot_harness.runtime import TaskRuntime
from mixed_pilot_harness.tools import WorkspaceTools
from mixed_pilot_harness.types import Message, TurnLimitError


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout


async def main():
    out = {}
    with tempfile.TemporaryDirectory(prefix="mixed-pilot-review-") as raw:
        root = Path(raw)
        ws = root / "workspace"
        ws.mkdir()
        (root / "synthetic-reference.txt").write_text("SYNTHETIC_GRADER_MARKER")
        tools = WorkspaceTools(ws)
        out["shell_parent_read"] = json.loads(await tools.shell(
            {"command": "cat ../synthetic-reference.txt"}))
        (ws / "linked.txt").symlink_to(root / "synthetic-reference.txt")
        out["search_symlink_read"] = await tools.search_files({"pattern": "SYNTHETIC"})
        started = time.monotonic()
        result = await tools.shell({"command": "sleep 0.8 & wait", "timeout": 0.05})
        out["shell_timeout"] = {"requested_seconds": 0.05,
                                "elapsed_seconds": round(time.monotonic() - started, 3),
                                "result": json.loads(result)}
        result = await TurnRunner(ScriptedProvider([
            {"text": "unfinished", "stop_reason": "length"}]), ws).run("probe")
        out["length_stop_returned_as_completed"] = result.to_dict()

        class FakeAnthropic(AnthropicProvider):
            async def _post_json(self, *args):
                return {"content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5,
                                  "cache_creation_input_tokens": 100,
                                  "cache_read_input_tokens": 900}}

        provider = FakeAnthropic(ProviderProfile(kind="anthropic", model="synthetic",
                                                 api_key_optional=True))
        response = await provider.complete([Message(role="user", content="probe")], [], "")
        out["cache_usage_after_adapter"] = asdict(response.usage)
        scorer = RubricsScorer(root / "unused-rubrics.json")
        criterion = {"id": "native", "method": "native_test", "points": 70,
                     "expected_ref": "reference/native.json#/FAIL_TO_PASS"}
        out["missing_native_group"] = scorer._native_criterion(
            criterion, 70, {"status": "passed"})
        out["nan_judge"] = scorer._parse_judgment(
            '{"criteria":[{"id":"x","score":NaN}]}',
            [{"id": "x", "method": "exact_match", "points": 100}])

        source = root / "source"
        source.mkdir()
        git(source, "init", "-q")
        git(source, "config", "user.email", "review@example.invalid")
        git(source, "config", "user.name", "Offline Review")
        (source / "app.py").write_text("value = 0\nfixed = 0\n")
        (source / "check.py").write_text("from app import value, fixed\nassert value == 0\n")
        git(source, "add", ".")
        git(source, "commit", "-qm", "frozen base")
        frozen_base = git(source, "rev-parse", "HEAD").strip()
        test_patch = root / "test.patch"
        test_patch.write_text("--- /dev/null\n+++ b/official_check.py\n@@ -0,0 +1,2 @@\n+from app import fixed\n+assert fixed == 1\n")
        run_dir = root / "commit-probe"
        run_dir.mkdir()
        runtime = TaskRuntime(run_dir, str(source))
        agent = runtime.prepare_agent()
        git(agent.workspace, "config", "user.email", "review@example.invalid")
        git(agent.workspace, "config", "user.name", "Offline Review")
        (agent.workspace / "app.py").write_text("value = 9\nfixed = 1\n")
        (agent.workspace / "check.py").write_text("from app import value, fixed\nassert value == 9\n")
        git(agent.workspace, "add", ".")
        git(agent.workspace, "commit", "-qm", "agent commit with weakened regression")
        patch, untracked = collect_changes(agent.workspace, run_dir / "output")
        evaluator = runtime.clone_agent_result(agent, "evaluator", patch, untracked)
        ev = await evaluate_once(evaluator.workspace, run_dir,
                                 shlex.quote(sys.executable) + " check.py && "
                                 + shlex.quote(sys.executable) + " official_check.py",
                                 test_patch, 5,
                                 native_groups={"PASS_TO_PASS": ["check.py::regression"]})
        out["mutable_head_evaluation"] = {
            "frozen_base": frozen_base,
            "agent_head": git(agent.workspace, "rev-parse", "HEAD").strip(),
            "exported_patch_bytes": Path(patch).stat().st_size,
            "evaluation_status": ev.status,
            "evaluator_checks": (evaluator.workspace / "check.py").read_text(),
            "app": (evaluator.workspace / "app.py").read_text(),
            "frozen_regression_expected": "assert value == 0"}

        def factory(workspace, run_dir, tracer):
            return TurnRunner(ScriptedProvider([{
                "tool_calls": [{"id": "x", "name": "write_file",
                                "arguments": {"path": "app.py", "content": "value = 0\nfixed = 1\n"}}],
                "usage": {"input_tokens": 17, "output_tokens": 3}}]),
                workspace, max_steps=1, trace=tracer)
        try:
            await run_code_task(factory, str(source), "probe", root / "budget", "budget",
                                "true")
        except TurnLimitError:
            failure_file = next((root / "budget").glob("*/result.json"))
            failure = json.loads(failure_file.read_text())
            out["step_limit_failure"] = {"turn": failure["turn"],
                                         "evaluation": failure["evaluation"],
                                         "patch": failure["patch"],
                                         "saved_output_files": [p.name for p in (failure_file.parent / "output").iterdir()]}

    config = load_config(ROOT / "harness.json")
    args = build_parser().parse_args(["agent", "--workspace", ".", "--profile", "router"])
    profile, _, _ = _profile_and_defaults(args, config)
    router = load_router_config(ROOT / profile.router_config, config.profiles)
    out["timeout_resolution"] = {"outer_router_timeout": profile.timeout,
                                  "actual_tier_timeouts": {k: v.timeout for k, v in router.tiers.items()}}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
