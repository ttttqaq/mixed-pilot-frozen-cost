from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import ssl
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from mixed_pilot_harness.cli import main
from mixed_pilot_harness.code_task import resolve_native_groups, run_code_task
from mixed_pilot_harness.config import ProviderProfile
from mixed_pilot_harness.providers import JsonHttpProvider, ScriptedProvider
from mixed_pilot_harness.runner import TurnRunner
from mixed_pilot_harness.trace import JsonlTrace
from mixed_pilot_harness.types import (
    ProviderError,
    ProviderResponse,
    ToolCall,
    TurnResult,
    Usage,
    WorkspaceError,
)


class TurnRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_turn_can_contain_tool_steps(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trace_path = root / "events.jsonl"
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "name": "write_file",
                                "arguments": {"path": "answer.txt", "content": "done\n"},
                            }
                        ]
                    },
                    {"text": "finished", "usage": {"input_tokens": 4, "output_tokens": 2}},
                ]
            )
            result = await TurnRunner(
                provider,
                root,
                trace=JsonlTrace(trace_path),
            ).run("write the answer")
            self.assertEqual(result.text, "finished")
            self.assertEqual(result.steps, 2)
            self.assertEqual(result.provider_calls, 2)
            self.assertEqual(result.tool_calls, 1)
            self.assertEqual((root / "answer.txt").read_text(), "done\n")
            events = [json.loads(line)["event"] for line in trace_path.read_text().splitlines()]
            self.assertIn("tool_use", events)
            self.assertIn("tool_result", events)
            self.assertEqual(events[-1], "task_completed")
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            task = next(item for item in records if item["event"] == "task_started")
            turn = next(item for item in records if item["event"] == "turn_started")
            leg = next(item for item in records if item["event"] == "model_leg_started")
            iteration = next(item for item in records if item["event"] == "iteration_started")
            tool = next(item for item in records if item["event"] == "tool_use")
            self.assertEqual(turn["parent_id"], task["node_id"])
            self.assertEqual(leg["parent_id"], turn["node_id"])
            self.assertEqual(iteration["parent_id"], leg["node_id"])
            self.assertEqual(tool["parent_id"], iteration["node_id"])
            sse = trace_path.with_suffix(".sse").read_text()
            self.assertIn("event: tool_use\n", sse)
            self.assertIn("data: {", sse)

    async def test_file_tool_rejects_workspace_escape(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {"id": "x", "name": "read_file", "arguments": {"path": "../secret"}}
                        ]
                    },
                    {"text": "escape rejected"},
                ]
            )
            result = await TurnRunner(provider, root).run("try")
            self.assertEqual(result.text, "escape rejected")


class ProviderRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_tls_eof_is_retried_twice(self):
        class FlakyProvider(JsonHttpProvider):
            calls = 0

            def _post_json_sync(self, url, payload, headers):
                self.calls += 1
                if self.calls < 3:
                    try:
                        raise ssl.SSLEOFError("EOF occurred in violation of protocol")
                    except ssl.SSLEOFError as exc:
                        raise ProviderError("provider transport error") from exc
                return {"ok": True}

        provider = FlakyProvider(ProviderProfile(kind="test", model="test"))
        with mock.patch(
            "mixed_pilot_harness.providers.asyncio.sleep", new=mock.AsyncMock()
        ) as sleep:
            result = await provider._post_json("https://example.test", {}, {})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(provider.calls, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_http_error_is_not_retried(self):
        class HttpFailureProvider(JsonHttpProvider):
            def __init__(self, profile, status):
                super().__init__(profile)
                self.status = status
                self.calls = 0

            def _post_json_sync(self, url, payload, headers):
                self.calls += 1
                error = urllib.error.HTTPError(url, self.status, "rejected", {}, None)
                raise ProviderError(
                    "provider HTTP %s: temporarily unavailable" % self.status
                ) from error

        for status in (401, 403):
            with self.subTest(status=status):
                provider = HttpFailureProvider(
                    ProviderProfile(kind="test", model="test"), status
                )
                with self.assertRaisesRegex(ProviderError, "HTTP %s" % status):
                    await provider._post_json("https://example.test", {}, {})
                self.assertEqual(provider.calls, 1)

    async def test_transient_http_status_is_retried(self):
        """524 is a Cloudflare origin timeout, not a permanent rejection."""

        class TransientProvider(JsonHttpProvider):
            def __init__(self, profile, status, fail_times):
                super().__init__(profile)
                self.status = status
                self.fail_times = fail_times
                self.calls = 0

            def _post_json_sync(self, url, payload, headers):
                self.calls += 1
                if self.calls <= self.fail_times:
                    error = urllib.error.HTTPError(url, self.status, "upstream", {}, None)
                    raise ProviderError("provider HTTP %s" % self.status) from error
                return {"ok": True}

        profile = ProviderProfile(kind="test", model="test")
        for status in (408, 429, 500, 502, 503, 504, 520, 524):
            with self.subTest(status=status):
                provider = TransientProvider(profile, status, fail_times=2)
                with mock.patch(
                    "mixed_pilot_harness.providers.asyncio.sleep", new=mock.AsyncMock()
                ):
                    result = await provider._post_json("https://example.test", {}, {})
                self.assertEqual(result, {"ok": True})
                self.assertEqual(provider.calls, 3)

        # Exhausting the budget still surfaces the error instead of hanging.
        provider = TransientProvider(profile, 524, fail_times=99)
        with mock.patch("mixed_pilot_harness.providers.asyncio.sleep", new=mock.AsyncMock()):
            with self.assertRaisesRegex(ProviderError, "HTTP 524"):
                await provider._post_json("https://example.test", {}, {})
        self.assertEqual(provider.calls, 3)


class CliTests(unittest.TestCase):
    def test_agent_cli_with_scripted_profile(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "workspace"
            workspace.mkdir()
            responses = root / "responses.json"
            responses.write_text(
                json.dumps(
                    [
                        {
                            "tool_calls": [
                                {
                                    "id": "write",
                                    "name": "write_file",
                                    "arguments": {"path": "cli.txt", "content": "ok\n"},
                                }
                            ]
                        },
                        {"text": "cli complete"},
                    ]
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "default_profile": "offline",
                        "profiles": {
                            "offline": {
                                "kind": "scripted",
                                "model": "test",
                                "responses_file": str(responses),
                            }
                        },
                    }
                )
            )
            env = os.environ.copy()
            source_root = str(Path(__file__).resolve().parents[1] / "src")
            env["PYTHONPATH"] = source_root
            env["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mixed_pilot_harness.cli",
                    "agent",
                    "--config",
                    str(config),
                    "--workspace",
                    str(workspace),
                    "--message",
                    "write it",
                    "--trace",
                    str(root / "events.jsonl"),
                    "--result",
                    str(root / "result.json"),
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout.strip(), "cli complete")
            self.assertEqual((workspace / "cli.txt").read_text(), "ok\n")
            self.assertEqual(json.loads((root / "result.json").read_text())["steps"], 2)

    def test_agent_restore_workspace_uses_prompt_file_and_keeps_full_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "source"
            workspace.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace, check=True)
            subprocess.run(["git", "config", "user.name", "Harness Test"], cwd=workspace, check=True)
            (workspace / "value.txt").write_text("before\n")
            subprocess.run(["git", "add", "value.txt"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True)
            prompt = root / "task.txt"
            prompt.write_text("Change value.txt to after.\n")
            responses = root / "responses.json"
            responses.write_text(
                json.dumps(
                    [
                        {
                            "tool_calls": [
                                {
                                    "id": "edit",
                                    "name": "replace_in_file",
                                    "arguments": {
                                        "path": "value.txt",
                                        "old": "before\n",
                                        "new": "after\n",
                                    },
                                },
                                {
                                    "id": "deliverable",
                                    "name": "write_file",
                                    "arguments": {
                                        "path": "deliverable.txt",
                                        "content": "preserved\n",
                                    },
                                },
                            ]
                        },
                        {"text": "restored run complete"},
                    ]
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "default_profile": "offline",
                        "profiles": {
                            "offline": {
                                "kind": "scripted",
                                "model": "test",
                                "responses_file": str(responses),
                            }
                        },
                    }
                )
            )
            run_dir = root / "agent-run"
            env = os.environ.copy()
            source_root = str(Path(__file__).resolve().parents[1] / "src")
            env["PYTHONPATH"] = source_root
            env["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mixed_pilot_harness.cli",
                    "agent",
                    "--config",
                    str(config),
                    "--workspace",
                    str(workspace),
                    "--task-file",
                    str(prompt),
                    "--restore-workspace",
                    "--run-dir",
                    str(run_dir),
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual((workspace / "value.txt").read_text(), "before\n")
            self.assertFalse((run_dir / "workspace").exists())
            result = json.loads((run_dir / "result.json").read_text())
            self.assertTrue(result["workspace_restored"])
            self.assertTrue(Path(result["patch"]).read_text())
            self.assertEqual(
                (Path(result["untracked_dir"]) / "deliverable.txt").read_text(),
                "preserved\n",
            )
            self.assertEqual((run_dir / "prompt.txt").read_text(), prompt.read_text())
            records = [
                json.loads(line)
                for line in (run_dir / "trace/events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[0]["event"], "task_started")
            self.assertEqual(records[-1]["event"], "workspace_restored")
            self.assertEqual(records[0]["prompt"], prompt.read_text())
            sse = (run_dir / "trace/events.sse").read_text()
            self.assertIn("event: provider_call_started", sse)
            self.assertIn("event: tool_result", sse)
            self.assertIn("event: workspace_restored", sse)

    def test_agent_restore_non_git_workspace_keeps_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Harness Test"], cwd=root, check=True)
            parent_file = root / "parent.txt"
            parent_file.write_text("before\n")
            subprocess.run(["git", "add", "parent.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            parent_file.write_text("unrelated parent change\n")

            workspace = root / "source"
            workspace.mkdir()
            (workspace / "input.txt").write_text("source data\n")
            responses = root / "responses.json"
            responses.write_text(
                json.dumps(
                    [
                        {
                            "tool_calls": [
                                {
                                    "id": "write",
                                    "name": "write_file",
                                    "arguments": {
                                        "path": "deliverables/result.txt",
                                        "content": "completed\n",
                                    },
                                }
                            ]
                        },
                        {"text": "non-git complete"},
                    ]
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "default_profile": "offline",
                        "profiles": {
                            "offline": {
                                "kind": "scripted",
                                "model": "test",
                                "responses_file": str(responses),
                            }
                        },
                    }
                )
            )
            run_dir = root / "run" / "non-git"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            env["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mixed_pilot_harness.cli",
                    "agent",
                    "--config",
                    str(config),
                    "--workspace",
                    str(workspace),
                    "--message",
                    "write the result",
                    "--restore-workspace",
                    "--run-dir",
                    str(run_dir),
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse((workspace / "deliverables").exists())
            self.assertFalse((run_dir / "workspace").exists())
            result = json.loads((run_dir / "result.json").read_text())
            snapshot = Path(result["workspace_snapshot"])
            self.assertEqual((snapshot / "input.txt").read_text(), "source data\n")
            self.assertEqual(
                (snapshot / "deliverables/result.txt").read_text(),
                "completed\n",
            )
            self.assertEqual(result["patch"], "")
            self.assertFalse((run_dir / "output/change.patch").exists())
            self.assertIn("deliverables/result.txt", result["untracked_files"])
            self.assertEqual(parent_file.read_text(), "unrelated parent change\n")
            records = [
                json.loads(line)
                for line in (run_dir / "trace/events.jsonl").read_text().splitlines()
            ]
            restored = records[-1]
            self.assertEqual(restored["event"], "workspace_restored")
            self.assertEqual(Path(restored["workspace_snapshot"]), snapshot)

    def test_bundled_code_task_clones_runs_and_evaluates_once(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            project_root = Path(__file__).resolve().parents[1]
            source_data_root = Path(
                os.environ.get(
                    "MIXED_PILOT_DATA_ROOT",
                    str(Path.home() / "Data" / "mixed-pilot-data"),
                )
            ).expanduser().resolve()
            source_task = source_data_root / "tasks" / "swebench-flask-5014"
            data_root = root / "data"
            task_dir = data_root / "tasks" / "swebench-flask-5014"
            task_dir.mkdir(parents=True)
            for name in ("workspace", "prompt.md", "rubrics.json", "reference"):
                (task_dir / name).symlink_to(source_task / name)
            manifest = json.loads((source_task / "manifest.json").read_text())
            manifest["agent_environment"]["setup_command"] = (
                "python -c \"import sys; print('agent setup', sys.prefix)\""
            )
            (task_dir / "manifest.json").write_text(json.dumps(manifest))
            responses = root / "responses.json"
            old = "        if \".\" in name:\n"
            new = (
                "        if not name:\n"
                "            raise ValueError(\"'name' may not be empty.\")\n"
                "        if \".\" in name:\n"
            )
            responses.write_text(
                json.dumps(
                    [
                        {
                            "tool_calls": [
                                {
                                    "id": "edit",
                                    "name": "replace_in_file",
                                    "arguments": {
                                        "path": "src/flask/blueprints.py",
                                        "old": old,
                                        "new": new,
                                    },
                                }
                            ]
                        },
                        {"text": "implemented"},
                    ]
                )
            )
            check_code = (
                "from pathlib import Path; "
                "assert 'if not name:' in Path('src/flask/blueprints.py').read_text(); "
                "assert 'test_empty_name_not_allowed' in Path('tests/test_blueprints.py').read_text()"
            )
            eval_command = "%s -c %s" % (shlex.quote(sys.executable), shlex.quote(check_code))
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "default_profile": "offline",
                        "profiles": {
                            "offline": {
                                "kind": "scripted",
                                "model": "test",
                                "responses_file": str(responses),
                            }
                        },
                        "tasks": {
                            "swebench-flask-5014": {
                                "eval_command": eval_command,
                                "test_patch": str(
                                    data_root
                                    / "tasks/swebench-flask-5014/reference/test.patch"
                                ),
                            }
                        },
                    }
                )
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root / "src")
            env["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mixed_pilot_harness.cli",
                    "code-task",
                    "--config",
                    str(config),
                    "--task",
                    "swebench-flask-5014",
                    "--dataset-root",
                    str(data_root),
                    "--attempt-root",
                    str(root / "attempts"),
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            result = json.loads(process.stdout)
            self.assertEqual(result["evaluation"]["status"], "passed")
            self.assertEqual(result["rubric"]["status"], "scored")
            self.assertEqual(result["rubric"]["score"], 100)
            self.assertEqual(result["turn"]["provider_calls"], 2)
            self.assertIn(
                "agent setup",
                (Path(result["run_dir"]) / "agent.setup.stdout.txt").read_text(),
            )
            workspace = Path(result["workspace"])
            self.assertEqual(workspace.parent.name, "agent")
            self.assertFalse((workspace / "reference").exists())
            environments = Path(result["run_dir"]) / "environments"
            self.assertFalse((environments / "evaluator").exists())
            self.assertFalse((environments / "scorer").exists())


class AnyCategoryTaskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "tasks" / "general-task"
        (self.task / "workspace").mkdir(parents=True)
        (self.task / "reference").mkdir()
        (self.task / "reference" / "gold.txt").write_text("answer")
        (self.task / "manifest.json").write_text(json.dumps({
            "category": "office_spreadsheet_document",
            "agent_environment": {"kind": "process", "setup_command": "printf ready > setup.txt"},
        }))
        (self.task / "prompt.md").write_text("Create the requested answer.")
        (self.task / "rubrics.json").write_text(json.dumps({
            "max_score": 100, "pass_threshold": 80,
            "criteria": [{"id": "answer", "method": "artifact_rubric", "points": 100}],
        }))
        self.config = self.root / "harness.json"
        self.config.write_text(json.dumps({
            "default_profile": "offline",
            "profiles": {"offline": {"kind": "scripted", "model": "test"}},
            "tasks": {"general-task": {"eval_command": "test -s answer.txt && test -s setup.txt"}},
        }))

    def invoke(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["code-task", "--config", str(self.config), "--task", "general-task",
                         "--dataset-root", str(self.root), "--attempt-root", str(self.root / "runs"),
                         *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_non_coding_task_runs_agent_eval_and_score_and_preserves_exit_status(self):
        for eval_ok, score in ((True, 100), (False, 100), (True, 0)):
            with self.subTest(eval_ok=eval_ok, score=score):
                agent = ScriptedProvider([
                    {"tool_calls": [{"id": "write", "name": "write_file",
                                     "arguments": {"path": "answer.txt", "content": "answer"}}]},
                    {"text": "done"},
                ])
                scorer = ScriptedProvider([
                    {"tool_calls": [{"id": "reference", "name": "read_file",
                                     "arguments": {"path": ".mixed-pilot-scorer/reference/gold.txt"}}]},
                    {"text": json.dumps({"criteria": [{"id": "answer", "score": score}]})},
                ])
                with mock.patch("mixed_pilot_harness.cli.create_provider", side_effect=[agent, scorer]):
                    code, stdout, stderr = self.invoke(*([] if eval_ok else ["--eval-command", "false"]))
                self.assertEqual(code, 0 if eval_ok and score == 100 else 1, stderr)
                result = json.loads(stdout)
                self.assertEqual(result["evaluation"]["status"], "passed" if eval_ok else "failed")
                self.assertEqual(result["rubric"]["status"], "scored")
                self.assertEqual(result["rubric"]["score"], score)
                self.assertEqual((agent.calls, scorer.calls), (2, 2))
                self.assertEqual((Path(result["workspace"]) / "answer.txt").read_text(), "answer")
                events = [json.loads(line) for line in
                          (Path(result["run_dir"]) / "scorer/trace/events.jsonl").read_text().splitlines()]
                self.assertTrue(any(e["event"] == "tool_result" and e["output"] == "answer" for e in events))
                self.assertFalse((self.task / "workspace" / "answer.txt").exists())
                self.assertFalse((self.task / "workspace" / "setup.txt").exists())
                self.assertFalse((Path(result["workspace"]) / "reference").exists())

    def test_invalid_task_inputs_fail_before_agent_starts(self):
        with mock.patch("mixed_pilot_harness.cli.create_provider") as factory:
            code, _, stderr = self.invoke("--reference-dir", str(self.root / "missing"))
            self.assertEqual(code, 2)
            self.assertIn("reference directory does not exist", stderr)
            config = json.loads(self.config.read_text())
            config["tasks"] = {}
            self.config.write_text(json.dumps(config))
            code, _, stderr = self.invoke()
            self.assertEqual(code, 2)
            self.assertIn("requires --eval-command", stderr)
            (self.task / "workspace").rmdir()
            code, _, stderr = self.invoke("--eval-command", "true")
            self.assertEqual(code, 2)
            self.assertIn("task workspace does not exist", stderr)
            factory.assert_not_called()
        self.assertFalse((self.root / "runs").exists())


class CodeTaskTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _git_source(root: Path, filename: str, content: str) -> Path:
        source = root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=source, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
        subprocess.run(["git", "config", "user.name", "Harness Test"], cwd=source, check=True)
        (source / filename).write_text(content)
        subprocess.run(["git", "add", filename], cwd=source, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=source, check=True)
        return source

    async def test_git_agent_patch_is_applied_in_one_shot_evaluation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=source, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
            subprocess.run(["git", "config", "user.name", "Harness Test"], cwd=source, check=True)
            (source / "value.txt").write_text("before\n")
            subprocess.run(["git", "add", "value.txt"], cwd=source, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=source, check=True)
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "replace",
                                "name": "replace_in_file",
                                "arguments": {
                                    "path": "value.txt",
                                    "old": "before\n",
                                    "new": "after\n",
                                },
                            }
                        ]
                    },
                    {"text": "implemented"},
                ]
            )

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            check_code = "from pathlib import Path; assert Path('value.txt').read_text() == 'after\\n'"
            command = "%s -c %s" % (shlex.quote(sys.executable), shlex.quote(check_code))
            result = await run_code_task(
                runner_factory=factory,
                source=str(source),
                prompt="change it",
                attempt_root=root / "attempts",
                label="git-demo",
                eval_command=command,
            )
            self.assertEqual(result.evaluation["status"], "passed")
            self.assertEqual((source / "value.txt").read_text(), "before\n")
            self.assertEqual((Path(result.workspace) / "value.txt").read_text(), "after\n")
            self.assertTrue(Path(result.patch).read_text())

    async def test_agent_shell_uses_task_venv(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            probe = (
                "import json, os, shutil, sys; from pathlib import Path; "
                "Path('agent-env.json').write_text(json.dumps({"
                "'prefix': sys.prefix, 'venv': os.environ.get('VIRTUAL_ENV', ''), "
                "'conda': os.environ.get('CONDA_PREFIX', ''), 'pip': shutil.which('pip')}))"
            )
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "probe",
                                "name": "shell",
                                "arguments": {
                                    "command": "python -c %s" % shlex.quote(probe)
                                },
                            }
                        ]
                    },
                    {"text": "isolated"},
                ]
            )

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            result = await run_code_task(
                runner_factory=factory,
                source=str(source),
                prompt="inspect the environment",
                attempt_root=root / "attempts",
                label="agent-env",
                eval_command="test -s agent-env.json",
                agent_environment={"kind": "python_venv"},
            )
            environment = Path(result.run_dir) / "environments" / "agent"
            observed = json.loads((Path(result.workspace) / "agent-env.json").read_text())
            self.assertEqual(Path(observed["prefix"]).resolve(), (environment / "venv").resolve())
            self.assertEqual(Path(observed["venv"]).resolve(), (environment / "venv").resolve())
            self.assertEqual(observed["conda"], "")
            self.assertEqual(
                Path(observed["pip"]).resolve(), (environment / "venv" / "bin" / "pip").resolve()
            )

    async def test_eval_failure_does_not_start_a_second_turn_or_mutate_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "base.txt").write_text("base\n")
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "write",
                                "name": "write_file",
                                "arguments": {"path": "new.txt", "content": "agent\n"},
                            }
                        ]
                    },
                    {"text": "implemented"},
                ]
            )

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            marker_code = "from pathlib import Path; Path('eval-only.txt').write_text('x'); raise SystemExit(1)"
            command = "%s -c %s" % (shlex.quote(sys.executable), shlex.quote(marker_code))
            result = await run_code_task(
                runner_factory=factory,
                source=str(source),
                prompt="change it",
                attempt_root=root / "attempts",
                label="demo",
                eval_command=command,
            )
            self.assertEqual(result.evaluation["status"], "failed")
            self.assertEqual(provider.calls, 2)
            self.assertFalse((source / "new.txt").exists())
            self.assertTrue((Path(result.workspace) / "new.txt").exists())
            self.assertFalse((Path(result.workspace) / "eval-only.txt").exists())
            self.assertFalse((Path(result.run_dir) / "eval-workspace").exists())

    async def test_agent_failure_writes_result_with_completed_counts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "value.txt").write_text("base\n")

            class FailingProvider:
                provider_name = "failing"

                def __init__(self):
                    self.calls = 0

                async def complete(self, messages, tools, system):
                    self.calls += 1
                    if self.calls == 1:
                        return ProviderResponse(
                            tool_calls=[
                                ToolCall(
                                    id="read",
                                    name="read_file",
                                    arguments={"path": "value.txt"},
                                )
                            ]
                        )
                    raise ProviderError("provider transport error: reset")

            provider = FailingProvider()

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            with self.assertRaisesRegex(ProviderError, "transport error"):
                await run_code_task(
                    runner_factory=factory,
                    source=str(source),
                    prompt="read it",
                    attempt_root=root / "attempts",
                    label="agent-failure",
                    eval_command="true",
                )
            run_dir = next((root / "attempts").iterdir())
            saved = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(saved["status"], "error")
            self.assertEqual(saved["phase"], "agent")
            self.assertEqual(saved["error_type"], "ProviderError")
            self.assertEqual(
                saved["turn"],
                {"status": "error", "steps": 1, "provider_calls": 1, "tool_calls": 1},
            )
            self.assertEqual(saved["evaluation"]["status"], "not_run")
            self.assertEqual(saved["rubric"]["status"], "not_run")

    async def test_agent_setup_failure_stops_before_provider(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            factory_calls = 0

            def factory(workspace, run_dir, tracer):
                nonlocal factory_calls
                factory_calls += 1
                return TurnRunner(ScriptedProvider([{"text": "unexpected"}]), workspace)

            with self.assertRaisesRegex(WorkspaceError, "agent environment setup failed"):
                await run_code_task(
                    runner_factory=factory,
                    source=str(source),
                    prompt="do not start",
                    attempt_root=root / "attempts",
                    label="setup-failure",
                    eval_command="true",
                    agent_environment={
                        "kind": "process",
                        "setup_command": "echo setup-failed >&2; exit 7",
                    },
                )
            run_dir = next((root / "attempts").iterdir())
            saved = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(factory_calls, 0)
            self.assertEqual(saved["phase"], "agent_setup")
            self.assertEqual(saved["turn"]["provider_calls"], 0)
            self.assertIn("setup-failed", (run_dir / "agent.setup.stderr.txt").read_text())

    async def test_evaluator_restores_test_files_the_agent_edited(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._git_source(root, "checks.txt", "base\n")
            test_patch = root / "test.patch"
            test_patch.write_text(
                "diff --git a/checks.txt b/checks.txt\n"
                "--- a/checks.txt\n"
                "+++ b/checks.txt\n"
                "@@ -1 +1,2 @@\n"
                " base\n"
                "+hidden\n"
            )
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "add-test",
                                "name": "replace_in_file",
                                "arguments": {
                                    "path": "checks.txt",
                                    "old": "base\n",
                                    "new": "base\nhidden\n",
                                },
                            }
                        ]
                    },
                    {"text": "implemented"},
                ]
            )

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            result = await run_code_task(
                runner_factory=factory,
                source=str(source),
                prompt="add the test",
                attempt_root=root / "attempts",
                label="duplicate-test",
                eval_command="test \"$(tail -n 1 checks.txt)\" = hidden",
                test_patch=test_patch,
            )
            # The agent's edit to an evaluator-owned test file is discarded, so the
            # official patch applies to a pristine file instead of being skipped.
            self.assertEqual(result.evaluation["status"], "passed")
            self.assertEqual(result.evaluation["test_patch_status"], "applied")
            self.assertEqual(result.evaluation["restored_test_files"], ["checks.txt"])

    async def test_agent_cannot_shadow_the_test_that_grades_it(self):
        """An appended duplicate check must not survive into the evaluator run."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._git_source(root, "checks.txt", "base\n")
            test_patch = root / "test.patch"
            test_patch.write_text(
                "diff --git a/checks.txt b/checks.txt\n"
                "--- a/checks.txt\n"
                "+++ b/checks.txt\n"
                "@@ -1 +1,2 @@\n"
                " base\n"
                "+official\n"
            )
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "shadow",
                                "name": "replace_in_file",
                                "arguments": {
                                    "path": "checks.txt",
                                    "old": "base\n",
                                    "new": "base\nweakened\n",
                                },
                            }
                        ]
                    },
                    {"text": "done"},
                ]
            )

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            result = await run_code_task(
                runner_factory=factory,
                source=str(source),
                prompt="weaken the test",
                attempt_root=root / "attempts",
                label="shadow-test",
                eval_command="test \"$(tail -n 1 checks.txt)\" = official",
                test_patch=test_patch,
            )
            self.assertEqual(result.evaluation["restored_test_files"], ["checks.txt"])
            self.assertEqual(result.evaluation["test_patch_status"], "applied")
            self.assertEqual(result.evaluation["status"], "passed")

    async def test_native_criteria_are_graded_from_their_own_test_ids(self):
        """One group failing must not be masked by the other group passing."""
        from mixed_pilot_harness.rubrics_scorer import RubricsScorer

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rubrics = root / "rubrics.json"
            rubrics.write_text(
                json.dumps(
                    {
                        "max_score": 90,
                        "pass_threshold": 80,
                        "criteria": [
                            {
                                "id": "fail_to_pass",
                                "points": 70,
                                "method": "native_test",
                                "expected_ref": "reference/native_record.json#/FAIL_TO_PASS",
                            },
                            {
                                "id": "pass_to_pass",
                                "points": 20,
                                "method": "native_test",
                                "expected_ref": "reference/native_record.json#/PASS_TO_PASS",
                            },
                        ],
                    }
                )
            )
            scorer = RubricsScorer(rubrics_path=rubrics)
            evaluation = {
                "status": "failed",
                "native_checks": {
                    "FAIL_TO_PASS": {"status": "failed", "test_count": 1},
                    "PASS_TO_PASS": {"status": "passed", "test_count": 59},
                },
            }
            criteria = json.loads(rubrics.read_text())["criteria"]
            scores = {
                item["id"]: scorer._native_criterion(item, float(item["points"]), evaluation)
                for item in criteria
            }
            self.assertEqual(scores["fail_to_pass"]["score"], 0.0)
            self.assertEqual(scores["pass_to_pass"]["score"], 20.0)

    async def test_scorer_receives_the_agent_response(self):
        """Answer-only tasks have no artifact; the response must reach the scorer."""
        from mixed_pilot_harness.rubrics_scorer import RubricsScorer
        from mixed_pilot_harness.runtime import TaskEnvironment

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rubrics = root / "rubrics.json"
            rubrics.write_text(
                json.dumps(
                    {
                        "max_score": 100,
                        "pass_threshold": 100,
                        "criteria": [{"id": "answer", "points": 100, "method": "exact_match"}],
                    }
                )
            )
            environment = TaskEnvironment(name="scorer", root=root / "env")
            environment.workspace.mkdir(parents=True)
            seen = {}

            class Runner:
                async def run(self, prompt):
                    seen["prompt"] = prompt
                    seen["response"] = (
                        environment.workspace / ".mixed-pilot-scorer/agent_response.txt"
                    ).read_text()
                    return TurnResult(
                        turn_id="t",
                        text='{"criteria":[{"id":"answer","score":100,"reason":"ok","evidence":[]}]}',
                        steps=1,
                        provider_calls=1,
                        tool_calls=0,
                        usage=Usage(),
                    )

                def set_shell_env(self, env):
                    pass

            scorer = RubricsScorer(rubrics_path=rubrics, runner_factory=lambda w, d: Runner())
            result = await scorer.score(
                environment, root, {"status": "passed"}, agent_response="FINAL ANSWER: 90"
            )
            self.assertEqual(seen["response"], "FINAL ANSWER: 90")
            self.assertIn("agent_response.txt", seen["prompt"])
            self.assertEqual(result.score, 100.0)

    async def test_repo_label_names_the_task_not_the_layout_slot(self):
        """Every dataset task lives in <task>/workspace, so the slot name is useless."""
        from mixed_pilot_harness.cli import _repo_label

        self.assertEqual(_repo_label("/data/tasks/gaia-paper-enrollment/workspace"), "gaia-paper-enrollment")
        self.assertEqual(_repo_label("/data/tasks/browsecomp-0001/workspace/"), "browsecomp-0001")
        self.assertEqual(_repo_label("/some/place/my-project"), "my-project")
        self.assertEqual(_repo_label("/some/place/my-project.git"), "my-project")
        self.assertEqual(_repo_label("workspace"), "workspace")

    async def test_native_groups_resolve_from_expected_ref(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference"
            reference.mkdir()
            (reference / "native_record.json").write_text(
                json.dumps({"FAIL_TO_PASS": ["t.py::a"], "PASS_TO_PASS": ["t.py::b", "t.py::c"]})
            )
            rubrics = root / "rubrics.json"
            rubrics.write_text(
                json.dumps(
                    {
                        "criteria": [
                            {
                                "id": "fail_to_pass",
                                "method": "native_test",
                                "expected_ref": "reference/native_record.json#/FAIL_TO_PASS",
                            },
                            {
                                "id": "pass_to_pass",
                                "method": "native_test",
                                "expected_ref": "reference/native_record.json#/PASS_TO_PASS",
                            },
                            {"id": "scope", "method": "manual_diff_review"},
                        ]
                    }
                )
            )
            groups = resolve_native_groups(rubrics, reference)
            self.assertEqual(groups, {"FAIL_TO_PASS": ["t.py::a"], "PASS_TO_PASS": ["t.py::b", "t.py::c"]})

    async def test_test_patch_failure_writes_result_and_cleans_eval_workspace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._git_source(root, "value.txt", "base\n")
            test_patch = root / "conflict.patch"
            test_patch.write_text(
                "diff --git a/value.txt b/value.txt\n"
                "--- a/value.txt\n"
                "+++ b/value.txt\n"
                "@@ -1 +1 @@\n"
                "-missing\n"
                "+hidden\n"
            )
            provider = ScriptedProvider([{"text": "no changes"}])

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            result = await run_code_task(
                runner_factory=factory,
                source=str(source),
                prompt="finish",
                attempt_root=root / "attempts",
                label="conflicting-test",
                eval_command="true",
                test_patch=test_patch,
            )
            run_dir = Path(result.run_dir)
            saved = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(saved["evaluation"]["status"], "error")
            self.assertEqual(saved["evaluation"]["phase"], "test_patch")
            self.assertIn("could not apply evaluator test patch", saved["evaluation"]["error"])
            self.assertTrue((run_dir / "eval.stderr.txt").is_file())
            self.assertFalse((run_dir / "eval-workspace").exists())

    async def test_evaluator_setup_uses_sanitized_environment(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "value.txt").write_text("base\n")
            provider = ScriptedProvider([{"text": "no changes"}])

            def factory(workspace, run_dir, tracer):
                return TurnRunner(provider, workspace, trace=tracer)

            old_values = {
                name: os.environ.get(name)
                for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH")
            }
            os.environ.update(
                {
                    "VIRTUAL_ENV": "/tmp/fake-agent-venv",
                    "CONDA_PREFIX": "/tmp/fake-agent-conda",
                    "PYTHONPATH": "/tmp/fake-agent-pythonpath",
                }
            )
            try:
                result = await run_code_task(
                    runner_factory=factory,
                    source=str(source),
                    prompt="finish",
                    attempt_root=root / "attempts",
                    label="isolated-eval",
                    eval_setup_command=(
                        "test -z \"$VIRTUAL_ENV$CONDA_PREFIX$PYTHONPATH\" "
                        "&& mkdir isolated && echo ready > isolated/ready"
                    ),
                    eval_command=(
                        "test -f isolated/ready && test \"$PYTHONNOUSERSITE\" = 1"
                    ),
                )
            finally:
                for name, value in old_values.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            self.assertEqual(result.evaluation["status"], "passed")
            self.assertEqual(result.evaluation["phase"], "evaluation")
            self.assertTrue(Path(result.evaluation["setup_stdout_path"]).is_file())
            self.assertFalse((Path(result.run_dir) / "eval-workspace").exists())

    async def test_rubric_scorer_reads_merged_patch_in_separate_environment(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._git_source(root, "answer.txt", "before\n")
            rubrics = root / "rubrics.json"
            rubrics.write_text(
                json.dumps(
                    {
                        "max_score": 100,
                        "pass_threshold": 80,
                        "criteria": [
                            {
                                "id": "artifact",
                                "description": "answer contains the requested result",
                                "points": 100,
                                "method": "artifact_rubric",
                            }
                        ],
                    }
                )
            )
            agent_provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "edit",
                                "name": "replace_in_file",
                                "arguments": {
                                    "path": "answer.txt",
                                    "old": "before\n",
                                    "new": "after\n",
                                },
                            }
                        ]
                    },
                    {"text": "implemented"},
                ]
            )
            scorer_provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {
                                "id": "inspect",
                                "name": "read_file",
                                "arguments": {"path": "answer.txt"},
                            }
                        ]
                    },
                    {
                        "text": "Scoring is complete.\n\n```json\n%s\n```"
                        % json.dumps(
                            {
                                "criteria": [
                                    {
                                        "id": "artifact",
                                        "score": 100,
                                        "reason": "requested result is present",
                                        "evidence": ["answer.txt"],
                                    }
                                ]
                            }
                        )
                    },
                ]
            )

            def agent_factory(workspace, run_dir, tracer):
                return TurnRunner(agent_provider, workspace, trace=tracer)

            def scorer_factory(workspace, scorer_dir):
                return TurnRunner(
                    scorer_provider,
                    workspace,
                    trace=JsonlTrace(scorer_dir / "trace" / "events.jsonl"),
                )

            result = await run_code_task(
                runner_factory=agent_factory,
                scorer_runner_factory=scorer_factory,
                source=str(source),
                prompt="change the answer",
                attempt_root=root / "attempts",
                label="scored-task",
                eval_command="test \"$(cat answer.txt)\" = after",
                rubrics_path=rubrics,
            )
            run_dir = Path(result.run_dir)
            self.assertEqual(result.rubric["status"], "scored")
            self.assertEqual(result.rubric["score"], 100)
            self.assertTrue(result.rubric["passed"])
            self.assertEqual((source / "answer.txt").read_text(), "before\n")
            self.assertEqual((Path(result.workspace) / "answer.txt").read_text(), "after\n")
            self.assertFalse((run_dir / "environments" / "evaluator").exists())
            self.assertFalse((run_dir / "environments" / "scorer").exists())
            self.assertTrue((run_dir / "rubric_result.json").is_file())
            scorer_trace = run_dir / "scorer" / "trace" / "events.jsonl"
            outputs = [
                item.get("output", "")
                for item in map(json.loads, scorer_trace.read_text().splitlines())
                if item.get("event") == "tool_result"
            ]
            self.assertTrue(any("after" in output for output in outputs))


if __name__ == "__main__":
    unittest.main()
