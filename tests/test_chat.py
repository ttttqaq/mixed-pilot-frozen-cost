"""Single-request chat checks without network access or credentials."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mixed_pilot_harness.cli import main
from mixed_pilot_harness.types import Message, ProviderError, ProviderResponse, ToolCall, Usage


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "harness.json"
        self.config.write_text(json.dumps({
            "default_profile": "default",
            "agent": {"provider_timeout": 45},
            "profiles": {
                "default": {"kind": "openai", "model": "default-model"},
                "other": {"kind": "anthropic", "model": "other-model"},
            },
        }))

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["chat", "--config", str(self.config), *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_one_call_without_tools_or_runner_and_preserves_response(self):
        response = ProviderResponse(
            text="hello", model="actual-model", stop_reason="length",
            usage=Usage(12, 3, raw={"cached_tokens": 4}),
            # Even an unsolicited tool request must never enter an agent loop.
            tool_calls=[ToolCall("x", "write_file", {"path": "unexpected.txt", "content": "x"})],
        )
        provider = AsyncMock()
        provider.complete.return_value = response
        result = self.root / "nested" / "result.json"
        with patch("mixed_pilot_harness.cli.create_provider", return_value=provider) as factory, \
             patch("mixed_pilot_harness.cli.TurnRunner") as runner:
            code, stdout, stderr = self.invoke("--message", "你好", "--result", str(result))
        self.assertEqual((code, stdout, stderr), (0, "hello\n", ""))
        provider.complete.assert_awaited_once_with(
            [Message("user", "你好")], [], "You are a helpful assistant.")
        runner.assert_not_called()
        self.assertEqual(factory.call_args.args[0].model, "default-model")
        self.assertEqual(factory.call_args.args[0].timeout, 45)
        self.assertEqual(factory.call_args.kwargs["config_dir"], self.root.resolve())
        saved = json.loads(result.read_text())
        self.assertEqual(saved["model"], "actual-model")
        self.assertEqual(saved["stop_reason"], "length")
        self.assertEqual(saved["usage"], {"input_tokens": 12, "output_tokens": 3,
                                           "raw": {"cached_tokens": 4}})

    def test_file_input_profile_overrides_and_empty_system(self):
        prompt = self.root / "prompt.txt"
        prompt.write_text("文件中的问题", encoding="utf-8")
        provider = AsyncMock()
        provider.complete.return_value = ProviderResponse(text="answer")
        with patch("mixed_pilot_harness.cli.create_provider", return_value=provider) as factory:
            code, _, _ = self.invoke("--task-file", str(prompt), "--profile", "other",
                                     "--model", "override", "--base-url", "https://example.invalid",
                                     "--system-prompt", "")
        self.assertEqual(code, 0)
        selected = factory.call_args.args[0]
        self.assertEqual((selected.name, selected.kind, selected.model, selected.base_url),
                         ("other", "anthropic", "override", "https://example.invalid"))
        provider.complete.assert_awaited_once_with([Message("user", "文件中的问题")], [], "")

    def test_stdin_and_custom_system(self):
        provider = AsyncMock()
        provider.complete.return_value = ProviderResponse(text="answer")
        with patch("mixed_pilot_harness.cli.create_provider", return_value=provider), \
             patch("sys.stdin", io.StringIO("piped question\n")):
            code, _, _ = self.invoke("--system-prompt", "Reply briefly.")
        self.assertEqual(code, 0)
        provider.complete.assert_awaited_once_with(
            [Message("user", "piped question\n")], [], "Reply briefly.")

    def test_conflicting_inputs_fail_before_provider_creation(self):
        with patch("mixed_pilot_harness.cli.create_provider") as factory:
            code, stdout, stderr = self.invoke("--message", "a", "--task-file", "unused.txt")
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("pass only one", stderr)
        factory.assert_not_called()

    def test_provider_failure_returns_cli_error(self):
        provider = AsyncMock()
        provider.complete.side_effect = ProviderError("synthetic failure")
        with patch("mixed_pilot_harness.cli.create_provider", return_value=provider):
            code, stdout, stderr = self.invoke("--message", "hello")
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("synthetic failure", stderr)
        provider.complete.assert_awaited_once()

    def test_cli_scripted_provider_uses_relative_config_files(self):
        (self.root / "responses.json").write_text(json.dumps([{"text": "single answer"}]))
        (self.root / "router.toml").write_text(
            '[router]\nengine = "fixed"\nfixed_tier = "c1"\n'
            + "".join('[tiers.c%d]\nprofile = "offline"\n' % tier for tier in range(4)))
        self.config.write_text(json.dumps({
            "default_profile": "offline",
            "profiles": {"offline": {"kind": "scripted", "model": "offline",
                                       "responses_file": "responses.json"},
                         "router": {"kind": "router", "model": "router",
                                    "router_config": "router.toml"}},
        }))
        for profile in ("offline", "router"):
            with self.subTest(profile=profile):
                result = subprocess.run(
                    [sys.executable, "-m", "mixed_pilot_harness.cli", "chat",
                     "--config", str(self.config), "--profile", profile, "--message", "hello"],
                    cwd=self.root, capture_output=True, text=True,
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "single answer\n")
