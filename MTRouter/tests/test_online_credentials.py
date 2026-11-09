"""Credential inputs must reach providers without entering saved training state."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mixed_pilot_harness.config import ProviderProfile
from mixed_pilot_harness.providers import JsonHttpProvider
from mixed_pilot_mtrouter.online.cli import api_credentials, main, parser
from mixed_pilot_mtrouter.online.config import load_config


class CredentialTests(unittest.TestCase):
    def test_yaml_keys_reach_actual_provider_interface_and_are_consumed(self):
        config = {"seed": 42, "api_keys": {"dashscope": "dummy-qwen", "zhipu": "dummy-glm", "newcli": "dummy-teacher"}}
        with patch.dict(os.environ, {}, clear=True):
            with api_credentials(config, SimpleNamespace()):
                for env, expected in [("DASHSCOPE_API_KEY", "dummy-qwen"), ("ZHIPU_API_KEY", "dummy-glm"), ("NEWCLI_API_KEY", "dummy-teacher")]:
                    provider = JsonHttpProvider(ProviderProfile(kind="openai", model="dummy", api_key_env=env))
                    self.assertEqual(provider._api_key(), expected)
                self.assertEqual(config, {"seed": 42})
            self.assertNotIn("DASHSCOPE_API_KEY", os.environ)
            self.assertNotIn("ZHIPU_API_KEY", os.environ)

    def test_cli_overrides_yaml_and_environment_is_restored_on_failure(self):
        args = parser().parse_args(["--config", "unused.yaml", "--dashscope-api-key", "cli-key"])
        config = {"api_keys": {"dashscope": "yaml-key", "zhipu": "yaml-glm"}}
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "previous-key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with api_credentials(config, args):
                    self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "cli-key")
                    self.assertEqual(os.environ["ZHIPU_API_KEY"], "yaml-glm")
                    raise RuntimeError("interrupted")
            self.assertEqual(dict(os.environ), {"DASHSCOPE_API_KEY": "previous-key"})

    def test_empty_yaml_preserves_existing_provider_credentials(self):
        config = {"api_keys": {"dashscope": "", "zhipu": None}}
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "existing"}, clear=True):
            with api_credentials(config, SimpleNamespace()):
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "existing")
                self.assertNotIn("ZHIPU_API_KEY", os.environ)

    def test_main_removes_keys_before_training_config_can_be_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "online.yaml"
            path.write_text(json.dumps({"api_keys": {"dashscope": "yaml-secret", "zhipu": "glm-secret"}}))
            output = io.StringIO()
            def run(config, args):
                self.assertNotIn("api_keys", config)
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "cli-secret")
                self.assertEqual(os.environ["ZHIPU_API_KEY"], "glm-secret")
                print(json.dumps(config))
                return 0
            with patch.dict(os.environ, {}, clear=True), patch("mixed_pilot_mtrouter.online.cli._run", run), redirect_stdout(output):
                self.assertEqual(main(["--config", str(path), "--dashscope-api-key", "cli-secret"]), 0)
            for secret in ["yaml-secret", "glm-secret", "cli-secret"]:
                self.assertNotIn(secret, output.getvalue())

    def test_invalid_key_fails_without_echoing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "online.yaml"
            path.write_text("api_keys:\n  zhipu: secret with whitespace\n")
            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(main(["--config", str(path)]), 2)
            self.assertIn("nonempty token", error.getvalue())
            self.assertNotIn("secret with whitespace", error.getvalue())
            path.write_text("api_keys:\n  dashscope: 123\n")
            with self.assertRaisesRegex(ValueError, "api_keys.dashscope must be a string"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
