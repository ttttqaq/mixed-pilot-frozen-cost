"""Offline native SWE runtime checks; unified orchestration is tested in test_inference."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mixed_pilot_harness.inference import swe_runtime as native
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.providers import ScriptedProvider


class SWENeoBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def repo(self):
        workspace = self.root / "source"
        workspace.mkdir()
        (workspace / "answer.txt").write_text("old\n")
        native.command(["git", "init", "-q", str(workspace)])
        native.command(["git", "-C", str(workspace), "add", "-A"])
        native.command(["git", "-C", str(workspace), "-c", "user.name=Test",
                        "-c", "user.email=test@localhost", "commit", "-qm", "base"])
        base = native.command(["git", "-C", str(workspace), "rev-parse", "HEAD"]).stdout.strip()
        return workspace, base

    def test_prepared_images_require_matching_task_provenance(self):
        row = {'instance_id': 'fixture', 'repo': 'owner/repo',
               'dockerhub_tag': 'original', 'base_commit': 'base'}
        path = self.root / 'prepared.json'
        entry = {'image': 'swe-neo-prepared:fixture',
                 'source_image': 'jefzda/sweap-images:original', 'base_commit': 'base'}
        native.write_json(path, {'images': {'fixture': entry}})
        with mock.patch.dict(os.environ, {'SWE_NEO_PREPARED_IMAGES': str(path)}):
            self.assertEqual(native.image_for(row), ('swe-neo-prepared:fixture', '/app'))
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                native.image_for({**row, 'base_commit': 'different'})
            with self.assertRaisesRegex(ValueError, 'No verified prepared image'):
                native.image_for({**row, 'instance_id': 'missing'})
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(native.image_for(row), ('jefzda/sweap-images:original', '/app'))


    def test_export_patch_retains_agent_commits_and_new_files(self):
        workspace, base = self.repo()
        (workspace / "answer.txt").write_text("fixed\n")
        native.command(["git", "-C", str(workspace), "add", "-A"])
        native.command(["git", "-C", str(workspace), "-c", "user.name=Test",
                        "-c", "user.email=test@localhost", "commit", "-qm", "agent commit"])
        (workspace / "new.txt").write_text("new\n")
        destination = self.root / "prediction.patch"
        native.export_patch(workspace, base, destination)
        self.assertIn("+fixed", destination.read_text())
        self.assertIn("+new", destination.read_text())
        self.assertEqual(native.command(["git", "-C", str(workspace), "status", "--short"]).stdout, "?? new.txt\n")

    def test_native_rules_require_both_test_groups(self):
        row = {"fail_to_pass": ["f"], "pass_to_pass": ["p"]}
        self.assertFalse(native.grade_tests(row, [{"name": "f", "status": "PASSED"}])["resolved"])
        both = [{"name": name, "status": "PASSED"} for name in ("f", "p", "extra")]
        self.assertTrue(native.grade_tests(row, both)["resolved"])
        self.assertFalse(native.grade_tests(row, both, exact=True)["resolved"])
        with self.assertRaises(ValueError):
            native.grade_tests({"fail_to_pass": []}, [])

    def test_go_source_compilation_is_failed_not_environment_error(self):
        row = {"FAIL_TO_PASS": ["TestJob"], "PASS_TO_PASS": []}
        value = native.classify_test_output(row, [],
            './scheduler_test.go:357:14: j.functions undefined (type *Job has no field or method functions)\nFAIL [build failed]')
        self.assertEqual(value["status"], "failed")
        self.assertEqual(value["failure_kind"], "source_compilation")
        self.assertEqual(value["fail_to_pass_missing"], ["TestJob"])

    def test_typescript_missing_local_implementation_is_failed(self):
        row = {"fail_to_pass": ["chunks"], "pass_to_pass": []}
        value = native.classify_test_output(row, [],
            "chunk.test.ts:1:19 - error TS2307: Cannot find module './chunk' or its corresponding type declarations.")
        self.assertEqual(value["status"], "failed")

    def test_browser_failure_is_error_even_with_source_compilation_failure(self):
        row = {"fail_to_pass": ["chunks"], "pass_to_pass": []}
        value = native.classify_test_output(row, [],
            "error TS2307: Cannot find module './chunk'\nCannot start ChromeHeadless\nrosetta error: failed to open elf")
        self.assertEqual(value["status"], "error")
        self.assertEqual(value["error_category"], "runtime_dependency")

    def test_go_required_failure_omitted_by_upstream_parser(self):
        row = {'FAIL_TO_PASS': ['TestImport_InvalidVersion'], 'PASS_TO_PASS': []}
        log = '--- FAIL: TestImport_InvalidVersion (0.00s)'
        self.assertEqual(native.classify_test_output(row, [], log)['status'], 'failed')
        self.assertEqual(native.classify_test_output(row, [], '--- FAIL: TestOther (0.00s)')['status'], 'error')
        self.assertEqual(native.classify_test_output(row, [], log + '\nnetwork is unreachable')['status'], 'error')
        self.assertEqual(native.classify_test_output(row, [], '--- PASS: TestImport_InvalidVersion (0.00s)')['status'], 'error')

    def test_unknown_zero_tests_remain_error(self):
        row = {"fail_to_pass": ["required"], "pass_to_pass": []}
        value = native.classify_test_output(row, [], "No tests found")
        self.assertEqual(value["status"], "error")

    def test_source_and_assertion_failures_do_not_require_all_tests_to_run(self):
        row = {'FAIL_TO_PASS': ['new_test'], 'PASS_TO_PASS': []}
        for log in ['test result: FAILED. 9 passed; 1 failed; 0 ignored;',
                    '  43 failing\n',
                    'server/evaluator_test.go:91:3: unknown field ExcludeNotFound in struct literal of type Request',
                    "Cannot find module './moveToFolder' from 'src/helpers/moveToFolder.test.ts'",
                    "  3:23  error  Unable to resolve path to module '../mbx-button'  import/no-unresolved"]:
            with self.subTest(log=log):
                self.assertEqual(native.classify_test_output(row, [], log)['status'], 'failed')
        for log in ['test result: ok. 9 passed; 0 failed;', '0 failing',
                    "Cannot find module 'missing-external-package' from 'test.js'",
                    "  3:23  error  Unable to resolve path to module 'external-package'  import/no-unresolved",
                    'panic: timeout']:
            with self.subTest(log=log):
                self.assertEqual(native.classify_test_output(row, [], log)['status'], 'error')

    def test_inference_preserves_installed_dependencies_and_cleans_own_volumes(self):
        workspace = self.root / "workspace"
        (workspace / "vendor").mkdir(parents=True)
        row = {"instance_id": "fixture", "repo": "owner/repo", "dockerhub_tag": "tag"}
        with mock.patch.object(native, "image_node_options", return_value="--require=/runtime/hook.cjs"), \
             mock.patch.object(native, "image_dependency_dirs", return_value=("/app/node_modules", "/app/vendor")), \
             mock.patch.object(native, "command", return_value=subprocess.CompletedProcess([], 0, "container-id", "")) as cmd:
            session = native.DockerSession(row, self.root / "docker", workspace=workspace)
            argv = cmd.call_args.args[0]
            self.assertIn("type=volume,target=/app/node_modules", argv)
            self.assertNotIn("type=volume,target=/app/vendor", argv)
            self.assertIn("NODE_OPTIONS=--require=/runtime/hook.cjs", argv)
            session.close()
            self.assertIn("--volumes", cmd.call_args.args[0])





    def test_pro_native_adapter_uses_fresh_session_and_structured_results(self):
        row = {"instance_id": "fixture", "repo": "owner/repo", "base_commit": "a" * 40,
               "dockerhub_tag": "tag", "selected_test_files_to_run": ["tests/test_x.py"],
               "before_repo_set_cmd": "git reset --hard base\ngit checkout solution -- tests/test_x.py",
               "base_dockerfile": "ENV LANG en_US.UTF-8\nENV A=one B=two\n",
               "fail_to_pass": ["f"], "pass_to_pass": ["p"]}
        patch = self.root / "prediction.patch"
        patch.write_text("diff")
        script = self.root / "official.sh"
        script.write_text("script")
        output = self.root / "native"
        captured = []
        class FakeDocker:
            def __init__(self, *args, **kwargs):
                captured.append(kwargs)
            def exec(self, entry, timeout):
                captured.append(entry)
                native.write_json(output / "tests.json", {"tests": [{"name": n, "status": "PASSED"} for n in ("f", "p")]})
                return subprocess.CompletedProcess([], 0, native.PATCH_READY, "")
            def close(self):
                pass
        with mock.patch.object(native, "prepare_native"), \
             mock.patch.object(native, "pro_file", return_value=script), \
             mock.patch.object(native, "DockerSession", FakeDocker):
            result = native.native_evaluate(row, patch, self.root / "cache", output)
        self.assertTrue(result["resolved"])
        self.assertNotIn("workspace", captured[0], "evaluation must use a fresh image")
        entry = captured[1]
        self.assertNotIn('export LANG', entry, 'Docker inherits ENV without lossy shell conversion')
        self.assertNotIn('export en_US', entry)
        self.assertLess(entry.index("git apply"), entry.index("git checkout solution"))
        self.assertTrue((output / "result.json").exists())

    def test_native_timeout_is_unscored_error(self):
        row = {"instance_id": "fixture", "repo": "owner/repo", "base_commit": "a" * 40,
               "dockerhub_tag": "tag", "selected_test_files_to_run": [],
               "fail_to_pass": ["f"], "pass_to_pass": []}
        patch = self.root / "empty.patch"
        patch.write_text("")
        script = self.root / "official.sh"
        script.write_text("")
        with mock.patch.object(native, "prepare_native"), \
             mock.patch.object(native, "pro_file", return_value=script), \
             mock.patch.object(native, "DockerSession") as session:
            session.return_value.exec.side_effect = subprocess.TimeoutExpired("docker", 1)
            result = native.native_evaluate(row, patch, self.root / "cache", self.root / "native")
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["resolved"])
        session.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
