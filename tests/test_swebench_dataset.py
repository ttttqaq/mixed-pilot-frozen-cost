"""Original SWE-bench provenance, relocation and native grader routing."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_swe_data as builder
import pull_swe_images as pull
from mixed_pilot_harness.inference import swe_runtime as native, swebench_runtime as official
from mixed_pilot_harness.inference.dataset import load_dataset
from mixed_pilot_harness.inference.maintenance import reparse


class OriginalSWETests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(instance_id=f"astropy__astropy-{i}", repo="astropy/astropy", base_commit="a" * 40,
                          problem_statement=f"Fix issue {i}", patch=f"gold-{i}", test_patch=f"test-{i}",
                          version="1.3", FAIL_TO_PASS='["test_fix"]', PASS_TO_PASS='["test_keep"]',
                          hints_text="PRIVATE HINT", environment_setup_commit="b" * 40) for i in range(5)]
        self.metadata = {"sha": builder.REVISION, "cardData": {"dataset_info": {
            "splits": [{"name": "test", "num_examples": len(self.rows)}]}}}

    def build(self, name="SWE-data-fixture"):
        target = self.root / name
        with redirect_stdout(io.StringIO()):
            builder.build(self.rows, self.metadata, "source-hash", target, count=3, seed=42)
        return target

    def test_reproducible_complete_native_records_and_private_references(self):
        target = self.build()
        second = self.build("second")
        raw = json.loads((target / "raw/swe-bench.json").read_text())
        self.assertEqual(raw, json.loads((second / "raw/swe-bench.json").read_text()))
        self.assertEqual(len({r['instance_id'] for r in raw['rows']}), 3)
        for entry, original in zip(json.loads((target / "train.json").read_text())['tasks'], raw['rows']):
            task = target / entry['path']
            manifest = json.loads((task / 'manifest.json').read_text())
            self.assertEqual(manifest['source']['split'], 'test')
            self.assertEqual(manifest['routing_metadata']['router_split'], 'train')
            self.assertEqual(json.loads((task / 'reference/source_record.json').read_text()), original)
            self.assertEqual(json.loads((task / 'reference/native_record.json').read_text())['FAIL_TO_PASS'], ['test_fix'])
            self.assertNotIn('PRIVATE HINT', (task / 'prompt.md').read_text())
            self.assertNotIn(original['patch'], (task / 'prompt.md').read_text())
            self.assertEqual(list((task / 'workspace').iterdir()), [task / 'workspace/README.md'])
        hashes = json.loads((target / 'checksums.json').read_text())
        self.assertTrue(all(builder.digest(target / p) == h for p, h in hashes.items()))
        self.assertNotIn('checksums.json', hashes)

    def test_relocated_dataset_can_plan_all_images_without_old_mac_map_or_official_install(self):
        target = self.build()
        moved = self.root / 'server/data'
        shutil.copytree(target, moved)
        moved = moved.resolve()
        with patch.dict(os.environ, {'SWE_NEO_PREPARED_IMAGES': '/missing/mac.json'}), \
             patch.object(pull, 'run_process') as docker, redirect_stdout(io.StringIO()):
            self.assertEqual(pull.main(['--dataset', str(moved), '--split', 'all',
                                       '--output', str(self.root / 'plan'), '--dry-run']), 0)
            docker.assert_not_called()
        tasks = json.loads((self.root / 'plan/plan.json').read_text())['tasks']
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(Path(t['record_path']).is_relative_to(moved) for t in tasks))
        self.assertTrue(all(t['source_image'].startswith('swebench/sweb.eval.x86_64.astropy_1776_astropy-') for t in tasks))
        self.assertEqual(len(load_dataset(moved, 'train')), 3)
        self.assertEqual(len(load_dataset(moved, 'all')), 3)

    def test_bad_population_and_existing_target_are_rejected(self):
        self.rows[0]['FAIL_TO_PASS'] = '[]'
        with self.assertRaisesRegex(ValueError, 'Empty'):
            self.build()
        self.assertFalse((self.root / 'SWE-data-fixture').exists())
        self.rows[0]['FAIL_TO_PASS'] = '["test_fix"]'
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()

    def test_original_test_strings_are_parsed_for_static_precheck(self):
        result = native.grade_tests(self.rows[0], [{'name': 'test_fix', 'status': 'PASSED'},
                                                   {'name': 'test_keep', 'status': 'PASSED'}])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['fail_to_pass_count'], 1)
        self.assertEqual(result['pass_to_pass_count'], 1)

    def test_original_evaluation_never_calls_pro_or_rebench(self):
        with patch.object(official, 'evaluate', return_value={'status': 'passed'}) as evaluate, \
             patch.object(native, 'bootstrap_rebench') as rebench, patch.object(native, 'pro_file') as pro:
            self.assertEqual(native.native_evaluate(self.rows[0], Path('patch'), self.root, self.root)['status'], 'passed')
            evaluate.assert_called_once()
            rebench.assert_not_called(); pro.assert_not_called()

    def test_missing_or_wrong_official_package_is_explicit(self):
        import importlib.metadata
        for exception in [importlib.metadata.PackageNotFoundError('swebench'), None]:
            with patch.object(importlib.metadata, 'version', side_effect=exception, return_value='5.0.1'):
                with self.assertRaisesRegex(RuntimeError, 'swebench'):
                    official.require_official()

    def test_official_evaluation_uses_fresh_container_and_merged_log(self):
        output = self.root / 'eval'
        pred = self.root / 'prediction.patch'
        pred.write_text('submitted patch')
        def execute(script, timeout):
            if 'test_output.txt' in script:
                (output / 'test_output.txt').write_text('official merged logs')
            return subprocess.CompletedProcess([], 0, 'ok', '')
        with patch.object(official, 'test_spec', return_value=SimpleNamespace(eval_script='official eval')), \
             patch.dict(sys.modules, {'swebench.harness.run_evaluation': SimpleNamespace(GIT_APPLY_CMDS=['git apply'])}), \
             patch.object(native, 'DockerSession') as session, \
             patch.object(official, 'grade_logs', return_value={'status': 'passed', 'resolved': True, 'official_report': {}}) as grade:
            session.return_value.exec.side_effect = execute
            result = native.native_evaluate(self.rows[0], pred, self.root, output)
            self.assertEqual(result['status'], 'passed')
            self.assertNotIn('workspace', session.call_args.kwargs)
            session.return_value.close.assert_called_once()
            self.assertIn('2>&1', session.return_value.exec.call_args.args[0])
            grade.assert_called_once_with(self.rows[0], output / 'prediction.patch', output / 'test_output.txt')

    def test_test_timeout_is_unscored_and_container_cleaned(self):
        pred = self.root / 'prediction.patch'; pred.write_text('submitted patch')
        with patch.object(official, 'test_spec', return_value=SimpleNamespace(eval_script='official eval')), \
             patch.dict(sys.modules, {'swebench.harness.run_evaluation': SimpleNamespace(GIT_APPLY_CMDS=['git apply'])}), \
             patch.object(native, 'DockerSession') as session:
            session.return_value.exec.side_effect = [subprocess.CompletedProcess([], 0, '', ''),
                                                     subprocess.TimeoutExpired('tests', 1)]
            result = native.native_evaluate(self.rows[0], pred, self.root, self.root / 'eval')
            self.assertEqual(result['status'], 'error'); self.assertIsNone(result['resolved'])
            self.assertEqual(result['error_category'], 'evaluation_timeout')
            session.return_value.close.assert_called_once()

    def test_reparse_uses_official_log_parser(self):
        folder = self.root / 'eval'; folder.mkdir()
        (folder / 'result.json').write_text(json.dumps({'status': 'failed', 'phase': 'test_parsing'}))
        (folder / 'test_output.txt').write_text('official log')
        with patch.object(official, 'grade_logs', return_value={'status': 'passed', 'resolved': True}) as grade:
            self.assertEqual(reparse(self.rows[0], folder)['status'], 'passed')
            grade.assert_called_once()


if __name__ == '__main__':
    unittest.main()
