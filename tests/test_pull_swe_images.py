"""Verify a new server can plan upstream pulls without any old host image map."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pull_swe_images as pull
from mixed_pilot_harness.inference import swe_runtime as native


class PullTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / 'data'
        rows = []
        for name, image in [('rebench', 'docker.io/swerebenchv2/owner-repo:123-base'),
                            ('pro', None)]:
            task = self.dataset / 'train/tasks' / name
            (task / 'reference').mkdir(parents=True)
            (task / 'prompt.md').write_text('Fix the issue')
            record = {'instance_id': name, 'repo': 'owner/repo', 'base_commit': 'base'}
            record.update({'image_name': image} if image else {'dockerhub_tag': 'owner.repo-instance'})
            (task / 'reference/source_record.json').write_text(json.dumps(record))
            manifest = {'id': name, 'source': {'split': 'train', 'dataset': 'SWE',
                        'upstream_id': name, 'base_commit': 'base'}}
            (task / 'manifest.json').write_text(json.dumps(manifest))
            rows.append({'id': name, 'path': 'train/tasks/' + name})
        (self.dataset / 'train.json').write_text(json.dumps({'split': 'train', 'task_count': 2, 'tasks': rows}))

    def args(self, output='pull'):
        return ['--dataset', str(self.dataset), '--output', str(self.root / output)]

    def test_dry_run_ignores_missing_mac_manifest_and_restores_environment(self):
        old = '/Users/old-mac/environment-images.json'
        with patch.dict(os.environ, {'SWE_NEO_PREPARED_IMAGES': old}), \
             patch.object(pull, 'run_process') as docker, redirect_stdout(io.StringIO()):
            self.assertEqual(pull.main(self.args() + ['--dry-run']), 0)
            self.assertEqual(os.environ['SWE_NEO_PREPARED_IMAGES'], old)
            docker.assert_not_called()
        images = json.loads((self.root / 'pull/image-plan.json').read_text())['images']
        self.assertEqual({v['image'] for v in images},
                         {'docker.io/swerebenchv2/owner-repo:123-base', 'jefzda/sweap-images:owner.repo-instance'})

    def test_relocated_dataset_generates_paths_on_destination_server(self):
        moved = self.root / 'server/data'
        shutil.copytree(self.dataset, moved)
        moved = moved.resolve()
        with patch.object(pull, 'run_process') as docker, redirect_stdout(io.StringIO()):
            code = pull.main(['--dataset', str(moved), '--output', str(self.root / 'moved'), '--dry-run'])
        self.assertEqual(code, 0); docker.assert_not_called()
        tasks = json.loads((self.root / 'moved/plan.json').read_text())['tasks']
        self.assertTrue(all(Path(t['record_path']).is_relative_to(moved) for t in tasks))
        self.assertTrue(all(Path(t['task_dir']).is_relative_to(moved) for t in tasks))

    def test_missing_old_plan_paths_point_to_dataset_option(self):
        plan = self.root / 'old-plan.json'
        plan.write_text(json.dumps({'tasks': [{'task_id': 'old', 'record_path': '/missing/old-mac/record.json'}]}))
        errors = io.StringIO()
        with patch.object(pull, 'run_process') as docker, redirect_stderr(errors):
            self.assertEqual(pull.main(['--plan', str(plan), '--output', str(self.root / 'bad')]), 2)
        docker.assert_not_called(); self.assertIn('--dataset', errors.getvalue())

    def test_fresh_docker_receives_upstream_images_for_both_benchmarks(self):
        def run(argv, **kwargs):
            if argv[1] == 'info': return subprocess.CompletedProcess(argv, 0, '29', '')
            if argv[1:3] == ['image', 'inspect']: return subprocess.CompletedProcess(argv, 1, '', 'missing')
            return subprocess.CompletedProcess(argv, 0, '', '')
        with patch.object(pull, 'run_process', side_effect=run) as docker, redirect_stdout(io.StringIO()):
            self.assertEqual(pull.main(self.args()), 0)
        pulls = [call.args[0] for call in docker.call_args_list if call.args[0][1] == 'pull']
        self.assertEqual(len(pulls), 2)
        self.assertTrue(all(argv[2:4] == ['--platform', native.DOCKER_PLATFORM] for argv in pulls))
        self.assertFalse(any('swe-neo-prepared:' in argv[-1] for argv in pulls))
        self.assertEqual(json.loads((self.root / 'pull/images.json').read_text())['ready'], 2)

    def test_original_scope_restores_override_on_failure(self):
        with patch.dict(os.environ, {'SWE_NEO_PREPARED_IMAGES': 'original-setting'}):
            with self.assertRaises(ValueError), native.original_images():
                self.assertNotIn('SWE_NEO_PREPARED_IMAGES', os.environ)
                os.environ['SWE_NEO_PREPARED_IMAGES'] = 'temporary-setting'
                raise ValueError('test')
            self.assertEqual(os.environ['SWE_NEO_PREPARED_IMAGES'], 'original-setting')


if __name__ == '__main__': unittest.main()
