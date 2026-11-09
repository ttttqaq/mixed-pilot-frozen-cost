"""Exercise real local Git repositories; no network, Docker, or model calls."""
import os
import shutil
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from mixed_pilot_harness.inference import swe_runtime as native
from mixed_pilot_harness.runtime import prepare_workspace


class SubmoduleBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def git(self, repo, *args):
        return native.command(["git", "-C", str(repo), *args]).stdout.strip()

    def commit(self, repo):
        self.git(repo, "add", "-A")
        self.git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "snapshot")
        return self.git(repo, "rev-parse", "HEAD")

    def fixture(self):
        sub, parent = self.root / "dependency", self.root / "parent"
        for repo in (sub, parent):
            repo.mkdir()
            self.git(repo, "init", "-q")
        (sub / "code.py").write_text("pinned dependency\n")
        pin = self.commit(sub)
        (sub / "code.py").write_text("newer branch tip\n")
        self.commit(sub)
        self.git(parent, "submodule", "add", "-q", str(sub), "vendor/module")
        self.git(parent / "vendor/module", "checkout", "-q", pin)
        (parent / ".gitignore").write_text("vendor/\n")
        (parent / "dependency_alias").symlink_to("vendor/module")
        (parent / "answer.py").write_text("old\n")
        base = self.commit(parent)
        return parent, base

    def snapshot(self, parent, base, cache, *, image_id='sha256:fixture', row=None):
        real = native.command
        def local_command(argv, **kw):
            if argv[:3] == ['docker', 'image', 'inspect']:
                return subprocess.CompletedProcess(argv, 0, image_id, '')
            self.assertNotIn('fetch', argv)
            self.assertNotIn('update', argv)
            return real(argv, **kw)
        def copy_source(row, image_id, workspace):
            shutil.copytree(parent, workspace, symlinks=True)
            return self.git(parent, 'ls-files', '--recurse-submodules', '-z').split('\0')
        row = row or dict(instance_id='fixture', repo='test/parent', image_name='fixture', base_commit=base)
        with mock.patch.object(native, 'command', side_effect=local_command), \
             mock.patch.object(native, 'copy_image_source', side_effect=copy_source):
            return native.repo_snapshot(row, cache)

    def test_pinned_submodule_is_flattened_and_real_edits_survive_export(self):
        parent, base = self.fixture()
        workspace, snapshot = self.snapshot(parent, base, self.root / "cache")
        self.assertEqual((workspace / "vendor/module/code.py").read_text(), "pinned dependency\n")
        self.assertFalse((workspace / "vendor/module/.git").exists())
        self.assertFalse((workspace / ".git/modules").exists())
        clone = self.root / "agent-workspace"
        prepare_workspace(str(workspace), clone)
        self.assertTrue((clone / "dependency_alias").is_symlink())
        self.assertEqual(self.git(clone, "status", "--porcelain"), "")
        workspace = clone
        patch = self.root / "prediction.patch"
        native.export_patch(workspace, snapshot, patch)
        self.assertEqual(patch.read_text(), "")
        (workspace / "answer.py").write_text("fixed\n")
        (workspace / "vendor/module/code.py").write_text("legitimate dependency fix\n")
        (workspace / "vendor/module/new.py").write_text("new source\n")
        self.git(workspace, "add", "-f", "vendor/module/new.py")
        self.commit(workspace)
        native.export_patch(workspace, snapshot, patch)
        self.assertIn("legitimate dependency fix", patch.read_text())
        self.assertIn("new source", patch.read_text())
        self.git(parent, "apply", "--check", str(patch))

    def test_incomplete_legacy_baseline_is_rejected_not_silently_filtered(self):
        repo = self.root / "legacy"
        repo.mkdir()
        self.git(repo, "init", "-q")
        (repo / ".gitmodules").write_text('[submodule "vendor/module"]\npath = vendor/module\nurl = https://example.org/module\n')
        base = self.commit(repo)
        (repo / "vendor/module").mkdir(parents=True)
        (repo / "vendor/module/code.py").write_text("recovered dependency")
        with self.assertRaisesRegex(RuntimeError, "Incomplete submodule baseline"):
            native.export_patch(repo, base, self.root / "prediction.patch")
        self.assertFalse((self.root / "prediction.patch").exists())

    def test_legacy_cache_is_preserved_and_rebuilt_separately(self):
        parent, base = self.fixture()
        cache = self.root / "cache"
        legacy = cache / "repos/fixture"
        (legacy / "workspace").mkdir(parents=True)
        (legacy / "workspace/.gitmodules").write_text("legacy")
        native.write_json(legacy / "snapshot.json", {"base_commit":base, "snapshot_commit":"old"})
        workspace, _ = self.snapshot(parent, base, cache)
        self.assertIn(native.SNAPSHOT_REVISION, str(workspace))
        self.assertEqual((legacy / "workspace/.gitmodules").read_text(), "legacy")

    def test_missing_optional_submodules_stay_missing_without_download(self):
        parent, base = self.fixture()
        self.git(parent, 'submodule', 'deinit', '-f', '--all')
        workspace, snapshot = self.snapshot(parent, base, self.root / 'cache')
        self.assertFalse((workspace / 'vendor/module/code.py').exists())
        patch = self.root / 'prediction.patch'
        native.export_patch(workspace, snapshot, patch)
        self.assertEqual(patch.read_text(), '')
        (workspace / 'answer.py').write_text('real fix\n')
        native.export_patch(workspace, snapshot, patch)
        self.assertIn('+real fix', patch.read_text())
        self.assertNotIn('vendor/module', patch.read_text())

    def test_image_source_and_cache_identity_are_preserved(self):
        parent, base = self.fixture()
        (parent / 'vendor/module/code.py').write_text('verified image content\n')
        cache = self.root / 'cache'
        workspace, _ = self.snapshot(parent, base, cache)
        self.assertEqual((workspace / 'vendor/module/code.py').read_text(), 'verified image content\n')
        newer, _ = self.snapshot(parent, base, cache, image_id='sha256:new-image')
        self.assertNotEqual(workspace, newer)
        self.assertTrue(workspace.exists())

    def test_pro_hidden_tests_and_history_are_not_exposed(self):
        parent, base = self.fixture()
        (parent / 'secret_test.py').write_text('hidden answer')
        base = self.commit(parent)
        row = dict(instance_id='fixture', repo='test/parent', dockerhub_tag='fixture',
                   base_commit=base, selected_test_files_to_run=['secret_test.py','vendor/module/code.py'])
        workspace, snapshot = self.snapshot(parent, base, self.root / 'cache', row=row)
        clone = self.root / 'agent'
        prepare_workspace(str(workspace), clone)
        self.assertFalse((clone / 'secret_test.py').exists())
        self.assertFalse((clone / 'vendor/module/code.py').exists())
        self.assertFalse((clone / '.git/modules').exists())
        self.assertEqual(self.git(clone, 'rev-list', '--count', 'HEAD'), '1')
        self.assertEqual(native.command(['git','-C',str(clone),'grep','hidden answer','HEAD'],check=False).returncode, 1)

    def test_dependency_scan_prunes_git_metadata(self):
        repo = self.root / "repo"
        (repo / ".git/modules/vendor/hidden").mkdir(parents=True)
        (repo / "vendor").mkdir()
        (repo / "node_modules").mkdir()
        def run_find(argv, **kw):
            if argv[:2] == ['docker', 'rm']:
                return subprocess.CompletedProcess(argv, 0, '', '')
            self.assertIn("-name .git -prune", argv[-1])
            return subprocess.run(["/bin/sh", "-c", argv[-1]], capture_output=True, text=True, check=True)
        native.image_dependency_dirs.cache_clear()
        with mock.patch.object(native, "command", side_effect=run_find):
            result = native.image_dependency_dirs("fixture", str(repo))
        self.assertEqual(set(result), {str(repo / "vendor"), str(repo / "node_modules")})
        native.image_dependency_dirs.cache_clear()

    def test_dependency_volume_cannot_restore_hidden_tests(self):
        workspace = self.root / 'agent'
        workspace.mkdir()
        row = dict(instance_id='fixture', repo='test/parent', dockerhub_tag='fixture',
                   base_commit='abc', selected_test_files_to_run=['/app/vendor/module/test.py'])
        with mock.patch.object(native, 'image_for', return_value=('fixture','/app')), \
             mock.patch.object(native, 'image_node_options', return_value=''), \
             mock.patch.object(native, 'image_dependency_dirs', return_value=('/app/vendor','/app/node_modules')), \
             mock.patch.object(native, 'command', return_value=subprocess.CompletedProcess([],0,'','')) as command:
            session = native.DockerSession(row, self.root / 'logs', workspace=workspace)
        argv = command.call_args.args[0]
        self.assertNotIn('type=volume,target=/app/vendor', argv)
        self.assertIn('type=volume,target=/app/node_modules', argv)
        self.assertEqual(session.dependency_dirs, ['/app/node_modules'])


if __name__ == "__main__":
    unittest.main()
