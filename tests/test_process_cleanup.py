"""Local subprocess regressions; Docker boundaries are mocked."""
import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mixed_pilot_harness.processes import run
from mixed_pilot_harness.tools import WorkspaceTools
from mixed_pilot_harness.inference import swe_runtime as native


class ProcessTests(unittest.TestCase):
    def test_confirmation_gets_eof_and_noninteractive_environment(self):
        script = ('import sys,os,json; print(json.dumps([sys.stdin.read(), '
                  'os.environ["GIT_TERMINAL_PROMPT"],os.environ["COREPACK_ENABLE_DOWNLOAD_PROMPT"]]))')
        result = run([sys.executable,'-c',script],timeout=2)
        self.assertEqual(json.loads(result.stdout),['','0','0'])

    def assert_stopped(self, pid):
        # Give the OS a bounded interval to reap an orphan; no process listing.
        for _ in range(100):
            try:
                os.kill(pid,0)
            except ProcessLookupError:
                return
            time.sleep(.01)
        self.fail(f'child {pid} survived process-group cleanup')

    def child_script(self, pid_file):
        return ('import subprocess,sys,time; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                f'open({str(pid_file)!r},"w").write(str(p.pid)); '
                'print("waiting for confirmation",flush=True); '
                'sys.stdin.read(); time.sleep(60)')

    def test_sync_timeout_kills_descendants_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'pid'
            start=time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired) as cm:
                run([sys.executable,'-c',self.child_script(file)],timeout=.4)
            self.assertLess(time.monotonic()-start,3)
            self.assertIn('waiting for confirmation',cm.exception.output)
            self.assert_stopped(int(file.read_text()))

    def test_docker_exec_timeout_removes_remote_container(self):
        session=object.__new__(native.DockerSession)
        session.name='owned-container';session.repo_dir='/app';session.close=Mock()
        with patch.object(native,'command',side_effect=subprocess.TimeoutExpired('docker',1)):
            with self.assertRaises(subprocess.TimeoutExpired):
                session.exec('sleep 60',1)
        session.close.assert_called_once()

    def test_source_copy_timeout_removes_container(self):
        session=Mock(name='source');session.name='source';session.repo_dir='/app'
        session.exec.return_value=subprocess.CompletedProcess([],0,'','')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(native,'DockerSession',return_value=session), \
             patch.object(native,'command',side_effect=subprocess.TimeoutExpired('docker cp',1)):
            with self.assertRaises(subprocess.TimeoutExpired):
                native.copy_image_source({'base_commit':'abc'},'sha256:test',Path(directory)/'source')
        session.close.assert_called_once()


class AsyncProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_shell_timeout_kills_child_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'pid'
            script=ProcessTests().child_script(file)
            tools=WorkspaceTools(Path(directory),timeout=.4)
            result=json.loads(await tools.shell({'command':shlex.join([sys.executable,'-c',script])}))
            self.assertFalse(result['ok'])
            self.assertIn('timed out',result['error'])
            ProcessTests().assert_stopped(int(file.read_text()))

    async def test_shell_cancellation_cleans_children(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'pid'
            script=ProcessTests().child_script(file)
            tools=WorkspaceTools(Path(directory),timeout=60)
            task=asyncio.create_task(tools.shell({'command':shlex.join([sys.executable,'-c',script])}))
            try:
                for _ in range(100):
                    if file.exists() and file.read_text():break
                    await asyncio.sleep(.01)
                self.assertTrue(file.exists())
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
            ProcessTests().assert_stopped(int(file.read_text()))

    async def test_container_shell_timeout_closes_remote_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session=Mock(name='container',repo_dir='/app')
            tools=native.DockerWorkspaceTools(Path(directory),session,1,Path(directory)/'logs')
            process=Mock(returncode=137,wait=AsyncMock(return_value=137))
            with patch('asyncio.create_subprocess_exec',AsyncMock(return_value=process)) as spawn:
                with self.assertRaises(TimeoutError):await tools.shell({'command':'interactive download'})
            session.close.assert_called_once()
            self.assertEqual(spawn.call_args.kwargs['stdin'],asyncio.subprocess.DEVNULL)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])

    async def test_removed_container_error_is_not_swallowed_by_tool_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            session=Mock(name='container',repo_dir='/app')
            tools=native.DockerWorkspaceTools(Path(directory),session,1,Path(directory)/'logs')
            process=Mock(returncode=137,wait=AsyncMock(return_value=137))
            with patch('asyncio.create_subprocess_exec',AsyncMock(return_value=process)) as spawn:
                with self.assertRaisesRegex(RuntimeError,'task container removed'):
                    await tools.execute('shell',{'command':'interactive download'})
                with self.assertRaisesRegex(RuntimeError,'task container removed'):
                    await tools.execute('shell',{'command':'retry'})
            session.close.assert_called_once()
            spawn.assert_awaited_once()
