"""Noninteractive subprocesses with bounded waits and process-group cleanup."""
from __future__ import annotations

import os
import signal
import subprocess

NONINTERACTIVE = dict(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never',
                     GIT_ASKPASS='/usr/bin/false', SSH_ASKPASS='/usr/bin/false',
                     COREPACK_ENABLE_DOWNLOAD_PROMPT='0', PIP_NO_INPUT='1')

def noninteractive_env(env=None):
    value = dict(os.environ if env is None else env)
    value.update(NONINTERACTIVE)
    value.setdefault('GIT_SSH_COMMAND', 'ssh -oBatchMode=yes -oConnectTimeout=15')
    value.setdefault('GIT_HTTP_LOW_SPEED_LIMIT', '1')
    value.setdefault('GIT_HTTP_LOW_SPEED_TIME', '30')
    return value


def kill_group(process):
    # Kill even when the group leader exited: descendants may still hold pipes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(argv, *, cwd=None, timeout=600, env=None, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True):
    process = subprocess.Popen(argv, cwd=cwd, env=noninteractive_env(env),
        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, text=text,
        start_new_session=True)
    try:
        out, err = process.communicate(timeout=timeout)
    except BaseException as exc:
        kill_group(process)
        try:
            out, err = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # A deliberately detached daemon may retain these descriptors.
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            process.wait(timeout=5)
        else:
            if isinstance(exc, subprocess.TimeoutExpired):
                exc.output, exc.stderr = out, err
        raise
    return subprocess.CompletedProcess(argv, process.returncode, out, err)
