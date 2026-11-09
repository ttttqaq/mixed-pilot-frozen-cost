#!/usr/bin/env python3
"""Standalone per-workspace native environment server/client with bounded lifecycle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / '.vw_session.json'


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def request(command, action=None):
    if not STATE.exists(): raise RuntimeError('Run python vw.py start first')
    session = json.loads(STATE.read_text())
    with socket.create_connection(('127.0.0.1', session['port']), timeout=30) as sock:
        sock.settimeout(120)
        sock.sendall((json.dumps({'command': command, 'action': action, 'token': session['token']}) + '\n').encode())
        line = sock.makefile().readline(4 * 1024 * 1024)
        if not line: raise RuntimeError('Environment server returned no response; inspect .vw_server.log')
        result = json.loads(line)
        if 'error' in result: raise RuntimeError(result['error'])
        return result


def start():
    if STATE.exists():
        try: return request('info')
        except (OSError, RuntimeError, ValueError): STATE.unlink(missing_ok=True)
    lock = ROOT / '.vw_start.lock'
    if lock.exists():
        try:
            owner = json.loads(lock.read_text())
            os.kill(owner['pid'], 0)
        except ProcessLookupError: lock.unlink(missing_ok=True)
        except (ValueError, KeyError):
            if time.time() - lock.stat().st_mtime > 120: lock.unlink(missing_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, json.dumps({'pid': os.getpid()}).encode())
    try:
        with (ROOT / '.vw_server.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--server'],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if STATE.exists(): return request('info')
            if proc.poll() is not None: raise RuntimeError('Startup failed; inspect .vw_server.log')
            time.sleep(0.1)
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        raise RuntimeError('Startup timed out; inspect .vw_server.log')
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def serve():
    info = json.loads((ROOT / 'task_info.json').read_text())
    benchmark = info['benchmark']
    if benchmark == 'scienceworld':
        java = Path(sys.prefix) / 'lib/jvm'
        if not (java / 'bin/java').exists(): java = Path(sys.base_prefix) / 'lib/jvm'
        if (java / 'bin/java').exists():
            os.environ.setdefault('JAVA_HOME', str(java))
            os.environ['PATH'] = str(java / 'bin') + os.pathsep + os.environ.get('PATH', '')
        from importlib.metadata import version
        from scienceworld import ScienceWorldEnv
        if version('scienceworld') != '1.2.3': raise RuntimeError('Requires scienceworld==1.2.3')
        env = ScienceWorldEnv('', envStepLimit=info['step_limit'])
        try:
            env.load(info['task_name'], info['variation_idx'], info['simplification_str'])
            observation, state = env.reset()
        except BaseException:
            env.close(); raise
        description = state['taskDesc']
        def advance(action):
            obs, reward, done, native = env.step(action)
            return {'observation': obs, 'reward': reward, 'score': native['score'], 'done': bool(done),
                    'won': native['score'] == 100}
        def valid(): return env.get_valid_action_object_combinations()
    else:
        from native_alfworld import ALFWorldEnvironment
        game = (ROOT / info['game_file']).resolve()
        if not game.is_relative_to(ROOT): raise ValueError('game_file escapes workspace')
        env = ALFWorldEnvironment(game, info['step_limit'])
        initial = env.load_task(); observation = initial['observation']; description = initial['task_description']
        advance = env.step
        def valid(): return list(env.state.admissible_commands)
    current = {'benchmark': benchmark, 'task_description': description,
               'observation': observation, 'score': 0.0, 'done': False, 'won': False, 'moves': 0}
    token = secrets.token_hex(24)
    last_request = [time.monotonic()]

    def persist(): atomic(ROOT / 'score.json', current)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(30)
            try:
                request = json.loads(self.rfile.readline(65536))
                if request.get('token') != token: raise ValueError('Invalid session token')
                last_request[0] = time.monotonic()
                command = request.get('command')
                if command == 'info': result = current
                elif command == 'valid': result = {'valid_actions': valid()}
                elif command == 'step':
                    action = request.get('action')
                    if not isinstance(action, str) or not action.strip(): raise ValueError('step requires an action')
                    if current['done']: raise ValueError('Episode is already done')
                    current.update(advance(action))
                    if not action.startswith('?'): current['moves'] += 1
                    if current['moves'] >= info['step_limit']: current['done'] = True
                    persist(); result = current
                elif command == 'stop':
                    persist(); result = dict(current, stopped=True)
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else: raise ValueError('Unknown command')
            except Exception as exc: result = {'error': f'{type(exc).__name__}: {exc}'}
            self.wfile.write((json.dumps(result) + '\n').encode()); self.wfile.flush()

    server = socketserver.TCPServer(('127.0.0.1', 0), Handler)
    def idle_watch():
        # Clean up orphaned episodes after 15 minutes with no requests.
        while time.monotonic() - last_request[0] < 900: time.sleep(5)
        server.shutdown()
    try:
        persist()
        atomic(STATE, {'port': server.server_address[1], 'pid': os.getpid(), 'token': token})
        STATE.chmod(0o600)
        threading.Thread(target=idle_watch, daemon=True).start()
        server.serve_forever(poll_interval=0.1)
    finally:
        env.close(); server.server_close(); STATE.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['start', 'info', 'step', 'valid', 'stop'])
    parser.add_argument('action', nargs='?')
    parser.add_argument('--server', action='store_true')
    args = parser.parse_args()
    try:
        if args.server: serve(); return 0
        if args.command is None: parser.error('command is required')
        result = start() if args.command == 'start' else request(args.command, args.action)
        print(json.dumps(result)); return 0
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'})); return 1


if __name__ == '__main__': raise SystemExit(main())
