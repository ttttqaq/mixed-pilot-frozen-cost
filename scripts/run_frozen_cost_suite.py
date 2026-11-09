#!/usr/bin/env python3
"""Launch/resume three isolated DB28 jobs, each on one free GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import socket
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False


def jobs(suite, gpus, python):
    credentials = ['--credentials-config', 'configs/online-router.reference-teacher.db94.yaml']
    old = ROOT/'outputs/ksr-reference-router-run01'
    online = [python, 'scripts/train_frozen_cost_online.py', '--config',
              'configs/online-router.frozen-cost.online9.db28.yaml',
              '--init-cold-checkpoint', str(old/'cold_start/checkpoints/step_000500.pt'),
              '--budget-cold-source', str(old), '--output-dir', str(suite/'online9'), *credentials]
    result = {'online': {'gpu': gpus[0], 'command': online, 'output': str(suite/'online9')}}
    for i, name in enumerate(('run01_replay', 'run02_replay'), 1):
        source = ROOT/'outputs'/('ksr-reference-router-run0'+str(i))
        result[name] = {'gpu': gpus[i], 'output': str(suite/name), 'command': [python,
            'scripts/replay_ksr_cold.py', '--config', 'configs/online-router.ksr-reference.db94.yaml',
            '--source-run', str(source), '--output-dir', str(suite/name), '--seed', '42', *credentials]}
    return result


def worker(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    with spec_path.with_suffix('.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 2
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['gpu']), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1')
        command = list(spec['command'])
        if spec.get('resume'):
            command.append('--resume')
        status_path = spec_path.with_suffix('.status.json')
        state = {'job': spec['job'], 'gpu': spec['gpu'], 'worker_pid': os.getpid(),
                 'host': socket.gethostname(), 'started': now(), 'command': command, 'status': 'starting'}
        save(status_path, state)
        with spec_path.with_suffix('.log').open('a') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', pid=proc.pid)
            save(status_path, state)
            code = proc.wait()
        state.update(status='exited', exit_code=code, finished=now())
        save(status_path, state)
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-dir', type=Path, default=ROOT/'outputs/frozen-cost-suite-20261103')
    p.add_argument('--gpus', default='2,3,4')
    p.add_argument('--python', default='/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python')
    p.add_argument('--min-free-mib', type=int, help='Explicitly allow shared GPU when this much memory is free')
    p.add_argument('--only', choices=('online', 'run01_replay', 'run02_replay'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker', type=Path)
    args = p.parse_args()
    if args.worker:
        return worker(args.worker)
    suite = args.suite_dir.expanduser().resolve()
    gpus = [int(v) for v in args.gpus.split(',')]
    if len(gpus) != 3 or len(set(gpus)) != 3:
        p.error('Specify three distinct GPU indices')
    definitions = jobs(suite, gpus, args.python)
    if args.dry_run:
        print(json.dumps(definitions, indent=2))
        return 0
    suite.mkdir(parents=True, exist_ok=True)
    processes = suite/'processes'
    processes.mkdir(exist_ok=True)
    plan_path = suite/'suite.json'
    plan = {'created': now(), 'host': socket.gethostname(), 'root': str(ROOT), 'jobs': definitions,
            'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        saved_jobs = {n: {k: v for k, v in d.items() if k != 'host'} for n, d in saved['jobs'].items()}
        if saved_jobs != definitions:
            raise ValueError('Existing suite job definitions/GPU assignments differ')
    else:
        save(plan_path, plan)
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    occupied = {int(x.split(',')[0]): (int(x.split(',')[1]), int(x.split(',')[2])) for x in memory.splitlines()}
    for name, spec in definitions.items():
        if args.only and name != args.only:
            continue
        if plan_path.exists():
            owner = json.loads(plan_path.read_text())['jobs'][name].get('host')
            if owner and owner != socket.gethostname():
                raise ValueError(name+' is assigned to '+owner)
        path = processes/(name+'.json')
        status_path = path.with_suffix('.status.json')
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('status') != 'exited' and alive(status.get('worker_pid')):
                print(name, 'already running', status.get('pid'))
                continue
            if status.get('exit_code') == 0:
                print(name, 'already exited successfully')
                continue
            if not args.resume:
                raise ValueError(name+' previously stopped; inspect the log, then explicitly --resume')
        used, total = occupied.get(spec['gpu'], (100000, 0))
        available = total-used >= args.min_free_mib if args.min_free_mib is not None else used <= 512
        if not available:
            raise ValueError('GPU '+str(spec['gpu'])+' is occupied; no process was displaced')
        save(path, dict(spec, job=name, resume=args.resume))
        with path.with_suffix('.worker.log').open('a') as log:
            proc = subprocess.Popen([args.python, str(Path(__file__).resolve()), '--worker', str(path)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(name, 'worker_pid', proc.pid, 'gpu', spec['gpu'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
