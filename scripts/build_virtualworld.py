#!/usr/bin/env python3
"""Build native, oracle-replayed VirtualWorld tasks from official public sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.inference.virtualworld import ALFWorldEnvironment

REPO = Path(__file__).resolve().parents[1]
_SW_ENV = None


def check_scienceworld(row):
    """One native worker owns one JVM; no environment is shared between processes."""
    global _SW_ENV
    from scienceworld import ScienceWorldEnv
    from multiprocessing.util import Finalize
    if _SW_ENV is None:
        _SW_ENV = ScienceWorldEnv('', envStepLimit=500)
        Finalize(_SW_ENV, _SW_ENV.close, exitpriority=10)
    env = _SW_ENV
    try:
        env.load(row['task_type'], row['variation_idx'], '', generateGoldPath=True)
        actions = env.get_gold_action_sequence()
        if not actions or actions[0].startswith('ERROR:') or len(actions) > 500:
            raise ValueError('No oracle within the 500-action limit')
        initial, info = env.reset()
        description = info['taskDesc']
        score, done, moves = 0, False, 0
        for action in actions:
            # Same native transition/score as ScienceWorldEnv.step, without eagerly
            # enumerating thousands of admissible object combinations for oracle QA.
            env.server.step(action)
            score = int(round(100 * env.server.getScore())); moves += 1
            done = bool(env.server.getCompleted() or score < 0 or moves >= 500)
            if done: break
        if score != 100:
            raise ValueError(f'Oracle replay failed: score={score}, moves={moves}')
        validation = {'status': 'passed', 'reset': True, 'oracle_replay': True,
                      'score': score, 'done': bool(done), 'actions_executed': moves,
                      'runtime_version': '1.2.3'}
        return {'initial': initial, 'description': description, 'actions': actions[:moves],
                'validation': validation}
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


def checked_candidates(rows, workers):
    """Bounded in-flight work and ordered consumption preserve seeded selection order."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator = iter(rows)
        pending = []
        for _ in range(workers * 4):
            row = next(iterator, None)
            if row is not None: pending.append((row, pool.submit(check_scienceworld, row)))
        while pending:
            row, future = pending.pop(0)
            yield row, future.result()
            row = next(iterator, None)
            if row is not None: pending.append((row, pool.submit(check_scienceworld, row)))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def balanced(rows, seed):
    """Seeded round-robin; exhaust small types before redistributing their quotas."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['task_type']].append(row)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    for name in names:
        groups[name].sort(key=lambda row: row['source_id'])
        rng.shuffle(groups[name])
    while any(groups.values()):
        for name in names:
            if groups[name]:
                yield groups[name].pop()


def emit(root, row, source, description, initial, actions, validation, payload=None):
    directory = root / 'tasks' / row['id']
    workspace = directory / 'workspace'
    reference = directory / 'reference'
    workspace.mkdir(parents=True)
    reference.mkdir()
    benchmark = row['benchmark']
    env = {'benchmark': benchmark, 'step_limit': 500 if benchmark == 'scienceworld' else 200,
           'runtime': {'scienceworld': '1.2.3'} if benchmark == 'scienceworld' else
                      {'alfworld': '0.4.2', 'textworld': '1.7.0', 'fast_downward_textworld': '20.6.4'}}
    if benchmark == 'scienceworld':
        env.update(task_name=row['task_type'], variation_idx=row['variation_idx'], simplification_str='')
        info = {k: env[k] for k in ['benchmark', 'task_name', 'variation_idx', 'simplification_str', 'step_limit']}
    else:
        env.update(task_type=row['task_type'], game_file='workspace/game.tw-pddl')
        info = {'benchmark': benchmark, 'game_file': 'game.tw-pddl', 'step_limit': 200}
        original = json.loads(payload['game'])
        public = {k: original[k] for k in ['pddl_domain', 'pddl_problem', 'grammar']}
        # Native engine accepts absent oracle metadata. Preserve original bytes for grading.
        write(workspace / 'game.tw-pddl', public)
        (reference / 'game.original.tw-pddl').write_bytes(payload['game'])
        (reference / 'traj_data.json').write_bytes(payload['trajectory'])
    write(workspace / 'task_info.json', info)
    shutil.copyfile(REPO / 'src/mixed_pilot_harness/inference/alfworld_runtime.py', workspace / 'native_alfworld.py')
    shutil.copyfile(REPO / 'scripts/virtualworld_client.py', workspace / 'vw.py')
    # Keep the familiar ScienceWorld command entry point.
    if benchmark == 'scienceworld':
        shutil.copyfile(REPO / 'scripts/virtualworld_client.py', workspace / 'sw.py')
    else:
        shutil.copyfile(REPO / 'scripts/virtualworld_client.py', workspace / 'alf.py')
    source_record = {'native_instance': row, 'task_description': description,
                     'runtime': env['runtime'], 'provenance': source}
    write(reference / 'source_record.json', source_record)
    write(reference / 'gold_path.json', {'actions': actions, 'final_score': 100.0,
                                        'completed': True, 'validation': validation})
    write(reference / 'runtime_validation.json', validation)
    source_manifest = {'dataset': 'ScienceWorld' if benchmark == 'scienceworld' else 'ALFWorld',
                       'upstream_id': row['source_id'], 'split': row['source_split'],
                       'revision': source['revision'], 'url': source['url'], 'license': source['license']}
    manifest = {'schema_version': '1.0', 'dataset_name': 'VirtualWorld-1000', 'id': row['id'],
                'title': source_manifest['dataset'] + ': ' + row['task_type'],
                'category': 'interactive_simulation', 'difficulty': 'native',
                'agent_environment': {'kind': 'process'}, 'source': source_manifest,
                'routing_metadata': {'router_split': row['split'], 'task_type': row['task_type']},
                'prompt': 'prompt.md',
                'workspace': {'path': 'workspace', 'mode': 'interactive_env', 'agent_visible': True,
                              'original_payload_preserved': benchmark == 'scienceworld',
                              'limitations': ['Requires pinned native Python dependencies; ScienceWorld also requires Java.',
                                              'ALFWorld PDDL initial state is readable in shell mode; infer.py uses native text actions.']},
                'agent': {'tools': ['shell', 'read_file', 'write_file'],
                          'recommended_max_turns': env['step_limit'],
                          'output_contract': {'type': 'native_environment_score', 'format': 'score.json (runtime generated)'}},
                'evaluation': {'rubrics': 'rubrics.json', 'reference_dir': 'reference',
                               'reference_agent_visible': False, 'verification_level': 'native_environment',
                               'method': 'scienceworld_terminal_score' if benchmark == 'scienceworld' else 'alfworld_native_won'},
                'privacy': {'variant': 'clean', 'contains_injected_private_data': False, 'injected_data': []},
                'environment': env}
    write(directory / 'manifest.json', manifest)
    write(directory / 'rubrics.json', {'schema_version': '1.0', 'task_id': row['id'],
          'max_score': 100, 'pass_threshold': 100, 'verification_level': 'native_environment',
          'criteria': [{'id': 'native_success', 'description': 'Native environment reaches full success.',
                        'points': 100, 'method': manifest['evaluation']['method'],
                        'expected_ref': 'reference/gold_path.json#/final_score'}]})
    prompt = f'''# Task

{description}

# Workspace

This is a native {source_manifest['dataset']} text simulation. Initial observation:

{initial}

Use `python vw.py start`, `python vw.py info`, `python vw.py valid`,
`python vw.py step "<one action>"`, and `python vw.py stop` from the workspace.
With mixed-pilot `infer.py`, issue one command in a ```text``` code block instead;
the harness owns the environment and supplies observations after each command.

# Deliverables

Complete the objective in the environment. The runtime records `score.json`.
Scores use a 0–100 scale; full success is 100. Self-reported files do not determine success.

# Constraints

- Maximum native actions: {env['step_limit']}.
- A terminal episode can be unsuccessful. Stop after native success or termination.
- Keep file operations within the workspace. Reference files are grader-only.
'''
    (directory / 'prompt.md').write_text(prompt)
    (directory / 'prompt.txt').write_text(prompt)
    return {'id': row['id'], 'path': 'tasks/' + row['id'], 'benchmark': benchmark,
            'task_type': row['task_type'], 'source_split': row['source_split']}


def scienceworld(root, sources, seed, rejected, workers=3):
    from scienceworld import ScienceWorldEnv
    from importlib.metadata import version
    if version('scienceworld') != '1.2.3':
        raise RuntimeError('ScienceWorld must be 1.2.3')
    revision = json.loads((sources / 'scienceworld-revision.json').read_text())['sha']
    provenance = {'revision': 'scienceworld==1.2.3', 'upstream_repo_revision': revision,
                  'url': 'https://github.com/allenai/ScienceWorld', 'license': 'Apache-2.0',
                  'extraction': 'native variation enumeration and generated oracle, empty simplifications'}
    env = ScienceWorldEnv('', envStepLimit=500)
    result = defaultdict(list)
    try:
        pools = defaultdict(list)
        for name in env.get_task_names():
            env.load(name, 0, '')
            for split in ['train', 'test']:
                for var in getattr(env, 'get_variations_' + split)():
                    pools[split].append({'id': f'scienceworld-{name}-var{var}', 'benchmark': 'scienceworld',
                        'task_type': name, 'variation_idx': var, 'source_id': f'{name}/var{var}',
                        'source_split': split, 'split': split})
        for split, count in [('train', 400), ('test', 100)]:
            existing = {}
            for path in (root / 'tasks').glob('scienceworld-*/manifest.json'):
                manifest = json.loads(path.read_text())
                if manifest['routing_metadata']['router_split'] != split: continue
                validation = json.loads((path.parent / 'reference/runtime_validation.json').read_text())
                if validation['status'] != 'passed' or validation['score'] != 100: raise ValueError('Invalid resume task')
                existing[manifest['source']['upstream_id']] = {'id': manifest['id'], 'path': 'tasks/' + manifest['id'],
                    'benchmark': 'scienceworld', 'task_type': manifest['environment']['task_name'], 'source_split': split}
            result[split].extend(existing.values())
            rejected_ids = {r['instance']['source_id'] for r in rejected
                            if r['instance']['benchmark'] == 'scienceworld'}
            candidates = [row for row in balanced(pools[split], seed)
                          if row['source_id'] not in existing and row['source_id'] not in rejected_ids]
            if len(existing) > count: raise ValueError('Resume count exceeds target')
            if len(existing) == count: continue
            for row, checked in checked_candidates(candidates, workers):
                try:
                    if 'error' in checked: raise ValueError(checked['error'])
                    result[split].append(emit(root, row, provenance, checked['description'], checked['initial'],
                                              checked['actions'], checked['validation']))
                    if len(result[split]) % 20 == 0:
                        print(f'ScienceWorld {split}: {len(result[split])}/{count}', flush=True)
                except Exception as exc:
                    rejected.append({'instance': row, 'error': f'{type(exc).__name__}: {exc}'})
                    print('Rejected ScienceWorld', row['source_id'], str(exc)[:150], flush=True)
                if len(result[split]) == count: break
            if len(result[split]) != count: raise RuntimeError(f'Insufficient ScienceWorld {split} candidates')
    finally:
        env.close()
    return result


def alfworld(root, sources, seed, rejected):
    gamezip = sources / 'json_2.1.3_tw-pddl.zip'
    recordzip = sources / 'json_2.1.1_json.zip'
    provenance = {'revision': 'ALFWorld-0.4.2/json_2.1.3_tw-pddl',
                  'upstream_repo_revision': json.loads((sources / 'alfworld-revision.json').read_text())['sha'],
                  'url': 'https://github.com/alfworld/alfworld', 'license': 'MIT',
                  'games_url': 'https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip',
                  'games_sha256': sha(gamezip.read_bytes()), 'records_sha256': sha(recordzip.read_bytes()),
                  'records_release': '0.2.2/json_2.1.1_json',
                  'extraction': 'unique trials; prioritize unique groups; reserve evaluation groups; original walkthrough replay'}
    result = defaultdict(list)
    with zipfile.ZipFile(gamezip) as games, zipfile.ZipFile(recordzip) as records:
        record_names = set(records.namelist())
        pools = defaultdict(list)
        for name in sorted(games.namelist()):
            if not name.endswith('/game.tw-pddl'): continue
            _, native_split, group, trial, _ = name.split('/')
            if native_split not in ['train', 'valid_seen', 'valid_unseen']: continue
            record_name = name.replace('/game.tw-pddl', '/traj_data.json')
            if record_name not in record_names: continue
            task_type = group.split('-')[0]
            source_id = '/'.join([native_split, group, trial])
            pools[native_split].append({'id': 'alfworld-' + native_split.replace('_', '-') + '-' + sha(source_id.encode())[:16],
                'benchmark': 'alfworld', 'task_type': task_type, 'source_id': source_id,
                'source_split': native_split, 'split': 'train' if native_split == 'train' else 'test',
                'group': group, 'game_entry': name, 'record_entry': record_name})
        used_groups = Counter()
        reserved_groups = {row['group'] for s in ['valid_seen', 'valid_unseen'] for row in pools[s]}
        temp = root / '.validation-game.tw-pddl'
        for native_split, count in [('train', 400), ('valid_seen', 50), ('valid_unseen', 50)]:
            accepted = 0
            candidates = list(balanced(pools[native_split], seed))
            first, rest, seen_groups = [], [], set()
            for row in candidates:
                if row['group'] in seen_groups: rest.append(row)
                else: first.append(row); seen_groups.add(row['group'])
            for row in first + rest:
                group_key = row['group']
                if native_split == 'train' and group_key in reserved_groups: continue
                cap = 2 if native_split == 'valid_unseen' else 1
                if used_groups[group_key] >= cap: continue
                env = None
                try:
                    raw_game = games.read(row['game_entry']); raw_record = records.read(row['record_entry'])
                    game = json.loads(raw_game); record = json.loads(raw_record)
                    if record['task_type'] != row['task_type']: raise ValueError('Trajectory task_type mismatch')
                    actions = game.get('walkthrough')
                    if not game.get('solvable') or not actions or len(actions) > 200:
                        raise ValueError('No solvable official walkthrough')
                    public = {k: game[k] for k in ['pddl_domain', 'pddl_problem', 'grammar']}
                    write(temp, public)
                    env = ALFWorldEnvironment(temp)
                    initial = env.load_task()
                    description = initial['observation'].split('Your task is to:')[-1].strip()
                    score, moves, done = 0, 0, False
                    for action in actions:
                        transition = env.step(action); moves += 1
                        score, done = transition['score'], transition['done']
                        if done: break
                    if score != 100: raise ValueError(f'Official walkthrough failed: score={score}, moves={moves}')
                    validation = {'status': 'passed', 'reset': True, 'oracle_replay': True, 'won': True,
                                  'score': score, 'done': bool(done), 'actions_executed': moves,
                                  'runtime_version': {'alfworld': '0.4.2', 'textworld': '1.7.0'}}
                    result[row['split']].append(emit(root, row, provenance, description, initial['observation'],
                        actions[:moves], validation, {'game': raw_game, 'trajectory': raw_record}))
                    used_groups[group_key] += 1; accepted += 1
                    if accepted % 20 == 0: print(f'ALFWorld {native_split}: {accepted}/{count}', flush=True)
                except Exception as exc:
                    rejected.append({'instance': row, 'error': f'{type(exc).__name__}: {exc}'})
                    print('Rejected ALFWorld', row['source_id'], str(exc)[:150], flush=True)
                finally:
                    if env is not None: env.close()
                if accepted == count: break
            if accepted != count: raise RuntimeError(f'Insufficient ALFWorld {native_split} candidates')
        temp.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/VirtualWorld-1000')
    parser.add_argument('--sources', type=Path, default=REPO / 'outputs/virtualworld-sources')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--benchmark', choices=['scienceworld', 'alfworld', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume-build', action='store_true', help='Resume verified ScienceWorld payloads only')
    args = parser.parse_args()
    java = Path(sys.prefix) / 'lib/jvm'
    if not (java / 'bin/java').exists():
        java = Path(sys.base_prefix) / 'lib/jvm'
    if (java / 'bin/java').exists():
        os.environ.setdefault('JAVA_HOME', str(java))
        os.environ['PATH'] = str(java / 'bin') + os.pathsep + os.environ.get('PATH', '')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rejected = []
    results = defaultdict(list)
    try:
        for name in ['scienceworld', 'alfworld']:
            if args.benchmark not in ['both', name]: continue
            existing = sorted((root / 'tasks').glob(name + '-*/manifest.json'))
            if existing and not (name == 'scienceworld' and args.resume_build):
                raise ValueError(f'Refusing to overwrite {name} tasks')
            if args.workers <= 0: raise ValueError('workers must be positive')
            if args.resume_build and (root / 'scienceworld.rejections.json').exists():
                rejected.extend(json.loads((root / 'scienceworld.rejections.json').read_text()))
            result = (scienceworld(root, args.sources, args.seed, rejected, args.workers) if name == 'scienceworld'
                      else alfworld(root, args.sources, args.seed, rejected))
            for split, rows in result.items(): results[split].extend(rows)
            write(root / (name + '.index.json'), {'seed': args.seed, 'splits': result})
    finally:
        write(root / (args.benchmark + '.rejections.json'), rejected)
    # Separate benchmark builds can be run concurrently and assembled afterward.
    if all((root / (name + '.index.json')).exists() for name in ['scienceworld', 'alfworld']):
        results = defaultdict(list)
        for name in ['scienceworld', 'alfworld']:
            index = json.loads((root / (name + '.index.json')).read_text())
            if index['seed'] != args.seed: raise ValueError('Benchmark seed mismatch')
            for split, rows in index['splits'].items(): results[split].extend(rows)
        for split, rows in results.items():
            rows.sort(key=lambda row: sha((str(args.seed) + row['id']).encode()))
            write(root / (split + '.json'), {'schema_version': '1.0', 'dataset': 'VirtualWorld-1000',
                                          'split': split, 'task_count': len(rows), 'tasks': rows})
        summary = {'dataset': 'VirtualWorld-1000', 'seed': args.seed, 'task_count': sum(map(len, results.values())),
                   'splits': {s: len(r) for s, r in results.items()},
                   'benchmarks': dict(Counter(row['benchmark'] for rows in results.values() for row in rows)),
                   'strata': {s: dict(Counter(row['benchmark'] + '/' + row['task_type'] for row in rows)) for s, rows in results.items()}}
        write(root / 'dataset.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
