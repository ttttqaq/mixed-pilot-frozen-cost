#!/usr/bin/env python3
"""Audit VirtualWorld structure, checksums, native identities and oracle evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.inference.dataset import load_dataset, inside


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root):
    root = root.resolve()
    tasks = load_dataset(root, 'all')
    counts = Counter(t['benchmark'] for t in tasks)
    if counts != {'scienceworld': 500, 'alfworld': 500}:
        raise ValueError(f'Expected 500 + 500 native tasks, got {counts}')
    splits = Counter((t['benchmark'], t['split']) for t in tasks)
    if splits != {('scienceworld', 'train'): 400, ('scienceworld', 'test'): 100,
                  ('alfworld', 'train'): 400, ('alfworld', 'test'): 100}:
        raise ValueError(f'Unexpected split balance: {splits}')
    dirs = sorted((root / 'tasks').iterdir())
    if len(dirs) != 1000: raise ValueError('Task directories and indexes differ')
    indexed = {t['source_task_id'] for t in tasks}
    native_groups = {}
    group_counts = Counter()
    for directory in dirs:
        if directory.name not in indexed: raise ValueError('Unindexed task: ' + directory.name)
        m = read(directory / 'manifest.json'); r = read(directory / 'rubrics.json')
        required = {'schema_version', 'id', 'title', 'category', 'source', 'prompt',
                    'workspace', 'agent', 'evaluation', 'privacy'}
        if not required.issubset(m) or m['schema_version'] != '1.0' or not re.fullmatch('[a-z0-9][a-z0-9-]+', m['id']):
            raise ValueError('Invalid mixed-pilot manifest schema')
        workspace = inside(directory, m['workspace']['path'])
        reference = inside(directory, m['evaluation']['reference_dir'])
        if m['id'] != directory.name or r['task_id'] != m['id']: raise ValueError('Task ID mismatch')
        if reference.is_relative_to(workspace) or m['evaluation']['reference_agent_visible'] is not False:
            raise ValueError('Reference must be outside workspace')
        if (directory / 'prompt.txt').read_bytes() != (directory / 'prompt.md').read_bytes():
            raise ValueError('Prompt aliases differ')
        if r['max_score'] != 100 or r['pass_threshold'] != 100 or sum(c['points'] for c in r['criteria']) != 100:
            raise ValueError('Invalid rubric score scale')
        validation = read(reference / 'runtime_validation.json')
        gold = read(reference / 'gold_path.json')
        if validation['status'] != 'passed' or validation['score'] != 100 or not validation['oracle_replay']:
            raise ValueError('Missing successful native oracle replay')
        if len(gold['actions']) != validation['actions_executed'] or len(gold['actions']) > m['environment']['step_limit']:
            raise ValueError('Gold path action count/limit mismatch')
        if any(p.is_symlink() for p in directory.rglob('*')): raise ValueError('Symlink in task payload')
        if any(p.name in {'gold_path.json', 'traj_data.json', 'source_record.json'} for p in workspace.rglob('*')):
            raise ValueError('Reference artifact in public workspace')
        if m['environment']['benchmark'] == 'alfworld':
            game = read(workspace / 'game.tw-pddl')
            if set(game) != {'pddl_domain', 'pddl_problem', 'grammar'}:
                raise ValueError('Public ALFWorld game contains oracle metadata')
            original = read(reference / 'game.original.tw-pddl')
            if any(original[k] != game[k] for k in game): raise ValueError('Public game native state was modified')
            source_id = m['source']['upstream_id'].split('/')
            group = source_id[1]
            experiment_split = m['routing_metadata']['router_split']
            if group in native_groups and native_groups[group] != experiment_split:
                raise ValueError('ALFWorld task group leaks across train/test')
            native_groups[group] = experiment_split
            group_counts[group] += 1
            cap = 2 if m['source']['split'] == 'valid_unseen' else 1
            if group_counts[group] > cap: raise ValueError('ALFWorld group exceeds sampling cap')
        else:
            info = read(workspace / 'task_info.json')
            if any(info[k] != m['environment'][k] for k in ['task_name', 'variation_idx', 'simplification_str', 'step_limit']):
                raise ValueError('ScienceWorld workspace descriptor differs from manifest')
            if m['source']['split'] != m['routing_metadata']['router_split']:
                raise ValueError('ScienceWorld official split changed')
    expected = set()
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        path = inside(root, name)
        if name in expected or digest(path) != checksum: raise ValueError('Duplicate or mismatched checksum: ' + name)
        expected.add(name)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p.name not in {'SHA256SUMS', 'validation_report.json'} and '__pycache__' not in p.parts}
    if actual != expected: raise ValueError('Checksum inventory does not cover the dataset exactly')
    sw_types = len({t['task_name'] for t in tasks if t['benchmark'] == 'scienceworld'})
    alf_types = len({t['task_name'] for t in tasks if t['benchmark'] == 'alfworld'})
    if sw_types != 30 or alf_types != 6: raise ValueError('Dataset must cover all 30 + 6 native task types')
    return {'status': 'passed', 'task_count': len(tasks), 'benchmarks': dict(counts),
            'splits': {s: sum(t['split'] == s for t in tasks) for s in ['train', 'test']},
            'checked_files': len(expected), 'native_oracle_replays': 1000,
            'scienceworld_task_types': sw_types, 'alfworld_task_types': alf_types}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    result = validate(args.dataset)
    if args.write_report:
        (args.dataset / 'validation_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
