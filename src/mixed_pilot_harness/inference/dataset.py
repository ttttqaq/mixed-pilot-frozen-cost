"""Load indexed train/test datasets or existing ScienceWorld/HLE manifests."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

from ..collection.tasks import _hle_manifest, _manifest_task, validate_tasks


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Dataset path escapes its root: {value}')
    return path


def load_dataset(root: Path, split: str):
    root = root.resolve()
    splits = ['train', 'test'] if split in {'all', 'train-test'} else [split]
    manifests = []
    indexed = any((root / f'{s}.json').exists() for s in splits)
    if indexed:
        for s in splits:
            data = read(root / f'{s}.json')
            rows = data['tasks']
            if data.get('split', s) != s or data.get('task_count', len(rows)) != len(rows):
                raise ValueError(f'Invalid {s} index')
            for row in rows:
                directory = inside(root, row['path'])
                manifests.append((directory / 'manifest.json', s, row.get('id')))
    else:
        candidates = ([root / 'manifest.json'] if (root / 'manifest.json').exists() else
                      sorted(root.glob('tasks/*/manifest.json')) or sorted(root.glob('*/manifest.json')))
        manifests = [(path, None, None) for path in candidates]
    tasks, seen = [], set()
    for path, index_split, index_id in manifests:
        m = read(path)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', m.get('id', '')):
            raise ValueError(f'Invalid task ID in {path}')
        actual_split = m.get('routing_metadata', {}).get('router_split') or m.get('source', {}).get('split')
        if index_split and actual_split != index_split:
            raise ValueError(f'Index/manifest split mismatch: {path}')
        if index_id and index_id != m['id']:
            raise ValueError(f'Index/manifest ID mismatch: {path}')
        if actual_split not in splits:
            continue
        if m.get('environment', {}).get('benchmark') == 'scienceworld':
            task = _manifest_task(m)
            task['split'] = actual_split
            validate_tasks([task])
            if (m.get('dataset_family') == 'VirtualWorld'
                    or m.get('dataset_name') == 'VirtualWorld-1000'):
                from .virtualworld import fingerprint
                task['source_split'] = m['source']['split']
                task['env_step_limit'] = m['environment']['step_limit']
                if type(task['env_step_limit']) is not int or task['env_step_limit'] <= 0:
                    raise ValueError('ScienceWorld step_limit must be positive')
                task['content_hashes'] = {
                    'manifest': fingerprint(path),
                    'prompt': fingerprint(inside(path.parent, m.get('prompt', 'prompt.md'))),
                    'task_info': fingerprint(inside(path.parent, 'workspace/task_info.json'))}
        elif m.get('environment', {}).get('benchmark') == 'alfworld':
            from .virtualworld import manifest_task
            task = manifest_task(m, path.parent)
        elif 'hle' in m.get('source', {}).get('dataset', '').lower():
            task = _hle_manifest(m, path.parent)
            validate_tasks([task])
        else:
            # SWE-neo descriptors are prepared with the existing native runtime.
            record = inside(path.parent, m.get('evaluation', {}).get('reference_dir', 'reference')) / 'source_record.json'
            row = read(record)
            from .swe_runtime import image_for
            image_for(row)
            source = m.get('source', {})
            if row.get('instance_id') != source.get('upstream_id') or row.get('base_commit') != source.get('base_commit'):
                raise ValueError(f'SWE source provenance mismatch: {path}')
            prompt = inside(path.parent, m.get('prompt', 'prompt.md'))
            task = {'task_id': m['id'], 'benchmark': 'swe', 'split': actual_split,
                    'task_name': m['id'], 'task_dir': str(path.parent), 'prompt_path': str(prompt),
                    'record_path': str(record), 'allowed_tools': m.get('agent', {}).get('tools', []),
                    'content_hashes': {'record': hashlib.sha256(record.read_bytes()).hexdigest(),
                                       'prompt': hashlib.sha256(prompt.read_bytes()).hexdigest()}}
        key = (task['split'], task['task_id'])
        if key in seen:
            raise ValueError(f'Duplicate task: {key}')
        seen.add(key)
        # The engine keys jobs by task_id; preserve original ID separately when needed.
        task['source_task_id'] = task['task_id']
        task['task_id'] = task['split'] + '--' + task['task_id']
        tasks.append(task)
    if not tasks:
        raise ValueError('No tasks selected; expected train/test indexes or task manifests')
    instances = set()
    for task in tasks:
        if task['benchmark'] == 'scienceworld':
            instance = ('scienceworld', task['task_name'], task['variation_idx'])
        elif task['benchmark'] == 'alfworld':
            instance = ('alfworld', task['source_id'])
        else:
            continue
        if instance in instances:
            raise ValueError(f'Duplicate native instance across dataset splits: {instance}')
        instances.add(instance)
    return tasks
