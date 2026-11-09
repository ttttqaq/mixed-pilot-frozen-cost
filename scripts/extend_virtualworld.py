#!/usr/bin/env python3
"""Stage additional distinct VirtualWorld training tasks without altering old payloads."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.inference.dataset import inside, load_dataset


def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def inventory(root):
    result = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        if name in result: raise ValueError('Duplicate checksum entry')
        result[name] = checksum
    return result


def prepare(source, target, stage, count, seed):
    source, target, stage = source.resolve(), target.resolve(), stage.resolve()
    if stage.exists(): raise ValueError('Staging directory already exists')
    original = inventory(target)
    for name, checksum in original.items():
        if digest(inside(target, name)) != checksum: raise ValueError('Existing checksum mismatch: ' + name)
    train = read(target / 'train.json')
    if read(target / 'test.json')['tasks']: raise ValueError('Expected a training-only subset')
    old_ids = {row['id'] for row in train['tasks']}
    if train['task_count'] != len(old_ids): raise ValueError('Existing index count or IDs invalid')
    source_rows = []
    for split in ['train', 'test']:
        index = read(source / (split + '.json'))
        if index['task_count'] != len(index['tasks']): raise ValueError('Invalid parent index count')
        source_rows.extend(dict(row, parent_split=split) for row in index['tasks'])
    source_ids = {row['id'] for row in source_rows}
    if len(source_ids) != len(source_rows) or not old_ids.issubset(source_ids):
        raise ValueError('Invalid parent IDs or existing tasks absent from parent')
    pool = sorted((row for row in source_rows if row['id'] not in old_ids), key=lambda row: row['id'])
    if not 0 < count <= len(pool): raise ValueError('Insufficient distinct candidates')
    selected = random.Random(seed).sample(pool, count)
    source_inventory = inventory(source)
    shutil.copytree(target, stage, ignore=shutil.ignore_patterns('.DS_Store', '__pycache__'))
    dataset_name = read(target / 'dataset.json')['dataset']
    selection = read(stage / 'selection.json')
    if {row['id'] for row in selection['tasks']} != old_ids: raise ValueError('Selection/index mismatch')
    round_number = len(selection.get('sampling_rounds', [None])) + 1
    checked = 0
    for row in selected:
        if row['path'] != 'tasks/' + row['id']: raise ValueError('Unexpected task path')
        origin = inside(source, row['path']); destination = stage / row['path']
        for path in origin.rglob('*'):
            if path.is_symlink(): raise ValueError('Task symlinks are not allowed')
            if not path.is_file() or path.name == '.DS_Store' or '__pycache__' in path.parts: continue
            name = str(path.relative_to(source))
            if source_inventory.get(name) != digest(path): raise ValueError('Source checksum mismatch: ' + name)
            copied = destination / path.relative_to(origin)
            copied.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, copied); checked += 1
        manifest_path = destination / 'manifest.json'; manifest = read(manifest_path)
        if manifest['id'] != row['id'] or manifest['routing_metadata']['router_split'] != row['parent_split']:
            raise ValueError('Parent manifest/index mismatch')
        manifest['parent_dataset_name'] = manifest.get('dataset_name', 'VirtualWorld-1000')
        manifest.update(dataset_name=dataset_name, dataset_family='VirtualWorld')
        manifest['routing_metadata'].update(router_split='train', parent_router_split=row['parent_split'])
        write(manifest_path, manifest)
        train['tasks'].append({k: v for k, v in row.items() if k != 'parent_split'})
        selection['tasks'].append(dict(row, sampling_round=round_number,
            original_manifest_sha256=source_inventory[row['path'] + '/manifest.json']))
    train['task_count'] = len(train['tasks']); write(stage / 'train.json', train)
    rounds = selection.setdefault('sampling_rounds', [{'round': 1, 'seed': selection['seed'],
        'sampling_pool_count': len(source_rows), 'count': len(old_ids),
        'task_ids': [row['id'] for row in selection['tasks'][:len(old_ids)]]}])
    extension = {'round': round_number, 'seed': seed, 'sampling_pool_count': len(pool), 'count': count,
                 'task_ids': [row['id'] for row in selected], 'excluded_existing_task_ids': sorted(old_ids)}
    rounds.append(extension); write(stage / 'selection.json', selection)
    summary = read(stage / 'dataset.json')
    summary.update(task_count=train['task_count'], splits={'train': train['task_count'], 'test': 0},
        benchmarks=dict(Counter(row['benchmark'] for row in train['tasks'])),
        parent_splits=dict(Counter(row['parent_split'] for row in selection['tasks'])),
        sampling='uniform random without replacement per round; exclude existing tasks before each extension',
        sampling_rounds=[{k: v for k, v in item.items() if k not in {'task_ids', 'excluded_existing_task_ids'}} for item in rounds])
    write(stage / 'dataset.json', summary)
    provenance = read(stage / 'provenance.json')
    provenance.setdefault('extensions', []).append(dict(extension,
        extended_at_utc=datetime.now(timezone.utc).isoformat(),
        previous_inventory_sha256=digest(target / 'SHA256SUMS'),
        parent_inventory_sha256=digest(source / 'SHA256SUMS')))
    provenance['sampling'] = summary['sampling']; write(stage / 'provenance.json', provenance)
    tasks = load_dataset(stage, 'all')
    if len(tasks) != train['task_count'] or any(t['split'] != 'train' for t in tasks):
        raise ValueError('Harness loading failed')
    for task in tasks:
        root = stage / 'tasks' / task['source_task_id']
        evidence = read(root / 'reference/runtime_validation.json')
        if evidence['status'] != 'passed' or evidence['score'] != 100: raise ValueError('Missing native evidence')
        manifest = read(root / 'manifest.json')
        if manifest['evaluation']['reference_agent_visible'] is not False: raise ValueError('Reference not hidden')
        if task['benchmark'] == 'scienceworld' and task.get('env_step_limit') != 500:
            raise ValueError('ScienceWorld action limit lost')
    for name, checksum in original.items():
        if name.startswith('tasks/') and digest(stage / name) != checksum: raise ValueError('Old task payload changed')
    report = {'status': 'passed', 'task_count': len(tasks), 'added_tasks': count,
        'previous_tasks': len(old_ids), 'overlap_with_previous_tasks': 0,
        'benchmarks': summary['benchmarks'], 'added_benchmarks': dict(Counter(r['benchmark'] for r in selected)),
        'harness_loaded_tasks': len(tasks), 'existing_files_verified': len(original),
        'added_source_files_verified': checked, 'original_task_payloads_unchanged': True,
        'inherited_native_oracle_replays': len(tasks), 'new_native_oracle_replays': 0, 'reference_isolation': True}
    write(stage / 'validation_report.json', report)
    (stage / 'README.md').write_text(f'''# {dataset_name}（已扩展）

目录名保留为 {dataset_name}，当前包含 {len(tasks)} 条训练任务，测试集为 0 条。
保留原有 {len(old_ids)} 条，使用随机种子 {seed} 从 `{source}` 的剩余 {len(pool)} 条任务
中无放回随机新增 {count} 条。新增样本与原样本没有重复，原任务文件内容未修改。

ScienceWorld {summary['benchmarks'].get('scienceworld', 0)} 条，ALFWorld {summary['benchmarks'].get('alfworld', 0)} 条。
来源 parent train {summary['parent_splits'].get('train', 0)} 条，parent test {summary['parent_splits'].get('test', 0)} 条。
原 test 来源任务用于训练后，应从父数据集评测集排除，具体名单见 selection.json。
所有任务统一 router_split=train，原始 source.split、parent_router_split 和评测 reference 保留。

selection.json 记录抽样轮次与新增 ID，provenance.json 记录扩展来源。
validation_report.json 记录去重、文件完整性、harness 加载检查；原生重放证据继承自父数据集，未重新运行环境。
SHA256SUMS 覆盖全部数据文件。

在 mixed-pilot 项目目录运行：

```bash
outputs/virtualworld-venv/bin/python scripts/infer.py --dataset {target} --split train --mode single-model --models deepseek-flash --step-limit 500 --output-dir run/virtualworld-100-plan --dry-run
```
''')
    files = sorted(p for p in stage.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (stage / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + str(p.relative_to(stage)) + '\n' for p in files))
    plan = stage.parent / (stage.name + '.publish.json')
    write(plan, {'target': str(target), 'stage': str(stage), 'existing_files': original,
                 'new_task_ids': [row['id'] for row in selected], 'report': report})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.target, args.stage, args.count, args.seed)
