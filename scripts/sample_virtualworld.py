#!/usr/bin/env python3
"""Sample a reproducible VirtualWorld subset, optionally excluding another dataset."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.inference.dataset import inside, load_dataset


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def native_identity(task):
    if task['benchmark'] == 'scienceworld':
        return ('scienceworld', task['task_name'], task['variation_idx'])
    if task['benchmark'] == 'alfworld':
        return ('alfworld', task['source_id'])
    raise ValueError('Only VirtualWorld native tasks can be sampled')


def build(source, output, count, seed, split='train', exclude=None):
    source, output = source.resolve(), output.resolve()
    if split not in {'train', 'test'}: raise ValueError('Expected train or test output split')
    if output.exists(): raise ValueError('Refusing to overwrite existing destination')
    if output.is_relative_to(source): raise ValueError('Destination cannot be inside source')
    pool = []
    for parent_split in ['train', 'test']:
        index = json.loads((source / (parent_split + '.json')).read_text())
        if index['task_count'] != len(index['tasks']): raise ValueError('Invalid source index count')
        pool.extend(dict(row, parent_split=parent_split) for row in index['tasks'])
    if len({row['id'] for row in pool}) != len(pool): raise ValueError('Duplicate source task IDs')
    parent_count = len(pool)
    parent_tasks = load_dataset(source, 'all')
    native_by_id = {task['source_task_id']: native_identity(task) for task in parent_tasks}
    if set(native_by_id) != {row['id'] for row in pool}: raise ValueError('Parent native/index IDs differ')
    excluded_ids, excluded_native, exclusion = set(), set(), None
    if exclude is not None:
        exclude = exclude.resolve()
        if output.is_relative_to(exclude): raise ValueError('Destination cannot be inside exclusion dataset')
        excluded_tasks = load_dataset(exclude, 'all')
        excluded_ids = {task['source_task_id'] for task in excluded_tasks}
        excluded_native = {native_identity(task) for task in excluded_tasks}
        # Check the actual task payloads, regardless of an old dataset_name in
        # metadata after a folder rename. Local IDs alone do not prove disjointness.
        for line in (exclude / 'SHA256SUMS').read_text().splitlines():
            checksum, name = line.split('  ', 1)
            if digest(inside(exclude, name)) != checksum:
                raise ValueError('Exclusion dataset checksum mismatch: ' + name)
        pool = [row for row in pool if row['id'] not in excluded_ids
                and native_by_id[row['id']] not in excluded_native]
        exclusion = {'dataset': str(exclude), 'task_count': len(excluded_tasks),
                     'task_ids': sorted(excluded_ids),
                     'native_identities': [list(key) for key in sorted(excluded_native)],
                     'inventory_sha256': digest(exclude / 'SHA256SUMS'),
                     'excluded_parent_tasks': parent_count - len(pool),
                     'overlap_policy': 'task ID OR native instance identity'}
    if not 0 < count <= len(pool): raise ValueError('Invalid sample count')
    selected = random.Random(seed).sample(sorted(pool, key=lambda row: row['id']), count)
    inventory = {}
    for line in (source / 'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        if name in inventory: raise ValueError('Duplicate source checksum')
        inventory[name] = checksum
    output.mkdir(parents=True)
    dataset = f'VirtualWorld-{count}'
    rows, selections, checked = [], [], 0
    for row in selected:
        origin = inside(source, row['path'])
        if row['path'] != 'tasks/' + row['id']: raise ValueError('Unexpected task path')
        destination = output / row['path']
        for path in origin.rglob('*'):
            if path.is_symlink(): raise ValueError('Source task symlink is not allowed')
            if not path.is_file() or path.name == '.DS_Store' or '__pycache__' in path.parts: continue
            name = str(path.relative_to(source))
            if inventory.get(name) != digest(path): raise ValueError('Source checksum mismatch: ' + name)
            target = destination / path.relative_to(origin)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            checked += 1
        manifest_path = destination / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['id'] != row['id'] or manifest['routing_metadata']['router_split'] != row['parent_split']:
            raise ValueError('Source index and manifest differ')
        manifest['parent_dataset_name'] = manifest.get('dataset_name', 'VirtualWorld-1000')
        manifest['dataset_name'] = dataset
        manifest['dataset_family'] = 'VirtualWorld'
        manifest['routing_metadata'].update(parent_router_split=row['parent_split'], router_split=split)
        write(manifest_path, manifest)
        index_row = {k: v for k, v in row.items() if k != 'parent_split'}
        rows.append(index_row)
        selections.append(dict(row, original_manifest_sha256=inventory[row['path'] + '/manifest.json']))
    for name in ['licenses', 'runtime']:
        for path in (source / name).rglob('*'):
            if path.is_file() and path.name != '.DS_Store' and '__pycache__' not in path.parts:
                relative = str(path.relative_to(source))
                if path.is_symlink() or inventory.get(relative) != digest(path):
                    raise ValueError('Source runtime/license checksum mismatch')
                target = output / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target); checked += 1
    for output_split in ['train', 'test']:
        write(output / (output_split + '.json'), {'schema_version': '1.0', 'dataset': dataset,
              'split': output_split, 'task_count': count if output_split == split else 0,
              'tasks': rows if output_split == split else []})
    summary = {'dataset': dataset, 'dataset_family': 'VirtualWorld', 'seed': seed, 'task_count': count,
               'splits': {s: count if s == split else 0 for s in ['train', 'test']},
               'benchmarks': dict(Counter(r['benchmark'] for r in rows)),
               'parent_dataset': str(source), 'sampling_pool_count': len(pool),
               'sampling': 'uniform random without replacement; sorted task IDs; train + test pool',
               'parent_splits': dict(Counter(r['parent_split'] for r in selected))}
    if exclusion:
        summary.update(parent_task_count=parent_count, excluded_dataset=str(exclude),
                       excluded_parent_tasks=parent_count - len(pool))
    write(output / 'dataset.json', summary)
    write(output / 'selection.json', {'seed': seed, 'tasks': selections,
                                      **({'exclusion': exclusion} if exclusion else {})})
    parent_provenance = json.loads((source / 'provenance.json').read_text())
    write(output / 'provenance.json', {'dataset': dataset, 'parent_dataset': str(source),
          'parent_inventory_sha256': digest(source / 'SHA256SUMS'), 'seed': seed,
          'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'runtime': parent_provenance['runtime'], 'sampling': summary['sampling'],
          'output_split': split, **({'exclusion': exclusion} if exclusion else {}),
          'validation': 'Selected source files verified against parent checksums; native replay evidence inherited, not rerun.'})
    source_test_ids = [r['id'] for r in selected if r['parent_split'] == 'test']
    exclusion_note = (f'抽样前排除 `{exclude}` 的 {len(excluded_ids)} 条任务，按 task ID 和原生实例身份双重去重。\n'
                      f'父数据集共 {parent_count} 条，排除后候选池 {len(pool)} 条；与排除数据集重合为 0。\n'
                      if exclusion else '')
    overlap_note = ('本训练集包含原 test 样本，后续使用父数据集 test 评估时，应排除 selection.json\n'
                    '中 parent_split 为 test 的这些任务，避免训练/评测重叠。'
                    if split == 'train' else
                    '本数据集用于测试，全部任务的 router_split=test；父数据集的 train/test 来源仅作为 provenance 保留。')
    (output / 'README.md').write_text(f'''# {dataset}

从 `{source}` 的候选池中均匀、无放回随机抽取 {count} 条。
{exclusion_note}随机种子 {seed}，抽样前按 task ID 排序。训练集 {summary['splits']['train']} 条，测试集 {summary['splits']['test']} 条。

ScienceWorld {summary['benchmarks'].get('scienceworld', 0)} 条；ALFWorld {summary['benchmarks'].get('alfworld', 0)} 条。
原 train {summary['parent_splits'].get('train', 0)} 条，原 test {len(source_test_ids)} 条。
manifest 的 router_split 统一为 {split}；parent_router_split 和原始 source.split 保留来源。
任务 workspace、prompt、rubrics 和 reference 完整复制；reference 仍仅供评测使用。
原生成功重放记录继承自父数据集，本次检查文件完整性与 harness 加载，未重新运行环境。

{overlap_note}

运行示例（在当前 mixed-pilot 项目目录执行）：

```bash
outputs/virtualworld-venv/bin/python scripts/infer.py \\
  --dataset /Users/liangjianfeng/Data/Router/{dataset} --split {split} \\
  --mode single-model --models deepseek-flash --step-limit 500 \\
  --output-dir run/{dataset.lower()}-dry-run --dry-run
```

完整选样记录见 selection.json，文件清单见 SHA256SUMS，验证见 validation_report.json。
''')
    tasks = load_dataset(output, 'all')
    if len(tasks) != count or any(t['split'] != split for t in tasks): raise ValueError('Harness subset load failed')
    if len(load_dataset(output, split)) != count: raise ValueError('Harness split load failed')
    id_overlap = sorted({task['source_task_id'] for task in tasks} & excluded_ids)
    native_overlap = {native_identity(task) for task in tasks} & excluded_native
    if id_overlap or native_overlap: raise ValueError('Sample overlaps exclusion dataset')
    for task in tasks:
        if task['benchmark'] == 'scienceworld' and task.get('env_step_limit') != 500:
            raise ValueError('ScienceWorld native step limit was lost')
        evidence = json.loads((output / 'tasks' / task['source_task_id'] / 'reference/runtime_validation.json').read_text())
        if evidence['status'] != 'passed' or evidence['score'] != 100: raise ValueError('Missing inherited native evidence')
    write(output / 'validation_report.json', {'status': 'passed', 'task_count': count,
          'benchmarks': summary['benchmarks'], 'splits': summary['splits'], 'harness_loaded_tasks': len(tasks),
          **({'excluded_dataset': str(exclude), 'excluded_task_count': len(excluded_ids),
              'overlap_with_excluded_task_ids': len(id_overlap),
              'overlap_with_excluded_native_instances': len(native_overlap)} if exclusion else {}),
          'source_files_checksum_verified': checked, 'inherited_native_oracle_replays': count,
          'new_native_oracle_replays': 0, 'reference_isolation': True})
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + str(p.relative_to(output)) + '\n' for p in files))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--exclude', type=Path, help='Exclude all task IDs and native instances in this dataset')
    args = parser.parse_args()
    build(args.source, args.output, args.count, args.seed, args.split, args.exclude)
