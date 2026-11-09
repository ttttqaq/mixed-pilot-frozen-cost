#!/usr/bin/env python3
"""Merge VirtualWorld train/test subsets and add distinct parent tasks to test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.inference.dataset import inside, load_dataset
from sample_virtualworld import digest, native_identity, write


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inventory(root, verify=False):
    values = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        if name in values:
            raise ValueError('Duplicate checksum: ' + name)
        path = inside(root, name)
        if not path.is_file() or (verify and digest(path) != checksum):
            raise ValueError('Invalid source checksum: ' + str(path))
        values[name] = checksum
    return values


def indexed(root):
    rows = []
    for split in ['train', 'test']:
        data = read(root / (split + '.json'))
        if data['split'] != split or data['task_count'] != len(data['tasks']):
            raise ValueError('Invalid source index: ' + str(root))
        rows.extend(dict(row, origin_split=split) for row in data['tasks'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source task IDs')
    tasks = load_dataset(root, 'all')
    identities = {t['source_task_id']: native_identity(t) for t in tasks}
    if set(identities) != {row['id'] for row in rows}:
        raise ValueError('Harness and index IDs differ')
    return rows, identities


def files(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed: ' + str(path))
        if path.is_file() and path.name != '.DS_Store' and '__pycache__' not in path.parts:
            result[path.relative_to(root).as_posix()] = path
    return result


def copy_checked(source, relative, target, checksums):
    origin = inside(source, relative)
    if origin.is_symlink() or checksums.get(relative) != digest(origin):
        raise ValueError('Source checksum mismatch: ' + str(origin))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)


def build(train_source, test_source, parent, output, add_count=30, seed=42):
    train_source, test_source, parent, output = [p.resolve() for p in (train_source, test_source, parent, output)]
    if output.exists():
        raise FileExistsError(output)
    for root in (train_source, test_source, parent):
        if output.is_relative_to(root):
            raise ValueError('Output must be outside source datasets')
    sources = {'train': train_source, 'test': test_source, 'parent': parent}
    snapshots = {key: {name: digest(root / name) for name in ('SHA256SUMS', 'train.json', 'test.json')}
                 for key, root in sources.items()}
    checksums = {key: inventory(root, verify=key != 'parent') for key, root in sources.items()}
    original_train, train_native = indexed(train_source)
    original_test, test_native = indexed(test_source)
    parent_rows, parent_native = indexed(parent)
    if len(original_train) != 100 or len(original_test) != 20 or len(parent_rows) != 1000:
        raise ValueError('Expected 100 train, 20 existing test and 1000 parent tasks')
    if set(train_native) & set(test_native) or set(train_native.values()) & set(test_native.values()):
        raise ValueError('Original train/test subsets already overlap')
    used_ids = set(train_native) | set(test_native)
    used_native = set(train_native.values()) | set(test_native.values())
    if not used_ids.issubset(parent_native):
        raise ValueError('Existing subset task absent from parent')
    for identifier, identity in {**train_native, **test_native}.items():
        if parent_native[identifier] != identity:
            raise ValueError('Parent/subset native identity mismatch')
    pool = sorted((row for row in parent_rows if row['id'] not in used_ids
                   and parent_native[row['id']] not in used_native), key=lambda row: row['id'])
    if not 0 < add_count <= len(pool):
        raise ValueError('Insufficient distinct parent tasks')
    additions = random.Random(seed).sample(pool, add_count)
    output.mkdir(parents=True)
    name = output.name
    entries, selections, verified_files = {'train': [], 'test': []}, [], 0
    batches = [('train', 'train', original_train, 'original_train'),
               ('test', 'test', original_test, 'original_test'),
               ('parent', 'test', additions, 'additional_test')]
    for source_key, target_split, rows, role in batches:
        root = sources[source_key]
        for row in rows:
            relative = 'tasks/' + row['id']
            if row['path'] != relative:
                raise ValueError('Unexpected task path')
            destination = output / relative
            if destination.exists():
                raise ValueError('Repeated task directory')
            original = inside(root, relative)
            payload = files(original)
            for child in payload:
                copy_checked(root, relative + '/' + child, destination / child, checksums[source_key])
                verified_files += 1
            manifest = read(destination / 'manifest.json')
            if manifest['id'] != row['id'] or manifest['routing_metadata']['router_split'] != row['origin_split']:
                raise ValueError('Source manifest/index mismatch')
            manifest.update(dataset_name=name, dataset_family='VirtualWorld')
            # Keep the earlier parent_router_split and official source.split.
            # This records the immediate merge input without rewriting ancestry.
            manifest['routing_metadata'].update(router_split=target_split,
                merge_origin_dataset=root.name, merge_origin_split=row['origin_split'])
            write(destination / 'manifest.json', manifest)
            entry = {key: value for key, value in row.items() if key != 'origin_split'}
            entries[target_split].append(entry)
            selections.append({**entry, 'split': target_split, 'selection_role': role,
                               'origin_dataset': str(root), 'origin_split': row['origin_split'],
                               'origin_manifest_sha256': checksums[source_key][relative + '/manifest.json']})
            for child in payload.keys() - {'manifest.json'}:
                if digest(destination / child) != checksums[source_key][relative + '/' + child]:
                    raise ValueError('Task payload changed during merge')
    for directory in ['runtime', 'licenses']:
        for relative in files(parent / directory):
            path = directory + '/' + relative
            copy_checked(parent, path, output / path, checksums['parent'])
            for source_key in ['train', 'test']:
                if checksums[source_key].get(path) != checksums['parent'][path]:
                    raise ValueError('Runtime/license versions differ across inputs')
            verified_files += 1
    for split, rows in entries.items():
        write(output / (split + '.json'), {'schema_version': '1.0', 'dataset': name,
              'split': split, 'task_count': len(rows), 'tasks': rows})
    counts = {split: dict(Counter(row['benchmark'] for row in rows)) for split, rows in entries.items()}
    summary = {'dataset': name, 'dataset_family': 'VirtualWorld', 'seed': seed,
               'task_count': sum(map(len, entries.values())),
               'splits': {split: len(rows) for split, rows in entries.items()},
               'benchmarks': dict(Counter(row['benchmark'] for rows in entries.values() for row in rows)),
               'benchmarks_by_split': counts, 'preserved_train_tasks': len(original_train),
               'preserved_test_tasks': len(original_test), 'added_test_tasks': add_count,
               'parent_dataset': str(parent), 'sampling_pool_count': len(pool),
               'sampling': 'uniform random without replacement; sorted parent IDs; exclude original train and test by ID and native identity'}
    write(output / 'dataset.json', summary)
    write(output / 'selection.json', {'seed': seed, 'tasks': selections,
          'additional_test': {'candidate_count': len(pool), 'count': add_count,
                              'task_ids': [r['id'] for r in additions], 'excluded_task_ids': sorted(used_ids)}})
    write(output / 'provenance.json', {'dataset': name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'sources': {key: {'dataset': str(root), 'checksums': snapshots[key]} for key, root in sources.items()},
          'runtime': read(parent / 'provenance.json')['runtime'], 'sampling': summary['sampling'], 'seed': seed,
          'validation': 'Original payloads preserved except manifest dataset/routing metadata; successful native replay evidence inherited, not rerun.'})
    loaded = {split: load_dataset(output, split) for split in ['train', 'test', 'all']}
    if [len(loaded[s]) for s in ['train', 'test', 'all']] != [100, 20 + add_count, 120 + add_count]:
        raise ValueError('Harness split counts differ')
    train_ids = {t['source_task_id'] for t in loaded['train']}
    test_ids = {t['source_task_id'] for t in loaded['test']}
    train_instances = {native_identity(t) for t in loaded['train']}
    test_instances = {native_identity(t) for t in loaded['test']}
    if train_ids != set(train_native) or not set(test_native).issubset(test_ids):
        raise ValueError('Original train/test membership changed')
    if train_ids & test_ids or train_instances & test_instances:
        raise ValueError('Train/test instance overlap')
    for task in loaded['all']:
        directory = output / 'tasks' / task['source_task_id']
        manifest = read(directory / 'manifest.json')
        workspace = inside(directory, manifest['workspace']['path'])
        reference = inside(directory, manifest['evaluation']['reference_dir'])
        if reference.is_relative_to(workspace) or manifest['evaluation']['reference_agent_visible'] is not False:
            raise ValueError('Reference isolation lost')
        if (directory / 'prompt.md').read_bytes() != (directory / 'prompt.txt').read_bytes():
            raise ValueError('Prompt aliases differ')
        evidence = read(reference / 'runtime_validation.json')
        if evidence['status'] != 'passed' or evidence['score'] != 100:
            raise ValueError('Missing successful inherited native evidence')
        if task['benchmark'] == 'scienceworld' and task.get('env_step_limit') != 500:
            raise ValueError('ScienceWorld step limit lost')
    for key, root in sources.items():
        if snapshots[key] != {name: digest(root / name) for name in snapshots[key]}:
            raise ValueError('Source changed during merge')
    report = {'status': 'passed', 'task_count': len(loaded['all']), 'splits': summary['splits'],
              'benchmarks_by_split': counts, 'original_train_ids_preserved': True,
              'original_test_ids_preserved': True, 'task_payloads_preserved_except_manifest_metadata': True,
              'added_test_tasks': add_count, 'sampling_pool_count': len(pool),
              'train_test_id_overlap': 0, 'train_test_native_instance_overlap': 0,
              'additional_test_overlap_with_original_train': 0, 'additional_test_overlap_with_original_test': 0,
              'harness_loaded_tasks': {s: len(tasks) for s, tasks in loaded.items()},
              'source_files_checksum_verified': verified_files, 'reference_isolation': True,
              'inherited_native_oracle_replays': len(loaded['all']), 'new_native_oracle_replays': 0,
              'source_datasets_modified': False}
    write(output / 'validation_report.json', report)
    (output / 'README.md').write_text(f'''# {name}

共 {summary['task_count']} 条：train 100 条，test {20 + add_count} 条。

- train：完整保留 `{train_source}` 的 100 条任务。
- test：保留 `{test_source}` 的 20 条，再从 `{parent}` 随机新增 {add_count} 条。
- 新增抽样先排除原 train 100 条和原 test 20 条，从剩余 {len(pool)} 条中无放回抽样；按 task ID 排序，种子 {seed}。
- train/test 按 task ID 与原生实例双重去重，重合均为 0；新增数据与原 test 也不重复。

train：ScienceWorld {counts['train'].get('scienceworld', 0)} 条，ALFWorld {counts['train'].get('alfworld', 0)} 条。
test：ScienceWorld {counts['test'].get('scienceworld', 0)} 条，ALFWorld {counts['test'].get('alfworld', 0)} 条。

兼容 mixed-pilot harness：使用 train.json/test.json，manifest 的 router_split 对应本地划分。
dataset_name 统一为 {name}，原始 source.split、parent_router_split、prompt、workspace、rubrics 和 reference 保留。
原始三个数据集未修改。划分按实例隔离，任务类型可以重合。
原生成功重放证据继承自来源，本次只校验文件、去重与 harness 加载，未重新运行原生环境。

在 mixed-pilot 项目目录运行：

```bash
outputs/virtualworld-venv/bin/python scripts/infer.py \\
  --dataset /Users/liangjianfeng/Data/Router/{name} --split test \\
  --mode single-model --models YOUR_PROFILE --step-limit 500 \\
  --output-dir run/virtualworld-subset-test --dry-run
```

将 YOUR_PROFILE 替换为 harness.json 中的模型配置名称；训练集改为 --split train，合并加载使用 --split all。
selection.json 保存全部任务来源与新增名单，provenance.json 保存输入指纹，SHA256SUMS 覆盖数据文件，validation_report.json 保存校验结果。
''', encoding='utf-8')
    payload = files(output)
    (output / 'SHA256SUMS').write_text(''.join(digest(path) + '  ' + name + '\n'
                                            for name, path in sorted(payload.items())), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-source', type=Path, required=True)
    parser.add_argument('--test-source', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--add-test-count', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must not already exist')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.virtualworld-merge-', dir=args.output.parent) as temp:
        stage = Path(temp) / args.output.name
        build(args.train_source, args.test_source, args.parent, stage, args.add_test_count, args.seed)
        if args.output.exists():
            raise FileExistsError(args.output)
        stage.rename(args.output)


if __name__ == '__main__':
    main()
