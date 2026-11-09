#!/usr/bin/env python3
"""Pull upstream registry images from a relocated dataset or plan. No LLM calls."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mixed_pilot_harness.processes import run as run_process
from mixed_pilot_harness.inference.dataset import load_dataset
from mixed_pilot_harness.inference import swe_runtime as native


def image_for(row):
    return native.source_image_for(row)[0]


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    source = value.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan', type=Path, help='Existing inference/preparation plan on this machine')
    source.add_argument('--dataset', type=Path, help='Dataset root on the destination server; regenerate local paths')
    value.add_argument('--split', choices=['train', 'test', 'all', 'train-test'], default='train')
    value.add_argument('--output', type=Path, required=True)
    value.add_argument('--attempts', type=int, default=5)
    value.add_argument('--timeout', type=int, default=1800)
    value.add_argument('--dry-run', action='store_true', help='Write registry references and local plan; no Docker or downloads')
    return value


def build_plan(args):
    # A shell may still export a manifest from the old Mac. It must not affect
    # the upstream-only download plan or cause a missing local file error.
    with native.original_images():
        tasks = (load_dataset(args.dataset, args.split) if args.dataset else
                 json.loads(args.plan.read_text())['tasks'])
    if not tasks:
        raise ValueError('No tasks selected')
    resolved, images = [], {}
    for task in tasks:
        if task.get('benchmark') not in {None, 'swe'}:
            raise ValueError('pull_swe_images supports SWE tasks only')
        record = Path(task['record_path']).resolve()
        if not record.is_file():
            raise ValueError(f'Missing source record: {record}; use --dataset to regenerate server paths')
        row = json.loads(record.read_text())
        image = image_for(row)
        if image.startswith('swe-neo-prepared:'):
            raise ValueError('Source record must reference an upstream registry image')
        resolved.append({**task, 'record_path': str(record), 'source_image': image})
        images.setdefault(image, []).append(task['task_id'])
    return resolved, images


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.attempts < 1 or args.timeout < 1:
        cli.error('attempts and timeout must be positive')
    try:
        tasks, images = build_plan(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f'pull_swe_images: {exc}', file=sys.stderr)
        return 2
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {'image_policy': 'upstream', 'platform': native.DOCKER_PLATFORM,
                'tasks': len(tasks), 'total_images': len(images),
                'images': [{'image': image, 'task_ids': ids} for image, ids in images.items()]}
    # This plan is generated on the destination host, including absolute paths
    # appropriate for that host; prepare_swe_dependencies.py consumes it directly.
    native.write_json(args.output / 'plan.json', {'tasks': tasks})
    native.write_json(args.output / 'image-plan.json', manifest)
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'tasks': len(tasks), 'images': len(images),
                          'platform': native.DOCKER_PLATFORM, 'image_policy': 'upstream',
                          'plan': str(args.output / 'plan.json')}, indent=2))
        return 0
    try:
        probe = run_process(['docker', 'info', '--format', '{{.ServerVersion}}'], text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'pull_swe_images: Docker is unavailable: {exc}', file=sys.stderr)
        return 2
    if probe.returncode:
        print('pull_swe_images: Docker daemon is unavailable; start Docker on this server', file=sys.stderr)
        return 2
    results = []
    def save():
        value = dict(updated_at=datetime.datetime.now().astimezone().isoformat(),
                     total_images=len(images), processed=len(results), new_api_cost=0,
                     ready=sum(r['status'] in ('cached', 'pulled') for r in results),
                     failed=sum(r['status'] == 'failed' for r in results), results=results)
        tmp = args.output / 'images.json.tmp'
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(args.output / 'images.json')
    for index, (image, task_ids) in enumerate(images.items(), 1):
        item = dict(image=image, task_ids=task_ids, status='pulling', attempts=0)
        results.append(item)
        inspect = run_process(['docker', 'image', 'inspect', '--format', '{{.Os}}/{{.Architecture}}', image],
                                 text=True, timeout=60)
        if inspect.returncode == 0 and inspect.stdout.strip() == native.DOCKER_PLATFORM:
            item['status'] = 'cached'
        else:
            print(f'[{index}/{len(images)}] Pulling {image}', flush=True)
            for attempt in range(1, args.attempts + 1):
                item['attempts'] = attempt
                save()
                log = args.output / f'image-{index:03d}-attempt-{attempt}.log'
                try:
                    with log.open('w') as stream:
                        result = run_process(['docker', 'pull', '--platform', native.DOCKER_PLATFORM, image],
                                                stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                    if result.returncode == 0:
                        item['status'] = 'pulled'
                        break
                    item['error'] = f'Exit {result.returncode}; see {log.name}'
                except subprocess.TimeoutExpired:
                    item['error'] = f'Timeout after {args.timeout}s; see {log.name}'
                if attempt < args.attempts:
                    time.sleep(min(60, 5 * 2 ** (attempt - 1)))
            else:
                item['status'] = 'failed'
        save()
        print(f'[{index}/{len(images)}] {item["status"]}: {image}', flush=True)
    return int(any(r['status'] == 'failed' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
