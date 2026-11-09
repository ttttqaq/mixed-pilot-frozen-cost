"""No-LLM evaluation maintenance; always write a new, auditable output directory."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..collection.collector import step_samples
from . import swe_runtime as native
from .dataset import load_dataset


def summarize(output, results, **metadata):
    scored = [r for r in results if r['status'] in {'passed', 'failed'}]
    passed = sum(r['status'] == 'passed' for r in scored)
    report = dict(**metadata, passed=passed, scored=len(scored), unscored=len(results)-len(scored),
                  pass_rate=passed/len(scored) if scored else None, new_api_cost=0.0, results=results)
    native.write_json(output / 'summary.json', report)
    rate = f'{passed/len(scored):.2%}' if scored else 'N/A'
    (output / 'SUMMARY.md').write_text(
        f'# Pass rate: {rate} ({passed}/{len(scored)})\n\n'
        f'# New API cost: 0\n\nUnscored: {len(results)-len(scored)}. Original run is unchanged.\n\n'
        f'Mode: {metadata.get("mode")}. Log-only reclassification does not rerun tests.\n')
    return report


def reparse(row, folder):
    result = json.loads((folder / 'result.json').read_text())
    from . import swebench_runtime
    if swebench_runtime.matches(row):
        if not (folder / 'test_output.txt').is_file() or result.get('phase') != 'test_parsing':
            return {**result, 'requires_reevaluation': True}
        value = swebench_runtime.grade_logs(row, folder / 'prediction.patch', folder / 'test_output.txt')
        return {**result, **value, 'classification_revision': 'swebench-official-v1'}
    logs = '\n'.join(p.read_text(errors='replace') for name in
        ('container.stdout.log', 'container.stderr.log', 'tests.stdout.log', 'tests.stderr.log')
        if (p := folder / name).exists())
    if native.PATCH_READY not in logs:
        category = ('environment_script' if 'not a valid identifier' in logs else
                    'patch_conflict' if re.search(r'with conflicts|already exists in working directory', logs)
                    else result.get('error_category', 'repository_setup'))
        return {**result, 'status': 'error', 'resolved': None, 'error_category': category,
                'requires_reevaluation': True}
    # No downloads/imports: retain the native parser's saved names and use only
    # explicit raw-log evidence to repair its classification.
    def normalize(name):
        # Same three timing patterns as the pinned rebench evaluator.
        for pattern in (r'\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$',
                        r'\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b',
                        r'\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$'):
            name = re.sub(pattern, '', name, flags=re.I)
        return name.strip()
    value = native.classify_test_output(row, result.get('tests', []), logs, normalize=normalize)
    metadata = {k:v for k,v in result.items() if k not in {'error', 'error_type', 'error_category'}}
    return {**metadata, **value, 'rule': 'all_required_f2p_and_p2p_pass',
            'classification_revision': native.RUNTIME_REVISION}


def evaluate_saved(args, *, logs_only=False):
    source = (args.reparse_only if logs_only else args.eval_only).resolve()
    plan = json.loads((source / 'plan.json').read_text())
    tree = json.loads((source / 'tree.json').read_text())
    tasks = {t['task_id']: t for t in plan['tasks']}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for node in tree['nodes']:
        if node['children'] or node.get('superseded') or tasks[node['task_id']]['benchmark'] != 'swe':
            continue
        task = tasks[node['task_id']]
        item = dict(node_id=node['id'], task_id=node['task_id'], original_status=node['status'])
        try:
            row = json.loads(Path(task['record_path']).read_text())
            candidates = sorted((source / 'workspaces' / node['id']).glob('replay-*/prediction.patch'),
                                key=lambda p: int(p.parent.name.split('-')[-1]))
            if not candidates:
                raise ValueError('No saved prediction.patch; inference/initialization did not reach evaluation')
            patch = candidates[-1]
            result = (reparse(row, patch.parent / 'native-eval') if logs_only else
                      native.native_evaluate(row, patch, args.cache, args.output_dir / node['id'],
                                             args.eval_timeout, args.cpus, args.memory))
            item.update(result, source_patch=str(patch))
        except Exception as exc:
            item.update(status='error', resolved=None, error_category='maintenance', error=str(exc))
        results.append(item)
        summarize(args.output_dir, results, mode='reparse' if logs_only else 'eval-only', source_run=str(source))
    exports = {} if logs_only else export_reevaluated(source, args.output_dir, results)
    return summarize(args.output_dir, results, mode='reparse' if logs_only else 'eval-only', source_run=str(source), **exports,
                     original_known_api_cost=json.loads((source / 'summary.json').read_text()).get('total_api_cost'))


def export_reevaluated(source, output, results):
    """Relabel existing trajectories; never regenerate model messages/actions."""
    (output / 'traces').mkdir()
    traces = steps = 0
    with (output / 'steps.jsonl').open('w') as stream:
        for result in results:
            path = source / 'traces' / (result['node_id'] + '.traj.json')
            if not path.exists():
                continue
            trace = json.loads(path.read_text())
            info = trace['info']
            valid = result['status'] in {'passed', 'failed'}
            success = result['status'] == 'passed' if valid else None
            score = 100.0 if success else 0.0 if valid else None
            cost_complete = all(step.get('cost') is not None for step in trace['steps'])
            info.update(original_exit_status=info['exit_status'], source_trace=str(path),
                        exit_status='reevaluated' if valid else 'evaluation_error',
                        labels_valid=valid and cost_complete, cost_complete=cost_complete,
                        final_score=score, success=success,
                        evaluation={'status': 'scored' if valid else 'error', 'method': 'swe_native',
                                    'score': score, 'success': success, 'cost': 0.0,
                                    'cost_complete': True, 'api_calls': 0, 'native_evaluation': result})
            native.write_json(output / 'traces' / path.name, trace)
            traces += 1
            for sample in step_samples(trace):
                stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
                steps += 1
    return {'exported_traces': traces, 'exported_steps': steps}


def precheck(args):
    if args.dataset is None:
        raise ValueError('--precheck requires --dataset')
    tasks = load_dataset(args.dataset, args.split)
    if args.limit:
        tasks = tasks[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for task in tasks:
        item = dict(task_id=task['task_id'])
        try:
            if task['benchmark'] != 'swe':
                raise ValueError('Native empty/gold precheck currently supports SWE only')
            row = json.loads(Path(task['record_path']).read_text())
            native.grade_tests(row, [])  # Validate nonempty required test set.
            if not row.get('patch'):
                raise ValueError('Missing reference patch')
            if args.dry_run:
                item.update(status='static_checked', runtime_checked=False)
            else:
                folder = args.output_dir / task['task_id']
                folder.mkdir()
                native.prepare_native(row, args.cache)
                native.pull_image(row, folder / 'preparation', args.pull_timeout)
                for name, content in [('empty', ''), ('gold', row['patch'])]:
                    patch = folder / (name + '.patch')
                    patch.write_text(content)
                    item[name] = native.native_evaluate(row, patch, args.cache, folder / name,
                                                       args.eval_timeout, args.cpus, args.memory)
                item.update(status='ready' if item['empty']['status']=='failed' and item['gold']['status']=='passed'
                            else 'invalid_baseline', runtime_checked=True)
        except Exception as exc:
            item.update(status='error', error=str(exc))
        results.append(item)
        native.write_json(args.output_dir / 'precheck.json', {'new_api_cost': 0, 'results': results})
    return {'new_api_cost': 0, 'ready': sum(r['status']=='ready' for r in results),
            'static_checked': sum(r['status']=='static_checked' for r in results), 'results': results}
