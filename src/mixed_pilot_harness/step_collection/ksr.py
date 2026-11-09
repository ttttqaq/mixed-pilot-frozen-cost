"""Configurable, observation-only KSR heuristics; no judge or hidden tests."""
from fnmatch import fnmatchcase
import json
from pathlib import Path


ERROR_MARKERS = ['not found', 'not a valid', 'unknown command', 'cannot', 'traceback',
                 'error:', 'failed', '"ok": false']
SUCCESS_MARKERS = ['"ok": true', '"exit_code": 0', '"returncode": 0', 'tests passed', 'successfully']


def load_config(path, profiles):
    raw = json.loads(Path(path).read_text())
    return validate_config(raw, profiles)


def validate_config(raw, profiles):
    allowed = {'strong_model','weak_model','strong_first_tasks','weak_first_tasks',
               'default_start','KSR_strong2weak','KSR_weak2strong'}
    if not isinstance(raw, dict) or set(raw)-allowed:
        raise ValueError('Unsupported KSR configuration fields')
    strong, weak = raw.get('strong_model'), raw.get('weak_model')
    if not isinstance(strong,str) or not isinstance(weak,str) or strong == weak:
        raise ValueError('KSR requires distinct strong_model and weak_model profile names')
    for name in (strong, weak):
        if name not in profiles or profiles[name].kind == 'router':
            raise ValueError(f'KSR requires concrete harness profiles: {name}')
    result = dict(raw)
    result.setdefault('default_start','weak')
    if result['default_start'] not in {'strong','weak'}:
        raise ValueError('KSR default_start must be strong or weak')
    for key in ('strong_first_tasks','weak_first_tasks'):
        values = result.setdefault(key, [])
        if not isinstance(values,list) or any(not isinstance(v,str) or not v for v in values):
            raise ValueError(f'{key} must be a list of task IDs or glob patterns')
    for key, signals in [('KSR_strong2weak',['easy_progress']),
                         ('KSR_weak2strong',['consecutive_errors','repeated_command','stalled_progress'])]:
        defaults = dict(signals=signals,min_branch_gap=3,error_window=2,repeat_window=3,
                        stall_window=4,progress_window=2,error_markers=ERROR_MARKERS,
                        success_markers=SUCCESS_MARKERS)
        rule = result.get(key,{})
        if not isinstance(rule,dict) or set(rule)-set(defaults):
            raise ValueError(f'Unsupported rule fields in {key}')
        rule = {**defaults, **rule}
        if not isinstance(rule['signals'],list) or not rule['signals'] or any(not isinstance(s,str) for s in rule['signals']) or set(rule['signals'])-{
                'easy_progress','consecutive_errors','repeated_command','stalled_progress'}:
            raise ValueError(f'Unknown or empty signals in {key}')
        for field in ('min_branch_gap','error_window','repeat_window','stall_window','progress_window'):
            if type(rule[field]) is not int or rule[field] < 1:
                raise ValueError(f'{key}.{field} must be a positive integer')
        for field in ('error_markers','success_markers'):
            if not isinstance(rule[field],list) or any(not isinstance(v,str) or not v for v in rule[field]):
                raise ValueError(f'{key}.{field} must be a list of nonempty strings')
        result[key] = rule
    return result


def root_profiles(config, task):
    # Match public dataset IDs, never the gold answer or final evaluation score.
    strong = any(fnmatchcase(task['task_id'],p) for p in config['strong_first_tasks'])
    weak = any(fnmatchcase(task['task_id'],p) for p in config['weak_first_tasks'])
    kinds = [kind for kind, matched in [('strong', strong), ('weak', weak)] if matched]
    return [config[kind+'_model'] for kind in (kinds or [config['default_start']])]


def task_schedule(config, tasks):
    """One independent tree per (task, starting profile), strong group first."""
    selected = [(task, root_profiles(config, task)) for task in tasks]
    return [(task, profile) for profile in (config['strong_model'], config['weak_model'])
            for task, profiles in selected if profile in profiles]


def detect(config, base_profile, steps):
    direction = 'KSR_strong2weak' if base_profile == config['strong_model'] else 'KSR_weak2strong'
    rule = config[direction]
    if len(steps) < rule['min_branch_gap']:
        return None
    def error(s):
        return bool(s.get('format_error')) or any(m.lower() in s.get('observation','').lower() for m in rule['error_markers'])
    def progress(s):
        return s.get('score_after',0) > s.get('score_before',0)
    for signal in rule['signals']:
        window = rule[{'easy_progress':'progress_window','consecutive_errors':'error_window',
                       'repeated_command':'repeat_window','stalled_progress':'stall_window'}[signal]]
        last = steps[-window:]
        if len(last) != window:
            continue
        if signal == 'consecutive_errors':
            matched = all(error(s) for s in last)
        elif signal == 'repeated_command':
            matched = bool(last[0].get('command')) and len({s.get('command') for s in last}) == 1
        elif signal == 'stalled_progress':
            # Flat scores alone are normal in SWE/HLE. Require repeated output too.
            matched = not any(progress(s) for s in last) and bool(last[0].get('observation')) and len({s.get('observation') for s in last}) == 1
        else:
            matched = all(not error(s) and (progress(s) or any(
                m.lower() in s.get('observation','').lower() for m in rule['success_markers'])) for s in last)
        if matched:
            return {'rule':signal,'direction':direction,'window':window,'evidence_call_ids':[s['call_id'] for s in last]}
    return None
