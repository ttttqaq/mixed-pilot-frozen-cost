"""Match task instances, not run IDs or manifest filename spelling."""
import json
from pathlib import Path


def task_key(row):
    return row['task_name'], row['variation_idx'], row.get('simplification_str','')


def select_ksr_tasks(tasks, baseline_root, strong, weak):
    outcomes = {}
    for profile in (strong, weak):
        for path in sorted((Path(baseline_root)/profile).glob('*.traj.json')):
            info=json.loads(path.read_text())['info']
            if not info.get('labels_valid') or not isinstance(info.get('success'),bool):
                continue
            bucket=outcomes.setdefault(task_key(info),{strong:[],weak:[]})
            bucket[profile].append({'trace':str(path.resolve()),'success':info['success']})
    selected=[];evidence=[]
    for task in tasks:
        group=outcomes.get(task_key(task),{})
        if any(x['success'] for x in group.get(strong,[])) and any(not x['success'] for x in group.get(weak,[])):
            selected.append(task);evidence.append({'task_id':task['task_id'],'rule':'any_strong_success_and_any_weak_failure','baselines':group})
    return selected,evidence
