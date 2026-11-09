"""One row per unique call; branch returns average descendants, never sum them."""
import json
import math

from ..collection.collector import atomic_json


def samples(nodes, calls, tasks, mode):
    active = {n['id']:n for n in nodes if not n.get('superseded')}
    task_map = {t['task_id']:t for t in tasks}
    totals = {}
    def complete(c):
        return c.get('cost') is not None and c.get('cost_complete',True)
    def value(node):
        if node['id'] in totals:
            return totals[node['id']]
        children = [active[c] for c in node['children']]
        if children:
            future = [value(c) for c in children]
            y = sum(v[0] for v in future)/len(future) if all(v[0] is not None for v in future) else None
            tail = sum(v[1] for v in future)/len(future) if all(v[1] is not None for v in future) else None
        else:
            score = node.get('final_score')
            benchmark = task_map[node['task_id']].get('benchmark','scienceworld')
            y = node.get('label_value', (score+100)/200 if benchmark=='scienceworld' else score/100) if node.get('labels_valid') and score is not None else None
            if y is not None and (not math.isfinite(y) or not 0 <= y <= 1):
                raise ValueError(f'Invalid normalized terminal label for {node["id"]}: {y}')
            tail = 0.0 if y is not None and node.get('evaluation',{}).get('cost_complete',True) else None
        own = [calls[c] for c in node['own_call_ids']]
        total = tail + sum(c['cost'] for c in own) if tail is not None and all(complete(c) for c in own) else None
        totals[node['id']] = (y,total,tail)
        return totals[node['id']]
    for node in active.values():
        value(node)
    rows = []
    for node in active.values():
        task = task_map[node['task_id']]
        y, _, remaining = totals[node['id']]
        prefix_complete = all(complete(calls[c]) for c in node['prefix_call_ids'])
        for cid in reversed(node['own_call_ids']):
            c = calls[cid]
            remaining = remaining+c['cost'] if remaining is not None and complete(c) else None
            valid = y is not None and remaining is not None and prefix_complete
            rows.append({'schema':'mtrouter-step-v1','sample_id':cid,'call_id':cid,
                'task_id':node['task_id'],'run_id':node['run_id'],'node_id':node['id'],
                'benchmark':task.get('benchmark','scienceworld'),'split':task.get('split'),
                'source_split':task.get('source_split',task.get('split')),
                'step_idx':c['step_idx'],'state':c['state'],
                # Router actions are harness profiles, including provider/model aliases.
                'action_model':c['selected_profile'],'action_profile':c['selected_profile'],
                'requested_model':c['requested_model'],'response':c.get('response'),
                'observation':c.get('observation'),'cost':c.get('cost'),
                'remaining_cost':remaining,'cost_currency':c.get('cost_currency'),
                'y':y,'labels_valid':valid,'cost_complete':remaining is not None and prefix_complete,
                'selection_probability':c.get('selection_probability'),
                'branch_point_id':node.get('branch_point_id') if cid==node['own_call_ids'][0] else None,
                'branch_enumerated':bool(mode=='ksr_branch1' and node['parent_id'] and cid==node['own_call_ids'][0]),
                'base_profile':node.get('base_profile'),
                'continuation_policy':'single_step_then_base_with_recursive_ksr' if mode=='ksr_branch1' else 'random_router',
                'label_method':'uniform_future_forks_expected_terminal_score' if mode=='ksr_branch1' else 'single_rollout_terminal_score',
                'action_scope':'single_step','candidate_count':node.get('candidate_count')})
    groups = {}
    for row in rows:
        if row['branch_enumerated']:
            groups.setdefault(row['branch_point_id'],[]).append(row)
    for row in rows:
        pair = groups.get(row['branch_point_id'],[])
        row['paired_actions_complete'] = row['branch_enumerated'] and len(pair)==2 and all(p['labels_valid'] for p in pair)
    return sorted(rows,key=lambda r:r['call_id'])


def export(collector):
    output = collector.out/'training'
    output.mkdir(exist_ok=True)
    rows = samples(collector.nodes,collector.calls,collector.tasks,collector.s.mode)
    temporary = output/'samples.jsonl.tmp'
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    temporary.replace(output/'samples.jsonl')
    report = {'schema':'mtrouter-step-v1','samples':len(rows),
              'valid_samples':sum(r['labels_valid'] for r in rows),
              'model_names':collector.pool,'currency':collector.pricing['currency'],
              'label_target':'normalized terminal return; recursive forks averaged with equal weights',
              'cost_target':'expected agent API cost from this call onward; evaluator excluded',
              'training_data_dir':str(output.resolve()),
              'total_known_cost':collector.total_cost,'cost_complete':collector.cost_complete}
    atomic_json(output/'summary.json',report)
    return report
