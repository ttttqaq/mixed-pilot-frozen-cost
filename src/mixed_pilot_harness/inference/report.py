"""Unique-call accounting, complete leaf trajectories, and visible pass/cost summaries."""
from __future__ import annotations
from dataclasses import asdict
import json

from ..branch_sampling.core import Collector, export_trace_steps
from ..collection.collector import atomic_json, step_samples
from ..types import Message, ToolCall


class InferenceCollector(Collector):
    def summary(self):
        report = super().summary()
        leaves = [n for n in self.nodes if not n['children'] and not n.get('superseded')]
        valid = [n for n in leaves if n['labels_valid']]
        passed = sum(bool(n.get('success', n['final_score'] >= 100)) for n in valid)
        report.update(passed=passed, scored_traces=len(valid), total_traces=len(leaves),
            unscored_traces=len(leaves)-len(valid), pass_rate=passed/len(valid) if valid else None,
            pass_rate_definition='passed / scored leaf trajectories; KSR leaves are correlated, not independent task trials',
            agent_cost=sum(c['cost'] or 0 for c in self.calls.values() if c.get('cost_currency',self.pricing['currency'])==self.pricing['currency']),
            evaluation_cost=sum(n.get('evaluation', {}).get('cost', 0) for n in self.nodes if n.get('config_snapshot',{}).get('pricing',self.pricing)['currency']==self.pricing['currency']))
        return report

    def save(self):
        super().save()
        report = self.summary()
        rate = f"{report['pass_rate']:.2%}" if report['pass_rate'] is not None else 'N/A'
        text = (f"# Pass rate: {rate} ({report['passed']}/{report['scored_traces']})\n\n"
                f"# Total known cost: {report['total_api_cost']:.6f} {report['currency']}\n\n"
                f"- Total tasks: {report['total_tasks']}\n"
                f"- Jobs: planned={report['planned_jobs']}, started={report['started_jobs']}, unstarted={report['unstarted_jobs']}\n"
                f"- Costs by currency (never summed across currencies): {report['cost_by_currency']}\n"
                f"- Agent: {report['agent_cost']:.6f}; evaluator: {report['evaluation_cost']:.6f}\n"
                f"- Unscored trajectories: {report['unscored_traces']}\n"
                f"- Superseded nodes (retained, excluded from pass rate): {report['superseded_nodes']}\n"
                f"- Cost complete: {report['cost_complete']}\n"
                f"- Stop: {report['stop_reason'] or 'running'}\n\n"
                f"{report['pass_rate_definition']}\n\n"
                "Incomplete/error trajectories retain null labels. EmptyResponseLimit counts as failure.\n"
                "Shared prefixes appear in multiple traces; total cost counts each API call once.\n")
        text += '\n## Costs by profile (known amounts)\n\n'
        text += '| Profile | Currency | Agent | Evaluator | Total | Complete | Uncertain calls |\n'
        text += '| --- | --- | ---: | ---: | ---: | --- | ---: |\n'
        for profile, currencies in report['profile_costs'].items():
            for currency, row in currencies.items():
                text += f"| {profile} | {currency} | {row['agent_cost']:.6f} | {row['evaluation_cost']:.6f} | {row['total_known_cost']:.6f} | {row['cost_complete']} | {row['uncertain_calls']} |\n"
        if not report['cost_complete']:
            text += '\n**Cost is incomplete: known amounts are lower bounds; failed requests may have been billed.**\n'
        (self.out / 'SUMMARY.md').write_text(text)


def export(collector):
    root = collector.out
    leaf = root / 'leaf-data'
    leaf.mkdir(exist_ok=True)
    export_trace_steps(leaf, collector.nodes, collector.calls)
    (root / 'traces').mkdir(exist_ok=True)
    tasks = {t['task_id']: t for t in collector.tasks}
    plan = json.loads((root / 'plan.json').read_text())
    config = {key: plan[key] for key in ['settings','profiles','pricing','prompt_config']}
    config['settings']['cost_limit'] = collector.s.path_cost_limit
    with (root / 'steps.jsonl').open('w') as samples:
        for node in collector.nodes:
            if node['children']:
                continue
            ids = node['prefix_call_ids'] + node['own_call_ids']
            # Include failed in-flight requests too; these must not acquire labels.
            ids += [cid for cid, c in collector.calls.items() if c['node_id']==node['id'] and cid not in ids]
            calls = [collector.calls[cid] for cid in ids]
            task = tasks[node['task_id']]
            valid = bool(node['labels_valid']) and all(c.get('cost') is not None and c.get('cost_complete',True) for c in calls)
            messages = []
            steps = []
            for c in calls:
                # Each state contains the exact prior history. The final transition adds the last observation.
                messages = [{'role':'system','content':c['state']['system']}] + c['state']['history']
                if 'response' in c:
                    messages = messages + [asdict(Message('assistant', c['response'],
                        tool_calls=[ToolCall(**x) for x in c.get('tool_calls', [])],
                        reasoning_content=c.get('reasoning_content'), reasoning_owner=c.get('reasoning_owner')))]
                    results = c.get('tool_results', [])
                    if results:
                        messages += [asdict(Message('tool', r['output'], tool_call_id=r['tool_call_id'], name=r['name'])) for r in results]
                    elif c.get('observation'):
                        messages += [asdict(Message('user', c['observation']))]
                steps.append({**c, 'propensity': c['selection_probability']})
            # Keep message-based consumers (including MTRouter training) aligned
            # with the explicit call ledger without changing provider requests.
            assistant_messages=[m for m in messages if m['role']=='assistant']
            responses=[c for c in calls if 'response' in c]
            if len(assistant_messages)==len(responses):
                for message,c in zip(assistant_messages,responses):
                    message.update(model_name=c['requested_model'],profile=c['selected_profile'],
                        propensity=c['selection_probability'],extra={'response':{
                            'usage':{**c['usage'],'cost':c['cost']},'cost_complete':c.get('cost_complete',True),'api_attempts':c.get('api_attempts',[])}})
            info = {**task, 'trace_id':node['id'], 'run_id':node['run_id'],
                    'base_profile':node.get('base_profile'),
                    'attempt':node.get('attempt',0), 'superseded':bool(node.get('superseded')),
                    'exit_status':node['status'], 'labels_valid':valid,
                    'cost_complete':all(c.get('cost') is not None and c.get('cost_complete',True) for c in calls) and node.get('evaluation',{}).get('cost_complete',True),
                    'final_score':node['final_score'], 'success':node.get('success'),
                    'policy':collector.s.mode, 'cost_currency':node.get('config_snapshot',config)['pricing']['currency'],
                    'config': node.get('config_snapshot',config),
                    'evaluation':node.get('evaluation'),
                    'model_stats':{'api_calls':len(calls),'instance_cost':sum(c.get('cost') or 0 for c in calls)},
                    'cost_note':'Path cost includes shared prefixes; summary.json is the unique-call bill.'}
            trace = {'info':info, 'messages':messages, 'steps':steps}
            atomic_json(root / 'traces' / (node['id']+'.traj.json'), trace)
            for sample in step_samples(trace):
                samples.write(json.dumps(sample, ensure_ascii=False)+'\n')
