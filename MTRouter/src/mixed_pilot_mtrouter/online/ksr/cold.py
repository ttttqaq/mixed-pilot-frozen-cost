"""Resumable KSR-brunch-1 collection, counted by unique Actor/environment steps."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mixed_pilot_harness.step_collection.ksr import root_profiles, detect
from ..runtime import _atomic, BudgetExceeded, UnknownCostError, RecoverableRequestError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def labels(data):
    """Canonical per-call expected KSR returns, plus complete path views.

    Unknown descendants invalidate their ancestors, not completed siblings.
    Success is native 0/1, never the old normalized terminal score.
    """
    nodes, calls = data['nodes'], data['calls']
    memo = {}
    def value(key):
        if key in memo: return memo[key]
        n = nodes[key]
        if n['children']:
            vs = [value(c) for c in n['children']]
            y = sum(v[0] for v in vs)/len(vs) if all(v[0] is not None for v in vs) else None
            tail = sum(v[1] for v in vs)/len(vs) if all(v[1] is not None for v in vs) else None
        else:
            y = n.get('reward') if n.get('labels_valid') else None
            tail = 0. if y is not None else None
        total = (None if tail is None or any(not calls[c].get('cost_complete',True) for c in n['own'])
                 else tail + sum(calls[c]['cost'] for c in n['own']))
        memo[key] = y, total, tail
        return memo[key]
    rows, traces = [], []
    for key, n in nodes.items():
        y, _, remaining = value(key)
        for cid in reversed(n['own']):
            s = calls[cid]
            remaining = None if remaining is None or not s.get('cost_complete',True) else remaining + s['cost']
            rows.append({'sample_id': 'ksr-'+data['fingerprint'][:16]+'-'+cid, 'call_id': cid,
                'task_id': n['task_id'], 'node_id': key, 'state': s['before_state'], 'model': s['profile'],
                'reward': y, 'cost': remaining, 'source': 'real', 'weight': 1.,
                'labels_valid': y is not None and remaining is not None,
                'policy_version': data['version'], 'continuation': 'single_step_then_base_with_recursive_ksr'})
        if not n['children']:
            ids = n['prefix'] + n['own']
            tail = 0. if n.get('labels_valid') else None
            steps = []
            for cid in reversed(ids):
                s = calls[cid]
                tail = None if tail is None or not s.get('cost_complete',True) else tail+s['cost']
                steps.append({'call_id': cid, 'model': s['profile'], 'state': s['before_state'],
                    'reward': n.get('reward') if n.get('labels_valid') else None,
                    'remaining_cost': tail, 'labels_valid': tail is not None})
            traces.append({'trace_id': key, 'task_id': n['task_id'], 'base_model': n['base'],
                'status': n['status'], 'call_ids': ids, 'steps': list(reversed(steps)),
                'success': bool(n['reward']) if n.get('labels_valid') else None,
                'label_semantics': 'realized_path_return; shared prefixes are references, not independent samples'})
    return rows, traces


class ColdCollector:
    def __init__(self, trainer, rules):
        self.t, self.cfg, self.rules = trainer, trainer.config['cold_start'], rules
        self.directory = trainer.out/'cold_start'
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory/'cache.json'
        signature = {'method': trainer.config['method'], 'rules': rules,
            'runtime': trainer.manifest.get('runtime_fingerprint'),
            'split': trainer.manifest['split_fingerprint'], 'prices': trainer.runner.pricing,
            'rollout': trainer.config['rollout'], 'encoder': trainer.config['encoder'],
            'pool': trainer.config['model_pool'], 'max_nodes': self.cfg['max_nodes'], 'max_depth': self.cfg['max_depth']}
        self.finish_current_task = self.cfg.get('finish_current_task', False)
        if self.finish_current_task:
            signature['finish_current_task'] = True
            signature['trajectory_step_threshold'] = self.cfg['trajectory_steps']
        self.fingerprint = digest(signature)
        reuse = self.cfg['reuse_from']
        source = self.path if self.path.exists() else Path(reuse) if reuse else None
        if source and source.is_dir(): source = source/'cold_start'/'cache.json'
        if source:
            self.data = json.loads(source.read_text())
            if self.data['fingerprint'] != self.fingerprint:
                raise ValueError('KSR cache configuration/data/source/price semantics differ')
            count = len(self.data['calls'])
            count_valid = count >= self.cfg['trajectory_steps'] if self.finish_current_task else count == self.cfg['trajectory_steps']
            if source != self.path and (not self.data.get('complete') or not count_valid):
                raise ValueError('Reuse requires a completed cache with compatible trajectory count semantics')
            if source != self.path:
                self.data['reuse_provenance'] = str(source.resolve())
            if self.data.get('schedule_rng_state'):
                def tuples(x): return tuple(tuples(v) for v in x) if isinstance(x,list) else x
                trainer.rng.setstate(tuples(self.data['schedule_rng_state']))
        else:
            groups = {}
            for task in trainer.train: groups.setdefault(task['benchmark'], []).append(task['task_id'])
            for group in groups.values(): trainer.rng.shuffle(group)
            order = []
            while any(groups.values()):
                for group in groups.values():
                    if group: order.append(group.pop())
            self.data = {'schema': 'ksr-cold-cache-v1', 'fingerprint': self.fingerprint,
                'version': 'cold-ksr-'+self.fingerprint, 'order': order, 'cursor': 0,
                'nodes': {}, 'queue': [], 'calls': {}, 'complete': False,
                'schedule_rng_state': trainer.rng.getstate()}

    def persist(self):
        _atomic(self.path, self.data)

    def new_node(self, task, base, model, prefix, parent=None):
        key = 'n%06d' % len(self.data['nodes'])
        depth = self.data['nodes'][parent]['depth']+1 if parent else 0
        root = self.data['nodes'][parent]['root'] if parent else key
        self.data['nodes'][key] = {'task_id': task, 'base': base, 'model': model, 'prefix': list(prefix),
            'own': [], 'children': [], 'status': 'pending', 'parent': parent, 'depth': depth, 'root': root}
        return key

    def export(self):
        rows, traces = labels(self.data)
        for name, values in [('samples.jsonl', rows), ('traces.jsonl', traces)]:
            path = self.directory/name
            temporary = path.with_suffix('.tmp')
            temporary.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in values))
            temporary.replace(path)
        return [r for r in rows if r['labels_valid']]

    async def collect(self):
        t, d = self.t, self.data
        if len(d['calls']) > self.cfg['trajectory_steps'] and not self.finish_current_task:
            raise ValueError('Saved collection exceeds configured step target')
        count = len(d['calls'])
        if (count > t.ksr_trained_steps or (d.get('complete') and not t._cold_fit_complete())) and (count % self.cfg['checkpoint_steps'] == 0 or d.get('complete')):
            await t.cold_checkpoint(self, count)
        while len(d['calls']) < self.cfg['trajectory_steps'] or (self.finish_current_task and d['queue']):
            if not d['queue']:
                if d['cursor'] >= len(d['order']):
                    raise ValueError('Train task schedule exhausted before the trajectory step target; preserve cache and increase task coverage')
                identity = d['order'][d['cursor']]
                d['cursor'] += 1
                for base in root_profiles(self.rules, t.task_map[identity]):
                    d['queue'].append(self.new_node(identity, base, base, []))
                self.persist()
            key = d['queue'][0]
            node = d['nodes'][key]
            path_ids = node['prefix'] + node['own']
            steps = [d['calls'][c] for c in path_ids]
            signal_steps = []
            for i, cid in enumerate(node['own']):
                s = d['calls'][cid]; tr = s['transition']; rec = tr.get('record') or {}
                previous = path_ids.index(cid)-1
                signal_steps.append({'call_id': cid, 'command': rec.get('command'),
                    'observation': rec.get('observation', ''), 'format_error': tr.get('format_error', False),
                    'score_after': tr['score'], 'score_before': d['calls'][path_ids[previous]]['transition']['score'] if previous >= 0 else 0.})
            reason = detect(self.rules, node['base'], signal_steps)
            tree_nodes = sum(n['root'] == node['root'] for n in d['nodes'].values())
            if reason and node['depth'] < self.cfg['max_depth'] and tree_nodes+2 <= self.cfg['max_nodes']:
                node['status'] = 'split'; node['reason'] = reason
                d['queue'].pop(0)
                children = [self.new_node(node['task_id'], node['base'], m, path_ids, key) for m in t.config['model_pool']]
                node['children'] = children
                d['queue'] = children + d['queue']
                self.persist()
                continue
            model = node['model'] if not node['own'] else node['base']
            result = await t.runner.rollout(t.task_map[node['task_id']], lambda s: node['base'],
                prefix=steps, forced_first=model, horizon=1, scope='cold_start', policy_version=d['version'])
            for s in result['steps']:
                if s.get('transition') is not None:
                    s['cost_complete'] = result['cost_complete']
                    d['calls'][s['call_id']] = s
                    node['own'].append(s['call_id'])
            node['status'] = result['status']
            self.persist()
            if result['status'] == 'global_budget': raise BudgetExceeded('Global budget interrupted KSR collection')
            if not result['cost_complete']: raise UnknownCostError('Incomplete KSR Actor fee; observed steps retained')
            if result['terminal'] and result['labels_valid']:
                node.update(reward=result['reward'], labels_valid=True)
                d['queue'].pop(0)
            elif result['status'] in {'task_collection_budget', 'replay_failed'}:
                d['queue'].pop(0)
            elif result['status'] != 'truncated':
                self.persist()
                raise RecoverableColdError('KSR rollout stopped: '+result['status'])
            count = len(d['calls'])
            d['complete'] = count >= self.cfg['trajectory_steps'] and (not self.finish_current_task or not d['queue'])
            self.persist()
            if count and (count % self.cfg['checkpoint_steps'] == 0 or d['complete']) and (count > t.ksr_trained_steps or (d['complete'] and not t._cold_fit_complete())):
                await t.cold_checkpoint(self, count)
        d['complete'] = True
        self.persist()
        # Reused data requires no new Actor calls; re-fit the two heads once.
        if t.ksr_trained_steps < len(d['calls']):
            await t.cold_checkpoint(self, len(d['calls']))
        return self.export()


class RecoverableColdError(RecoverableRequestError):
    pass
