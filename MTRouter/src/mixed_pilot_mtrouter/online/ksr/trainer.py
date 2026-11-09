"""Reference Teacher online loop with a reusable, trajectory-counted KSR cold start."""
from copy import deepcopy
import hashlib
import json
import math
import uuid

from ..learning import state_key
from ..runtime import _atomic
from ..teacher import append_jsonl
from ..trainer import OnlineTrainer, result_target
from .learning import SeparateLearner
from .runtime import GuardedRunner
from .cold import ColdCollector
from .teacher import ResilientReferenceTeacher


class KSRTrainer(OnlineTrainer):
    learner_class = SeparateLearner
    runner_class = GuardedRunner

    def __init__(self, *args, **kwargs):
        self.ksr_trained_steps = 0
        super().__init__(*args, **kwargs)
        self.evidence = ResilientReferenceTeacher(self.config, self.runner, self.out, self.evidence.references)
        if self.resume:
            self.ksr_trained_steps = json.loads((self.out/'state.json').read_text()).get('ksr_trained_steps', 0)

    def _save(self, completed, *, cold_done=None):
        if cold_done is not None: self.cold_done = cold_done
        self.completed_iterations = completed
        name = 'checkpoints/'+uuid.uuid4().hex+'.pt'
        metadata = {'completed_iterations': completed, 'cold_done': self.cold_done,
            'collected_cold_steps': self.ksr_trained_steps, 'simulation': self.simulation,
            'router_policy_version': self._policy_id()}
        self.learner.save(self.out/name, metadata)
        _atomic(self.out/'state.json', {'completed_iterations': completed, 'cold_done': self.cold_done,
            'cold_stage': self.cold_stage, 'cold_query_refs': self.cold_query_refs,
            'cold_queries_done': self.cold_queries_done, 'ksr_trained_steps': self.ksr_trained_steps,
            'checkpoint': name, 'best_valid': self.best_valid, 'last_evaluated': self.last_evaluated,
            'rng_state': self.rng.getstate(), 'router_replay': list(self.router_rows.values()),
            'pairs': list(self.pairs.values())})
        self.learner.save(self.out/'latest.pt', metadata)

    def _after_cold_fit(self, count, *, complete=False):
        pass

    def _cold_fit_complete(self):
        return True

    async def cold_checkpoint(self, collector, count):
        rows = collector.export()
        if not rows:
            report = {'samples': 0, 'updates': 0, 'note': 'No grounded terminal labels yet; weights remain unchanged'}
        else:
            for row in rows: self._put_router(row)
            report = self.learner.train_router(rows)
        self._after_cold_fit(count, complete=collector.data.get('complete', False))
        self.ksr_trained_steps = count
        self._save(0)
        directory = collector.directory/'checkpoints'
        directory.mkdir(exist_ok=True)
        _atomic(directory/('step_%06d.json' % count), collector.data)
        self.learner.save(directory/('step_%06d.pt' % count), {
            'completed_iterations': 0, 'collected_cold_steps': count,
            'valid_labels': len(rows), 'cache_fingerprint': collector.fingerprint,
            'simulation': self.simulation, 'objective': 'separate_Y_C'})
        self.event('ksr_cold_checkpoint', trajectory_steps=count, router=report,
            total_saved_steps=len(collector.data['calls']), valid_labels=len(rows))

    async def _cold_start(self):
        from mixed_pilot_harness.step_collection.ksr import load_config
        rules = load_config(self.config['cold_start']['ksr_config'], self.runner.harness.profiles)
        if {rules['strong_model'], rules['weak_model']} != set(self.config['model_pool']):
            raise ValueError('KSR model pool differs from Router model pool')
        collector = ColdCollector(self, rules)
        try:
            rows = await collector.collect()
        finally:
            collector.persist()
            collector.export()
        if not rows:
            raise ValueError('KSR reached the sample target without any valid terminal labels')
        self.event('cold_start_finished', policy='KSR-brunch-1', trajectory_steps=len(collector.data['calls']),
            valid_labels=len(rows), reused=bool(collector.data.get('reuse_provenance')))
        if self.config['evaluation']['after_cold_start'] and self.config['evaluation']['enabled']:
            await self._evaluate(0)
        self._save(0, cold_done=True)

    def _select(self, visits, count):
        pool = list({(v['state']['task_id'], state_key(v['state'])): v for v in visits}.values())
        if not self.config['selection']['enabled']:
            return self.rng.sample(pool, min(count, len(pool)))
        # No beta or U: query uncertainty/coverage in the two supervised targets.
        for v in pool:
            vals = list(self.learner.predict_actions(v['state']).values())
            v['acquisition'] = sum(z['reward_std'] + z['cost_std']/self.config['budget']['cost_reference'] for z in vals)
            v['acquisition'] += abs(vals[0]['reward']-vals[1]['reward'])
        n = min(count, len(pool))
        random_n = min(n, math.ceil(n*self.config['selection']['random_fraction']))
        chosen = self.rng.sample(pool, random_n)
        remaining = [v for v in pool if v not in chosen]
        return chosen + sorted(remaining, key=lambda v: v['acquisition'], reverse=True)[:n-random_n]

    async def _query(self, visit, policy, version, *, force_full=False, audit_slots=0, random_audit=None):
        start = len(visit['prefix'])
        pool, short = self.config['model_pool'], self.config['short_rollout']
        full = force_full or not short['enabled'] or not self.config['teacher']['enabled']
        branches = []
        for model in pool:
            r = await self.runner.rollout(visit['task'], policy, prefix=visit['prefix'], forced_first=model,
                horizon=None if full else short['horizon'], scope='branch', policy_version=version,
                expected_root=visit['state'])
            branches.append(r)
            if r['status'] == 'global_budget':
                from ..runtime import BudgetExceeded
                raise BudgetExceeded('Global budget interrupted A/B')
            if not r['cost_complete']:
                from ..runtime import UnknownCostError
                raise UnknownCostError('Incomplete branch cost')
        tails = [await self._tail(r, policy) if not full else None for r in branches]
        targets = [result_target(r, start, tail) for r, tail in zip(branches, tails)]
        def ambiguous():
            return (any(t is None for t in targets) or
                (abs(targets[0]['reward']-targets[1]['reward']) <= self.config['audit']['reward_threshold']
                 and abs(targets[0]['cost']-targets[1]['cost']) <= self.config['audit']['cost_threshold']))
        expanded = short['horizon']
        while not full and not random_audit and short['adaptive'] and ambiguous() and expanded < short['max_horizon']:
            extension = min(short['extension'], short['max_horizon']-expanded)
            for i, r in enumerate(branches):
                if r['status'] == 'truncated':
                    branches[i] = await self.runner.rollout(visit['task'], policy, prefix=r['all_steps'],
                        horizon=extension, scope='branch', policy_version=version, expected_root=r['state'])
            expanded += extension
            tails = [await self._tail(r, policy) for r in branches]
            targets = [result_target(r, start, tail) for r, tail in zip(branches, tails)]
        before = deepcopy(targets)
        missing = any(t is None for t in targets)
        audited = (missing or (self.config['audit']['enabled'] and audit_slots > 0 and
                  (random_audit or ambiguous()))) and any(r['status'] == 'truncated' for r in branches)
        if audited:
            for i, r in enumerate(branches):
                if r['status'] == 'truncated':
                    branches[i] = await self.runner.rollout(visit['task'], policy, prefix=r['all_steps'],
                        scope='audit', policy_version=version, expected_root=r['state'])
            targets = [result_target(r, start) for r in branches]
        for r in branches:
            await self._real_rows(r, start=start)
        if any(t is None for t in targets) or any(len(r['all_steps']) <= start for r in branches):
            self.event('pair_invalid', reason='missing_grounded_Y_C', statuses=[r['status'] for r in branches])
            return None, int(audited)
        first = [r['all_steps'][start] for r in branches]
        if state_key(first[0]['before_state']) != state_key(first[1]['before_state']):
            raise RuntimeError('A/B public states differ')
        actual = all(t['source'] == 'real' for t in targets)
        weight = 1. if actual else self.config['optimization']['pseudo_weight']
        pair_id = uuid.uuid4().hex
        pair = {'pair_id': pair_id, 'state': first[0]['before_state'], 'a': pool[0], 'b': pool[1],
            'reward_delta': targets[0]['reward']-targets[1]['reward'],
            'cost_delta': targets[0]['cost']-targets[1]['cost'],
            'targets': dict(zip(pool, targets)), 'weight': weight, 'policy_version': version,
            'source': 'real' if actual else 'llm_teacher', 'audited': audited,
            'branch_trajectory_ids': [r['trajectory_id'] for r in branches],
            'intervention_call_ids': [s['call_id'] for s in first]}
        self.pairs[pair_id] = pair
        append_jsonl(self.out/'pairs.jsonl', pair)
        for model, step, target in zip(pool, first, targets):
            self._put_router({'sample_id': step['call_id'], 'state': step['before_state'], 'model': model,
                'reward': target['reward'], 'cost': target['cost'], 'source': target['source'],
                'weight': 1. if target['source'] == 'real' else weight, 'policy_version': version,
                'pair_id': pair_id})
        if audited and all(t is not None for t in before):
            append_jsonl(self.out/'teacher_audits.jsonl', {'pair_id': pair_id, 'policy_version': version,
                'reward_errors': [a['reward']-b['reward'] for a,b in zip(before, targets)],
                'cost_errors': [a['cost']-b['cost'] for a,b in zip(before, targets)]})
        return pair, int(audited)

    def _summary(self, stop):
        report = super()._summary(stop)
        cache = self.out/'cold_start/cache.json'
        collected = len(json.loads(cache.read_text())['calls']) if cache.exists() else 0
        report.update(schema=self.config['method'], cold_trajectory_steps=collected,
                      cold_checkpoint_steps=self.ksr_trained_steps,
                      beta=self.config['decision']['beta'], training_objective='separate_Y_C')
        _atomic(self.out/'summary.json', report)
        return report
