"""KSR cold initialization, then online Y evidence with immutable cold C."""
from copy import deepcopy
import math
import uuid

from ..learning import state_key
from ..runtime import _atomic, BudgetExceeded, UnknownCostError
from ..teacher import append_jsonl
from ..trainer import FrozenPolicy
from ..ksr.trainer import KSRTrainer
from .learning import FrozenCostLearner
from .teacher import RewardOnlyTeacher


def reward_target(result, tail=None):
    if result['terminal'] and result['labels_valid']:
        return {'reward': result['reward'], 'source': 'real', 'ready': True}
    if result['status'] == 'truncated' and tail is not None and tail.get('ready'):
        return {'reward': tail['reward'], 'source': 'llm_teacher', 'ready': True}
    return None


class FrozenCostPolicy(FrozenPolicy):
    def describe(self, state):
        result = super().describe(state)
        result.update(beta=self.config['decision']['beta'], cost_model='immutable_cold_start',
                      cost_model_fingerprint=self.learner.cost_fingerprint,
                      endpoint_frozen_costs={m: v['cost'] for m, v in self.learner.predict_actions(state).items()})
        return result


class FrozenCostTrainer(KSRTrainer):
    learner_class = FrozenCostLearner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence = RewardOnlyTeacher(self.config, self.runner, self.out, self.evidence.references)
        if self.cold_done and self.learner.frozen_cost_router is None:
            raise ValueError('Completed cold start is missing the frozen C model')

    def _policy(self, version, *, greedy=False):
        return FrozenCostPolicy(self.learner, self.config, version, self.rng, greedy=greedy, output=self.out)

    def _after_cold_fit(self, count, *, complete=False):
        if complete and count >= self.config['cold_start']['trajectory_steps']:
            self.learner.freeze_cost()

    def _cold_fit_complete(self):
        return self.learner.frozen_cost_router is not None

    async def _cold_start(self):
        if self.learner.cold_initialization:
            self.ksr_trained_steps = self.learner.cold_initialization['trajectory_steps']
            self.event('cold_checkpoint_imported', **self.learner.cold_initialization)
            self._save(0)
            if self.config['evaluation']['after_cold_start'] and self.config['evaluation']['enabled']:
                await self._evaluate(0)
            self._save(0, cold_done=True)
        else:
            await super()._cold_start()

    def _select(self, visits, count):
        pool = list({(v['state']['task_id'], state_key(v['state'])): v for v in visits}.values())
        if not self.config['selection']['enabled']:
            return self.rng.sample(pool, min(count, len(pool)))
        for v in pool:
            vals = list(self.learner.predict_actions(v['state']).values())
            v['acquisition'] = sum(z['reward_std'] for z in vals) + abs(vals[0]['reward']-vals[1]['reward'])
        n = min(count, len(pool))
        random_n = min(n, math.ceil(n*self.config['selection']['random_fraction']))
        chosen = self.rng.sample(pool, random_n)
        remaining = [v for v in pool if v not in chosen]
        return chosen + sorted(remaining, key=lambda v: v['acquisition'], reverse=True)[:n-random_n]

    async def _real_rows(self, result, *, start=0):
        if not result['terminal'] or not result['labels_valid'] or result['split'] != 'train':
            return
        for step in result['all_steps'][start:]:
            self._put_router({'sample_id': step['call_id'], 'state': step['before_state'],
                'model': step['profile'], 'reward': result['reward'], 'source': 'real', 'weight': 1.,
                'policy_version': result['policy_version'], 'continuation': 'frozen_epsilon_policy',
                'trajectory_id': result['trajectory_id']})

    async def _query(self, visit, policy, version, *, force_full=False, audit_slots=0, random_audit=None):
        start = len(visit['prefix'])
        pool, short = self.config['model_pool'], self.config['short_rollout']
        full = force_full or not short['enabled'] or not self.config['teacher']['enabled']
        branches = []
        for model in pool:
            result = await self.runner.rollout(visit['task'], policy, prefix=visit['prefix'], forced_first=model,
                horizon=None if full else short['horizon'], scope='branch', policy_version=version,
                expected_root=visit['state'])
            branches.append(result)
            if result['status'] == 'global_budget':
                raise BudgetExceeded('Global budget interrupted A/B')
            if not result['cost_complete']:
                raise UnknownCostError('Incomplete branch cost')
        tails = [await self._tail(r, policy) if not full else None for r in branches]
        targets = [reward_target(r, tail) for r, tail in zip(branches, tails)]

        def ambiguous():
            return (any(t is None for t in targets) or
                    abs(targets[0]['reward']-targets[1]['reward']) <= self.config['audit']['reward_threshold'])

        expanded = short['horizon']
        while not full and not random_audit and short['adaptive'] and ambiguous() and expanded < short['max_horizon']:
            extension = min(short['extension'], short['max_horizon']-expanded)
            for i, r in enumerate(branches):
                if r['status'] == 'truncated':
                    branches[i] = await self.runner.rollout(visit['task'], policy, prefix=r['all_steps'],
                        horizon=extension, scope='branch', policy_version=version, expected_root=r['state'])
            expanded += extension
            tails = [await self._tail(r, policy) for r in branches]
            targets = [reward_target(r, tail) for r, tail in zip(branches, tails)]
        before = deepcopy(targets)
        missing = any(t is None for t in targets)
        audited = (missing or (self.config['audit']['enabled'] and audit_slots > 0 and
                   (random_audit or ambiguous()))) and any(r['status'] == 'truncated' for r in branches)
        if audited:
            for i, r in enumerate(branches):
                if r['status'] == 'truncated':
                    branches[i] = await self.runner.rollout(visit['task'], policy, prefix=r['all_steps'],
                        scope='audit', policy_version=version, expected_root=r['state'])
            targets = [reward_target(r) for r in branches]
        for r in branches:
            await self._real_rows(r, start=start)
        if any(t is None for t in targets) or any(len(r['all_steps']) <= start for r in branches):
            self.event('pair_invalid', reason='missing_grounded_Y', statuses=[r['status'] for r in branches])
            return None, int(audited)
        first = [r['all_steps'][start] for r in branches]
        if state_key(first[0]['before_state']) != state_key(first[1]['before_state']):
            raise RuntimeError('A/B public states differ')
        actual = all(t['source'] == 'real' for t in targets)
        weight = 1. if actual else self.config['optimization']['pseudo_weight']
        pair_id = uuid.uuid4().hex
        frozen_costs = {m: v['cost'] for m, v in self.learner.predict_actions(first[0]['before_state']).items()}
        pair = {'pair_id': pair_id, 'state': first[0]['before_state'], 'a': pool[0], 'b': pool[1],
            'reward_delta': targets[0]['reward']-targets[1]['reward'], 'targets': dict(zip(pool, targets)),
            'frozen_costs': frozen_costs, 'cost_source': 'frozen_cold_router',
            'weight': weight, 'policy_version': version, 'source': 'real' if actual else 'llm_teacher',
            'audited': audited, 'branch_trajectory_ids': [r['trajectory_id'] for r in branches],
            'intervention_call_ids': [s['call_id'] for s in first]}
        self.pairs[pair_id] = pair
        append_jsonl(self.out/'pairs.jsonl', pair)
        for model, step, target in zip(pool, first, targets):
            self._put_router({'sample_id': step['call_id'], 'state': step['before_state'], 'model': model,
                'reward': target['reward'], 'source': target['source'],
                'weight': 1. if target['source'] == 'real' else weight,
                'policy_version': version, 'pair_id': pair_id})
        if audited and all(t is not None for t in before):
            append_jsonl(self.out/'teacher_audits.jsonl', {'pair_id': pair_id, 'policy_version': version,
                'reward_errors': [a['reward']-b['reward'] for a, b in zip(before, targets)]})
        return pair, int(audited)

    def _summary(self, stop):
        report = super()._summary(stop)
        report.update(training_objective='joint_Y_C_cold_then_Y_only',
                      cold_step_threshold=self.config['cold_start']['trajectory_steps'],
                      finish_current_task=True,
                      cost_frozen=self.learner.frozen_cost_router is not None,
                      cost_model_fingerprint=self.learner.cost_fingerprint,
                      cold_initialization=self.learner.cold_initialization)
        if self.learner.cold_initialization:
            report['cold_trajectory_steps'] = self.ksr_trained_steps
            report['new_cold_actor_calls'] = 0
        _atomic(self.out/'summary.json', report)
        return report
