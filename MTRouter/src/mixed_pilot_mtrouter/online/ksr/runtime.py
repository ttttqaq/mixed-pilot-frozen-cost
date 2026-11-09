"""Per-task unique-call budgets and exact public-state replay guards."""
from dataclasses import asdict
import json
import math

from ..runtime import (Runner, RecoverableRequestError, UnknownCostError, ReplayError,
                       InvalidTeacherEvidenceError, BudgetExceeded, _atomic, _message, _canonical)
from ..learning import state_key


class ActorBudgetStop(RecoverableRequestError):
    def __init__(self, kind):
        self.kind = kind
        super().__init__(kind)


def request_bound(messages, system, tools, rates, available, max_output):
    """Conservative UTF-8 token envelope, uncached maximum rate across bands.

    This is a guard against exceeding the configured price ledger, not a provider
    invoice guarantee. Unknown usage stops the run; response costs are checked.
    """
    raw = {'messages': [asdict(_message(m)) for m in messages], 'system': str(system),
           'tools': [asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in tools or []]}
    input_bound = len(json.dumps(raw, ensure_ascii=False).encode('utf-8')) + 1024 + 64 * len(messages)
    bands = rates.get('bands', [rates])
    if input_bound > max(b.get('max_input_tokens', float('inf')) for b in bands):
        return None
    pin = max(max(b['input_per_million'], b.get('cached_input_per_million', 0.)) for b in bands)
    pout = max(b['output_per_million'] for b in bands)
    fixed = input_bound * pin / 1e6
    cap = max_output if not pout else min(max_output, math.floor((available-fixed) * 1e6 / pout))
    if available <= fixed or cap < 1:
        return None
    return {'input_tokens_bound': input_bound, 'max_tokens': int(cap),
            'cost_bound': (input_bound * pin + cap * pout) / 1e6}


class GuardedRunner(Runner):
    def _state(self, task, episode, *args, **kwargs):
        state = super()._state(task, episode, *args, **kwargs)
        self.path_episode = episode
        self.path_remaining = state['remaining_budget']
        return state

    def task_spent(self, task_id, version):
        rows = [r for r in self.ledger.records.values() if r['scope'] != 'teacher'
                and r.get('metadata', {}).get('task_id') == task_id
                and r.get('metadata', {}).get('policy_version') == version]
        # Use the same known+reserved upper bound as the global ledger. Treating
        # a bounded HTTP failure as an entire task budget prevents its retry and
        # leaves evaluation outcomes unscored. Unbounded fees still fail closed.
        return sum((float(r.get('cost') or 0.) + float(r.get('reserved_cost', 0.)))
                   if r.get('cost_complete') or r.get('reserved_cost', 0.) > 0
                   else self.task_budget for r in rows)

    async def _complete_once(self, profile, messages, system='', tools=None, scope='teacher', metadata=None, profile_overrides=None):
        if scope == 'teacher':
            # A failed Teacher reply may omit usage. Reserve its possible fee,
            # reject its evidence and let the bounded Teacher retry/fallback run.
            p = self.harness.profile(profile)
            maximum = (profile_overrides or {}).get('max_tokens',
                self.config['teacher']['llm'].get('max_tokens', p.max_tokens)) or 8192
            request = {'messages': [asdict(_message(m)) for m in messages], 'system': str(system),
                       'tools': [asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in tools or []]}
            input_bound = 2 * len(_canonical(request).encode('utf-8')) + 4096
            rates = self.pricing['profiles'][profile]
            bands = rates.get('bands', [rates])
            upper = (input_bound * max(b['input_per_million'] for b in bands)
                     + maximum * max(b['output_per_million'] for b in bands)) / 1e6
            if self.ledger.budget_cost + upper > self.ledger.limit:
                raise BudgetExceeded('Remaining global budget cannot cover the Teacher request envelope')
            previous = set(self.ledger.records)
            try:
                return await super()._complete_once(profile, messages, system, tools, scope, metadata, profile_overrides)
            except UnknownCostError as exc:
                new = set(self.ledger.records) - previous
                if len(new) != 1:
                    raise
                cid = new.pop()
                row = self.ledger.records[cid]
                if row.get('status') != 'pricing_error' or row.get('response') is None:
                    raise
                self._reserve_uncertain(cid)
                error = InvalidTeacherEvidenceError('Teacher usage is missing/invalid; fee reserved for '+cid)
                error.call_id = cid
                error.fee_reserved = True
                raise error from exc
        metadata = dict(metadata or {})
        path_left = getattr(self, 'path_remaining', self.task_budget)
        task_left = self.task_budget - self.task_spent(metadata.get('task_id'), metadata.get('policy_version'))
        global_left = self.ledger.limit - self.ledger.budget_cost
        available = min(path_left, task_left, global_left)
        kind = 'path_request_budget' if path_left <= task_left + 1e-10 else 'task_collection_budget'
        p = self.harness.profile(profile)
        overrides = dict(profile_overrides or {})
        envelope = request_bound(messages, system, tools, self.pricing['profiles'][profile],
                                 available, int(overrides.get('max_tokens', p.max_tokens or 8192)))
        if envelope is None:
            if global_left < min(path_left, task_left):
                raise BudgetExceeded('Remaining global budget cannot cover the Actor request envelope')
            stop = ActorBudgetStop(kind)
            if kind == 'path_request_budget' or scope in {'test', 'valid'}:
                # Grade the exact live environment BEFORE Runner closes it.
                # ScienceWorld can evolve stochastically even under identical
                # commands; replaying merely to grade can change the outcome.
                stop.native_evaluation = await self.path_episode.evaluate()
            raise stop
        overrides['max_tokens'] = envelope['max_tokens']
        metadata['budget_envelope'] = envelope
        result = await super()._complete_once(profile, messages, system, tools, scope, metadata, overrides)
        if result['cost'] > envelope['cost_bound'] + 1e-9:
            raise RecoverableRequestError('Provider usage exceeded the request envelope; incurred charge retained')
        return result

    async def _restore(self, task, prefix):
        expected = getattr(self, 'expected_root', None)
        for _ in range(max(1, self.replay_attempts)):
            ep, history, initial, replay = await super()._restore(task, prefix)
            if expected is None or state_key(self._state(task, ep, history, prefix, expected['policy_version'])) == state_key(expected):
                return ep, history, initial, replay
            ep.close()
        raise ReplayError('Expected public root could not be restored before API dispatch')

    async def rollout(self, task, policy, *, expected_root=None, **kwargs):
        self.expected_root = expected_root
        try:
            return self._evaluation_cost_integrity(await super().rollout(task, policy, **kwargs))
        except ActorBudgetStop as exc:
            result = exc.rollout_result
            result.update(status=exc.kind, terminal=False, labels_valid=False, reward=None)
            if hasattr(exc, 'native_evaluation'):
                # A single path cannot afford another bounded call. Native grading
                # is valid here; a shared tree's collection cutoff is NOT failure.
                value = getattr(exc, 'native_evaluation', {})
                if value.get('status') == 'scored' and value.get('cost_complete') and not value.get('api_calls', 0) and not value.get('cost', 0):
                    result.update(evaluation=value, terminal=True, labels_valid=result['cost_complete'],
                        outcome_valid=True, success=bool(value['success']),
                        reward=float(value['success']), outcome_reward=float(value['success']), final_score=value['score'],
                        budget_grading='live_environment_before_close')
            _atomic(self.output / 'trajectories' / (result['trajectory_id']+'.json'), result)
            return self._evaluation_cost_integrity(result)
        finally:
            self.expected_root = None

    def _evaluation_cost_integrity(self, result):
        if result['scope'] in {'test', 'valid'}:
            # Resumed prefixes may have uncertain failed attempts outside the
            # new call_ids. Native outcome evidence remains valid, but a C
            # target/known evaluation fee must not hide those old reservations.
            uncertain = any(not r.get('cost_complete') for r in self.ledger.records.values()
                if r['scope'] != 'teacher' and r.get('metadata', {}).get('task_id') == result['task_id']
                and r.get('metadata', {}).get('policy_version') == result['policy_version'])
            if uncertain or not result['cost_complete']:
                result.update(cost_complete=False, labels_valid=False, reward=None)
                _atomic(self.output/'trajectories'/(result['trajectory_id']+'.json'), result)
        return result
