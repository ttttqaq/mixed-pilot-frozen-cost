"""A historical cold-cost offset and a budget threshold checked between rounds."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from ..learning import state_key
from ..runtime import Ledger, BudgetExceeded, UnknownCostError, RecoverableRequestError, _atomic, _hash
from ..ksr.runtime import GuardedRunner


def historical_cold_cost(root, currency):
    root = Path(root).resolve()
    known, reserved, rows, digest = 0., 0., [], hashlib.sha256()
    for path in sorted((root/'calls').glob('c*.json')):
        raw = path.read_bytes()
        row = json.loads(raw)
        if row['scope'] != 'cold_start':
            continue
        if row.get('currency') != currency or row.get('call_id') != path.stem:
            raise ValueError('Historical cold call identity/currency mismatch')
        cost, reserve = float(row.get('cost') or 0.), float(row.get('reserved_cost', 0.))
        if not all(math.isfinite(v) and v >= 0 for v in (cost, reserve)):
            raise ValueError('Invalid historical cold fee')
        if (row.get('status') == 'in_flight' or
                (not row.get('cost_complete') and reserve <= 0)):
            raise UnknownCostError('Unbounded historical cold fee: '+path.stem)
        if row.get('cost_complete') and row.get('cost') is None:
            raise UnknownCostError('Historical complete call is missing its fee')
        digest.update(path.name.encode()); digest.update(raw)
        known += cost; reserved += reserve; rows.append(row['call_id'])
    if not rows:
        raise ValueError('Historical run has no cold_start calls')
    return {'source_run': str(root), 'currency': currency, 'scope': 'cold_start',
            'known_cost': known, 'reserved_cost': reserved, 'budget_cost': known+reserved,
            'call_count': len(rows), 'calls_sha256': digest.hexdigest()}


class RoundBudgetLedger(Ledger):
    def __init__(self, output, limit, currency):
        output = Path(output)
        settings = json.loads((output/'config.resolved.json').read_text())['experiment_budget']
        self.threshold = float(settings['threshold'])
        if not math.isfinite(self.threshold) or self.threshold <= 0 or self.threshold != float(limit):
            raise ValueError('experiment_budget.threshold must equal the configured positive budget.total')
        self.historical = historical_cold_cost(settings['historical_run'], currency)
        self.phase_path = output/'round-budget-phase.json'
        self.phase = json.loads(self.phase_path.read_text()) if self.phase_path.exists() else {'active': False}
        source = output/'historical-cold-cost.json'
        if source.exists() and json.loads(source.read_text()) != self.historical:
            raise ValueError('Historical cold accounting changed since initialization')
        _atomic(source, self.historical)
        # GuardedRunner still checks each request's per-path/task envelope. The
        # user-authorized global threshold is tested at durable round boundaries.
        super().__init__(output, 1.e100, currency)

    @property
    def total_budget_cost(self):
        return self.historical['budget_cost'] + self.budget_cost

    @property
    def cost_complete(self):
        return super().cost_complete and self.historical['reserved_cost'] == 0

    @property
    def threshold_reached(self):
        return self.total_budget_cost >= self.threshold

    @property
    def exhausted(self):
        return not self.phase.get('active') and self.threshold_reached

    def check(self):
        if not self.spending_safe:
            raise UnknownCostError('Round ledger contains unbounded or in-flight fees')
        if not self.phase.get('active'):
            raise BudgetExceeded('No active authorized round/assessment phase')

    def activate(self, iteration):
        if self.phase.get('active') and self.phase.get('iteration') != iteration:
            raise ValueError('Cannot overlap budget phases')
        self.phase = {'active': True, 'iteration': iteration}
        _atomic(self.phase_path, self.phase)
        self._write_summary()

    def deactivate(self):
        self.phase = {**self.phase, 'active': False}
        _atomic(self.phase_path, self.phase)
        self._write_summary()

    def report(self):
        result = super().report()
        total = self.total_budget_cost
        result.update(limit=self.threshold, remaining=max(0., self.threshold-total),
                      historical_cold=self.historical, new_known_cost=self.cost,
                      new_reserved_cost=self.reserved_cost, new_budget_cost=self.budget_cost,
                      experiment_known_cost=self.historical['known_cost']+self.cost,
                      experiment_budget_cost=total, threshold_reached=self.threshold_reached,
                      budget_exhausted=self.exhausted, overshoot=max(0., total-self.threshold),
                      phase=deepcopy(self.phase),
                      budget_semantics='soft threshold: finish admitted round and its complete evaluation')
        return result


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


class RoundBudgetRunner(GuardedRunner):
    """Reuse completed rollouts and resume saved partial prefixes after failure."""
    journal_iteration = None

    async def rollout(self, task, policy, **kwargs):
        if self.journal_iteration is None:
            raise ValueError('Set rollout journal iteration before dispatch')
        original_prefix = kwargs.get('prefix', [])
        signature = {'iteration': self.journal_iteration, 'task_id': task['task_id'],
            'policy_version': kwargs.get('policy_version'), 'scope': kwargs.get('scope'),
            'forced_first': kwargs.get('forced_first'), 'horizon': kwargs.get('horizon'),
            'prefix': [s['call_id'] for s in original_prefix],
            'expected_root': state_key(kwargs['expected_root']) if kwargs.get('expected_root') else None}
        path = self.output/'round-rollouts'/(_hash(signature)+'.json')
        previous = json.loads(path.read_text()) if path.exists() else None
        rng = getattr(policy, 'rng', None)
        if previous and previous.get('trajectory_id'):
            raw_path = self.output/'trajectories'/(previous['trajectory_id']+'.json')
            if raw_path.exists():
                raw = json.loads(raw_path.read_text())
                # The raw runner persists every transition, including before a
                # process kill that bypassed the journal's exception handler.
                if previous.get('result') is None:
                    previous['result'] = raw
                    if any(s.get('transition') is None for s in raw['all_steps']):
                        raise RecoverableRequestError('Paid response has no persisted transition; repair '+str(raw_path))
                    if rng is not None and previous.get('rng_before') is not None:
                        rng.setstate(tuples(previous['rng_before']))
                        for i, step in enumerate(raw['steps']):
                            if not (i == 0 and signature['forced_first'] is not None):
                                if policy(step['before_state']) != step['profile']:
                                    raise RecoverableRequestError('Cannot restore rollout RNG without changing its actions')
                        previous['rng_after'] = rng.getstate()
                    previous['complete'] = bool(raw.get('outcome_valid') or raw.get('status') == 'truncated')
            elif previous.get('result') is None:
                paid = [r for r in self.ledger.records.values()
                        if r.get('metadata', {}).get('trajectory_id') == previous['trajectory_id']]
                if paid:
                    raise RecoverableRequestError('Paid rollout has no transition snapshot; repair '+previous['trajectory_id'])
                previous = None
        if previous:
            if previous.get('rng_after') is not None and rng is not None:
                rng.setstate(tuples(previous['rng_after']))
            if previous['complete']:
                return previous['result']
            prefix = previous['result'].get('all_steps', [])
            if prefix:
                if any(s.get('transition') is None for s in prefix):
                    raise RecoverableRequestError('Cannot resume a response without transition evidence')
                kwargs['prefix'] = prefix
                kwargs['expected_root'] = previous['result']['state']
                if len(prefix) > len(original_prefix):
                    kwargs['forced_first'] = None
                if kwargs.get('horizon') is not None:
                    remaining = kwargs['horizon']-(len(prefix)-len(original_prefix))
                    if remaining <= 0 and not prefix[-1]['transition']['done']:
                        result = {**previous['result'], 'status': 'truncated', 'terminal': False,
                                  'labels_valid': False, 'reward': None}
                        _atomic(path, {**previous, 'complete': True, 'result': result})
                        return result
                    kwargs['horizon'] = max(1, remaining)
        attempt = (previous or {}).get('attempt', 0)+1
        trajectory_id = 'rb-'+path.stem+'-%03d'%attempt
        kwargs['trajectory_id'] = trajectory_id
        pending = {'signature': signature, 'complete': False, 'result': None,
                   'trajectory_id': trajectory_id, 'attempt': attempt,
                   'rng_before': rng.getstate() if rng else None}
        _atomic(path, pending)
        try:
            result = await super().rollout(task, policy, **kwargs)
        except BaseException as exc:
            if hasattr(exc, 'rollout_result'):
                _atomic(path, {**pending, 'complete': False, 'result': exc.rollout_result,
                               'rng_after': rng.getstate() if rng else None})
            raise
        complete = not (kwargs.get('scope') in {'test', 'valid'} and
                        not result.get('outcome_valid', result.get('labels_valid', False)))
        _atomic(path, {**pending, 'complete': complete, 'result': result,
                       'rng_after': rng.getstate() if rng else None})
        return result
