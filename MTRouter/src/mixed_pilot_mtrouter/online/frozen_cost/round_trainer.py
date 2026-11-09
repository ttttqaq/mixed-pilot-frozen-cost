"""Finish an admitted online round and assessment before enforcing its threshold."""
import json
import math

from ..runtime import _atomic, BudgetExceeded, UnknownCostError, RecoverableRequestError, InvalidTeacherEvidenceError
from ..teacher import append_jsonl
from .trainer import FrozenCostTrainer
from .round_budget import RoundBudgetLedger, RoundBudgetRunner, tuples


class RoundBudgetTrainer(FrozenCostTrainer):
    ledger_class = RoundBudgetLedger
    runner_class = RoundBudgetRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config['evaluation']['after_cold_start']:
            raise ValueError('Round-budget online initialization must disable after_cold_start evaluation')
        if not self.cold_done and not self.learner.cold_initialization:
            raise ValueError('Round-budget online experiment requires explicit cold checkpoint initialization')
        self.round_path = self.out/'online-round.json'
        self.round_state = json.loads(self.round_path.read_text()) if self.round_path.exists() else None

    def _round_save(self, **values):
        self.round_state.update(values)
        _atomic(self.round_path, self.round_state)

    async def _collect_round(self, iteration):
        # A restart replays the logical round using journaled results, with the
        # same RNG transitions. Only unfinished requests can dispatch again.
        self.rng.setstate(tuples(self.round_state['rng_start']))
        self.evidence.begin_phase()
        version = self._policy_id()
        if version != self.round_state['policy_version']:
            raise ValueError('Uncommitted round policy does not match its starting checkpoint')
        policy = self._policy(version)
        visits = []
        tasks = self.rng.sample(self.train, min(len(self.train), self.config['online']['tasks_per_iteration']))
        for task in tasks:
            result = await self.runner.rollout(task, policy, scope='collection', policy_version=version)
            await self._real_rows(result)
            visits.extend(self._visits(task, result))
        selected = self._select(visits, self.config['online']['states_per_iteration'])
        audit_cfg, audits = self.config['audit'], 0
        random_count = min(len(selected), audit_cfg['max_pairs_per_iteration'],
            math.ceil(len(selected)*audit_cfg['random_fraction'])) if audit_cfg['enabled'] else 0
        random_indexes = set(self.rng.sample(range(len(selected)), random_count))
        order = sorted(range(len(selected)), key=lambda i: i not in random_indexes)
        for i in order:
            _, used = await self._query(selected[i], policy, version,
                audit_slots=max(0, audit_cfg['max_pairs_per_iteration']-audits), random_audit=i in random_indexes)
            audits += used
        router = self._train_router(version)
        report = {'iteration': iteration, 'policy_version': version, 'next_policy_version': self._policy_id(),
            'visited_states': len(visits), 'audited_pairs': audits, 'teacher': 'fixed_reference_success_only',
            'router': router, 'iteration_cost': self.ledger.cost-self.round_state['start_cost'],
            'total_cost': self.ledger.cost, 'experiment_budget_cost': self.ledger.total_budget_cost,
            'cost_complete': self.ledger.cost_complete,
            'real_router_samples': sum(r['source']=='real' for r in self.router_rows.values()),
            'pseudo_router_samples': sum(r['source']!='real' for r in self.router_rows.values())}
        self._round_save(training_report=report)
        # state.json is the model/optimizer commit point. A crash after this save
        # must resume evaluation, never train the same round a second time.
        self._save(iteration)
        return report

    async def run(self):
        stop = 'iterations_completed'
        try:
            if not self.resume:
                self._save(0)
            if not self.cold_done:
                await self._cold_start()
            while True:
                pending = self.round_state and self.round_state.get('stage') != 'finished'
                if pending:
                    iteration = self.round_state['iteration']
                else:
                    if self.ledger.phase.get('active'):
                        self.ledger.deactivate()
                    if self.ledger.threshold_reached:
                        stop = 'budget_round_completed'
                        break
                    if self.completed_iterations >= self.config['online']['iterations']:
                        break
                    iteration = self.completed_iterations+1
                    self._save(self.completed_iterations)
                    self.round_state = {'iteration': iteration, 'stage': 'collecting',
                        'rng_start': self.rng.getstate(), 'policy_version': self._policy_id(),
                        'start_cost': self.ledger.cost, 'cost_model_fingerprint': self.learner.cost_fingerprint}
                    _atomic(self.round_path, self.round_state)
                    self.event('iteration_started', iteration=iteration,
                               policy_version=self.round_state['policy_version'], continuation='frozen_epsilon_policy')
                self.ledger.activate(iteration)
                self.runner.journal_iteration = iteration
                self.ledger.check()
                if self.learner.cost_fingerprint != self.round_state['cost_model_fingerprint']:
                    raise ValueError('Frozen cost model changed within the experiment')
                if self.completed_iterations < iteration:
                    report = await self._collect_round(iteration)
                else:
                    report = self.round_state['training_report']
                named = self.out/f'iterations/iteration-{iteration:04d}.pt'
                if not named.exists():
                    self.learner.save(named, {**report, 'completed_iterations': iteration,
                        'cold_done': True, 'collected_cold_steps': self.ksr_trained_steps,
                        'cost_model_fingerprint': self.learner.cost_fingerprint})
                self._round_save(stage='trained')
                need_eval = (iteration % self.config['evaluation']['every_iterations'] == 0 or
                             iteration == self.config['online']['iterations'] or self.ledger.threshold_reached)
                if need_eval and self.last_evaluated != iteration:
                    if not self.config['evaluation']['enabled']:
                        raise ValueError('Round-budget experiment requires evaluation enabled')
                    self._round_save(stage='evaluating')
                    evaluation = await self._evaluate(iteration)
                    if not evaluation or not evaluation['complete']:
                        raise RecoverableRequestError('Round assessment incomplete; preserve progress and resume')
                    self._save(iteration)
                self._round_save(stage='finished', threshold_reached=self.ledger.threshold_reached)
                self.ledger.deactivate()
                append_jsonl(self.out/'training_log.jsonl', report)
                self.event('iteration_finished', **report)
                if self.ledger.threshold_reached:
                    stop = 'budget_round_completed'
                    break
        except RecoverableRequestError as exc:
            stop = 'invalid_teacher_evidence' if isinstance(exc, InvalidTeacherEvidenceError) else 'request_retries_exhausted'
            self.event('stopped', reason=stop, error=str(exc))
        except (UnknownCostError, BudgetExceeded) as exc:
            stop = 'unknown_cost' if isinstance(exc, UnknownCostError) else 'global_budget'
            self.event('stopped', reason=stop, error=str(exc))
        except BaseException as exc:
            self.event('interrupted', error=f'{type(exc).__name__}: {exc}')
            self._summary('interrupted')
            raise
        return self._summary(stop)

    def _summary(self, stop):
        report = super()._summary(stop)
        report.update(budget_mode='round_threshold', historical_cold=self.ledger.historical,
                      experiment_budget_cost=self.ledger.total_budget_cost,
                      budget_threshold=self.ledger.threshold,
                      budget_overshoot=max(0., self.ledger.total_budget_cost-self.ledger.threshold),
                      round_state=self.round_state)
        _atomic(self.out/'summary.json', report)
        return report
