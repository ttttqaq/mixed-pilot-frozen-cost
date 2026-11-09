"""Round boundary accounting and restart behavior using synthetic Actor replies."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch

import test_ksr_reference_router as ksr_tests
from mixed_pilot_mtrouter.online.frozen_cost import METHOD
from mixed_pilot_mtrouter.online.frozen_cost.learning import FrozenCostLearner
from mixed_pilot_mtrouter.online.frozen_cost.round_budget import RoundBudgetLedger, historical_cold_cost
from mixed_pilot_mtrouter.online.frozen_cost.round_trainer import RoundBudgetTrainer
from mixed_pilot_mtrouter.online.runtime import UnknownCostError, RecoverableRequestError


class RoundBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ksr_tests.KSRReferenceTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.historical = self.fixture.root/'historical'
        (self.historical/'calls').mkdir(parents=True)
        self.write_old('c000000000', scope='cold_start', cost=.6, reserved=0., complete=True)
        self.write_old('c000000001', scope='test', cost=10., reserved=0., complete=True)
        self.cfg = deepcopy(self.fixture.cfg)
        self.cfg.update(method=METHOD, initialization={'cold_checkpoint': 'fixture-old-cold.pt'},
                        experiment_budget={'historical_run': str(self.historical), 'threshold': 50.})
        self.cfg['cold_start']['finish_current_task'] = True
        self.cfg['budget']['total'] = 50.
        self.cfg['evaluation'].update(enabled=True, after_cold_start=False, every_iterations=3)
        self.cfg['online'].update(iterations=9)

    def write_old(self, cid, *, scope, cost, reserved, complete):
        row = {'call_id': cid, 'scope': scope, 'currency': 'CNY', 'cost': cost,
               'reserved_cost': reserved, 'cost_complete': complete, 'status': 'completed'}
        (self.historical/'calls'/(cid+'.json')).write_text(json.dumps(row))

    def learner(self):
        learner = FrozenCostLearner(self.cfg)
        learner.freeze_cost()
        learner.cold_initialization = {'trajectory_steps': 500, 'checkpoint': 'fixture-old-cold.pt'}
        return learner

    def trainer(self, cls=RoundBudgetTrainer, resume=False):
        return self.fixture.trainer(self.cfg, cls=cls, resume=resume,
                                    **({} if resume else {'learner': self.learner()}))

    def test_historical_scope_excludes_test_and_counts_failed_fee_reservations(self):
        self.write_old('c000000002', scope='cold_start', cost=None, reserved=.1, complete=False)
        old = historical_cold_cost(self.historical, 'CNY')
        self.assertAlmostEqual(old['known_cost'], .6)
        self.assertAlmostEqual(old['reserved_cost'], .1)
        self.assertEqual(old['call_count'], 2)
        self.write_old('c000000003', scope='cold_start', cost=None, reserved=0., complete=False)
        with self.assertRaises(UnknownCostError):
            historical_cold_cost(self.historical, 'CNY')

    def test_ledger_defers_threshold_only_for_admitted_phase_and_keeps_unknown_fee_guard(self):
        self.cfg['budget']['total'] = .61
        self.cfg['experiment_budget']['threshold'] = .61
        t = self.trainer()
        ledger = t.ledger
        ledger.activate(1)
        cid = ledger.begin('collection', 'fixture')
        with self.assertRaises(UnknownCostError):
            ledger.check()
        ledger.finish(cid, cost=.02, complete=True)
        self.assertTrue(ledger.threshold_reached)
        self.assertFalse(ledger.exhausted)
        ledger.check()
        ledger.deactivate()
        self.assertTrue(ledger.exhausted)
        report = ledger.report()
        self.assertAlmostEqual(report['new_known_cost'], .02)
        self.assertAlmostEqual(report['experiment_known_cost'], .62)
        self.assertAlmostEqual(report['overshoot'], .01)

    def test_crossing_threshold_finishes_first_round_and_full_assessment_then_stops(self):
        self.cfg['budget']['total'] = .600001
        self.cfg['experiment_budget']['threshold'] = .600001
        self.fixture.valid = [dict(task_id='test--fixture-'+str(i), benchmark='scienceworld', split='test') for i in range(5)]
        t = self.trainer()
        fingerprint = t.learner.cost_fingerprint
        report = asyncio.run(t.run())
        self.assertEqual(report['stop_reason'], 'budget_round_completed')
        self.assertEqual(report['completed_iterations'], 1)
        self.assertGreater(report['budget_overshoot'], 0)
        self.assertEqual(t.learner.cost_fingerprint, fingerprint)
        evaluation = [json.loads(x) for x in (t.out/'evaluation.jsonl').read_text().splitlines()]
        self.assertEqual([r['iteration'] for r in evaluation], [1])
        self.assertEqual(evaluation[0]['scored_tasks'], 5)
        self.assertTrue(evaluation[0]['complete'])
        self.assertTrue((t.out/'iterations/iteration-0001.pt').is_file())
        before_calls, before_updates = len(t.ledger.records), t.learner.router_updates
        resumed = self.trainer(resume=True)
        repeated = asyncio.run(resumed.run())
        self.assertEqual(repeated['stop_reason'], 'budget_round_completed')
        self.assertEqual(len(resumed.ledger.records), before_calls)
        self.assertEqual(resumed.learner.router_updates, before_updates)

    def test_nine_rounds_save_all_checkpoints_and_evaluate_only_three_six_nine(self):
        t = self.trainer()
        report = asyncio.run(t.run())
        self.assertEqual(report['stop_reason'], 'iterations_completed')
        self.assertEqual(report['completed_iterations'], 9)
        self.assertEqual(sorted(p.name for p in (t.out/'iterations').glob('*.pt')),
                         ['iteration-%04d.pt'%i for i in range(1, 10)])
        evaluations = [json.loads(x) for x in (t.out/'evaluation.jsonl').read_text().splitlines()]
        self.assertEqual([r['iteration'] for r in evaluations], [3, 6, 9])

    def test_resume_collecting_round_reuses_already_paid_rollouts(self):
        class Pause(RoundBudgetTrainer):
            async def _real_rows(self, result, **kwargs):
                await super()._real_rows(result, **kwargs)
                raise RecoverableRequestError('stop after persisted rollout')
        self.cfg['online']['iterations'] = 1
        first = self.trainer(cls=Pause)
        report = asyncio.run(first.run())
        self.assertEqual(report['completed_iterations'], 0)
        first_calls = set(first.ledger.records)
        first_results = list((first.out/'round-rollouts').glob('*.json'))
        self.assertTrue(first_results)
        completed = json.loads(first_results[0].read_text())['result']
        call_ids = [s['call_id'] for s in completed['steps']]
        resumed = self.trainer(resume=True)
        report = asyncio.run(resumed.run())
        self.assertEqual(report['completed_iterations'], 1)
        self.assertTrue(first_calls <= set(resumed.ledger.records))
        same_root = [json.loads(p.read_text()) for p in (resumed.out/'round-rollouts').glob('*.json')]
        self.assertEqual(sum(r['result']['trajectory_id']==completed['trajectory_id'] for r in same_root), 1)
        self.assertTrue(all(cid in resumed.ledger.records for cid in call_ids))

    def test_resume_evaluation_keeps_named_weight_and_does_not_retrain_round(self):
        class Pause(RoundBudgetTrainer):
            async def _evaluate(self, iteration):
                # Persist a complete assessment, then simulate failure before the
                # trainer commits last_evaluated. Existing progress prevents fees.
                await super()._evaluate(iteration)
                raise RecoverableRequestError('interrupted before evaluation commit')
        self.cfg['online']['iterations'] = 1
        first = self.trainer(cls=Pause)
        report = asyncio.run(first.run())
        self.assertEqual(report['completed_iterations'], 1)
        self.assertTrue((first.out/'iterations/iteration-0001.pt').is_file())
        calls, updates = len(first.ledger.records), first.learner.router_updates
        resumed = self.trainer(resume=True)
        report = asyncio.run(resumed.run())
        self.assertEqual(report['stop_reason'], 'iterations_completed')
        self.assertEqual(len(resumed.ledger.records), calls)
        self.assertEqual(resumed.learner.router_updates, updates)

    def test_hard_kill_journal_recovers_raw_transition_and_fails_closed_without_evidence(self):
        t = self.trainer()
        t.ledger.activate(1)
        t.runner.journal_iteration = 1
        version = t._policy_id()
        policy = t._policy(version)
        task = self.fixture.train[0]
        args = dict(forced_first=self.cfg['model_pool'][0], horizon=1,
                    scope='branch', policy_version=version)
        first = asyncio.run(t.runner.rollout(task, policy, **args))
        count = len(t.ledger.records)
        path = next((t.out/'round-rollouts').glob('*.json'))
        record = json.loads(path.read_text())
        record.update(complete=False, result=None)
        record.pop('rng_after', None)
        path.write_text(json.dumps(record))
        recovered = asyncio.run(t.runner.rollout(task, policy, **args))
        self.assertEqual(first['trajectory_id'], recovered['trajectory_id'])
        self.assertEqual(len(t.ledger.records), count)
        raw_path = t.out/'trajectories'/(first['trajectory_id']+'.json')
        raw = json.loads(raw_path.read_text())
        raw['all_steps'][-1]['transition'] = None
        raw_path.write_text(json.dumps(raw))
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(RecoverableRequestError, 'no persisted transition'):
            asyncio.run(t.runner.rollout(task, policy, **args))
        self.assertEqual(len(t.ledger.records), count)


if __name__ == '__main__':
    unittest.main()
