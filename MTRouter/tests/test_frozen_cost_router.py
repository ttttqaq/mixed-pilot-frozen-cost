"""Behavioral coverage for task-complete cold start, fixed C and online Y-only fit."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch

import test_ksr_reference_router as ksr_tests
from mixed_pilot_mtrouter.online.frozen_cost import METHOD
from mixed_pilot_mtrouter.online.frozen_cost.config import validate_config
from mixed_pilot_mtrouter.online.frozen_cost.initialization import load_cold_checkpoint
from mixed_pilot_mtrouter.online.frozen_cost.learning import FrozenCostLearner, tensor_fingerprint
from mixed_pilot_mtrouter.online.frozen_cost.teacher import parse_reward, SYSTEM
from mixed_pilot_mtrouter.online.frozen_cost.trainer import FrozenCostTrainer
from mixed_pilot_mtrouter.online.ksr.learning import SeparateLearner
from mixed_pilot_mtrouter.online.runtime import RecoverableRequestError
from mixed_pilot_mtrouter.online.smoke import SmokeProvider, SmokeEpisode


class FrozenCostTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ksr_tests.KSRReferenceTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.cfg = deepcopy(self.fixture.cfg)
        self.cfg['method'] = METHOD
        self.cfg['cold_start']['finish_current_task'] = True
        self.cfg['initialization'] = {'cold_checkpoint': None}

    def trainer(self, cfg=None, cls=FrozenCostTrainer, **kwargs):
        return self.fixture.trainer(cfg or self.cfg, cls=cls, **kwargs)

    def state(self, suffix=''):
        return {'history': [{'role': 'user', 'content': 'fixture'+suffix}], 'remaining_budget': .8,
                'remaining_steps': 40, 'step_idx': 10, 'split': 'train', 'task_id': 'train--fixture'}

    def rows(self):
        return [dict(state=self.state(str(i)), model=m, reward=float(j), cost=.02+j*.1,
                     source='real', weight=1.)
                for i in range(5) for j, m in enumerate(self.cfg['model_pool'])]

    def test_joint_cold_fit_is_identical_to_original_learner(self):
        baseline = SeparateLearner(self.cfg)
        baseline.train_router(self.rows())
        current = FrozenCostLearner(self.cfg)
        current.train_router(self.rows())
        self.assertIsNone(current.frozen_cost_router)
        for k, v in baseline.router.state_dict().items():
            self.assertTrue(torch.equal(v, current.router.state_dict()[k]), k)

    def test_Y_updates_shared_layers_while_C_values_and_parameters_stay_identical(self):
        learner = FrozenCostLearner(self.cfg)
        learner.train_router(self.rows())
        learner.freeze_cost()
        before = {str(i): learner.predict_actions(self.state(str(i))) for i in range(4)}
        live = deepcopy(learner.router.state_dict())
        frozen = deepcopy(learner.frozen_cost_router.state_dict())
        rows = self.rows()
        for r in rows:
            r.pop('cost')
        for _ in range(6):
            report = learner.train_router(rows)
        self.assertEqual(report['cost_loss'], 0.)
        self.assertTrue(any(not torch.equal(v, learner.router.state_dict()[k])
                            for k, v in live.items() if '.backbone.' in k or '.models.' in k))
        self.assertTrue(any(not torch.equal(v, learner.router.state_dict()[k])
                            for k, v in live.items() if '.reward.' in k))
        self.assertTrue(all(torch.equal(v, learner.frozen_cost_router.state_dict()[k]) for k, v in frozen.items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in learner.frozen_cost_router.parameters()))
        for i in range(4):
            after = learner.predict_actions(self.state(str(i)))
            for m in learner.model_names:
                self.assertEqual(before[str(i)][m]['cost'], after[m]['cost'])
                self.assertEqual(before[str(i)][m]['cost_std'], after[m]['cost_std'])
        self.assertEqual(learner.cost_fingerprint, tensor_fingerprint(frozen))

    def test_online_gradient_ignores_C_fields_and_beta(self):
        a = FrozenCostLearner(self.cfg)
        a.freeze_cost()
        other = deepcopy(self.cfg)
        other['decision']['beta'] = 20.
        other['budget']['lambda_'] = 2.
        b = FrozenCostLearner(other)
        b.freeze_cost()
        rows = self.rows()
        poisoned = deepcopy(rows)
        for r in poisoned:
            r['cost'] = float('nan')
        torch.manual_seed(99)
        a.train_router(rows)
        torch.manual_seed(99)
        b.train_router(poisoned)
        self.assertTrue(all(torch.equal(v, b.router.state_dict()[k]) for k, v in a.router.state_dict().items()))

    def test_checkpoint_retains_original_C_after_online_update_and_reload(self):
        learner = FrozenCostLearner(self.cfg)
        learner.freeze_cost()
        expected = learner.predict_actions(self.state())
        learner.train_router(self.rows())
        path = self.fixture.root/'learner.pt'
        learner.save(path, {'cold_done': True})
        restored = FrozenCostLearner.load(path)
        from mixed_pilot_mtrouter.online.learning import LearningSystem
        with self.assertRaisesRegex(ValueError, 'FrozenCostLearner'):
            LearningSystem.load(path)
        self.assertEqual(learner.cost_fingerprint, restored.cost_fingerprint)
        for m in learner.model_names:
            self.assertEqual(expected[m]['cost'], restored.predict_actions(self.state())[m]['cost'])
        version = restored.policy_fingerprint()
        restored.train_router(self.rows())
        self.assertNotEqual(version, restored.policy_fingerprint())
        self.assertEqual(learner.cost_fingerprint, restored.cost_fingerprint)
        payload = torch.load(path, weights_only=True)
        payload['frozen_cost_router'] = None
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, 'missing'):
            FrozenCostLearner.load(path)

    def test_step_threshold_finishes_both_current_task_roots_without_starting_next(self):
        self.cfg['cold_start'].update(trajectory_steps=5, checkpoint_steps=5)
        t = self.trainer()
        asyncio.run(t._cold_start())
        data = json.loads((t.out/'cold_start/cache.json').read_text())
        self.assertEqual(len(data['calls']), 8)
        self.assertEqual(data['cursor'], 1)
        self.assertEqual(data['queue'], [])
        self.assertTrue(data['complete'])
        self.assertTrue(all(n.get('labels_valid') for n in data['nodes'].values()))
        middle = FrozenCostLearner.load(t.out/'cold_start/checkpoints/step_000005.pt')
        final = FrozenCostLearner.load(t.out/'cold_start/checkpoints/step_000008.pt')
        self.assertIsNone(middle.frozen_cost_router)
        self.assertIsNotNone(final.frozen_cost_router)

    def test_real_500_threshold_finishes_task_at_504_and_retains_all_terminal_labels(self):
        self.cfg['cold_start'].update(trajectory_steps=500, checkpoint_steps=100)
        self.fixture.train = [dict(task_id='train--fixture-'+str(i), benchmark='scienceworld', split='train') for i in range(70)]
        t = self.trainer()
        asyncio.run(t._cold_start())
        data = json.loads((t.out/'cold_start/cache.json').read_text())
        self.assertEqual(len(data['calls']), 504)
        self.assertEqual(data['cursor'], 63)
        rows = [json.loads(x) for x in (t.out/'cold_start/samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 504)
        self.assertTrue(all(r['labels_valid'] for r in rows))
        self.assertEqual(sorted(p.name for p in (t.out/'cold_start/checkpoints').glob('*.pt')),
                         ['step_%06d.pt'%n for n in (100, 200, 300, 400, 500, 504)])

    def test_threshold_finishes_recursive_branches_in_current_task(self):
        class BranchEpisode(SmokeEpisode):
            async def advance(self, response):
                tr = await super().advance(response)
                self.score = self.progress*25
                if response.text == 'SMOKE_WEAK_OBSERVE':
                    tr.record['observation'] = 'Nothing happens'
                    tr.messages[0].content = 'Nothing happens'
                return tr
        self.cfg['cold_start'].update(trajectory_steps=3, checkpoint_steps=3)
        self.fixture.rules['KSR_weak2strong'] = {'signals': ['consecutive_errors'],
            'min_branch_gap': 3, 'error_window': 2, 'error_markers': ['Nothing happens']}
        (self.fixture.root/'ksr.json').write_text(json.dumps(self.fixture.rules))
        t = self.trainer()
        t.runner.episode_factory = BranchEpisode
        asyncio.run(t._cold_start())
        data = json.loads((t.out/'cold_start/cache.json').read_text())
        self.assertGreater(len(data['calls']), 3)
        self.assertEqual(data['cursor'], 1)
        self.assertEqual(data['queue'], [])
        self.assertTrue(any(n['children'] for n in data['nodes'].values()))
        rows = [json.loads(x) for x in (t.out/'cold_start/samples.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['labels_valid'] for r in rows))

    def test_cache_reuse_keeps_task_completion_semantics_and_rejects_changed_threshold(self):
        self.cfg['cold_start'].update(trajectory_steps=5, checkpoint_steps=5)
        first = self.trainer()
        asyncio.run(first._cold_start())
        reused = deepcopy(self.cfg)
        reused['output_dir'] = str(self.fixture.root/'reuse')
        reused['cold_start']['reuse_from'] = str(first.out)
        second = self.trainer(reused)
        asyncio.run(second._cold_start())
        self.assertEqual(len(second.ledger.records), 0)
        self.assertEqual(second.ksr_trained_steps, 8)
        self.assertIsNotNone(second.learner.frozen_cost_router)
        changed = deepcopy(reused)
        changed['output_dir'] = str(self.fixture.root/'changed')
        changed['cold_start']['trajectory_steps'] = 6
        third = self.trainer(changed)
        with self.assertRaisesRegex(ValueError, 'semantics differ'):
            asyncio.run(third._cold_start())

    def test_resume_after_threshold_finishes_queue_and_preserves_unique_calls(self):
        class Pause(FrozenCostTrainer):
            async def cold_checkpoint(self, collector, count):
                await super().cold_checkpoint(collector, count)
                if count == 5:
                    raise RecoverableRequestError('fixture interruption')
        self.cfg['cold_start'].update(trajectory_steps=5, checkpoint_steps=5)
        first = self.trainer(cls=Pause)
        first._save(0)
        with self.assertRaises(RecoverableRequestError):
            asyncio.run(first._cold_start())
        self.assertIsNone(first.learner.frozen_cost_router)
        resumed = self.trainer(resume=True)
        asyncio.run(resumed._cold_start())
        self.assertEqual(len(resumed.ledger.records), 8)
        self.assertIsNotNone(resumed.learner.frozen_cost_router)
        self.assertEqual(len(json.loads((resumed.out/'cold_start/cache.json').read_text())['calls']), 8)

    def test_Y_only_teacher_and_five_rounds_produce_no_online_C_targets(self):
        self.cfg['online']['iterations'] = 5
        t = self.trainer()
        result = asyncio.run(t.run())
        self.assertEqual(result['completed_iterations'], 5)
        self.assertTrue(result['cost_frozen'])
        reports = [json.loads(x) for x in (t.out/'training_log.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['router']['cost_loss'] == 0. for r in reports))
        self.assertEqual(len({r['router']['cost_model_fingerprint'] for r in reports}), 1)
        estimates = [json.loads(x)['estimate'] for x in (t.out/'teacher_predictions.jsonl').read_text().splitlines()]
        self.assertTrue(estimates)
        self.assertTrue(all('reward' in e and 'cost' not in e for e in estimates))
        self.assertTrue(all('cost' not in r for r in t.router_rows.values() if str(r['policy_version']).startswith('policy-')))
        for pair in t.pairs.values():
            self.assertTrue(all('cost' not in target for target in pair['targets'].values()))

    def test_teacher_schema_requires_only_valid_Y(self):
        self.assertNotIn('remaining_agent_cost', SYSTEM)
        estimate = parse_reward('{"expected_reward":0.7,"assessment_status":"ok"}')
        self.assertEqual(estimate['reward'], .7)
        self.assertNotIn('cost', estimate)
        for y in ['NaN', 'true', '1.1', '-0.1']:
            with self.assertRaises(ValueError):
                parse_reward('{"expected_reward":'+y+',"assessment_status":"ok"}')

    def test_unavailable_teacher_still_falls_back_to_real_Y(self):
        class FailingTeacher(SmokeProvider):
            async def complete(self, *args, **kwargs):
                if 'training-only assessor' in str(args[2]):
                    from mixed_pilot_harness.types import ProviderError
                    raise ProviderError('Responses stream reported an error')
                return await super().complete(*args, **kwargs)
        self.cfg['resilience'].update(enabled=True, retry_delay=0., qwen_min_interval=0.)
        t = self.trainer()
        t.runner.provider_factory = lambda p, **kw: FailingTeacher(p, strong_model=self.cfg['strong_model'])
        report = asyncio.run(t.run())
        self.assertEqual(report['stop_reason'], 'iterations_completed')
        self.assertTrue(all(r['source'] == 'real' for r in t.router_rows.values()))
        self.assertTrue(t.ledger.spending_safe)

    def test_import_old_cold_checkpoint_preserves_weights_but_starts_new_ledger(self):
        self.fixture.manifest['runtime_artifact_hashes'] = dict.fromkeys(('harness_config', 'pricing', 'model_metadata'), 'fixture')
        old = self.fixture.trainer()
        asyncio.run(old._cold_start())
        self.cfg['output_dir'] = str(self.fixture.root/'new-run')
        path = old.out/'cold_start/checkpoints/step_000020.pt'
        self.cfg['initialization']['cold_checkpoint'] = str(path)
        learner = load_cold_checkpoint(self.cfg, self.fixture.manifest)
        saved = SeparateLearner.load(path)
        self.assertTrue(all(torch.equal(v, learner.router.state_dict()[k]) for k, v in saved.router.state_dict().items()))
        t = self.trainer(learner=learner)
        asyncio.run(t._cold_start())
        self.assertEqual(len(t.ledger.records), 0)
        resumed = self.trainer(resume=True)
        self.assertEqual(resumed.learner.cold_initialization['checkpoint'], str(path))
        self.assertEqual(t.learner.cost_fingerprint, resumed.learner.cost_fingerprint)
        bad = deepcopy(self.fixture.manifest)
        bad['split_fingerprint'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            load_cold_checkpoint(self.cfg, bad)

    def test_configuration_enforces_cold_freeze_and_nonnegative_beta(self):
        # Fixture-only model_metadata=None is not a production config.
        from mixed_pilot_mtrouter.online.frozen_cost.config import DEFAULTS
        cfg = deepcopy(DEFAULTS)
        validate_config(cfg)
        cfg['decision']['beta'] = -1.
        with self.assertRaises(ValueError):
            validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
