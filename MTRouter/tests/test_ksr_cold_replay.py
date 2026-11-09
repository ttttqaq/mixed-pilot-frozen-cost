"""Historical replay keeps time-local labels, training RNG and paid progress."""
import asyncio
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_ksr_reference_router as ksr_tests
from mixed_pilot_mtrouter.online.frozen_cost.replay import ColdReplay, evaluation_spending
from mixed_pilot_mtrouter.online.ksr.cold import labels
from mixed_pilot_mtrouter.online.ksr.trainer import KSRTrainer
from mixed_pilot_mtrouter.online.smoke import smoke_factories


class ColdReplayTests(unittest.TestCase):
    def setUp(self):
        self.f = ksr_tests.KSRReferenceTests(methodName='runTest')
        self.f.setUp()
        self.addCleanup(self.f.tearDown)
        self.f.manifest['runtime_artifact_hashes'] = dict.fromkeys(('harness_config', 'pricing', 'model_metadata'), 'fixture')
        source = self.f.trainer()
        asyncio.run(source._cold_start())
        self.source = source.out
        self.cfg = deepcopy(self.f.cfg)
        self.cfg['teacher']['enabled'] = False
        self.cfg['evaluation']['enabled'] = True
        self.cfg['budget']['total'] = 50.

    def replay(self, name='replay', *, cfg=None, eval_steps=(10, 20), resume=False, cls=ColdReplay):
        cfg = cfg or self.cfg
        episode, provider = smoke_factories(cfg)
        return cls(cfg, self.f.train, self.f.valid, self.f.manifest, self.f.harness, self.f.pricing,
            self.source, self.f.root/name, eval_steps=eval_steps, resume=resume, simulation=True,
            episode_factory=episode, provider_factory=provider)

    def test_uses_each_historical_snapshot_and_never_final_future_labels(self):
        replay = self.replay()
        result = asyncio.run(replay.run())
        self.assertEqual(result['stop_reason'], 'cold_replay_completed')
        self.assertEqual(result['evaluated_steps'], [10, 20])
        self.assertEqual(result['online_iterations'], 0)
        self.assertEqual(result['new_cold_actor_calls'], 0)
        logs = [json.loads(x) for x in (replay.out/'training_log.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in logs], [5, 10, 15, 20])
        for row in logs:
            data = json.loads((self.source/'cold_start/checkpoints'/('step_%06d.json'%row['step'])).read_text())
            expected = sum(r['labels_valid'] for r in labels(data)[0])
            self.assertEqual(row['samples'], expected)
        early = json.loads((self.source/'cold_start/checkpoints/step_000010.json').read_text())
        final = json.loads((self.source/'cold_start/checkpoints/step_000020.json').read_text())
        final_prefix = [r for r in labels(final)[0] if r['call_id'] in early['calls'] and r['labels_valid']]
        self.assertLess(logs[1]['samples'], len(final_prefix))
        for step in [10, 20]:
            scope = json.loads((replay.out/(str(step)+'step')/'ledger.json').read_text())['by_scope']
            self.assertEqual(set(scope), {'test'})
            self.assertTrue((replay.out/(str(step)+'step')/'cold.pt').is_file())

    def test_intermediate_evaluation_preserves_training_rng_and_final_weights(self):
        baseline = self.replay('baseline', eval_steps=())
        asyncio.run(baseline.run())
        expected = deepcopy(baseline.learner.router.state_dict())
        class NoisyEvaluation(KSRTrainer):
            async def _evaluate(self, version):
                random.random(); np.random.rand(); torch.rand(100); self.rng.random()
                return await super()._evaluate(version)
        with patch('mixed_pilot_mtrouter.online.frozen_cost.replay.KSRTrainer', NoisyEvaluation):
            tested = self.replay('with-evaluation')
            asyncio.run(tested.run())
        self.assertTrue(all(torch.equal(value, tested.learner.router.state_dict()[key])
                            for key, value in expected.items()))

    def test_completed_resume_does_not_retrain_or_repeat_evaluation_calls(self):
        first = self.replay()
        result = asyncio.run(first.run())
        before = {str(step): evaluation_spending(first.out/(str(step)+'step')) for step in [10, 20]}
        updates = first.learner.router_updates
        second = self.replay(resume=True)
        again = asyncio.run(second.run())
        self.assertEqual(result['budget_cost'], again['budget_cost'])
        self.assertEqual(updates, second.learner.router_updates)
        self.assertEqual(before, {str(step): evaluation_spending(second.out/(str(step)+'step')) for step in [10, 20]})

    def test_resume_after_intermediate_evaluation_preserves_final_fit(self):
        baseline = self.replay('baseline')
        asyncio.run(baseline.run())
        class Pause(ColdReplay):
            async def _evaluate_and_commit(self, step):
                value = await super()._evaluate_and_commit(step)
                if step == 10:
                    raise RuntimeError('pause after paid stage')
                return value
        paused = self.replay('paused', cls=Pause)
        with self.assertRaisesRegex(RuntimeError, 'pause'):
            asyncio.run(paused.run())
        before = evaluation_spending(paused.out/'10step')
        resumed = self.replay('paused', resume=True)
        report = asyncio.run(resumed.run())
        self.assertEqual(report['evaluated_steps'], [10, 20])
        self.assertEqual(before, evaluation_spending(paused.out/'10step'))
        self.assertTrue(all(torch.equal(v, resumed.learner.router.state_dict()[k])
                            for k, v in baseline.learner.router.state_dict().items()))

    def test_changed_source_snapshot_blocks_resume_before_any_new_calls(self):
        first = self.replay()
        asyncio.run(first.run())
        path = self.source/'cold_start/checkpoints/step_000010.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'source'):
            self.replay(resume=True)

    def test_two_data_sources_use_identical_initialization_for_same_seed(self):
        first = self.replay('one')
        expected = deepcopy(first.learner.router.state_dict())
        source_cfg = json.loads((self.source/'config.resolved.json').read_text())
        source_cfg['seed'] = 43
        (self.source/'config.resolved.json').write_text(json.dumps(source_cfg))
        second = self.replay('two')
        self.assertEqual(second.source['source_seed'], 43)
        self.assertEqual(second.source['replay_seed'], 42)
        self.assertTrue(all(torch.equal(v, second.learner.router.state_dict()[k])
                            for k, v in expected.items()))


if __name__ == '__main__':
    unittest.main()
