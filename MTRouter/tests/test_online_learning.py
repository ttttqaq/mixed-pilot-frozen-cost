"""Learning tests use an explicit hash encoder, no downloaded model or API."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import torch

from mixed_pilot_mtrouter.online.learning import LearningSystem, serialize_history, state_key


def config(**overrides):
    value = {
        "seed": 17,
        "model_pool": ["small", "strong"],
        "encoder": {"backend": "hash", "hash_dim": 16, "device": "cpu", "persist_cache": True},
        "rollout": {"task_budget": 1.0, "step_limit": 20},
        "budget": {"cost_reference": .1, "lambda_": .05},
        "model": {"hidden_dims": [32, 32], "dropout": 0., "ensemble_size": 2,
                  "meta_embed_dim": 8, "residual_embed_dim": 4, "model_dim": 8},
        "teacher": {"hidden_dims": [24, 24], "dropout": 0., "ensemble_size": 2,
                    "min_real_samples": 4, "epochs": 40},
        "optimization": {"epochs": 40, "learning_rate": .02, "weight_decay": 0.,
                         "batch_size": 16, "anchor_weight": 0., "pairwise_weight": .2,
                         "cost_loss_weight": 1., "grad_clip": 2.},
    }
    value.update(overrides)
    return value


def state(version="policy-0"):
    return {"task_id": "train--task", "benchmark": "scienceworld", "split": "train",
            "policy_version": version, "step_idx": 2, "remaining_steps": 18,
            "remaining_budget": .8, "system": "Use one tool action.",
            "history": [{"role": "user", "content": "Find a plant."},
                        {"role": "assistant", "content": "Inspect", "tool_calls": [
                            {"name": "shell", "arguments": {"command": "look"}}]},
                        {"role": "tool", "content": "A plant is visible.", "name": "shell"}]}


class OnlineLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny test matrices should not use all cores on a large training host.
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_features_include_tool_arguments_and_budget_but_not_future_label(self):
        learner = LearningSystem(config())
        original = state()
        changed = deepcopy(original)
        changed["history"][1]["tool_calls"][0]["arguments"]["command"] = "take plant"
        self.assertIn('"command": "look"', serialize_history(original))
        self.assertNotEqual(state_key(original), state_key(changed))
        self.assertFalse(torch.equal(learner.features.encode(original), learner.features.encode(changed)))
        future = deepcopy(original)
        future.update(final_score=100, success=True, prefix=[{"future_secret": "answer"}])
        self.assertTrue(torch.equal(learner.features.encode(original), learner.features.encode(future)))
        budget = deepcopy(original); budget["remaining_budget"] = .1
        before = len(learner.features.cache)
        a, b = learner.features.encode(original), learner.features.encode(budget)
        self.assertEqual(len(learner.features.cache), before)
        self.assertTrue(torch.equal(a[:-3], b[:-3]))
        self.assertNotEqual(float(a[-3]), float(b[-3]))

    def test_hash_features_are_deterministic_across_instances(self):
        first = LearningSystem(config())
        other = LearningSystem(config(seed=901))
        self.assertTrue(torch.equal(first.features.encode(state()), other.features.encode(state())))

    def test_policy_fingerprint_tracks_router_and_epsilon_but_not_teacher_or_rng(self):
        cfg = config()
        cfg["optimization"]["epochs"] = 2
        cfg["teacher"]["epochs"] = 2
        learner = LearningSystem(cfg)
        original = learner.policy_fingerprint(.1)
        self.assertTrue(original.startswith("policy-"))
        self.assertNotEqual(original, learner.policy_fingerprint(0.))
        learner.router.train()
        learner.set_seed(909)
        self.assertEqual(original, learner.policy_fingerprint(.1))
        rows = [{"sample_id": f"anchor-{i}", "state": state(), "model": "small", "source": "real",
                 "reward": .9, "cost": .02} for i in range(8)]
        self.assertFalse(hasattr(learner, "teacher"))
        self.assertFalse(hasattr(learner, "teacher_optimizer"))
        self.assertEqual(original, learner.policy_fingerprint(.1))
        learner.train_router(rows)
        self.assertNotEqual(original, learner.policy_fingerprint(.1))

    def test_real_router_gradient_learns_outcome_and_cost_and_pairwise_gap(self):
        learner = LearningSystem(config())
        rows = [{"state": state(), "model": model, "reward": reward, "cost": cost,
                 "source": "real", "sample_id": f"{model}-{i}"}
                for i in range(12) for model, reward, cost in [("small", 0., .01), ("strong", 1., .08)]]
        before = learner.predict_actions(state())
        desired_gap = 1. - .05 * (.08 - .01) / .1
        result = learner.train_router(rows, [{"state": state(), "a": "strong", "b": "small", "delta": desired_gap}])
        after = learner.predict_actions(state())
        self.assertGreater(result["updates"], 0)
        self.assertEqual(result["pairs"], 1)
        self.assertGreater(after["strong"]["reward"] - after["small"]["reward"], .6)
        self.assertLess(abs(after["strong"]["cost"] - .08), .025)
        self.assertLess(abs(after["small"]["cost"] - .01), .025)
        self.assertGreater(after["strong"]["reward"], before["strong"]["reward"])





    def test_checkpoint_restores_prediction_optimizer_rng_and_cache(self):
        cfg = config()
        cfg["optimization"]["epochs"] = 2
        learner = LearningSystem(cfg)
        rows = [{"sample_id": f"sample-{i}", "state": state(), "model": "small", "source": "real",
                 "reward": .8, "cost": .03} for i in range(8)]
        learner.train_router(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            expected = learner.predict_actions(state())
            expected_fingerprint = learner.policy_fingerprint(.1)
            learner.save(path, {"iteration": 2, "policy_version": "policy-0"})
            saved=torch.load(path,weights_only=True)
            self.assertEqual(saved["format"],"mixed-pilot-reference-router-v2")
            self.assertNotIn("teacher",saved)
            self.assertNotIn("teacher_optimizer",saved)
            learner.train_router(rows)
            next_expected = learner.predict_actions(state())
            restored = LearningSystem.load(path, {"encoder": {"device": "cpu"}})
            self.assertEqual(restored.metadata["iteration"], 2)
            self.assertEqual(expected, restored.predict_actions(state()))
            self.assertEqual(expected_fingerprint, restored.policy_fingerprint(.1))
            self.assertEqual(len(learner.features.cache), len(restored.features.cache))
            restored.train_router(rows)
            self.assertEqual(next_expected, restored.predict_actions(state()))
            with self.assertRaisesRegex(ValueError, "model_pool"):
                LearningSystem.load(path, {"model_pool": ["strong", "small"]})
            with self.assertRaisesRegex(ValueError, "hash_dim"):
                LearningSystem.load(path, {"encoder": {"hash_dim": 24}})
            with self.assertRaisesRegex(ValueError, "cost_reference"):
                LearningSystem.load(path, {"budget": {"cost_reference": 1.}})

    def test_invalid_targets_and_unknown_actions_fail_explicitly(self):
        learner = LearningSystem(config())
        with self.assertRaisesRegex(ValueError, "reward target"):
            learner.train_router([{"state": state(), "model": "small", "reward": 2., "cost": .1}])
        with self.assertRaisesRegex(ValueError, "Unknown router action"):
            learner.train_router([{"state": state(), "model": "other", "reward": 1., "cost": .1}])


if __name__ == "__main__":
    unittest.main()
