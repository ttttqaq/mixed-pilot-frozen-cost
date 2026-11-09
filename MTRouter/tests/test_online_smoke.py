"""Synthetic smoke fixtures never call a network or native benchmark runtime."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.types import Message
from mixed_pilot_mtrouter.online.runtime import Ledger, Runner
from mixed_pilot_mtrouter.online.smoke import SmokeEpisode, smoke_config, smoke_factories, SMOKE_MARKER


class SmokeTests(unittest.TestCase):
    def config(self):
        return {"model_pool": ["weak-profile", "strong-profile"], "strong_model": "strong-profile",
                "encoder": {"backend": "hf", "device": "cuda:7"},
                "model": {"hidden_dims": [256, 256], "ensemble_size": 3, "dropout": .1},
                "teacher": {"ensemble_size": 3, "min_real_samples": 16,
                            "llm": {"enabled": True, "profile": "strong-profile", "max_calls_per_iteration": 16}},
                "rollout": {"task_budget": 1.0, "step_limit": 50, "replay_attempts": 2, "reward": "success"},
                "optimization": {"epochs": 10, "batch_size": 64},
                "online": {"iterations": 5, "tasks_per_iteration": 4, "states_per_iteration": 4},
                "cold_start": {"tasks": 4, "queries": 4, "anchor_states_per_trace": 4},
                "short_rollout": {"horizon": 3, "max_horizon": 9, "adaptive": True},
                "evaluation": {"every_iterations": 2, "max_tasks": None},
                "budget": {"total": 50, "lambda_": .05},
                "dataset": {"root": "untouched", "train_split": "train", "validation_split": "valid"}}

    def test_smoke_config_is_copy_and_preserves_splits_model_pool_and_flags(self):
        original = self.config()
        before = deepcopy(original)
        adapted = smoke_config(original)
        self.assertEqual(original, before)
        self.assertEqual(adapted["encoder"]["backend"], "hash")
        self.assertEqual(adapted["encoder"]["device"], "cpu")
        self.assertEqual(adapted["model"]["hidden_dims"], [16, 16])
        self.assertEqual(adapted["teacher"]["llm"]["max_calls_per_iteration"], 20)
        self.assertEqual(adapted["dataset"], original["dataset"])
        self.assertEqual(adapted["model_pool"], original["model_pool"])
        self.assertEqual(adapted["strong_model"], original["strong_model"])
        self.assertTrue(adapted["teacher"]["llm"]["enabled"])
        self.assertEqual(adapted["budget"]["lambda_"], .05)

    def test_fixture_providers_use_arbitrary_configured_identities(self):
        config = smoke_config(self.config())
        _, factory = smoke_factories(config)
        strong = factory(ProviderProfile(kind="scripted", model="strong-api", name="strong-profile"))
        weak = factory(ProviderProfile(kind="scripted", model="weak-api", name="weak-profile"))
        good = asyncio.run(strong.complete([], [], "agent"))
        bad = asyncio.run(weak.complete([], [], "agent"))
        self.assertEqual(good.text, "SMOKE_STRONG_ADVANCE")
        self.assertEqual(bad.text, "SMOKE_WEAK_OBSERVE")
        self.assertEqual(good.model, "strong-api")
        self.assertGreater(good.usage.input_tokens, 0)

    def test_assessor_json_is_valid_and_finite(self):
        _, factory = smoke_factories(self.config())
        provider = factory(ProviderProfile(kind="scripted", model="x", name="strong-profile"))
        prompt = {"history": "SYNTHETIC progress=1.0; step=1", "remaining_steps": 10,
                  "remaining_budget": .1}
        response = asyncio.run(provider.complete([Message("user", json.dumps(prompt))], [],
                                                "You are a training-only assessor"))
        value = json.loads(response.text)
        self.assertEqual(value["expected_reward"], .9)
        self.assertGreaterEqual(value["remaining_agent_cost"], 0)
        self.assertLessEqual(value["remaining_agent_cost"], .1)
        self.assertEqual(value["assessment_status"], "ok")
        self.assertIn("Synthetic", value["reason"])

    def test_real_ledger_and_counterfactual_replay_with_synthetic_environment(self):
        config = smoke_config(self.config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = HarnessConfig(root / "harness.json", "weak-profile", {
                name: ProviderProfile(kind="scripted", model=name, name=name) for name in config["model_pool"]})
            pricing = {"currency": "CNY", "profiles": {name: {"input_per_million": 1,
                       "output_per_million": 1} for name in config["model_pool"]}}
            ledger = Ledger(root, 30, "CNY")
            episode_factory, provider_factory = smoke_factories(config)
            runner = Runner(config, harness, pricing, ledger, episode_factory=episode_factory,
                            provider_factory=provider_factory)
            task = {"task_id": "train--fixture", "benchmark": "alfworld", "split": "train"}
            strong = asyncio.run(runner.rollout(task, lambda s: "strong-profile", trajectory_id="strong"))
            weak = asyncio.run(runner.rollout(task, lambda s: "strong-profile", forced_first="weak-profile",
                                            trajectory_id="weak"))
            self.assertEqual(strong["reward"], 1)
            self.assertEqual(weak["reward"], 0)
            self.assertTrue(strong["evaluation"]["synthetic"])
            self.assertEqual(strong["evaluation"]["method"], SMOKE_MARKER)
            self.assertEqual(strong["steps"][0]["before_state"]["snapshot_id"], weak["steps"][0]["before_state"]["snapshot_id"])
            self.assertEqual(len(ledger.records), 8)
            self.assertAlmostEqual(ledger.cost, .002)
            short = asyncio.run(runner.rollout(task, lambda s: "strong-profile", prefix=strong["steps"][:1],
                                             forced_first="weak-profile", horizon=1, trajectory_id="short"))
            self.assertEqual(short["status"], "truncated")
            self.assertIsNone(short["reward"])
            self.assertEqual(len(ledger.records), 9)


if __name__ == "__main__":
    unittest.main()
