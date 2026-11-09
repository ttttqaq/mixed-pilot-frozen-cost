"""No API credentials, network, GPU, or JVM required for these checks."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mixed_pilot_harness.collection.cli import main
from mixed_pilot_harness.collection.collector import CollectionSettings, collect, step_samples
from mixed_pilot_harness.collection.policy import SamplingPolicy
from mixed_pilot_harness.collection.pricing import estimate_cost, load_prices
from mixed_pilot_harness.collection.tasks import load_tasks, select_tasks, validate_tasks
from mixed_pilot_harness.collection.upstream.scienceworld import ScienceWorldEnvironment
from mixed_pilot_harness.collection.upstream.scienceworld_tasks import enumerate_scienceworld_tasks
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.types import ProviderError, ProviderResponse, Usage


TASK = {"task_id": "boil-var1", "benchmark": "scienceworld", "task_name": "boil",
        "variation_idx": 1, "simplification_str": "", "split": "train"}
PRICING = {"currency": "CNY", "as_of": "test-fixture", "profiles": {
    "a": {"input_per_million": 1, "output_per_million": 2},
    "b": {"input_per_million": 10, "output_per_million": 20},
}}


class FakeEnv:
    instances = []

    def __init__(self, **kwargs):
        self.closed = False
        self.actions = []
        self.instances.append(self)

    def load_task(self, name, variation):
        return {"observation": "kitchen", "task_description": "Boil water", "valid_actions": []}

    def step(self, action):
        self.actions.append(action)
        done = len(self.actions) == 2
        return {"observation": "water", "score": 100 if done else 50, "done": done}

    def close(self):
        self.closed = True


class FakeProvider:
    def __init__(self, texts, model="a"):
        self.texts, self.model = iter(texts), model
        self.histories = []

    async def complete(self, messages, tools, system):
        self.histories.append([m.content for m in messages])
        item = next(self.texts)
        if isinstance(item, Exception):
            raise item
        return ProviderResponse(text=item, model=self.model,
                                usage=Usage(input_tokens=100, output_tokens=10))


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = HarnessConfig(self.root / "config.json", "a", {
            "a": ProviderProfile("scripted", "model-a"),
            "b": ProviderProfile("scripted", "model-b"),
        })
        FakeEnv.instances = []

    async def run_collection(self, provider, settings=None, factory=None):
        with redirect_stdout(io.StringIO()):
            return await collect([dict(TASK)], self.config,
                                 settings or CollectionSettings("baseline", ["a"], "a"),
                                 PRICING, self.root / "out", env_factory=FakeEnv,
                                 provider_factory=factory or (lambda *a, **kw: provider))

    async def test_trace_export_cost_to_go_state_and_resume(self):
        p = FakeProvider(["```text\nlook around\n```", "```text\nactivate stove\n```"])
        summary = await self.run_collection(p)
        self.assertEqual(summary["success_rate"], 1)
        self.assertAlmostEqual(summary["known_cost"], 0.00024)
        samples = [json.loads(x) for x in Path(summary["steps_file"]).read_text().splitlines()]
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0]["remaining_cost"], .00024)
        self.assertAlmostEqual(samples[1]["remaining_cost"], .00012)
        self.assertEqual(samples[1]["state"]["previous_actions"], ["a"])
        self.assertEqual(len(samples[1]["state"]["history"]), 3)
        self.assertEqual(samples[0]["label_kind"], "single_rollout")
        self.assertTrue(FakeEnv.instances[0].closed)
        self.assertEqual(samples[0]["terminal_score"], 100)
        self.assertTrue(samples[0]["terminal_success"])
        trace = json.loads(next((self.root / "out").glob("*.traj.json")).read_text())
        self.assertEqual(trace["info"]["final_score"], 100)
        self.assertNotIn("environment_score", json.dumps(trace))
        self.assertNotIn("environment_done", json.dumps(trace))
        def fail(*a, **kw):
            raise AssertionError("resume must not create a provider")
        resumed = await self.run_collection(None, factory=fail)
        self.assertEqual(resumed["skipped_existing"], 1)
        self.assertAlmostEqual(resumed["known_cost"], .00024)

    async def test_format_error_is_billed_and_recoverable(self):
        p = FakeProvider(["invalid", "```text\nlook around\n```", "```text\nactivate stove\n```"])
        summary = await self.run_collection(p)
        trace = json.loads(next((self.root / "out").glob("*.traj.json")).read_text())
        self.assertEqual(trace["info"]["format_errors"], 1)
        self.assertEqual(len(list(step_samples(trace))), 3)
        self.assertAlmostEqual(summary["known_cost"], .00036)
        self.assertIn("EXACTLY ONE", p.histories[1][-1])

    async def test_empty_response_limit_preserves_usage_and_counts_as_task_failure(self):
        p = FakeProvider(["", " \n", "", "must not be requested"])
        summary = await self.run_collection(p)
        trace = json.loads(next((self.root / "out").glob("*.traj.json")).read_text())
        self.assertEqual(trace["info"]["exit_status"], "EmptyResponseLimit")
        self.assertTrue(trace["info"]["cost_complete"])
        self.assertTrue(trace["info"]["labels_valid"])
        self.assertFalse(trace["info"]["success"])
        self.assertEqual(trace["info"]["final_score"], 0)
        self.assertEqual(len(p.histories), 3)
        self.assertEqual(len(trace["steps"]), 3)
        self.assertEqual(trace["steps"][0]["response"], "")
        self.assertEqual(trace["steps"][0]["usage"]["output_tokens"], 10)
        self.assertIn("stop_reason", trace["steps"][0])
        self.assertAlmostEqual(summary["known_cost"], .00036)
        self.assertEqual(summary["valid_traces"], 1)
        self.assertEqual(summary["success_rate"], 0)
        self.assertEqual(FakeEnv.instances[0].actions, [])
        self.assertIn("EXACTLY ONE", p.histories[1][-1])

    async def test_empty_failure_continues_next_task_and_is_reused_on_resume(self):
        tasks = [dict(TASK), {**TASK, "task_id": "second-task"}]
        provider = FakeProvider(["", "", "", "```text\nlook around\n```",
                                 "```text\nactivate stove\n```"])
        settings = CollectionSettings("baseline", ["a"], "a")
        with redirect_stdout(io.StringIO()):
            summary = await collect(tasks, self.config, settings, PRICING, self.root / "out",
                                    env_factory=FakeEnv, provider_factory=lambda *a, **kw: provider)
            resumed = await collect(tasks, self.config, settings, PRICING, self.root / "out",
                                    env_factory=FakeEnv, provider_factory=lambda *a, **kw: provider,
                                    retry_errors=True)
        self.assertEqual(summary["valid_traces"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["known_cost"], .00060)
        self.assertEqual(resumed["skipped_existing"], 2)
        self.assertEqual(resumed["archived_attempts"], 0)
        self.assertEqual(len(provider.histories), 5)

    async def test_nonempty_response_resets_empty_streak(self):
        p = FakeProvider(["", "", "```text\nlook around\n```",
                          "", "", "```text\nactivate stove\n```"])
        summary = await self.run_collection(p)
        trace = json.loads(next((self.root / "out").glob("*.traj.json")).read_text())
        self.assertEqual(trace["info"]["exit_status"], "EnvironmentDone")
        self.assertEqual(trace["info"]["format_errors"], 4)
        self.assertEqual(len(trace["steps"]), 6)
        self.assertEqual(summary["success_rate"], 1)
        self.assertAlmostEqual(summary["known_cost"], .00072)

    async def test_increasing_runs_keeps_completed_repeats(self):
        await self.run_collection(FakeProvider(["```text\ntask completed\n```"]))
        p = FakeProvider(["```text\ntask completed\n```"])
        result = await self.run_collection(p, CollectionSettings("baseline", ["a"], "a", runs=2))
        self.assertEqual(result["skipped_existing"], 1)
        self.assertEqual(result["valid_traces"], 2)
        self.assertEqual(len(p.histories), 1)

    async def test_roulette_selects_from_exact_pool_and_records_both(self):
        class LongEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                return {"observation": "still working", "score": 0, "done": False}
        def factory(profile, **kw):
            return FakeProvider(["```text\nlook around\n```"] * 20, model=profile.model)
        with redirect_stdout(io.StringIO()):
            result = await collect([dict(TASK)], self.config,
                CollectionSettings("roulette", ["a", "b"], max_steps=20), PRICING,
                self.root / "out", env_factory=LongEnv, provider_factory=factory)
        rows = [json.loads(x) for x in Path(result["steps_file"]).read_text().splitlines()]
        self.assertEqual({r["action"] for r in rows}, {"a", "b"})
        self.assertEqual({r["propensity"] for r in rows}, {.5})
        self.assertAlmostEqual(result["known_cost"], sum(r["step_cost"] for r in rows))

    async def test_self_declared_completion_is_not_task_success(self):
        summary = await self.run_collection(FakeProvider(["```text\ntask completed\n```"]))
        self.assertEqual(summary["success_rate"], 0)
        self.assertEqual(FakeEnv.instances[0].actions, [])

    async def test_api_failure_preserved_and_excluded_from_labels(self):
        summary = await self.run_collection(FakeProvider([ProviderError("timeout")]))
        self.assertFalse(summary["cost_complete"])
        self.assertEqual(summary["valid_traces"], 0)
        self.assertEqual(Path(summary["steps_file"]).read_text(), "")
        trace = json.loads(next((self.root / "out").glob("*.traj.json")).read_text())
        self.assertEqual(trace["info"]["model_stats"]["api_calls"], 1)
        self.assertTrue(FakeEnv.instances[0].closed)

    async def test_missing_usage_is_not_free(self):
        class MissingUsage:
            async def complete(self, *args):
                return ProviderResponse(text="```text\nlook around\n```", model="a")
        summary = await self.run_collection(MissingUsage())
        self.assertEqual(summary["valid_traces"], 0)
        self.assertFalse(summary["cost_complete"])

    async def test_cost_limit_stops_before_next_api_call(self):
        p = FakeProvider(["```text\nlook around\n```"])
        settings = CollectionSettings("baseline", ["a"], "a", cost_limit=.0001)
        summary = await self.run_collection(p, settings)
        self.assertEqual(len(p.histories), 1)
        self.assertEqual(summary["valid_traces"], 1)
        self.assertEqual(summary["success_rate"], 0)
        trace = json.loads(next((self.root / "out").glob("*.traj.json")).read_text())
        self.assertEqual(trace["info"]["exit_status"], "CostLimitExceeded")


class ReaderAndPricingTests(unittest.TestCase):
    def test_policy_determinism_and_propensity(self):
        a = SamplingPolicy(["a", "b"], "roulette", 7)
        b = SamplingPolicy(["a", "b"], "roulette", 7)
        selections = [a.choose() for _ in range(50)]
        self.assertEqual(selections, [b.choose() for _ in range(50)])
        self.assertEqual({x[0] for x in selections}, {"a", "b"})
        self.assertEqual({x[1] for x in selections}, {.5})
        self.assertEqual(SamplingPolicy(["a", "b"], "baseline", 7, "b").choose(), ("b", 1))

    def test_official_split_reader(self):
        class Catalog:
            closed = False
            def __init__(self, *a, **kw): pass
            def get_task_names(self): return ["boil"]
            def get_max_variations(self, name): return 10
            def load(self, *a, **kw): pass
            def get_variations_train(self): return [1, 3]
            def close(self): Catalog.closed = True
        import types
        module = types.ModuleType("scienceworld")
        module.ScienceWorldEnv = Catalog
        with patch.dict("sys.modules", {"scienceworld": module}):
            rows = enumerate_scienceworld_tasks(split="train")
        self.assertEqual([t["variation_idx"] for t in rows], [1, 3])
        self.assertTrue(Catalog.closed)

    def test_manifest_reader_avoids_reference_and_rejects_duplicate_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"id": "boil-1", "source": {"split": "train"},
                "environment": {"benchmark": "scienceworld", "task_name": "boil", "variation_idx": 1}}))
            rows = load_tasks(root)
            self.assertEqual(rows[0]["task_name"], "boil")
            with self.assertRaises(ValueError):
                validate_tasks([dict(TASK), {**TASK, "task_id": "alias", "split": "test"}])
            with self.assertRaises(ValueError):
                validate_tasks([{**TASK, "task_id": "../outside"}])

    def test_selection_balances_types_and_keeps_splits(self):
        rows = [{**TASK, "task_id": f"{name}-{i}", "task_name": name, "variation_idx": i}
                for name in ("boil", "melt", "grow") for i in range(10)]
        rows.append({**TASK, "task_id": "test-100", "variation_idx": 100, "split": "test"})
        selected = select_tasks(rows, split="train", limit=6)
        self.assertEqual({n: sum(x["task_name"] == n for x in selected) for n in ("boil", "melt", "grow")},
                         {"boil": 2, "melt": 2, "grow": 2})
        self.assertEqual(selected, select_tasks(list(reversed(rows)), split="train", limit=6))

    def test_cache_peak_and_off_peak_prices(self):
        usage = Usage(1000, 100, raw={"prompt_cache_hit_tokens": 800})
        rates = {"input_per_million": 2, "output_per_million": 8,
                 "cached_input_per_million": .04, "schedule": "deepseek_cn_peak"}
        peak, _ = estimate_cost(usage, rates, "2026-09-16T02:00:00+00:00")
        off_peak, _ = estimate_cost(usage, rates, "2026-09-16T12:00:00+00:00")
        self.assertAlmostEqual(peak, (200 * 2 + 800 * .04 + 100 * 8) / 1e6)
        self.assertAlmostEqual(off_peak, peak / 2)
        with self.assertRaises(ValueError):
            estimate_cost(Usage(), rates, "2026-09-16T02:00:00+00:00")

    def test_environment_reset_clears_cached_score(self):
        class Native:
            def load(self, *a, **kw): pass
            def reset(self): return "kitchen", {"taskDesc": "Boil", "valid": []}
            def close(self): pass
        env = ScienceWorldEnvironment(task_name="boil")
        env._env = Native()
        env._last_score = 100
        env.load_task("boil", 1)
        self.assertEqual(env.get_score(), 0)
        env.close()

    def test_dry_run_from_file_needs_no_jvm_or_credentials(self):
        root = Path(__file__).resolve().parents[1]
        with redirect_stdout(io.StringIO()) as output:
            status = main(["--dry-run", "--tasks-file", str(root / "configs/scienceworld.tasks.example.json"),
                "--split", "all", "--config", str(root / "configs/collection.providers.json"),
                "--pricing", str(root / "configs/collection.prices.json"), "--model-pool",
                "qwen-flash-2025-07-28", "deepseek-flash", "--runs", "2"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["traces"], 4)


if __name__ == "__main__":
    unittest.main()
