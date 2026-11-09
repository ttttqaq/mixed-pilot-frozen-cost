"""Native rollout contracts exercised without Java, GPUs, or paid APIs."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mixed_pilot_harness.collection.episodes import Transition
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.types import Message, ProviderError, ProviderResponse, ToolCall, Usage
from mixed_pilot_mtrouter.online.runtime import Ledger, Runner, UnknownCostError


class Episode:
    system = "Complete the simulated task."
    tools = []

    def __init__(self, task, *, salt="", broken=False):
        self.task, self.salt, self.broken = task, salt, broken
        self.score = 0.0
        self.count = 0
        self.closed = False

    def start(self):
        return [Message("user", "initial:" + self.task["task_id"] + self.salt)]

    async def advance(self, response):
        if self.broken:
            raise RuntimeError("simulator disconnected after billed response")
        if response.tool_calls:
            return Transition([Message("user", "invalid tool arguments")],
                              {"observation": "invalid tool arguments"}, format_error=True)
        self.count += 1
        self.score += 25 if response.text == "good" else 0
        observation = "observed:" + response.text + ":" + str(self.count) + self.salt
        return Transition([Message("user", observation)], {"observation": observation},
                          done=self.count == 4, status="native_done" if self.count == 4 else "")

    async def evaluate(self):
        return {"status": "scored", "score": self.score, "success": self.score == 100,
                "cost": 0.0, "cost_complete": True, "api_calls": 0}

    def close(self):
        self.closed = True


class Provider:
    def __init__(self, profile, calls, mode="normal"):
        self.profile, self.calls, self.mode = profile, calls, mode

    async def complete(self, messages, tools, system):
        self.calls.append(self.profile.name)
        if self.mode == "unknown":
            raise RuntimeError("network connection lost")
        if self.mode == "known_zero":
            raise ProviderError("credential unavailable", cost_known_zero=True)
        response = ProviderResponse(text="good" if self.profile.name == "a" else "bad",
                                    usage=Usage(10000, 10000), cost_complete=self.mode != "incomplete")
        if self.mode == "tool":
            response = replace(response, text="", tool_calls=[ToolCall("x", "shell", {},
                                                                        raw_arguments="{", parse_error="malformed")])
        return response


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = []
        self.config = {"rollout": {"step_limit": 10, "task_budget": 1.0, "replay_attempts": 2},
                       "teacher": {"llm": {"max_tokens": 123, "temperature": 0.0}}}
        self.harness = HarnessConfig(self.root / "harness.json", "a", {
            name: ProviderProfile(kind="scripted", model=name, name=name) for name in ["a", "b"]})
        self.prices = {"currency": "CNY", "profiles": {
            name: {"input_per_million": 1, "output_per_million": 1} for name in ["a", "b"]}}
        self.task = {"task_id": "train--one", "benchmark": "scienceworld", "split": "train"}
        self.ledger = Ledger(self.root, 100, "CNY")
        self.mode = "normal"
        self.runner = Runner(self.config, self.harness, self.prices, self.ledger,
                             episode_factory=lambda task: Episode(task),
                             provider_factory=lambda p, **kw: Provider(p, self.calls, self.mode))

    def tearDown(self):
        self.tmp.cleanup()

    def run_rollout(self, **kwargs):
        return asyncio.run(self.runner.rollout(self.task, lambda state: "a", **kwargs))

    def test_single_step_intervention_then_frozen_policy_and_zero_cost_replay(self):
        original = self.run_rollout(horizon=2, trajectory_id="original")
        self.assertEqual(self.calls, ["a", "a"])
        prefix = original["steps"][:1]
        branch = self.run_rollout(prefix=prefix, forced_first="b", horizon=3, trajectory_id="branch")
        self.assertEqual(self.calls, ["a", "a", "b", "a", "a"])
        self.assertEqual([s["profile"] for s in branch["steps"]], ["b", "a", "a"])
        self.assertEqual(branch["steps"][0]["before_state"]["snapshot_id"], original["steps"][1]["before_state"]["snapshot_id"])
        self.assertAlmostEqual(branch["cost"], .06)
        self.assertAlmostEqual(branch["path_cost"], .08)
        self.assertAlmostEqual(self.ledger.cost, .10)
        self.assertEqual(len(self.ledger.records), 5)
        self.assertTrue(branch["terminal"])
        self.assertTrue(branch["labels_valid"])
        self.assertEqual(branch["reward"], 0.0)
        self.assertTrue(branch["replay"][-1]["matched"])

    def test_short_cutoff_has_no_false_terminal_or_failure_label(self):
        result = self.run_rollout(horizon=1)
        self.assertEqual(result["status"], "truncated")
        self.assertFalse(result["terminal"])
        self.assertFalse(result["labels_valid"])
        self.assertIsNone(result["reward"])
        self.assertEqual(result["state"]["remaining_steps"], 9)
        self.assertAlmostEqual(result["state"]["remaining_budget"], .98)

    def test_replay_mismatch_rejected_before_new_model_call(self):
        original = self.run_rollout(horizon=1)
        self.runner.episode_factory = lambda task: Episode(task, salt="changed")
        result = self.run_rollout(prefix=original["steps"], horizon=1)
        self.assertEqual(result["status"], "replay_failed")
        self.assertFalse(result["labels_valid"])
        self.assertEqual(self.calls, ["a"])
        self.assertEqual(result["cost"], 0)

    def test_tampered_transition_rejected_even_with_same_initial_history(self):
        original = self.run_rollout(horizon=1)
        prefix = json.loads(json.dumps(original["steps"]))
        prefix[0]["transition"]["score"] = 99
        result = self.run_rollout(prefix=prefix, horizon=1)
        self.assertEqual(result["status"], "replay_failed")
        self.assertEqual(self.calls, ["a"])

    def test_native_task_limits_get_real_terminal_labels(self):
        self.runner.step_limit = 2
        result = self.run_rollout()
        self.assertEqual(result["status"], "step_limit")
        self.assertTrue(result["terminal"])
        self.assertTrue(result["labels_valid"])
        self.assertEqual(result["reward"], 0)
        self.assertAlmostEqual(result["cost"], .04)
        self.assertEqual(result["evaluation"]["score"], 50)

    def test_normalized_score_is_distinct_from_binary_success(self):
        self.runner.step_limit = 2
        self.runner.reward_mode = "normalized_score"
        result = self.run_rollout()
        self.assertEqual(result["reward"], .75)
        self.assertFalse(result["success"])

    def test_global_budget_stops_at_next_request_and_keeps_partial_unscored(self):
        self.ledger.limit = .03
        result = self.run_rollout()
        self.assertEqual(self.calls, ["a", "a"])
        self.assertEqual(result["status"], "global_budget")
        self.assertFalse(result["terminal"])
        self.assertIsNone(result["reward"])
        self.assertAlmostEqual(result["cost"], .04)

    def test_unknown_charge_persisted_and_blocks_further_calls(self):
        self.mode = "unknown"
        result = self.run_rollout()
        self.assertEqual(result["status"], "unknown_cost")
        self.assertFalse(result["cost_complete"])
        self.assertFalse(self.ledger.cost_complete)
        resumed = Ledger(self.root, 100, "CNY")
        with self.assertRaises(UnknownCostError):
            resumed.check()
        second = self.run_rollout()
        self.assertEqual(second["status"], "unknown_cost")
        self.assertEqual(self.calls, ["a"])
        self.assertEqual(len(list((self.root / "calls").glob("*.json"))), 1)

    def test_known_cost_from_incomplete_transport_is_not_lost(self):
        self.mode = "incomplete"
        result = self.run_rollout()
        self.assertEqual(result["status"], "unknown_cost")
        self.assertAlmostEqual(result["cost"], .02)
        self.assertFalse(self.ledger.report()["cost_complete"])

    def test_environment_failure_preserves_already_billed_request(self):
        self.runner.episode_factory = lambda task: Episode(task, broken=True)
        result = self.run_rollout()
        self.assertEqual(result["status"], "runtime_error")
        self.assertAlmostEqual(result["cost"], .02)
        self.assertFalse(result["labels_valid"])
        record = next(iter(self.ledger.records.values()))
        self.assertTrue(record["cost_complete"])
        self.assertEqual(record["response"]["text"], "good")

    def test_teacher_scope_generation_limits_and_resume_identity(self):
        completion = asyncio.run(self.runner.complete("b", [Message("user", "estimate tail")],
                                                      scope="teacher", profile_overrides={"max_tokens": 77}))
        self.assertEqual(completion["response"].text, "bad")
        first = completion["call_id"]
        self.assertEqual(self.ledger.records[first]["generation"]["max_tokens"], 77)
        self.assertAlmostEqual(self.ledger.report()["by_scope"]["teacher"]["known_cost"], .02)
        resumed = Ledger(self.root, 100, "CNY")
        cid = resumed.begin("valid", "a")
        self.assertNotEqual(first, cid)
        resumed.finish(cid, cost=0, complete=True)
        with self.assertRaises(ValueError):
            resumed.finish(cid, cost=0, complete=True)
        self.assertAlmostEqual(resumed.cost, .02)

    def test_teacher_reasoning_controls_do_not_change_actor_profile(self):
        seen = []
        def factory(profile, **kwargs):
            seen.append(profile)
            return Provider(profile, self.calls)
        self.runner.provider_factory = factory
        extra = {"thinking":{"type":"enabled"},"reasoning_effort":"low",
                 "response_format":{"type":"json_object"}}
        asyncio.run(self.runner.complete("b", [Message("user","estimate")],scope="teacher",
                                         profile_overrides={"extra_body":extra}))
        self.run_rollout(horizon=1)
        self.assertEqual(seen[0].extra_body,extra)
        self.assertEqual(seen[1].extra_body,{})
        self.assertEqual(self.harness.profile("b").extra_body,{})
        with self.assertRaises(ValueError):
            asyncio.run(self.runner.complete("a",[],scope="collection",profile_overrides={"extra_body":extra}))

    def test_low_reasoning_and_json_mode_reach_the_actual_http_payload(self):
        from unittest.mock import patch
        from mixed_pilot_harness.providers import OpenAICompatibleProvider
        captured = []
        self.harness.profiles["b"] = replace(self.harness.profile("b"),kind="openai",api_key_optional=True)
        self.runner.provider_factory = lambda profile,**kw:OpenAICompatibleProvider(profile)
        def response(provider,url,payload,headers):
            captured.append(json.loads(json.dumps(payload)))
            return {"model":"b","choices":[{"message":{"content":"{}"},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":100,"completion_tokens":10}}
        extra = {"thinking":{"type":"enabled"},"reasoning_effort":"low","response_format":{"type":"json_object"}}
        with patch("mixed_pilot_harness.providers.JsonHttpProvider._post_json_sync",new=response):
            asyncio.run(self.runner.complete("b",[Message("user","estimate")],scope="teacher",profile_overrides={"extra_body":extra}))
            asyncio.run(self.runner.complete("b",[Message("user","act")],scope="collection"))
        self.assertEqual(captured[0]["reasoning_effort"],"low")
        self.assertEqual(captured[0]["response_format"],{"type":"json_object"})
        self.assertEqual(captured[0]["thinking"],{"type":"enabled"})
        self.assertNotIn("reasoning_effort",captured[1])
        self.assertNotIn("response_format",captured[1])

    def test_invalid_tool_arguments_are_recorded_without_invented_success(self):
        self.mode = "tool"
        result = self.run_rollout(horizon=1)
        self.assertTrue(result["steps"][0]["transition"]["format_error"])
        self.assertEqual(result["status"], "truncated")
        self.assertIsNone(result["reward"])
        self.assertEqual(result["state"]["history"][-1]["content"], "invalid tool arguments")

    def test_resume_with_inflight_call_refuses_spending(self):
        cid = self.ledger.begin("collection", "a")
        recovered = Ledger(self.root, 100, "CNY")
        self.assertIn(cid, recovered.records)
        with self.assertRaises(UnknownCostError):
            recovered.check()

    def test_complete_call_cannot_have_an_unknown_cost(self):
        cid = self.ledger.begin("collection", "a")
        with self.assertRaises(ValueError):
            self.ledger.finish(cid, cost=None, complete=True)
        self.assertFalse(self.ledger.cost_complete)

    def test_known_zero_provider_rejection_does_not_poison_ledger(self):
        self.mode = "known_zero"
        failed = self.run_rollout()
        self.assertEqual(failed["status"], "runtime_error")
        self.assertTrue(failed["cost_complete"])
        self.assertTrue(self.ledger.cost_complete)
        self.assertEqual(self.ledger.cost, 0)
        self.mode = "normal"
        self.runner.providers.clear()
        result = self.run_rollout(horizon=1)
        self.assertEqual(result["status"], "truncated")

    def test_actor_must_stay_inside_its_model_pool(self):
        self.runner.pool = ["a"]
        result = self.run_rollout(forced_first="b", horizon=1)
        self.assertEqual(result["status"], "runtime_error")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger.cost, 0)

    def test_prefix_from_a_changed_native_task_is_rejected(self):
        original = self.run_rollout(horizon=1)
        changed = dict(self.task, content_hashes={"manifest": "different"})
        result = asyncio.run(self.runner.rollout(changed, lambda state: "a", prefix=original["steps"], horizon=1))
        self.assertEqual(result["status"], "replay_failed")
        self.assertEqual(self.calls, ["a"])

    def test_terminal_prefix_can_be_replayed_without_billing_new_call(self):
        original = self.run_rollout()
        replay = self.run_rollout(prefix=original["all_steps"], horizon=1)
        self.assertTrue(replay["terminal"])
        self.assertTrue(replay["labels_valid"])
        self.assertEqual(replay["reward"], 1)
        self.assertEqual(replay["cost"], 0)
        self.assertEqual(len(self.calls), 4)


if __name__ == "__main__":
    unittest.main()
