"""Offline fault injection: no real provider, environment, or API is used."""
import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

import torch

from mixed_pilot_harness.providers import HTTP_RETRY_POLICY
from mixed_pilot_harness.types import Message, ProviderError, ProviderResponse, Usage
from mixed_pilot_mtrouter.online.config import DEFAULTS
from mixed_pilot_mtrouter.online.runtime import Ledger, UnknownCostError, _ambiguity_mapping


class ResilienceTests(unittest.TestCase):
    def setUp(self):
        from test_online_runtime import RuntimeTests
        self.fixture = RuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.runner = self.fixture.runner
        self.runner.resilience = deepcopy(DEFAULTS["resilience"])
        self.runner.resilience.update(retry_delay=0, max_retry_delay=0, stop_after_retries=False)
        self.attempts = []

    def provider(self, errors):
        errors = list(errors)
        async def complete(messages, tools, system):
            self.attempts.append([m.content for m in messages])
            if errors:
                code = errors.pop(0)
                cause = TimeoutError("simulated timeout") if code == "timeout" else HTTPError("https://fixture.invalid", code, "simulated", {}, None)
                raise ProviderError("temporary fixture failure", cost_known_zero=code != "timeout") from cause
            return ProviderResponse(text="good", usage=Usage(100, 10))
        self.runner.provider_factory = lambda *a, **kw: SimpleNamespace(complete=complete)

    def run_rollout(self, **kwargs):
        with redirect_stdout(io.StringIO()), patch("asyncio.sleep", new_callable=AsyncMock):
            return self.fixture.run_rollout(**kwargs)

    def test_rate_limit_retries_same_state_and_advances_environment_once(self):
        self.provider([429, 429])
        result = self.run_rollout(horizon=1)
        self.assertEqual(len(self.attempts), 3)
        self.assertEqual(self.attempts[0], self.attempts[1])
        self.assertEqual(self.attempts[1], self.attempts[2])
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["transition"]["score"], 25)
        self.assertTrue(self.fixture.ledger.cost_complete)
        self.assertEqual(self.fixture.ledger.reserved_cost, 0)
        self.assertIsNone(HTTP_RETRY_POLICY.get())

    def test_ambiguity_permutation_requires_unique_semantically_identical_choices(self):
        prefix = "Ambiguous request: choose\n"
        expected = prefix + "0:\tapple\n1:\tpear\n"
        actual = prefix + "0:\tpear\n1:\tapple\n"
        self.assertEqual(_ambiguity_mapping(expected, actual), {"0":"1", "1":"0"})
        for other in [prefix + "0:\tpear\n1:\tbanana\n", prefix + "0:\tapple\n1:\tapple\n", "different\n"+actual]:
            self.assertIsNone(_ambiguity_mapping(expected, other))

    def test_replay_remaps_numeric_choice_to_the_same_native_action(self):
        from mixed_pilot_harness.collection.collector import load_prompt_config
        from mixed_pilot_harness.collection.episodes import ScienceWorldEpisode
        class Native:
            reversed = False
            last_command = None
            def __init__(self, **kwargs):
                self.choices = ["pear", "apple"] if Native.reversed else ["apple", "pear"]
            def load_task(self, *args):
                return {"task_description":"test", "observation":"initial"}
            def step(self, command):
                Native.last_command = command
                if command == "ask":
                    obs = "Ambiguous request: choose\n" + "".join(str(i)+":\t"+v+"\n" for i,v in enumerate(self.choices))
                    return {"observation":obs,"score":0,"done":False}
                success = self.choices[int(command)] == "apple"
                return {"observation":"selected apple" if success else "selected pear", "score":100 if success else 0,"done":True}
            def close(self):
                pass
        commands = iter(["ask", "0"])
        async def complete(*args, **kwargs):
            return ProviderResponse(text="```text\n"+next(commands)+"\n```",usage=Usage(100,10))
        self.runner.provider_factory = lambda *a,**kw:SimpleNamespace(complete=complete)
        self.runner.episode_factory = lambda task:ScienceWorldEpisode(task,load_prompt_config(),Native)
        self.fixture.task.update(task_name="fixture",variation_idx=0)
        original = self.run_rollout(horizon=1)
        Native.reversed = True
        branch = self.run_rollout(prefix=original["all_steps"],horizon=1)
        self.assertEqual(Native.last_command,"1")
        self.assertTrue(branch["success"])
        self.assertTrue(branch["labels_valid"])
        self.assertEqual(branch["steps"][0]["transition"]["record"]["command"],"0")
        self.assertTrue(any(r.get("ambiguity_choice_permutation_restored") for r in branch["replay"]))

    def test_timeout_then_success_reserves_unknown_fee_without_fake_complete_label(self):
        self.provider(["timeout"])
        result = self.run_rollout()
        self.assertTrue(result["terminal"])
        self.assertFalse(result["labels_valid"], "Uncertain inference cost cannot become a true cost target")
        self.assertIsNone(result["reward"])
        self.assertFalse(self.fixture.ledger.cost_complete)
        self.assertTrue(self.fixture.ledger.spending_safe)
        self.assertGreater(self.fixture.ledger.reserved_cost, 0)
        restored = Ledger(self.fixture.root, 100, "CNY")
        restored.check()
        self.assertAlmostEqual(restored.budget_cost, restored.cost + restored.reserved_cost)

    def test_exhausted_requests_skip_sample_and_allow_next_task(self):
        self.provider(["timeout"] * 4)
        result = self.run_rollout(horizon=1)
        self.assertEqual(len(self.attempts), 4)
        self.assertEqual(result["status"], "runtime_error")
        self.assertFalse(result["labels_valid"])
        self.assertTrue(self.fixture.ledger.spending_safe)
        self.assertEqual(self.fixture.ledger.cost, 0)
        self.runner.providers.clear()
        self.provider([])
        result = self.run_rollout(horizon=1)
        self.assertEqual(result["status"], "truncated")

    def test_permanent_auth_failure_is_not_retried(self):
        self.provider([401])
        result = self.run_rollout(horizon=1)
        self.assertEqual(len(self.attempts), 1)
        self.assertFalse(result["labels_valid"])
        self.assertEqual(self.fixture.ledger.budget_cost, 0)

    def test_unknown_reservations_count_against_budget(self):
        self.provider(["timeout"] * 4)
        self.fixture.ledger.limit = .00001
        result = self.run_rollout(horizon=1)
        self.assertEqual(result["status"], "global_budget")
        self.assertEqual(len(self.attempts), 1)
        self.assertTrue(self.fixture.ledger.exhausted)
        self.assertEqual(self.fixture.ledger.report()["known_cost"], 0)
        self.assertGreater(self.fixture.ledger.report()["budget_cost"], 0)

    def test_backoff_is_bounded_and_uses_configured_seconds(self):
        self.provider([503, 503, 503])
        self.runner.resilience.update(retry_delay=5, max_retry_delay=12)
        with redirect_stdout(io.StringIO()), patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = self.fixture.run_rollout(horizon=1)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [5, 10, 12])
        self.assertEqual(result["status"], "truncated")

    def test_rate_limits_wait_longer_than_general_transport_errors(self):
        self.provider([429,429,429])
        self.runner.resilience.update(rate_limit_wait=30,max_retry_delay=60)
        with redirect_stdout(io.StringIO()), patch("asyncio.sleep",new_callable=AsyncMock) as sleep:
            result = self.fixture.run_rollout(horizon=1)
        self.assertEqual([call.args[0] for call in sleep.await_args_list],[30,60,60])
        self.assertEqual(result["status"],"truncated")
