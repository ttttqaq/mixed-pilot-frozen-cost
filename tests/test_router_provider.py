from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mixed_pilot_harness.config import ProviderProfile, load_config
from mixed_pilot_harness.providers import create_provider
from mixed_pilot_harness.router.config import (
    RouterConfig,
    RouterConfigError,
    RouterDecisionConfig,
    load_router_config,
)
from mixed_pilot_harness.router.decision import RouteSelectionError, select_route
from mixed_pilot_harness.router.input_adapter import (
    build_router_request,
    project_opensquilla_v4_input,
)
from mixed_pilot_harness.router.opensquilla_v4 import OpenSquillaV4Engine
from mixed_pilot_harness.router.provider import RouterProvider
from mixed_pilot_harness.runner import TurnRunner
from mixed_pilot_harness.trace import JsonlTrace
from mixed_pilot_harness.types import Message, ProviderError, ToolCall, ToolDefinition


class RouterConfigTests(unittest.TestCase):
    def test_harness_profile_name_is_retained_as_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "harness.json"
            path.write_text(
                json.dumps(
                    {
                        "default_profile": "model-a",
                        "profiles": {
                            "model-a": {"kind": "openai", "model": "model-a"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_config(path)
            self.assertEqual(loaded.profile("model-a").name, "model-a")

    def test_profile_targets_and_model_override_are_resolved(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bundle = root / "bundle"
            config = root / "router.toml"
            config.write_text(
                """
[router]
engine = "opensquilla_v4"
bundle_dir = "bundle"

[decision]
mode = "raw"
high_risk_floor = false

[tiers.c0]
profile = "base"
model = "small"
[tiers.c1]
profile = "base"
[tiers.c2]
profile = "base"
[tiers.c3]
profile = "base"
""",
                encoding="utf-8",
            )
            loaded = load_router_config(
                config,
                {
                    "base": ProviderProfile(
                        kind="openai",
                        model="normal",
                        base_url="https://example.test/v1",
                        api_key_env="TEST_KEY",
                    )
                },
            )
            self.assertEqual(loaded.bundle_dir, bundle.resolve())
            self.assertEqual(loaded.tiers["c0"].model, "small")
            self.assertEqual(loaded.tiers["c1"].model, "normal")
            self.assertEqual(loaded.tiers["c3"].api_key_env, "TEST_KEY")
            self.assertEqual(loaded.decision.mode, "raw")
            self.assertFalse(loaded.decision.high_risk_floor)

    def test_router_target_recursion_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "router.toml"
            path.write_text(
                """
[router]
engine = "fixed"
[tiers.c0]
profile = "nested"
[tiers.c1]
profile = "normal"
[tiers.c2]
profile = "normal"
[tiers.c3]
profile = "normal"
""",
                encoding="utf-8",
            )
            profiles = {
                "nested": ProviderProfile(kind="router", model="router"),
                "normal": ProviderProfile(kind="scripted", model="normal"),
            }
            with self.assertRaisesRegex(RouterConfigError, "cannot target another router"):
                load_router_config(path, profiles)

    def test_invalid_decision_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "router.toml"
            path.write_text(
                """
[router]
engine = "fixed"
[decision]
mode = "turnish"
[tiers.c0]
kind = "scripted"
model = "test"
[tiers.c1]
kind = "scripted"
model = "test"
[tiers.c2]
kind = "scripted"
model = "test"
[tiers.c3]
kind = "scripted"
model = "test"
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RouterConfigError, "decision.mode"):
                load_router_config(path)


class RouterDecisionModeTests(unittest.TestCase):
    probabilities = {"R0": 0.70, "R1": 0.20, "R2": 0.08, "R3": 0.02}
    turn_flags = {"debug": True, "long_context": True, "high_risk": False}

    def test_legacy_uses_opensquilla_postprocessed_route(self):
        selected = select_route(
            self.probabilities,
            "R2",
            self.turn_flags,
            RouterDecisionConfig(mode="legacy"),
        )
        self.assertEqual(selected.raw_route_class, "R0")
        self.assertEqual(selected.selected_route_class, "R2")
        self.assertEqual(selected.applied_rules, ["opensquilla_legacy_postprocess"])

    def test_raw_ignores_all_legacy_heuristics(self):
        selected = select_route(
            self.probabilities,
            "R2",
            self.turn_flags,
            RouterDecisionConfig(mode="raw"),
        )
        self.assertEqual(selected.selected_route_class, "R0")
        self.assertEqual(selected.applied_rules, ["raw_argmax"])

    def test_api_step_ignores_turn_rules_but_keeps_high_risk_floor(self):
        without_risk = select_route(
            self.probabilities,
            "R2",
            self.turn_flags,
            RouterDecisionConfig(mode="api_step"),
        )
        with_risk = select_route(
            self.probabilities,
            "R2",
            {**self.turn_flags, "high_risk": True},
            RouterDecisionConfig(mode="api_step"),
        )
        floor_disabled = select_route(
            self.probabilities,
            "R2",
            {**self.turn_flags, "high_risk": True},
            RouterDecisionConfig(mode="api_step", high_risk_floor=False),
        )
        self.assertEqual(without_risk.selected_route_class, "R0")
        self.assertEqual(with_risk.selected_route_class, "R2")
        self.assertEqual(with_risk.applied_rules[-1], "high_risk_floor:R2")
        self.assertEqual(floor_disabled.selected_route_class, "R0")

    def test_missing_probability_class_is_rejected(self):
        with self.assertRaisesRegex(RouteSelectionError, "missing classes"):
            select_route(
                {"R0": 1.0},
                "R0",
                {},
                RouterDecisionConfig(mode="raw"),
            )

    def test_v4_adapter_uses_selected_mode_and_surfaces_audit_metadata(self):
        engine = object.__new__(OpenSquillaV4Engine)
        engine.config = RouterConfig(
            path=Path("router.toml"),
            engine="opensquilla_v4",
            tiers={},
            decision=RouterDecisionConfig(mode="raw"),
        )
        engine.model_version = "test-v4"
        engine._request_type = lambda **values: SimpleNamespace(**values)
        engine._core = SimpleNamespace(
            predict=lambda request: SimpleNamespace(
                probabilities=dict(self.probabilities),
                decision=SimpleNamespace(
                    route_class="R2",
                    margin=0.5,
                    difficulty_score=0.4,
                    thinking_mode="T3",
                    prompt_policy="P2",
                    flags=dict(self.turn_flags),
                    aux_downgrade_applied=False,
                    sticky_applied=False,
                ),
            )
        )
        decision = engine.route(
            build_router_request([Message("user", "Fix it")], [], "", [])
        )
        self.assertEqual(decision.route_class, "R0")
        self.assertEqual(decision.tier, "c0")
        self.assertEqual(decision.confidence, 0.70)
        self.assertEqual(decision.metadata["decision_mode"], "raw")
        self.assertEqual(decision.metadata["legacy_route_class"], "R2")
        self.assertEqual(decision.metadata["selected_route_class"], "R0")


class RouterInputAdapterTests(unittest.TestCase):
    def test_complete_tool_trail_is_projected_to_step_features(self):
        messages = [
            Message("user", "Fix src/parser.py safely."),
            Message(
                "assistant",
                "Inspecting.",
                tool_calls=[ToolCall("read", "read_file", {"path": "src/parser.py"})],
            ),
            Message(
                "tool",
                "Traceback: parse failed",
                tool_call_id="read",
                name="read_file",
            ),
        ]
        request = build_router_request(
            messages,
            [ToolDefinition("read_file", "read", {})],
            "system prompt",
            [],
        )
        projected = project_opensquilla_v4_input(
            request, load_router_config_input_defaults()
        )
        self.assertEqual(request.messages, messages)
        self.assertEqual(request.system, "system prompt")
        self.assertEqual(request.api_step_index, 2)
        self.assertEqual(projected.current_user_text, "Fix src/parser.py safely.")
        self.assertIn("TOOL_RESULT", projected.history_texts[-1])
        self.assertIn("Traceback", projected.flags_text)
        self.assertTrue(projected.context_metadata["has_tool_results"])
        self.assertTrue(projected.context_metadata["has_file_reference"])

    def test_optional_other_agent_trails_are_kept_as_older_context(self):
        request = build_router_request(
            [Message("user", "Implement it"), Message("assistant", "Working")],
            [],
            "",
            [],
            other_agent_trails=["research agent found an API constraint"],
        )
        projected = project_opensquilla_v4_input(
            request, load_router_config_input_defaults()
        )
        self.assertIn("research agent", request.other_agent_trails[0])
        self.assertIn("AGENT_TRAIL", projected.history_texts[0])
        self.assertIn("ASSISTANT", projected.history_texts[-1])


def load_router_config_input_defaults():
    from mixed_pilot_harness.router.config import RouterInputConfig

    return RouterInputConfig()


class RouterProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_router_kind_delegates_each_step_to_configured_profile(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            responses = root / "responses.json"
            responses.write_text(
                json.dumps(
                    [
                        {
                            "model": "tier-two",
                            "tool_calls": [
                                {
                                    "id": "write",
                                    "name": "write_file",
                                    "arguments": {"path": "answer.txt", "content": "ok\n"},
                                }
                            ],
                        },
                        {"model": "tier-two", "text": "finished"},
                    ]
                ),
                encoding="utf-8",
            )
            router_toml = root / "router.toml"
            router_toml.write_text(
                """
[router]
engine = "fixed"
fixed_tier = "c2"

[tiers.c0]
profile = "target"
[tiers.c1]
profile = "target"
[tiers.c2]
profile = "target"
[tiers.c3]
profile = "target"
""",
                encoding="utf-8",
            )
            profiles = {
                "target": ProviderProfile(
                    kind="scripted",
                    model="tier-two",
                    responses_file=str(responses),
                )
            }
            provider = create_provider(
                ProviderProfile(
                    kind="router",
                    model="squilla-router",
                    router_config=str(router_toml),
                ),
                config_dir=root,
                profiles=profiles,
            )
            self.assertIsInstance(provider, RouterProvider)
            trace_path = root / "events.jsonl"
            result = await TurnRunner(
                provider, root, trace=JsonlTrace(trace_path)
            ).run("write the answer")
            self.assertEqual(result.steps, 2)
            self.assertEqual(result.text, "finished")
            self.assertEqual((root / "answer.txt").read_text(), "ok\n")
            self.assertEqual(provider.last_decision.tier, "c2")
            self.assertEqual(provider.last_decision.metadata["api_step_index"], 2)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            decisions = [item for item in records if item["event"] == "router_decision"]
            self.assertEqual([item["step"] for item in decisions], [1, 2])
            self.assertEqual([item["api_step_index"] for item in decisions], [1, 2])
            self.assertEqual([item["tier"] for item in decisions], ["c2", "c2"])
            self.assertEqual(decisions[0]["selected_route_class"], "R2")
            self.assertEqual(decisions[0]["target_profile"], "target")
            self.assertEqual(decisions[0]["target_provider"], "scripted")
            self.assertEqual(decisions[0]["target_model"], "tier-two")
            completed = [
                item for item in records if item["event"] == "provider_call_completed"
            ]
            self.assertTrue(
                all(
                    decision["sequence"] < completion["sequence"]
                    for decision, completion in zip(decisions, completed)
                )
            )

    async def test_router_decision_is_traced_before_target_provider_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            responses = root / "responses.json"
            responses.write_text("[]", encoding="utf-8")
            router_toml = root / "router.toml"
            router_toml.write_text(
                """
[router]
engine = "fixed"
fixed_tier = "c1"

[tiers.c0]
profile = "failing-model"
[tiers.c1]
profile = "failing-model"
[tiers.c2]
profile = "failing-model"
[tiers.c3]
profile = "failing-model"
""",
                encoding="utf-8",
            )
            provider = create_provider(
                ProviderProfile(
                    kind="router",
                    model="squilla-router",
                    router_config=str(router_toml),
                ),
                config_dir=root,
                profiles={
                    "failing-model": ProviderProfile(
                        kind="scripted",
                        model="failing-model",
                        responses_file=str(responses),
                    )
                },
            )
            trace_path = root / "events.jsonl"
            with self.assertRaisesRegex(ProviderError, "sequence exhausted"):
                await TurnRunner(
                    provider, root, trace=JsonlTrace(trace_path)
                ).run("fail after routing")
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            decision = next(item for item in records if item["event"] == "router_decision")
            failure = next(item for item in records if item["event"] == "iteration_failed")
            self.assertEqual(decision["target_profile"], "failing-model")
            self.assertEqual(decision["tier"], "c1")
            self.assertLess(decision["sequence"], failure["sequence"])


if __name__ == "__main__":
    unittest.main()
