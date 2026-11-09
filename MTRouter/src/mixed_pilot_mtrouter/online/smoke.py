"""Deterministic, offline fixtures for the online training plumbing.

This module NEVER evaluates native benchmark performance. Real train/valid
manifests may still be used to exercise split access guards; the environment,
actor and assessor responses below are explicitly synthetic and make no network
request. Actual frozen prices are retained so the unique-call ledger is tested.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import json
import math
import re


SMOKE_MARKER = "synthetic-online-smoke-v1"


def smoke_config(config):
    """Return a cheap copy, retaining task splits, model names and price table."""
    out = deepcopy(config)
    out.setdefault("encoder", {}).update(backend="hash", device="cpu", hash_dim=32)
    out.setdefault("model", {}).update(hidden_dims=[16, 16], ensemble_size=2, dropout=0.0)
    out["teacher"]["llm"].update(max_calls_per_iteration=20, min_interval=0.)
    out.setdefault("optimization", {}).update(epochs=2, batch_size=8)
    out.setdefault("online", {}).update(iterations=2, tasks_per_iteration=2, states_per_iteration=2)
    out.setdefault("cold_start", {}).update(tasks=2, queries=1)
    out.setdefault("short_rollout", {}).update(horizon=1, max_horizon=1, adaptive=False)
    out.setdefault("evaluation", {}).update(every_iterations=1, max_tasks=2)
    out.setdefault("rollout", {}).update(trace_delay=0.0, retry_wait=0.0)
    out.setdefault("resilience", {})["qwen_min_interval"] = 0.0
    out.setdefault("budget", {})["total"] = 30.0
    return out


class SmokeEpisode:
    """Four deterministic synthetic checks; strong and weak actions differ."""
    tools = []
    system = ("SYNTHETIC OFFLINE SMOKE FIXTURE. This is not a native benchmark. "
              "Execute one synthetic check per reply. No real tool is used.")

    def __init__(self, task):
        # Only supplied metadata is read; no task files, references, or test index.
        self.task = dict(task)
        self.count = 0
        self.progress = 0.0
        self.score = 0.0
        self.closed = False

    def start(self):
        from mixed_pilot_harness.types import Message
        return [Message("user", "<task_description>SYNTHETIC SMOKE: finish four checks; "
                        "reach progress 3.5.</task_description>\n<initial_observation>"
                        "SYNTHETIC progress=0.0; step=0; goal=3.5; task=" + self.task["task_id"]
                        + "</initial_observation>")]

    async def advance(self, response):
        from mixed_pilot_harness.collection.episodes import Transition
        from mixed_pilot_harness.types import Message
        if self.closed:
            raise RuntimeError("Synthetic episode was already closed")
        if self.count >= 4:
            raise RuntimeError("Synthetic episode already terminated")
        if response.tool_calls:
            return Transition([Message("user", "SYNTHETIC tool calls are invalid")],
                              {"observation": "SYNTHETIC tool calls are invalid", "fixture": SMOKE_MARKER},
                              format_error=True)
        command = response.text.strip()
        self.count += 1
        self.progress += 1.0 if command == "SMOKE_STRONG_ADVANCE" else 0.25
        won = self.progress >= 3.5
        self.score = 100.0 if won else 0.0
        observation = "SYNTHETIC progress=%.2f; step=%d; goal=3.5" % (self.progress, self.count)
        return Transition([Message("user", observation)],
                          {"observation": observation, "command": command, "fixture": SMOKE_MARKER},
                          done=self.count == 4, status="synthetic_done" if self.count == 4 else "")

    async def evaluate(self, **kwargs):
        return {"status": "scored", "score": self.score, "success": self.score == 100,
                "cost": 0.0, "cost_complete": True, "api_calls": 0,
                "method": SMOKE_MARKER, "synthetic": True,
                "note": "Plumbing verification only; not native benchmark performance"}

    def close(self):
        self.closed = True


class SmokeProvider:
    provider_name = SMOKE_MARKER

    def __init__(self, profile, strong_model="glm-5.3"):
        self.profile = profile
        self.strong_model = strong_model
        self.calls = 0

    async def complete(self, messages, tools, system):
        from mixed_pilot_harness.types import ProviderResponse, Usage
        self.calls += 1
        if "training-only assessor" in str(system).lower():
            try:
                raw = messages[-1].content if messages else "{}"
                request = json.loads(raw)
            except (ValueError, TypeError, AttributeError):
                request = {}
            history = str(request.get("history", ""))
            matches = re.findall(r"SYNTHETIC progress=([0-9.]+); step=(\d+)", history)
            progress, step = (float(matches[-1][0]), int(matches[-1][1])) if matches else (0.0, 0)
            remaining = max(0, min(4 - step, int(request.get("remaining_steps", 4))))
            budget = max(0.0, float(request.get("remaining_budget", 1.0)))
            possible = progress + remaining >= 3.5
            cost = min(budget, remaining * .003)
            text = json.dumps({"expected_reward": .9 if possible else .1,
                               "remaining_agent_cost": cost, "assessment_status": "ok",
                               "reason": "Synthetic smoke forecast; not real LLM evidence"})
            if 'do not estimate or output C' in str(system):
                payload = json.loads(text)
                payload.pop('remaining_agent_cost')
                text = json.dumps(payload)
        else:
            name = self.profile.name or self.profile.model
            text = "SMOKE_STRONG_ADVANCE" if name == self.strong_model else "SMOKE_WEAK_OBSERVE"
        return ProviderResponse(text=text, model=self.profile.model, stop_reason="end_turn",
                                usage=Usage(input_tokens=200, output_tokens=50,
                                            raw={"fixture": SMOKE_MARKER}), cost_complete=True)


def smoke_provider_factory(profile, *, strong_model="glm-5.3", **kwargs):
    return SmokeProvider(profile, strong_model=strong_model)


def smoke_factories(config):
    """Return runtime injection factories honoring the configured A/B names."""
    return SmokeEpisode, partial(smoke_provider_factory, strong_model=config["strong_model"])


SMOKE_EPISODE_FACTORY = SmokeEpisode
SMOKE_PROVIDER_FACTORY = smoke_provider_factory
