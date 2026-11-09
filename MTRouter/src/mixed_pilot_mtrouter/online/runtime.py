"""Native online rollouts, verified prefix replay, and a unique-call ledger.

A replay is evidence-equivalent prefix restoration, not a claim that every
hidden simulator bit was cloned. Short horizon truncation never creates a
terminal failure label. Calls, including auxiliary teacher calls and failed
attempts, are persisted before the next environment action.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import time
from typing import Any


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class BudgetExceeded(RuntimeError):
    """No further billed call may be started under the collection threshold."""


class UnknownCostError(RuntimeError):
    """A request may have been billed but its complete fee is unknown."""


class ReplayError(RuntimeError):
    """The stored prefix did not reproduce its recorded public evidence."""


class RecoverableRequestError(RuntimeError):
    """Retries exhausted; accounting is reserved and this sample may be skipped."""


class InvalidTeacherEvidenceError(RecoverableRequestError):
    """A paid assessor response was truncated or did not satisfy its schema."""


def _ambiguity_mapping(expected, actual):
    """Only an exact permutation of unique numbered ScienceWorld choices is safe."""
    def parse(text):
        if not isinstance(text, str) or not text.startswith("Ambiguous request:"):
            return None
        lines = text.splitlines()
        choices, header = {}, []
        for line in lines:
            match = re.fullmatch(r"(\d+):\t(.+)", line)
            if match:
                if match[1] in choices or match[2] in choices.values():
                    return None
                choices[match[1]] = match[2]
            elif choices:
                return None
            else:
                header.append(line)
        return (header, choices) if choices else None
    left, right = parse(expected), parse(actual)
    if left is None or right is None or left[0] != right[0] or sorted(left[1].values()) != sorted(right[1].values()):
        return None
    inverse = {description: number for number, description in right[1].items()}
    return {number: inverse[description] for number, description in left[1].items()}


class Ledger:
    """One immutable identity per actual request, with recoverable call states.

    ``limit`` is a next-call stopping threshold, not an invoice guarantee: the
    final response can exceed it. In-flight or ambiguous requests found after a
    restart halt further spending instead of being silently counted as free.
    """

    def __init__(self, output, limit, currency):
        if isinstance(limit, bool) or not math.isfinite(float(limit)) or float(limit) <= 0:
            raise ValueError("Ledger limit must be finite and positive")
        if not isinstance(currency, str) or not currency:
            raise ValueError("Ledger requires an explicit currency")
        self.output = Path(output)
        self.directory = self.output / "calls"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit = float(limit)
        self.currency = currency
        self.records = {}
        for path in sorted(self.directory.glob("c*.json")):
            row = json.loads(path.read_text())
            if row.get("currency") != currency:
                raise ValueError("Cannot resume a ledger with a different currency")
            if row.get("call_id") != path.stem:
                raise ValueError("Call identity and filename disagree")
            if row.get("cost") is not None and (not math.isfinite(float(row["cost"])) or float(row["cost"]) < 0):
                raise ValueError("Persisted ledger has an invalid cost")
            if row.get("cost_complete") is True and row.get("cost") is None:
                raise ValueError("Persisted complete call is missing its cost")
            if "reserved_cost" in row and (not math.isfinite(float(row["reserved_cost"])) or row["reserved_cost"] < 0):
                raise ValueError("Persisted uncertain fee reservation is invalid")
            self.records[path.stem] = row
        numeric = [int(k[1:]) for k in self.records if re.fullmatch(r"c\d+", k)]
        self._next = max(numeric, default=-1) + 1
        self._write_summary()

    @property
    def cost(self):
        return sum(float(row.get("cost") or 0.0) for row in self.records.values())

    @property
    def cost_complete(self):
        return all(row.get("cost_complete") is True for row in self.records.values())

    @property
    def reserved_cost(self):
        return sum(float(row.get("reserved_cost", 0)) for row in self.records.values())

    @property
    def budget_cost(self):
        return self.cost + self.reserved_cost

    @property
    def spending_safe(self):
        return all(row.get("cost_complete") is True or
                   (row.get("status") != "in_flight" and
                    type(row.get("reserved_cost")) in (int, float) and
                    math.isfinite(row["reserved_cost"]) and row["reserved_cost"] > 0)
                   for row in self.records.values())

    def reserve(self, cid, upper_bound, basis):
        """Reserve uncertain fees without relabeling them as known or free."""
        if not math.isfinite(upper_bound) or upper_bound <= 0:
            raise UnknownCostError("Cannot safely bound uncertain request fees")
        row = self.records[cid]
        if row.get("status") == "in_flight":
            raise UnknownCostError("An active request cannot be reconciled automatically")
        if not row.get("cost_complete"):
            row["reserved_cost"] = max(float(row.get("reserved_cost", 0)), upper_bound)
            row["reservation_basis"] = basis
            _atomic(self.directory / (cid + ".json"), row)
            self._write_summary()

    @property
    def exhausted(self):
        return self.budget_cost >= self.limit

    def check(self):
        if not self.spending_safe:
            raise UnknownCostError("Ledger contains an in-flight or incompletely priced request; reconcile it before resuming")
        if self.exhausted:
            raise BudgetExceeded("Total collection budget threshold reached")

    def begin(self, scope, profile, metadata=None):
        self.check()
        cid = f"c{self._next:09d}"
        self._next += 1
        row = {"call_id": cid, "scope": str(scope), "profile": str(profile),
               "currency": self.currency, "started_at": _now(), "status": "in_flight",
               "cost": None, "cost_complete": False, "metadata": metadata or {}}
        self.records[cid] = row
        _atomic(self.directory / (cid + ".json"), row)
        self._write_summary()
        return cid

    def finish(self, cid, *, cost, complete, status="completed", **details):
        row = self.records[cid]
        if row["status"] != "in_flight":
            raise ValueError("A unique API call cannot be billed twice")
        if cost is not None and (not math.isfinite(float(cost)) or float(cost) < 0):
            raise ValueError("Invalid API call cost")
        if complete and cost is None:
            raise ValueError("A completely priced call requires an explicit cost")
        row.update(details)
        row.update(cost=None if cost is None else float(cost), cost_complete=bool(complete),
                   status=status, finished_at=_now())
        _atomic(self.directory / (cid + ".json"), row)
        self._write_summary()

    def report(self):
        scopes, profiles = {}, {}
        for row in self.records.values():
            for mapping, key in ((scopes, row["scope"]), (profiles, row["profile"])):
                stat = mapping.setdefault(key, {"known_cost": 0.0, "calls": 0, "uncertain_calls": 0})
                stat["known_cost"] += float(row.get("cost") or 0)
                stat["calls"] += 1
                stat["uncertain_calls"] += row.get("cost_complete") is not True
        return {"schema": "online-unique-call-ledger-v1", "currency": self.currency,
                "total_cost": self.cost, "total_api_cost": self.cost, "known_cost": self.cost,
                "reserved_cost": self.reserved_cost, "budget_cost": self.budget_cost,
                "spending_safe": self.spending_safe,
                "cost_complete": self.cost_complete, "calls": len(self.records),
                "unknown_calls": sum(r.get("cost_complete") is not True for r in self.records.values()),
                "limit": self.limit, "remaining": max(0.0, self.limit - self.budget_cost),
                "budget_exhausted": self.exhausted, "by_scope": scopes, "by_profile": profiles,
                "budget_semantics": "check before request; a final request may exceed the threshold"}

    def _write_summary(self):
        _atomic(self.output / "ledger.json", self.report())


def _message(value):
    from mixed_pilot_harness.types import Message, ToolCall
    if isinstance(value, Message):
        return value
    row = dict(value)
    row["tool_calls"] = [c if isinstance(c, ToolCall) else ToolCall(**c) for c in row.get("tool_calls", [])]
    return Message(**row)


def _response(value):
    from mixed_pilot_harness.types import ProviderResponse, ToolCall, Usage
    if isinstance(value, ProviderResponse):
        return value
    row = dict(value)
    row["usage"] = Usage(**row.get("usage", {}))
    row["tool_calls"] = [ToolCall(**c) for c in row.get("tool_calls", [])]
    return ProviderResponse(**row)


class Runner:
    """One-step interventions followed by a frozen caller-supplied policy."""

    def __init__(self, config, harness, pricing, ledger, *, episode_factory=None, provider_factory=None):
        self.config, self.harness, self.pricing, self.ledger = config, harness, pricing, ledger
        self.output = ledger.output
        self.episode_factory = episode_factory
        self.provider_factory = provider_factory
        self.providers = {}
        self.last_profile_call = {}
        self.pool = list(config.get("model_pool") or harness.profiles)
        rollout = config.get("rollout", {})
        self.task_budget = float(rollout.get("task_budget", 1.0))
        self.step_limit = int(rollout.get("step_limit", 50))
        self.replay_attempts = int(rollout.get("replay_attempts", 2))
        self.retry_wait = float(rollout.get("retry_wait", 0.0))
        self.trace_delay = float(rollout.get("trace_delay", 0.0))
        self.reward_mode = rollout.get("reward", "success")
        self.resilience = config.get("resilience", {})
        if not math.isfinite(self.task_budget) or self.task_budget <= 0 or min(self.step_limit, self.replay_attempts) <= 0:
            raise ValueError("Invalid task budget/step/replay limits")
        if self.retry_wait < 0 or self.trace_delay < 0 or self.reward_mode not in {"success", "normalized_score"}:
            raise ValueError("Invalid rollout options")
        if pricing.get("currency") != ledger.currency:
            raise ValueError("Pricing and ledger currencies disagree")

    def _episode(self, task):
        if self.episode_factory is not None:
            return self.episode_factory(task)
        from mixed_pilot_harness.collection.collector import load_prompt_config
        from mixed_pilot_harness.collection.episodes import ScienceWorldEpisode
        if task["benchmark"] == "alfworld":
            from mixed_pilot_harness.inference.virtualworld import ALFWorldEpisode
            return ALFWorldEpisode(task, load_prompt_config())
        if task["benchmark"] == "scienceworld":
            from mixed_pilot_harness.collection.upstream.scienceworld import ScienceWorldEnvironment
            return ScienceWorldEpisode(task, load_prompt_config(), ScienceWorldEnvironment)
        raise ValueError("Online native runtime only supports ScienceWorld and ALFWorld")

    def _reserve_uncertain(self, cid):
        row = self.ledger.records[cid]
        request = row.get("request")
        generation = row.get("generation", {})
        if request is None or not generation.get("max_tokens"):
            raise UnknownCostError("Missing persisted request needed to reserve " + cid)
        # Byte-level tokens cannot exceed UTF-8 bytes. Double the full untrimmed
        # public request plus framing allowance; ignore caching and use peak rates.
        input_bound = 2 * len(_canonical(request).encode("utf-8")) + 4096
        rates = self.pricing["profiles"][row["profile"]]
        bands = rates.get("bands", [rates])
        upper = (input_bound * max(b["input_per_million"] for b in bands)
                 + generation["max_tokens"] * max(b["output_per_million"] for b in bands)) / 1e6
        uncertain = sum(a.get("status") == "error" and not a.get("cost_known_zero", False)
                        for a in row.get("api_attempts", []))
        count = max(1, uncertain)
        self.ledger.reserve(cid, max(upper * count, 1e-9), {
            "method": "conservative_uncertain_attempt_reservation", "attempts": count,
            "input_token_upper_bound": input_bound, "output_token_limit": generation["max_tokens"],
            "known_actual_fee": False})

    def recover_uncertain_calls(self):
        for cid, row in self.ledger.records.items():
            if row.get("cost_complete") or row.get("reserved_cost", 0) > 0:
                continue
            if row["status"] == "in_flight":
                self.ledger.finish(cid, cost=row.get("cost"), complete=False, status="interrupted")
            self._reserve_uncertain(cid)

    async def complete(self, profile, messages, system="", tools=None, scope="teacher", metadata=None, profile_overrides=None):
        if not self.resilience.get("enabled", False):
            return await self._complete_once(profile, messages, system, tools, scope, metadata, profile_overrides)
        from mixed_pilot_harness.providers import HTTP_RETRY_POLICY, _is_retryable_transport_error, _error_chain
        from mixed_pilot_harness.types import ProviderError
        # Each real attempt gets its own ledger ID; disable nested HTTP retries.
        token = HTTP_RETRY_POLICY.set((0, 0))
        start = self.ledger.budget_cost
        overrides = dict(profile_overrides or {})
        overrides.setdefault("timeout", self.resilience["teacher_timeout" if scope == "teacher" else "request_timeout"])
        try:
            total_retries = max(self.resilience["max_retries"], self.resilience.get("rate_limit_retries", 0))
            for attempt in range(total_retries + 1):
                previous_ids = set(self.ledger.records)
                try:
                    if "qwen" in profile.lower():
                        delay = self.resilience.get("qwen_min_interval", 0) - (time.monotonic() - self.last_profile_call.get(profile, -float("inf")))
                        if delay > 0:
                            await asyncio.sleep(delay)
                        self.last_profile_call[profile] = time.monotonic()
                    result = await self._complete_once(profile, messages, system, tools, scope, metadata, overrides)
                    result["cost"] = self.ledger.budget_cost - start
                    return result
                except (UnknownCostError, ProviderError, OSError) as exc:
                    new_ids = set(self.ledger.records) - previous_ids
                    if not new_ids:
                        raise
                    cid = next(iter(new_ids))
                    row = self.ledger.records[cid]
                    if not row.get("cost_complete"):
                        self._reserve_uncertain(cid)
                    if row.get("response") is not None and row.get("cost") is not None:
                        return {"response": _response(row["response"]), "cost": self.ledger.budget_cost - start,
                                "call_id": cid}
                    cause = exc.__cause__ or exc
                    rate_limited = any(getattr(item, "code", None) == 429 for item in _error_chain(cause))
                    retryable = _is_retryable_transport_error(cause)
                    if not retryable:
                        if self.resilience.get("stop_after_retries", False):
                            raise RecoverableRequestError("Non-retryable provider failure for " + cid + ": " + str(exc)) from exc
                        raise
                    allowed_retries = self.resilience.get("rate_limit_retries", self.resilience["max_retries"]) if rate_limited else self.resilience["max_retries"]
                    if attempt >= allowed_retries:
                        raise RecoverableRequestError("Request retries exhausted for " + cid) from exc
                    self.ledger.check()
                    base_delay = self.resilience.get("rate_limit_wait", self.resilience["retry_delay"]) if rate_limited else self.resilience["retry_delay"]
                    delay = min(self.resilience["max_retry_delay"], base_delay * 2**attempt)
                    if rate_limited:
                        for item in _error_chain(cause):
                            header = getattr(item, "headers", None)
                            retry_after = header.get("Retry-After") if header else None
                            if retry_after:
                                try:
                                    seconds = float(retry_after)
                                    if math.isfinite(seconds):
                                        delay = max(delay, min(600, seconds))
                                except ValueError:
                                    pass
                    from .teacher import append_jsonl
                    append_jsonl(self.output / "request_retries.jsonl", {"call_id": cid, "scope": scope,
                        "profile": profile, "retry": attempt + 1, "wait_seconds": delay,
                        "error": str(exc), "error_type": type(cause).__name__, "rate_limited": rate_limited, "time": _now()})
                    print(_canonical({"event": "request_retry", "call_id": cid, "scope": scope,
                                      "wait_seconds": delay}), flush=True)
                    await asyncio.sleep(delay)
        finally:
            HTTP_RETRY_POLICY.reset(token)

    async def _complete_once(self, profile, messages, system="", tools=None, scope="teacher", metadata=None, profile_overrides=None):
        """Return ``{response: ProviderResponse, cost: float, call_id: str}``.

        All assistant/teacher calls use this method, including invalid responses;
        a failure after request dispatch remains a persisted uncertain charge.
        """
        from mixed_pilot_harness.collection.pricing import estimate_cost
        from mixed_pilot_harness.providers import create_provider
        p = self.harness.profile(profile)
        if scope != "teacher" and profile not in self.pool:
            raise ValueError("Actor chose a model outside the configured A/B pool")
        if p.kind == "router":
            raise ValueError("Online actor and teacher profiles must be concrete providers")
        overrides = {}
        if scope == "teacher":
            llm = self.config.get("teacher", {}).get("llm", {})
            overrides.update({k: llm[k] for k in ("max_tokens", "temperature", "timeout") if k in llm and llm[k] is not None})
        overrides.update(profile_overrides or {})
        allowed = {"max_tokens", "temperature", "timeout"}
        if scope == "teacher":
            allowed.add("extra_body")
        if set(overrides) - allowed:
            raise ValueError("Only generation limits can be overridden")
        if "extra_body" in overrides:
            if (not isinstance(overrides["extra_body"], dict) or
                    set(overrides["extra_body"]) - {"thinking", "reasoning_effort", "response_format"}):
                raise ValueError("Teacher may only override reasoning effort and JSON output format")
            overrides["extra_body"] = {**p.extra_body, **overrides["extra_body"]}
        if overrides:
            p = replace(p, **overrides)
        rates = self.pricing.get("profiles", {}).get(profile)
        if rates is None:
            raise ValueError("Missing price for " + str(profile))
        key = (profile, _canonical(overrides))
        if key not in self.providers:
            factory = self.provider_factory or create_provider
            if self.provider_factory is None and p.kind == "online-responses-teacher":
                from .responses import ResponsesTeacherProvider
                factory = lambda profile, **kwargs: ResponsesTeacherProvider(profile, self.config["teacher"]["llm"]["min_interval"])
            self.providers[key] = factory(p, config_dir=self.harness.path.parent, profiles=self.harness.profiles)
        cid = self.ledger.begin(scope, profile, metadata)
        row = self.ledger.records[cid]
        try:
            normalized_messages = [_message(m) for m in messages]
            row.update(request={"messages": [asdict(m) for m in normalized_messages], "system": str(system),
                                "tools": [asdict(t) if hasattr(t, "__dataclass_fields__") else t for t in (tools or [])]},
                       model=p.model, generation={"max_tokens": p.max_tokens, "temperature": p.temperature,
                                                "timeout": p.timeout,
                                                "extra_body": dict(p.extra_body)})
            _atomic(self.ledger.directory / (cid + ".json"), row)
        except Exception as exc:
            self.ledger.finish(cid, cost=0.0, complete=True, status="request_error", error=str(exc))
            raise
        try:
            response = await self.providers[key].complete(normalized_messages, tools or [], system)
        except BaseException as exc:
            known_zero = bool(getattr(exc, "cost_known_zero", False))
            self.ledger.finish(cid, cost=0.0 if known_zero else None, complete=known_zero,
                               status="provider_error", error=f"{type(exc).__name__}: {exc}",
                               api_attempts=getattr(exc, "api_attempts", []))
            if not known_zero and isinstance(exc, Exception):
                raise UnknownCostError("Provider failed after dispatch; fee is uncertain for " + cid) from exc
            raise
        # Store the complete reply before any parsing, pricing, or environment use.
        row["response"] = asdict(response)
        _atomic(self.ledger.directory / (cid + ".json"), row)
        try:
            cost, details = estimate_cost(response.usage, rates, row["started_at"])
        except Exception as exc:
            self.ledger.finish(cid, cost=None, complete=False, status="pricing_error", error=str(exc))
            raise UnknownCostError("Cannot price response " + cid) from exc
        self.ledger.finish(cid, cost=cost, complete=response.cost_complete,
                           status="completed" if response.cost_complete else "incomplete_cost",
                           cost_details=details, api_attempts=response.api_attempts)
        if not response.cost_complete:
            raise UnknownCostError("Earlier transport attempt may have been billed for " + cid)
        return {"response": response, "cost": cost, "call_id": cid}

    def _state(self, task, episode, history, steps, policy_version):
        spent = sum(float(s["cost"]) for s in steps)
        public = {"task_id": task["task_id"], "benchmark": task["benchmark"],
                  "split": task.get("split", "train"), "history": [asdict(m) for m in history],
                  "system": episode.system, "step_idx": len(steps), "spent_cost": spent,
                  "remaining_budget": max(0.0, self.task_budget - spent),
                  "remaining_steps": max(0, self.step_limit - len(steps)),
                  "policy_version": policy_version,
                  "task_fingerprint": _hash({k: task.get(k) for k in ("task_id", "benchmark", "content_hashes", "source_id", "task_name", "variation_idx")})}
        public["snapshot_id"] = _hash({"task_id": task["task_id"], "benchmark": task["benchmark"],
                                       "task_fingerprint": public["task_fingerprint"],
                                       "history": public["history"], "system": public["system"],
                                       "step_idx": len(steps), "spent_cost": spent,
                                       "policy_version": policy_version})
        return public

    async def _restore(self, task, prefix):
        failures = []
        for attempt in range(self.replay_attempts):
            episode = None
            try:
                episode = self._episode(task)
                history = [_message(m) for m in episode.start()]
                initial_hash = _hash([asdict(m) for m in history])
                if prefix:
                    expected = prefix[0].get("initial_history_hash")
                    if expected and expected != initial_hash:
                        raise ReplayError("Initial public history does not match saved prefix")
                for i, step in enumerate(prefix):
                    before = step.get("before_state", {})
                    if before.get("task_id", task["task_id"]) != task["task_id"] or before.get("benchmark", task["benchmark"]) != task["benchmark"]:
                        raise ReplayError("Prefix belongs to a different task")
                    expected_task = _hash({k: task.get(k) for k in ("task_id", "benchmark", "content_hashes", "source_id", "task_name", "variation_idx")})
                    if before.get("task_fingerprint", expected_task) != expected_task:
                        raise ReplayError("Native task or its content hashes changed")
                    if before.get("history") is not None and before["history"] != [asdict(m) for m in history]:
                        raise ReplayError("History mismatch before " + step["call_id"])
                    if before.get("system") is not None and before["system"] != episode.system:
                        raise ReplayError("System prompt mismatch before " + step["call_id"])
                    response = _response(step["response"])
                    history.append(response.assistant_message())
                    transition = await episode.advance(response)
                    actual = {"record": transition.record, "score": float(episode.score),
                              "done": bool(transition.done), "format_error": bool(transition.format_error),
                              "status": transition.status}
                    if actual != step["transition"] and task["benchmark"] == "scienceworld":
                        wanted = step["transition"]
                        expected_text = wanted.get("record", {}).get("observation")
                        actual_text = actual.get("record", {}).get("observation")
                        mapping = _ambiguity_mapping(expected_text, actual_text)
                        candidate = {**actual, "record": {**actual["record"], "observation": expected_text}}
                        if mapping is not None and candidate == wanted and all(
                                isinstance(m.content, str) and actual_text in m.content for m in transition.messages):
                            # Preserve the recorded public numbering AND remap the
                            # next numeric action to its actual semantic choice.
                            episode._replay_choice_mapping = mapping
                            transition.messages = [replace(m, content=m.content.replace(actual_text, expected_text))
                                                   for m in transition.messages]
                            actual = candidate
                            failures.append({"attempt": attempt + 1, "call_id": step["call_id"],
                                             "ambiguity_choice_permutation_restored": True})
                    history.extend(_message(m) for m in transition.messages)
                    if actual != step["transition"]:
                        raise ReplayError("Transition evidence mismatch at " + step["call_id"])
                    if transition.done and i + 1 != len(prefix):
                        raise ReplayError("Saved prefix continues beyond native termination")
                return episode, history, initial_hash, failures
            except Exception as exc:
                if episode is not None:
                    episode.close()
                failures.append({"attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}"})
                if attempt + 1 < self.replay_attempts and self.retry_wait:
                    await asyncio.sleep(self.retry_wait)
        raise ReplayError(_canonical(failures))

    async def rollout(self, task, policy, *, prefix=None, forced_first=None, horizon=None,
                      scope="collection", policy_version=0, trajectory_id=None):
        prefix = list(prefix or [])
        if horizon is not None and (isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1):
            raise ValueError("horizon must be a positive number of NEW API calls")
        for step in prefix:
            if step.get("cost") is None or not math.isfinite(float(step["cost"])) or step["cost"] < 0:
                raise ValueError("Cannot replay an incompletely priced prefix")
        trajectories = self.output / "trajectories"
        trajectories.mkdir(parents=True, exist_ok=True)
        token = str(trajectory_id) if trajectory_id is not None else "t%09d" % len(list(trajectories.glob("*.json")))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", token):
            raise ValueError("Unsafe trajectory identity")
        path = trajectories / (token + ".json")
        if path.exists():
            raise ValueError("Trajectory already exists: " + token)
        result = {"schema": "online-rollout-v1", "trajectory_id": token, "task_id": task["task_id"],
                  "benchmark": task["benchmark"], "split": task.get("split", "train"),
                  "scope": scope, "policy_version": policy_version, "forced_first": forced_first,
                  "horizon": horizon, "prefix": prefix, "steps": [], "all_steps": list(prefix),
                  "state": None, "terminal": False, "reward": None, "success": None,
                  "labels_valid": False, "outcome_valid": False, "outcome_reward": None, "status": "initializing", "cost": 0.0,
                  "path_cost": sum(s["cost"] for s in prefix), "cost_complete": True,
                  "started_at": _now(), "replay": []}
        start_cost = self.ledger.cost
        call_ids_before = set(self.ledger.records)
        episode = None
        history = []
        initial_hash = None
        native_done = bool(prefix and prefix[-1]["transition"]["done"])
        raised = None
        try:
            self.ledger.check()
            episode, history, initial_hash, failures = await self._restore(task, prefix)
            result["replay"] = failures + [{"matched": True, "prefix_steps": len(prefix),
                                            "semantics": "exact public history, transition, native score and done"}]
            result["status"] = "running"
            while True:
                all_steps = result["all_steps"]
                state = self._state(task, episode, history, all_steps, policy_version)
                result["state"] = state
                if native_done:
                    result["terminal"] = True
                    result["status"] = all_steps[-1]["transition"]["status"] or "native_done"
                    break
                if state["remaining_steps"] <= 0:
                    result.update(terminal=True, status="step_limit")
                    break
                if state["remaining_budget"] <= 0:
                    result.update(terminal=True, status="task_budget")
                    break
                self.ledger.check()
                if horizon is not None and len(result["steps"]) >= horizon:
                    result["status"] = "truncated"
                    break
                choice = forced_first if not result["steps"] and forced_first is not None else policy(state)
                if inspect.isawaitable(choice):
                    choice = await choice
                if not isinstance(choice, str):
                    raise ValueError("Policy must return a concrete profile name")
                complete = await self.complete(choice, history, system=episode.system, tools=episode.tools,
                                               scope=scope, metadata={"trajectory_id": token,
                                               "task_id": task["task_id"], "split": task.get("split", "train"),
                                               "step_idx": len(all_steps), "policy_version": policy_version,
                                               "snapshot_id": state["snapshot_id"]})
                response = complete["response"]
                step = {"call_id": complete["call_id"], "profile": choice, "cost": complete["cost"],
                        "before_state": state, "response": asdict(response),
                        "initial_history_hash": initial_hash, "transition": None}
                result["steps"].append(step)
                result["all_steps"].append(step)
                _atomic(path, result)
                history.append(response.assistant_message())
                transition = await episode.advance(response)
                history.extend(_message(m) for m in transition.messages)
                step["transition"] = {"record": transition.record, "score": float(episode.score),
                                      "done": bool(transition.done), "format_error": bool(transition.format_error),
                                      "status": transition.status}
                native_done = bool(transition.done)
                result["state"] = self._state(task, episode, history, result["all_steps"], policy_version)
                result["cost"] = self.ledger.cost - start_cost
                result["path_cost"] = sum(s["cost"] for s in result["all_steps"])
                _atomic(path, result)
                recent = result["all_steps"][-3:]
                if len(recent) == 3 and all(not s["response"].get("text", "").strip() and not s["response"].get("tool_calls") for s in recent):
                    result.update(terminal=True, status="empty_response_limit")
                    break
            if result["terminal"]:
                evaluation = await episode.evaluate()
                if result["status"] == "empty_response_limit":
                    evaluation = {"status": "scored", "score": 0.0, "success": False,
                                  "cost": 0.0, "cost_complete": True, "method": "empty_response_limit"}
                if evaluation.get("api_calls", 0) or evaluation.get("cost", 0):
                    raise ValueError("Native VirtualWorld evaluation must not invoke an unledgered API")
                result["evaluation"] = evaluation
                valid = evaluation.get("status") == "scored" and evaluation.get("cost_complete") is True
                if valid:
                    score = float(evaluation["score"])
                    success = bool(evaluation["success"])
                    reward = float(success) if self.reward_mode == "success" else ((score + 100) / 200 if task["benchmark"] == "scienceworld" else score / 100)
                    if result["status"] == "empty_response_limit":
                        reward = 0.0
                    if not math.isfinite(reward) or not 0 <= reward <= 1:
                        raise ValueError("Invalid native terminal reward")
                    result.update(reward=reward, success=success, final_score=score, labels_valid=True,
                                  outcome_valid=True, outcome_reward=reward)
                else:
                    result["status"] = "evaluation_error"
        except BudgetExceeded as exc:
            result.update(status="global_budget", error=str(exc), terminal=False, labels_valid=False, reward=None)
        except UnknownCostError as exc:
            result.update(status="unknown_cost", error=str(exc), terminal=False, labels_valid=False, reward=None)
        except ReplayError as exc:
            result.update(status="replay_failed", error=str(exc), terminal=False, labels_valid=False, reward=None)
        except BaseException as exc:
            result.update(status="interrupted" if not isinstance(exc, Exception) else "runtime_error",
                          error=f"{type(exc).__name__}: {exc}", labels_valid=False, reward=None)
            if not isinstance(exc, Exception) or (isinstance(exc, RecoverableRequestError)
                                                  and self.resilience.get("stop_after_retries", False)):
                raised = exc
        finally:
            if episode is not None:
                try:
                    episode.close()
                except Exception as exc:
                    result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            new_calls = [k for k in self.ledger.records if k not in call_ids_before]
            result.update(cost=self.ledger.cost - start_cost, call_ids=new_calls,
                          cost_complete=all(self.ledger.records[k].get("cost_complete") is True for k in new_calls),
                          path_cost=sum(s["cost"] for s in result["all_steps"]), finished_at=_now())
            result["labels_valid"] = bool(result["labels_valid"] and result["cost_complete"])
            if not result["labels_valid"]:
                result["reward"] = None
            _atomic(path, result)
        if raised is not None:
            raised.rollout_result = result
            raise raised
        if result["terminal"] and result["labels_valid"] and self.trace_delay:
            await asyncio.sleep(self.trace_delay)
        return result
