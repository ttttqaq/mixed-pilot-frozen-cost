"""Judge one collection API step without exposing future trajectory outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..chat import chat_once
from ..config import HarnessConfig
from .prompt import PROMPT_TEMPLATE, PROMPT_VERSION, SYSTEM_PROMPT


def _message(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("role") not in {"user", "assistant", "tool", "system"}:
        raise ValueError("trace history contains an invalid message")
    if not isinstance(value.get("content"), str):
        raise ValueError("trace history message content must be text")
    # Drop model identity, usage and routing metadata from the judge's input.
    return {k: value[k] for k in ("role", "content", "tool_calls", "tool_call_id", "name") if k in value}


def prepare_judgment(trace_path: Path, step: int, *, reference_answer: Optional[str] = None) -> dict:
    """Prepare a prompt from one .traj.json. ``step`` is ONE-based.

    This reads only the target step's input history and recorded transition into
    the prompt. Terminal score, success, later messages and later steps are never
    used as answers. Missing ground truth falls back to the recorded task goal.
    """
    path = Path(trace_path).resolve()
    raw = path.read_bytes()
    trace = json.loads(raw)
    if not isinstance(trace, dict) or not isinstance(trace.get("info"), dict) or not isinstance(trace.get("steps"), list):
        raise ValueError("expected a collection .traj.json with info and steps (not events.jsonl or steps.jsonl)")
    info, steps = trace["info"], trace["steps"]
    if type(step) is not int or not 1 <= step <= len(steps):
        raise ValueError("step must be between 1 and %d (one-based)" % len(steps))
    target = steps[step - 1]
    if not isinstance(target, dict) or target.get("step_idx") != step - 1:
        raise ValueError("trace step_idx does not match the selected one-based step")
    state = target.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("history"), list) or not state["history"]:
        raise ValueError("selected step is missing its pre-step state.history")
    history = [_message(m) for m in state["history"]]
    if sum(m["role"] == "assistant" for m in history) != step - 1:
        raise ValueError("pre-step history has the wrong number of assistant messages")
    if not isinstance(target.get("response"), str):
        raise ValueError("selected step has no recorded model response")
    if "observation" in target and not isinstance(target["observation"], str):
        raise ValueError("selected step observation must be text")
    if "tool_results" in target and not isinstance(target["tool_results"], list):
        raise ValueError("selected step tool_results must be a list")
    first_user = next((m["content"] for m in history if m["role"] == "user"), "")
    if not first_user:
        raise ValueError("pre-step history has no task input")
    match = re.search(r"<task_description>\s*(.*?)\s*</task_description>", first_user, re.DOTALL)
    task_goal = match.group(1) if match else first_user
    if reference_answer is not None and (not isinstance(reference_answer, str) or not reference_answer.strip()):
        raise ValueError("reference answer must be nonempty text")
    observations = [m for m in history if m["role"] in {"user", "tool"}]
    final_answer = {
        "source": "provided_reference" if reference_answer is not None else "task_goal_only",
        "reference_answer": reference_answer,
        "completion_criteria": task_goal,
    }
    context = {
        "benchmark": info.get("benchmark"),
        "task_name": info.get("task_name"),
        "variation_idx": info.get("variation_idx"),
        "simplification_str": info.get("simplification_str"),
        "task_goal": task_goal,
        "agent_system": state.get("system", ""),
        "available_tools": state.get("tools", []),
        "current_environment": {"source": "latest_pre_step_observation", "message": observations[-1]},
        "final_answer": final_answer,
    }
    candidate = {
        "step": step, "step_idx": step - 1,
        "model_response": target["response"],
        "command": target.get("command"),
        "tool_calls": target.get("tool_calls", []),
        "environment_feedback": {
            "observation": target.get("observation"),
            "tool_results": target.get("tool_results", []),
            "recorded": "observation" in target or "tool_results" in target,
        },
        "format_error": target.get("format_error", False),
    }
    dump = lambda value: json.dumps(value, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.substitute(context_json=dump(context), history_json=dump(history), step_json=dump(candidate))
    return {
        "schema_version": "1.0", "status": "prepared", "prompt_version": PROMPT_VERSION,
        "trace": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "trace_id": info.get("trace_id", path.stem), "task_id": info.get("task_id"),
                  "run_id": info.get("run_id"), "signature": info.get("signature"),
                  "selected_profile": target.get("selected_profile"),
                  "returned_model": target.get("returned_model")},
        "step": step, "step_idx": step - 1,
        "reference_source": final_answer["source"],
        "request": {"system": SYSTEM_PROMPT, "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()},
    }


def parse_judgment(text: str) -> dict:
    """Accept a JSON object or one complete JSON fence; reject partial scores."""
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    value = json.loads(fence.group(1) if fence else text)
    if not isinstance(value, dict):
        raise ValueError("judge must return a JSON object")
    for key, maximum in (("score", 100), ("success_probability", 1), ("confidence", 1)):
        number = value.get(key)
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= maximum:
            raise ValueError("judge %s must be a finite number in [0,%s]" % (key, maximum))
    if "estimated_remaining_steps" not in value:
        raise ValueError("judge must include estimated_remaining_steps")
    remaining = value["estimated_remaining_steps"]
    if remaining is not None and (type(remaining) is not int or remaining < 0):
        raise ValueError("estimated_remaining_steps must be a nonnegative integer or null")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("judge must include a nonempty reason")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(not isinstance(e, str) or not e.strip() for e in evidence):
        raise ValueError("judge must include a nonempty list of evidence strings")
    if value["score"] == 100 and (value["success_probability"] != 1 or remaining != 0):
        raise ValueError("score=100 requires success_probability=1 and estimated_remaining_steps=0")
    return {key: value[key] for key in ("score", "success_probability", "estimated_remaining_steps", "confidence", "reason", "evidence")}


async def judge_step(
    trace_path: Path, step: int, *, config: HarnessConfig, profile: str,
    reference_answer: Optional[str] = None,
) -> dict:
    """One logical chat call; return auditable output, including invalid replies.

    Caller checks ``status == 'scored'``. Provider transport failures propagate
    through the existing harness error handling; no extra scoring retries occur.
    """
    result = prepare_judgment(trace_path, step, reference_answer=reference_answer)
    if not profile:
        raise ValueError("select an explicit judge profile")
    selected = config.profile(profile)
    result["judge"] = {"profile": profile, "requested_model": selected.model,
                       "kind": selected.kind, "config_path": str(config.path),
                       "temperature": selected.temperature, "max_tokens": selected.max_tokens}
    result["started_at"] = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    response = await chat_once(config, result["request"]["prompt"], profile_name=profile,
                               system=result["request"]["system"])
    result.update(latency_seconds=time.monotonic() - start,
                  finished_at=datetime.now(timezone.utc).isoformat(), response=asdict(response))
    try:
        if response.tool_calls or response.stop_reason in {"length", "max_tokens", "content_filter"}:
            raise ValueError("judge response is incomplete, filtered, or contains tool calls")
        result.update(status="scored", judgment=parse_judgment(response.text))
    except (ValueError, TypeError) as exc:
        result.update(status="invalid_response", judgment=None, error=str(exc))
    return result
