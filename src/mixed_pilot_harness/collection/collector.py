"""Fixed/random full-Trace collection adapted from MTRouter's Collector.

Uses benchmark adapters for ScienceWorld and local HLE tasks with this
repository's Provider API. No router weights or training packages are imported.
See upstream/PROVENANCE.json and upstream/LICENSE.md for attribution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

from ..config import HarnessConfig
from ..providers import create_provider
from ..types import Message
from . import UPSTREAM_REVISION
from .policy import SamplingPolicy
from .pricing import estimate_cost
from .episodes import HLEEpisode, HLE_JUDGE_SYSTEM, HLE_SYSTEM, ScienceWorldEpisode
from .upstream.scienceworld import ScienceWorldEnvironment


MAX_CONSECUTIVE_EMPTY_RESPONSES = 3


@dataclass
class CollectionSettings:
    mode: str
    model_pool: list[str]
    profile: str | None = None
    runs: int = 1
    max_steps: int = 50
    cost_limit: float = 0.5
    total_cost_limit: float = 10.0
    seed: int = 42
    benchmark: str = "scienceworld"
    hle_evaluator: str = "judge"
    judge_profile: str | None = None
    tool_timeout: float = 30.0


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_prompt_config() -> dict:
    import yaml
    return yaml.safe_load((Path(__file__).parent / "upstream/scienceworld.yaml").read_text())["agent"]


def configuration(config: HarnessConfig, settings: CollectionSettings, pricing: dict) -> dict:
    # Never persist credentials or custom authentication headers.
    profiles = {}
    names = list(dict.fromkeys(settings.model_pool + ([settings.judge_profile] if settings.judge_profile else [])))
    for name in names:
        p = config.profile(name)
        profiles[name] = {k: getattr(p, k) for k in (
            "kind", "model", "base_url", "endpoint", "max_tokens", "temperature", "timeout", "extra_body")}
    try:
        version = metadata.version("scienceworld")
    except metadata.PackageNotFoundError:
        version = "not-installed"
    return {"settings": asdict(settings), "profiles": profiles, "pricing": pricing,
            "scienceworld_version": version, "upstream_revision": UPSTREAM_REVISION,
            "prompt_config": (load_prompt_config() if settings.benchmark == "scienceworld" else
                              {"system_template": HLE_SYSTEM, "judge_system": HLE_JUDGE_SYSTEM}),
            "collector_schema": "2.0"}


def fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def episode_signature(spec: dict, task: dict, run_id: int) -> str:
    # More planned repeats/budget must not re-bill completed repeats.
    episode_spec = {**spec, "settings": {k: v for k, v in spec["settings"].items()
                                        if k not in {"runs", "total_cost_limit"}}}
    return fingerprint({"spec": episode_spec, "task": task, "run_id": run_id})


def compatible_token_upgrade(previous: dict, spec: dict, task: dict, run_id: int) -> bool:
    """Allow only larger max_tokens; retain the original trace and configuration."""
    old = previous.get("info", {}).get("config", {})
    if set(old.get("profiles", {})) != set(spec["profiles"]):
        return False
    adjusted = json.loads(json.dumps(spec))
    for name, profile in old["profiles"].items():
        before = profile.get("max_tokens", 0)
        if before <= 0 or before > spec["profiles"][name]["max_tokens"]:
            return False
        adjusted["profiles"][name]["max_tokens"] = before
    return previous["info"].get("signature") == episode_signature(adjusted, task, run_id)


def step_samples(trace: dict):
    """Single observations of return/cost-to-go, NOT repeated-state means.

    Failed/API-incomplete episodes are excluded; their financial diagnostics
    remain in the trace. Native terminal score and binary success are separate.
    """
    info = trace["info"]
    if info.get("superseded") or not info.get("labels_valid") or not info.get("cost_complete"):
        return
    remaining = 0.0
    tails = []
    for step in reversed(trace["steps"]):
        remaining += step["cost"]
        tails.append(remaining)
    tails.reverse()
    for step, cost in zip(trace["steps"], tails):
        yield {
            "trace_id": info["trace_id"], "task_id": info["task_id"], "split": info["split"],
            "benchmark": info.get("benchmark", "scienceworld"),
            "source_split": info.get("source_split", info["split"]),
            "run_id": info["run_id"], "step_idx": step["step_idx"],
            "state": step["state"], "action": step["selected_profile"],
            "requested_model": step["requested_model"], "returned_model": step["returned_model"],
            "propensity": step["propensity"], "terminal_score": info["final_score"],
            "terminal_success": info["success"], "step_cost": step["cost"],
            "remaining_cost": cost, "cost_currency": info["cost_currency"],
            "continuation_policy": info["policy"], "label_kind": "single_rollout",
            **({'router_profile':step.get('router_profile'), 'router_decision':step['router_decision']}
               if 'router_decision' in step else {}),
        }


async def collect_episode(task: dict, run_id: int, config: HarnessConfig,
                          settings: CollectionSettings, pricing: dict, spec: dict,
                          path: Path, *, env_factory: Callable = ScienceWorldEnvironment,
                          provider_factory: Callable = create_provider) -> dict:
    signature = episode_signature(spec, task, run_id)
    seed = int(signature[:16], 16) ^ settings.seed
    policy = SamplingPolicy(settings.model_pool, settings.mode, seed, settings.profile)
    trace = {"info": {
        **task, "trace_id": path.stem, "run_id": run_id, "signature": signature,
        "config": spec, "started_at": utcnow(), "exit_status": "Running",
        "policy": {"mode": settings.mode, "pool": settings.model_pool, "profile": settings.profile},
        "cost_currency": pricing["currency"], "cost_basis": "token_estimate",
        "cost_complete": True, "labels_valid": False,
    }, "messages": [], "steps": []}
    info = trace["info"]
    episode = None
    providers = {}
    total_cost = 0.0
    evaluation = {"status": "not_run", "cost": 0.0, "cost_complete": True, "api_calls": 0}
    final_score, success = None, None
    format_errors = attempts = 0
    consecutive_empty_responses = 0
    model_history = []
    status = "EnvironmentError"
    atomic_json(path, trace)
    try:
        if settings.benchmark == "hle":
            episode = HLEEpisode(task, path, settings.tool_timeout)
            info["workspace"] = str(episode.workspace.resolve())
        else:
            episode = ScienceWorldEpisode(task, spec["prompt_config"], env_factory)
        system = episode.system
        messages = episode.start()
        trace["messages"] = [{"role": "system", "content": system}] + [asdict(m) for m in messages]
        info["tools"] = [asdict(tool) for tool in episode.tools]
        atomic_json(path, trace)
        status = "LimitsExceeded"
        for index in range(settings.max_steps):
            if total_cost >= settings.cost_limit:
                status = "CostLimitExceeded"
                break
            profile_name, propensity = policy.choose()
            profile = config.profile(profile_name)
            if profile_name not in providers:
                providers[profile_name] = provider_factory(
                    profile, config_dir=config.path.parent, profiles=config.profiles)
            state = {"history": [asdict(m) for m in messages], "system": system,
                     "previous_actions": list(model_history), "spent_cost": total_cost,
                     "remaining_budget": settings.cost_limit - total_cost}
            if episode.tools:
                state["tools"] = info["tools"]
            started_at = utcnow()
            timer = time.monotonic()
            attempts += 1
            try:
                response = await providers[profile_name].complete(messages, episode.tools, system)
            except Exception as exc:
                # A timeout does not prove the upstream request was free.
                info["cost_complete"] = False
                info["failure"] = {"type": type(exc).__name__, "message": str(exc),
                                   "selected_profile": profile_name, "step_idx": index,
                                   "started_at": started_at}
                status = "ProviderError"
                break
            info["cost_complete"] &= response.cost_complete
            step = {"cost_complete":response.cost_complete,"api_attempts":response.api_attempts,
                    "context_management":response.context_management,"step_idx": index, "state": state, "selected_profile": profile_name,
                    "requested_model": profile.model, "returned_model": response.model,
                    "propensity": propensity, "started_at": started_at,
                    "latency_seconds": time.monotonic() - timer,
                    "response": response.text, "stop_reason": response.stop_reason,
                    "reasoning_content": response.reasoning_content, "reasoning_owner": response.reasoning_owner,
                    "usage": asdict(response.usage), "cost": None,
                    "format_error": False}
            if response.tool_calls:
                step["tool_calls"] = [asdict(call) for call in response.tool_calls]
            trace["steps"].append(step)
            try:
                cost, cost_details = estimate_cost(response.usage, pricing["profiles"][profile_name], started_at)
            except ValueError as exc:
                info["cost_complete"] = False
                info["failure"] = {"type": "CostAccountingError", "message": str(exc)}
                status = "CostAccountingError"
                break
            step.update(cost=cost, cost_details=cost_details)
            total_cost += cost
            model_history.append(profile_name)
            assistant = response.assistant_message()
            messages.append(assistant)
            trace["messages"].append({
                **asdict(assistant), "model_name": profile.model,
                "router_info": {"selected_model": profile.model, "propensity": propensity},
                "extra": {"response": {"model": response.model, "usage": {
                    **response.usage.raw, "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens, "cost": cost,
                }}},
            })
            transition = await episode.advance(response)
            step.update(transition.record)
            step["format_error"] = transition.format_error
            format_errors += int(transition.format_error)
            messages.extend(transition.messages)
            trace["messages"].extend(asdict(m) for m in transition.messages)
            empty = not response.text.strip() and not response.tool_calls
            consecutive_empty_responses = consecutive_empty_responses + 1 if empty else 0
            if consecutive_empty_responses >= MAX_CONSECUTIVE_EMPTY_RESPONSES:
                status = "EmptyResponseLimit"
                info["failure"] = {
                    "type": status,
                    "message": "stopped after %d consecutive empty assistant responses"
                               % MAX_CONSECUTIVE_EMPTY_RESPONSES,
                    "selected_profile": profile_name, "step_idx": index,
                }
                # Exhausting the response allowance is a task failure, not an
                # infrastructure error. Keep usage and include it in pass rates.
                final_score, success = 0.0, False
                info["labels_valid"] = True
                evaluation = {"status": "scored", "method": "empty_response_limit",
                              "score": final_score, "success": success,
                              "reason": info["failure"]["message"], "cost": 0.0,
                              "cost_complete": True, "api_calls": 0}
                atomic_json(path, trace)
                break
            atomic_json(path, trace)
            if transition.done:
                status = transition.status
                break
        if status in {"Submitted", "EnvironmentDone", "LimitsExceeded", "CostLimitExceeded"}:
            try:
                evaluation = await episode.evaluate(config=config, settings=settings, pricing=pricing,
                                                    provider_factory=provider_factory)
                info["cost_complete"] &= evaluation["cost_complete"]
                if evaluation["status"] == "scored":
                    final_score, success = evaluation["score"], evaluation["success"]
                    info["labels_valid"] = True
                else:
                    status = "EvaluationError"
            except Exception as exc:
                status = "EvaluationError"
                evaluation.update(status="error", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        status = "EnvironmentError"
        info["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            if episode is not None:
                episode.close()
        except Exception as exc:
            status = "EnvironmentError"
            info["labels_valid"] = False
            info["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        info.update(exit_status=status, finished_at=utcnow(), final_score=final_score,
                    success=success, format_errors=format_errors, evaluation=evaluation,
                    collection_cost=total_cost + evaluation["cost"],
                    model_stats={"api_calls": attempts, "instance_cost": total_cost,
                                 "cost_complete": info["cost_complete"]})
        atomic_json(path, trace)
    return trace


async def collect(tasks: list[dict], config: HarnessConfig, settings: CollectionSettings,
                  pricing: dict, output: Path, *, env_factory: Callable = ScienceWorldEnvironment,
                  provider_factory: Callable = create_provider,
                  resume_token_upgrade: bool = False, retry_errors: bool = False) -> dict:
    if any(task.get("benchmark") != settings.benchmark for task in tasks):
        raise ValueError("all tasks must match the collection benchmark")
    if settings.benchmark == "hle" and settings.hle_evaluator == "judge" and not settings.judge_profile:
        raise ValueError("HLE judge evaluation requires --judge-profile")
    spec = configuration(config, settings, pricing)
    traces = []
    collected_cost = 0.0
    skipped = 0
    output.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers/resume races for one experiment directory.
    lock = output / ".collection.lock"
    try:
        handle = lock.open("x")
    except FileExistsError as exc:
        raise ValueError("collection directory is locked; use a different directory, or remove a stale lock after checking the process") from exc
    try:
        import os
        handle.write(str(os.getpid()))
        handle.close()
        archived_infos = [json.loads(p.read_text())["info"]
                          for p in (output / "failed-attempts").glob("*.traj.json")]
        collected_cost += sum(i.get("collection_cost", i["model_stats"]["instance_cost"])
                              for i in archived_infos)
        stop = False
        for task in tasks:
            for run_id in range(settings.runs):
                signature = episode_signature(spec, task, run_id)
                path = output / ("%s-run%d-%s.traj.json" % (task["task_id"], run_id, signature[:16]))
                if resume_token_upgrade and not path.exists():
                    candidates = []
                    for candidate in output.glob("%s-run%d-*.traj.json" % (task["task_id"], run_id)):
                        previous = json.loads(candidate.read_text())
                        if compatible_token_upgrade(previous, spec, task, run_id):
                            candidates.append(candidate)
                        else:
                            raise ValueError("resume found incompatible trajectory: " + str(candidate))
                    if len(candidates) > 1:
                        raise ValueError("multiple previous trajectories for task: " + task["task_id"])
                    if candidates:
                        path = candidates[0]
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous.get("info", {}).get("signature") != signature and not (
                        resume_token_upgrade and compatible_token_upgrade(previous, spec, task, run_id)
                    ):
                        raise ValueError("existing trajectory signature mismatch: " + str(path))
                    if retry_errors and previous["info"].get("exit_status") != "Running" and (
                        not previous["info"].get("labels_valid") or not previous["info"].get("cost_complete")
                    ):
                        archive = output / "failed-attempts" / (utcnow().replace(":", "-") + "-" + path.name)
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        previous_workspace = output / "workspaces" / path.stem
                        archived_workspace = None
                        if previous_workspace.exists():
                            archived_workspace = archive.parent / (archive.stem + ".workspace")
                            previous_workspace.rename(archived_workspace)
                        path.rename(archive)
                        archived_infos.append(previous["info"])
                        collected_cost += previous["info"].get("collection_cost", previous["info"]["model_stats"]["instance_cost"])
                        print(json.dumps({"event": "retry_archived", "task": task["task_id"],
                                          "previous_trace": str(archive),
                                          "previous_workspace": str(archived_workspace) if archived_workspace else None}), flush=True)
                        path = output / ("%s-run%d-%s.traj.json" % (task["task_id"], run_id, signature[:16]))
                    else:
                        if previous["info"].get("exit_status") == "Running":
                            raise ValueError("interrupted trace retained at %s; move it aside before retrying" % path)
                        traces.append(previous)
                        collected_cost += previous["info"].get("collection_cost", previous["info"]["model_stats"]["instance_cost"])
                        skipped += 1
                        if not previous["info"]["cost_complete"] or not previous["info"]["labels_valid"]:
                            stop = True
                            break
                        continue
                if collected_cost >= settings.total_cost_limit:
                    stop = True
                    break
                trace = await collect_episode(task, run_id, config, settings, pricing, spec, path,
                                              env_factory=env_factory, provider_factory=provider_factory)
                traces.append(trace)
                collected_cost += trace["info"]["collection_cost"]
                print(json.dumps({"task": task["task_id"], "run": run_id,
                                  "status": trace["info"]["exit_status"],
                                  "score": trace["info"]["final_score"],
                                  "cost": trace["info"]["model_stats"]["instance_cost"],
                                  "evaluation_cost": trace["info"]["evaluation"]["cost"],
                                  "currency": pricing["currency"]}), flush=True)
                if not trace["info"]["labels_valid"] or not trace["info"]["cost_complete"]:
                    stop = True
                    break
            if stop:
                break
        valid = [t for t in traces if t["info"]["labels_valid"] and t["info"]["cost_complete"]]
        summary = {"planned_traces": len(tasks) * settings.runs, "recorded_traces": len(traces),
                   "skipped_existing": skipped, "valid_traces": len(valid),
                   "success_rate": sum(t["info"]["success"] for t in valid) / len(valid) if valid else None,
                   "mean_score": sum(t["info"]["final_score"] for t in valid) / len(valid) if valid else None,
                   "known_cost": collected_cost, "cost_currency": pricing["currency"],
                   "agent_cost": sum(i["model_stats"]["instance_cost"] for i in
                                     [t["info"] for t in traces] + archived_infos),
                   "evaluation_cost": sum(i.get("evaluation", {}).get("cost", 0.0) for i in
                                          [t["info"] for t in traces] + archived_infos),
                   "archived_attempts": len(archived_infos),
                   "archived_cost_complete": all(i.get("cost_complete") for i in archived_infos),
                   "cost_complete": all(t["info"]["cost_complete"] for t in traces),
                   "config": spec}
        dataset_id = fingerprint({"spec": spec, "tasks": tasks})[:16]
        if resume_token_upgrade:
            for old_summary in output.glob("*.summary.json"):
                if old_summary.name == dataset_id + ".summary.json":
                    continue
                history = output / "history" / utcnow().replace(":", "-")
                history.mkdir(parents=True, exist_ok=True)
                old_summary.rename(history / old_summary.name)
                old_steps = output / old_summary.name.replace(".summary.json", ".steps.jsonl")
                if old_steps.exists():
                    old_steps.rename(history / old_steps.name)
        summary["reused_trace_ids"] = [t["info"]["trace_id"] for t in traces if t["info"]["config"] != spec]
        atomic_json(output / (dataset_id + ".summary.json"), summary)
        samples = output / (dataset_id + ".steps.jsonl")
        temporary = samples.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for trace in traces:
                for row in step_samples(trace):
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(samples)
        summary["steps_file"] = str(samples)
        return summary
    finally:
        lock.unlink(missing_ok=True)
