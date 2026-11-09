"""Standalone collection CLI; no changes to the existing agent/code-task loop."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from ..config import load_config
from .collector import CollectionSettings, collect
from .policy import SamplingPolicy
from .pricing import load_prices
from .tasks import load_tasks, select_tasks, validate_tasks


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ScienceWorld/HLE full Trace and per-step data collection")
    p.add_argument("--doctor", action="store_true", help="Inspect dependencies without API calls/downloads")
    p.add_argument("--benchmark", choices=["scienceworld", "hle"], default="scienceworld")
    p.add_argument("--tasks-file", type=Path, help="JSON/JSONL tasks, mixed-pilot manifest, or data/tasks directory")
    p.add_argument("--task-filter", default="*", help="Task-name glob (HLE: subject glob)")
    p.add_argument("--split", choices=["train", "dev", "test", "all"], default="train")
    p.add_argument("--simplification", default="", help="Native ScienceWorld simplification string")
    p.add_argument("--limit", type=int, help="Limit task instances, balanced across task types")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--list-tasks", action="store_true", help="Read tasks only; never call a model")
    p.add_argument("--save-tasks", type=Path, help="Write selected task list for subsequent collection")
    p.add_argument("--config", type=Path, default=Path("harness.json"))
    p.add_argument("--pricing", type=Path, default=Path("configs/collection.prices.json"))
    p.add_argument("--model-pool", nargs="+", help="EXACT harness profile names; overrides any example pool")
    p.add_argument("--mode", choices=["baseline", "roulette"], default="roulette")
    p.add_argument("--profile", help="Fixed profile in baseline mode")
    p.add_argument("--hle-evaluator", choices=["judge", "exact"], default="judge",
                   help="HLE: exact-match fast path plus LLM judge, or conservative exact matching only")
    p.add_argument("--judge-profile", help="Explicit terminal evaluator profile (HLE judge mode)")
    p.add_argument("--tool-timeout", type=float, default=30.0, help="HLE: per-tool timeout in seconds")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--step-limit", type=int, default=50)
    p.add_argument("--cost-limit", type=float, default=0.5, help="Per-Trace stop threshold in pricing currency")
    p.add_argument("--total-cost-limit", type=float, default=10.0, help="Collection stop threshold, including resumed traces")
    p.add_argument("--output-dir", type=Path, default=Path("run/collection"))
    p.add_argument("--dry-run", action="store_true", help="Validate plan without creating providers or paid calls")
    p.add_argument("--resume-token-upgrade", action="store_true", help="Reuse completed traces when only max_tokens increases")
    p.add_argument("--retry-errors", action="store_true", help="Archive terminal invalid traces and retry them")
    return p


def doctor() -> dict:
    packages = {}
    for name in ("scienceworld", "py4j", "PyYAML", "Jinja2", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    java = shutil.which("java")
    java_ok, detail = False, "java not found on PATH"
    if java:
        try:
            result = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=10)
            java_ok = result.returncode == 0
            detail = (result.stderr or result.stdout).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)
    return {"python": sys.version, "executable": sys.executable, "packages": packages,
            "java": {"path": java, "usable": java_ok, "detail": detail},
            "scienceworld_ready": java_ok and all(packages[n] for n in ("scienceworld", "py4j", "PyYAML", "Jinja2"))}


def _validate_api_profiles(config, pool: list[str], *, live: bool) -> None:
    identities = set()
    for name in pool:
        profile = config.profile(name)
        if profile.kind not in {"openai", "openai-compatible", "scripted"}:
            raise ValueError("collection currently supports OpenAI-compatible or scripted profiles: " + name)
        identity = (profile.base_url, profile.model)
        if identity in identities:
            raise ValueError("duplicate endpoint/model in pool: " + name)
        identities.add(identity)
        if profile.kind == "scripted":
            continue
        url = profile.endpoint or profile.base_url
        parsed = urlparse(url)
        if "{" in url or "}" in url or parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("invalid/placeholder endpoint for " + name)
        if parsed.username or parsed.password:
            raise ValueError("put credentials in an environment variable, not in the endpoint URL")
        if profile.max_tokens <= 0 or profile.timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        reserved = {"model", "messages", "tools", "tool_choice", "stream", "max_tokens", "temperature"}
        if reserved.intersection(profile.extra_body):
            raise ValueError("extra_body cannot override core fields: " + name)
        if live and not profile.api_key_optional and not os.environ.get(profile.api_key_env, ""):
            raise ValueError("missing credential environment variable for %s: %s" % (name, profile.api_key_env))


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.doctor:
            report = doctor()
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if args.benchmark == "hle" or report["scienceworld_ready"] else 1
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.runs <= 0 or args.step_limit <= 0:
            raise ValueError("--runs and --step-limit must be positive")
        if not math.isfinite(args.tool_timeout) or args.tool_timeout <= 0:
            raise ValueError("--tool-timeout must be positive")
        if not all(math.isfinite(x) and x > 0 for x in (args.cost_limit, args.total_cost_limit)):
            raise ValueError("cost limits must be finite and positive")
        if args.save_tasks and args.save_tasks.exists():
            raise ValueError("--save-tasks refuses to overwrite an existing task list")
        if args.benchmark == "hle":
            if args.tasks_file:
                rows = load_tasks(args.tasks_file, benchmark="hle")
                tasks = select_tasks(rows, split=args.split, pattern=args.task_filter,
                                     limit=args.limit, seed=args.seed)
            else:
                if not args.list_tasks or args.split != "test":
                    raise ValueError("HLE collection requires --tasks-file pointing to local manifests; "
                                     "the remote reader only supports --list-tasks --split test")
                from .upstream.hle_tasks import enumerate_hle_tasks
                tasks = enumerate_hle_tasks(subject_filter=args.task_filter,
                                            slice_spec=":" + str(args.limit) if args.limit else None)
                tasks = [{k: v for k, v in t.items() if k not in {"answer", "_original"}} for t in tasks]
                for task in tasks:
                    task.update(benchmark="hle", split="test")
        else:
            if not args.tasks_file:
                report = doctor()
                if not report["scienceworld_ready"]:
                    raise ValueError("ScienceWorld/Java dependencies missing; run --doctor and see docs/MTR_COLLECTION.md")
                from .upstream.scienceworld_tasks import enumerate_scienceworld_tasks
                rows = enumerate_scienceworld_tasks(task_filter=args.task_filter, split=args.split,
                                                    simplification_str=args.simplification)
                validate_tasks(rows)
            else:
                rows = load_tasks(args.tasks_file)
            tasks = select_tasks(rows, split=args.split, pattern=args.task_filter, limit=args.limit, seed=args.seed)
        if not tasks:
            raise ValueError("no tasks selected; check split and filters")
        if args.save_tasks:
            args.save_tasks.parent.mkdir(parents=True, exist_ok=True)
            args.save_tasks.write_text(json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False) + "\n")
        if args.list_tasks:
            print(json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False))
            return 0
        if not args.model_pool:
            raise ValueError("explicit --model-pool is required; no default paid model pool")
        SamplingPolicy(args.model_pool, args.mode, args.seed, args.profile)
        config = load_config(args.config)
        _validate_api_profiles(config, args.model_pool, live=not args.dry_run)
        judge_profile = None
        if args.benchmark == "hle" and args.hle_evaluator == "judge":
            if not args.judge_profile:
                raise ValueError("HLE judge evaluation requires --judge-profile; use --hle-evaluator exact for strict matching only")
            judge_profile = args.judge_profile
            _validate_api_profiles(config, [judge_profile], live=not args.dry_run)
        pricing = load_prices(args.pricing, list(dict.fromkeys(args.model_pool + ([judge_profile] if judge_profile else []))))
        settings = CollectionSettings(mode=args.mode, model_pool=args.model_pool, profile=args.profile,
                                      runs=args.runs, max_steps=args.step_limit, cost_limit=args.cost_limit,
                                      total_cost_limit=args.total_cost_limit, seed=args.seed,
                                      benchmark=args.benchmark, hle_evaluator=args.hle_evaluator,
                                      judge_profile=judge_profile, tool_timeout=args.tool_timeout)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "tasks": len(tasks), "traces": len(tasks) * args.runs,
                              "model_pool": args.model_pool, "mode": args.mode, "profile": args.profile,
                              "split": args.split, "currency": pricing["currency"],
                              "benchmark": args.benchmark, "judge_profile": judge_profile,
                              "evaluation_method": args.hle_evaluator if args.benchmark == "hle" else "native",
                              "max_api_steps_per_trace": args.step_limit,
                              "per_trace_cost_stop": args.cost_limit, "collection_cost_stop": args.total_cost_limit,
                              "task_ids": [t["task_id"] for t in tasks]}, indent=2, ensure_ascii=False))
            return 0
        if args.benchmark == "scienceworld" and not doctor()["scienceworld_ready"]:
            raise ValueError("ScienceWorld/Java dependencies missing; run --doctor")
        summary = asyncio.run(collect(tasks, config, settings, pricing, args.output_dir,
                                      resume_token_upgrade=args.resume_token_upgrade, retry_errors=args.retry_errors))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["valid_traces"] == summary["planned_traces"] and summary["cost_complete"] else 1
    except (ValueError, OSError, ImportError) as exc:
        print("mixed-pilot-collect: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
