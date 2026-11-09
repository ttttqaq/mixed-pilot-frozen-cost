#!/usr/bin/env python3
"""Reproducible sampling and resumable CLI per-step judging; never invent scores."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mixed_pilot_harness.config import load_config
from mixed_pilot_harness.per_step_judge import prepare_judgment
from mixed_pilot_harness.per_step_judge.core import parse_judgment


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def make_plan(args):
    source = args.trace_dir.resolve()
    candidates = sorted(source.glob("*.traj.json"))
    if len(candidates) < args.count:
        raise ValueError(f"requested {args.count} traces, only {len(candidates)} found")
    selected = random.Random(args.seed).sample(candidates, args.count)
    traces = []
    for path in selected:
        trace = json.loads(path.read_text())
        info = trace["info"]
        if not trace["steps"]:
            raise ValueError(f"trace has no steps: {path}")
        prompts = [prepare_judgment(path, step) for step in range(1, len(trace["steps"]) + 1)]
        traces.append({
            "path": str(path), "name": path.stem,
            "sha256": prompts[0]["trace"]["sha256"], "task_id": info["task_id"],
            "task_name": info.get("task_name", info["task_id"]), "run_id": info["run_id"],
            "native_score": info.get("final_score"), "native_success": info.get("success"),
            "step_count": len(prompts),
            "prompt_hashes": [p["request"]["prompt_sha256"] for p in prompts],
            "prompt_version": prompts[0]["prompt_version"],
            "system_sha256": hashlib.sha256(prompts[0]["request"]["system"].encode()).hexdigest(),
        })
    return {"schema_version": "1.0", "seed": args.seed, "count": args.count,
            "candidate_count": len(candidates), "trace_dir": str(source),
            "profile": args.profile, "config_path": str(args.config.resolve()),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "reference_source": "task_goal_only", "traces": traces,
            "planned_steps": sum(t["step_count"] for t in traces)}


def inspect_step(out, trace, step, profile):
    directory = out / "results" / trace["name"] / f"step-{step:03d}"
    markers = sorted(directory.glob("attempt-*.started.json"))
    row = {"step": step, "status": "pending", "score": None, "attempts": len(markers)}
    if not markers:
        return row
    stem = markers[-1].name.removesuffix(".started.json")
    path = directory / (stem + ".json")
    row.update(status="failed", result_path=str(path), log_path=str(directory / (stem + ".stderr.txt")))
    try:
        result = json.loads(path.read_text())
        row["usage"] = result.get("response", {}).get("usage", {})
        if result.get("status") != "scored":
            raise ValueError(result.get("error", "judge did not return a score"))
        if (result["trace"]["sha256"] != trace["sha256"] or result["step"] != step
                or result["judge"]["profile"] != profile
                or result["request"]["prompt_sha256"] != trace["prompt_hashes"][step - 1]):
            raise ValueError("result does not match the frozen trace/step/profile/prompt")
        judgment = parse_judgment(json.dumps(result["judgment"]))
        row.update(status="scored", **judgment)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row["error"] = str(exc)
    return row


def summarize(out, plan):
    traces = []
    for trace in plan["traces"]:
        rows = [inspect_step(out, trace, step, plan["profile"])
                for step in range(1, trace["step_count"] + 1)]
        scores = [r["score"] for r in rows if r["status"] == "scored"]
        traces.append({k: trace[k] for k in ["name", "path", "task_name", "run_id", "native_score", "native_success", "step_count"]}
                      | {"steps": rows, "mean_score": sum(scores) / len(scores) if scores else None})
    rows = [r for t in traces for r in t["steps"]]
    # Include billed invalid-response attempts, not only successful labels.
    usage = {"input_tokens": 0, "output_tokens": 0, "recorded_responses": 0}
    for marker in (out / "results").glob("*/step-*/attempt-*.started.json"):
        path = marker.with_name(marker.name.removesuffix(".started.json") + ".json")
        try:
            u = json.loads(path.read_text())["response"]["usage"]
            usage["input_tokens"] += u["input_tokens"]
            usage["output_tokens"] += u["output_tokens"]
            usage["recorded_responses"] += 1
        except (OSError, ValueError, KeyError, TypeError):
            pass
    summary = {"profile": plan["profile"], "reference_source": plan["reference_source"],
               "planned_traces": len(traces), "planned_steps": len(rows),
               "scored_steps": sum(r["status"] == "scored" for r in rows),
               "failed_steps": sum(r["status"] == "failed" for r in rows),
               "pending_steps": sum(r["status"] == "pending" for r in rows),
               "usage_all_recorded_attempts": usage, "traces": traces}
    write_json(out / "summary.json", summary)
    lines = ["# Qwen Flash 逐步评分", "",
             f"Judge: `{plan['profile']}`；抽样 seed={plan['seed']}；参考标准：任务目标（未提供标准答案）。",
             "原始终局分数仅用于展示对比，未输入 judge。分数是步骤后的状态价值估计，不是原生环境分数。", "",
             f"已评分 **{summary['scored_steps']}/{len(rows)}**；失败 {summary['failed_steps']}；待运行 {summary['pending_steps']}。", "",
             "| 任务 | run | 原生分数 | 步数 | 逐步 judge score（从 step 1 开始） |",
             "|---|---:|---:|---:|---|"]
    for t in traces:
        seq = ", ".join(str(r["score"]) if r["status"] == "scored" else r["status"] for r in t["steps"])
        lines.append(f"| {t['task_name']} | {t['run_id']} | {t['native_score']} | {t['step_count']} | {seq} |")
    for t in traces:
        lines += ["", f"## {t['task_name']} / run {t['run_id']}", "", f"[原始 trace]({t['path']})", "",
                  "| Step | Score | 成功概率 | 预计剩余步数 | 理由 |", "|---:|---:|---:|---:|---|"]
        for r in t["steps"]:
            reason = r.get("reason", r.get("error", r["status"])).replace("|", "\\|").replace("\n", " ")
            label = f"[{r['step']}]({r['result_path']})" if r["status"] == "scored" else str(r["step"])
            lines.append(f"| {label} | {r['score']} | {r.get('success_probability', '')} | {r.get('estimated_remaining_steps', '')} | {reason} |")
    (out / "scores.md").write_text("\n".join(lines) + "\n")
    return summary


async def run(args, plan):
    out = args.output_dir.resolve()
    jobs = []
    for trace in plan["traces"]:
        for step in range(1, trace["step_count"] + 1):
            row = inspect_step(out, trace, step, args.profile)
            if row["status"] == "pending" or (row["status"] == "failed" and args.retry_failed):
                jobs.append((trace, step, row["attempts"] + 1))
    summary = summarize(out, plan)
    if args.dry_run or not jobs:
        return summary
    config = load_config(args.config)
    profile = config.profile(args.profile)
    if profile.kind != "scripted" and not profile.api_key_optional and not os.environ.get(profile.api_key_env, ""):
        raise ValueError(f"missing {profile.api_key_env}; plan and pending report saved, no API calls made")
    semaphore = asyncio.Semaphore(args.workers)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}

    async def one(trace, step, attempt):
        async with semaphore:
            directory = out / "results" / trace["name"] / f"step-{step:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"attempt-{attempt:03d}"
            result = directory / (stem + ".json")
            write_json(directory / (stem + ".started.json"), {
                "started_at": datetime.now(timezone.utc).isoformat(), "step": step,
                "trace_sha256": trace["sha256"], "profile": args.profile})
            command = [sys.executable, "-m", "mixed_pilot_harness.cli", "per-step-judge",
                       "--config", str(args.config.resolve()), "--profile", args.profile,
                       "--trace", trace["path"], "--step", str(step), "--result", str(result)]
            with (directory / (stem + ".stderr.txt")).open("w") as err:
                proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL,
                                                            stderr=err, env=env, cwd=ROOT)
                try:
                    code = await proc.wait()
                except asyncio.CancelledError:
                    if proc.returncode is None:
                        proc.terminate()
                    await proc.wait()
                    raise
            write_json(directory / (stem + ".finished.json"), {"returncode": code})
            summary = summarize(out, plan)
            row = inspect_step(out, trace, step, args.profile)
            print(f"[{summary['scored_steps']}/{plan['planned_steps']}] {trace['task_name']} "
                  f"run{trace['run_id']} step{step}: {row['score'] if row['status'] == 'scored' else row['status']}", flush=True)

    await asyncio.gather(*(one(*job) for job in jobs))
    return summarize(out, plan)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, default=ROOT / "run/collection/baselines-scienceworld-3models-2runs/scienceworld/qwen-flash-2025-07-28")
    p.add_argument("--output-dir", type=Path, default=ROOT / "run/step-judgments/qwen10-deepseek-seed42")
    p.add_argument("--config", type=Path, default=ROOT / "configs/collection.providers.json")
    p.add_argument("--profile", default="deepseek-flash")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="Retry unsuccessful/interrupted steps, retaining earlier attempts")
    args = p.parse_args(argv)
    lock = None
    try:
        if args.count < 1 or args.workers < 1:
            raise ValueError("count and workers must be positive")
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        lock = (out / ".batch.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another batch is using this output directory") from None
        load_config(args.config).profile(args.profile)
        plan = make_plan(args)
        manifest = out / "selection.json"
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError("inputs/config/prompt changed; choose a new --output-dir")
        write_json(manifest, plan)
        print(f"Selected {len(plan['traces'])} traces / {plan['planned_steps']} steps; output: {out}", flush=True)
        summary = asyncio.run(run(args, plan))
        print(f"Scored {summary['scored_steps']}/{summary['planned_steps']}; report: {out / 'scores.md'}")
        return 0 if args.dry_run or summary["scored_steps"] == summary["planned_steps"] else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"judge-trace-batch: {exc}", file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
