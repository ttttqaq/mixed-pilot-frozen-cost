#!/usr/bin/env python3
"""Build an offline accuracy/resource scatter dashboard from inference runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import webbrowser


ROOT = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: expected a finite nonnegative number, got {value!r}")
    return value


def token_usage(call, context):
    """Use normalized usage first; cached/reasoning tokens are already included."""
    usage = call.get("usage") or {}
    raw = usage.get("raw") or {}
    values = []
    for key, alias in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        value = usage.get(key)
        if value is None:
            value = usage.get(alias, raw.get(alias))
        if value is None:
            return None
        number(value, f"{context}: {key}")
        if int(value) != value:
            raise ValueError(f"{context}: {key} must be an integer")
        values.append(int(value))
    return values


def summarize_run(directory, label=None):
    directory = Path(directory).expanduser().resolve()
    summary = read_json(directory / "summary.json")
    plan = read_json(directory / "plan.json") if (directory / "plan.json").exists() else {}
    planned_ids = sorted({t["task_id"] for t in plan.get("tasks", [])})
    task_ids = planned_ids or sorted({j["task_id"] for j in summary.get("jobs", [])})
    task_count = summary.get("total_tasks")
    if task_count is None:
        task_count = len(task_ids)
    number(task_count, f"{directory}: total_tasks")
    if task_count == 0 or int(task_count) != task_count:
        raise ValueError(f"{directory}: cannot determine a positive integer task count (need total_tasks or plan.tasks)")
    task_count = int(task_count)

    rate = summary.get("pass_rate")
    if rate is not None:
        number(rate, f"{directory}: pass_rate")
        if rate > 1:
            raise ValueError(f"{directory}: pass_rate must be between 0 and 1")
    currency = summary.get("currency")
    totals = summary.get("cost_by_currency")
    if not totals:
        if not currency or summary.get("total_api_cost") is None:
            raise ValueError(f"{directory}: missing cost_by_currency or total_api_cost/currency")
        totals = {currency: summary["total_api_cost"]}
    totals = {str(k): number(v, f"{directory}: cost in {k}") for k, v in totals.items()}
    currency = currency or next(iter(totals))

    call_paths = sorted((directory / "calls").glob("*.json"))
    if not (directory / "calls").is_dir():
        raise ValueError(f"{directory}: missing calls/ ledger; trace paths cannot safely replace unique-call accounting")
    seen = set()
    profiles = set()
    input_tokens = output_tokens = missing_usage = uncertain_calls = 0
    for path in call_paths:
        call = read_json(path)
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"{path}: missing call_id")
        if call_id in seen:
            raise ValueError(f"{path}: duplicate call_id {call_id}; ambiguous ledger")
        seen.add(call_id)
        if call.get("selected_profile"):
            profiles.add(call["selected_profile"])
        tokens = token_usage(call, path)
        if tokens is None:
            missing_usage += 1
        else:
            input_tokens += tokens[0]
            output_tokens += tokens[1]
        uncertain_calls += call.get("cost") is None or not call.get("cost_complete", True)

    expected_calls = summary.get("api_attempts")
    if expected_calls is not None and len(seen) != expected_calls:
        raise ValueError(f"{directory}: summary records {expected_calls} logical calls but calls/ contains {len(seen)}; finish/re-export the run first")
    if not seen and (rate is not None or summary.get("api_responses", 0)):
        raise ValueError(f"{directory}: calls/ is empty although summary contains inference results")

    notes = []
    cost_complete = summary.get("cost_complete") is True and not uncertain_calls
    if not cost_complete:
        notes.append("Cost 不完整：显示已知费用，可能低于实际消耗。")
    if missing_usage:
        notes.append(f"{missing_usage} 次调用缺失 usage；tokens 仅统计已记录值。")
    if summary.get("unscored_traces", 0):
        notes.append(f"{summary['unscored_traces']} 条轨迹未评分，未进入 pass_rate 分母。")
    if summary.get("unstarted_jobs", 0):
        notes.append(f"{summary['unstarted_jobs']} 个 job 未启动；资源均值仍除以全部 {task_count} 个 task。")
    if summary.get("stop_reason") not in (None, "tasks_finished"):
        notes.append(f"运行结束状态：{summary['stop_reason']}。")
    if len(totals) > 1:
        notes.append("运行包含多个币种；cost 图只展示所选币种的费用部分。")
    if summary.get("superseded_nodes", 0):
        notes.append("资源总量包含已被替代的重试轨迹；pass_rate 使用当前有效叶子。")
    if summary.get("mode") == "ksr":
        notes.append("KSR pass_rate 按已评分叶子计算；多个分支可能属于同一个 task。")
    if rate is None:
        notes.append("尚无 pass_rate；保留统计行，不绘制散点。")

    return dict(
        label=label or directory.name, directory=str(directory), profiles=sorted(profiles),
        mode=summary.get("mode", plan.get("settings", {}).get("mode", "unknown")),
        task_count=task_count, task_set_hash=(hashlib.sha256("\n".join(task_ids).encode()).hexdigest() if task_ids else None),
        started_jobs=summary.get("started_jobs"), planned_jobs=summary.get("planned_jobs"),
        passed=summary.get("passed"), scored_traces=summary.get("scored_traces"),
        total_traces=summary.get("total_traces"), unscored_traces=summary.get("unscored_traces", 0),
        pass_rate=rate, pass_rate_definition=summary.get("pass_rate_definition", "passed / scored leaf trajectories"),
        currency=currency, total_cost_by_currency=totals,
        avg_cost_by_currency={k: v / task_count for k, v in totals.items()},
        cost_complete=cost_complete, cost_basis=summary.get("cost_basis", "unknown"),
        input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens,
        avg_tokens=(input_tokens + output_tokens) / task_count,
        missing_usage_calls=missing_usage, tokens_complete=missing_usage == 0 and cost_complete,
        step_count=len(seen), avg_steps=len(seen) / task_count, notes=notes,
    )


def build_report(directories, output_dir, labels=None, title="Profile 推理 · Accuracy / Cost"):
    labels = labels or [None] * len(directories)
    if len(labels) != len(directories):
        raise ValueError("--label must be supplied once per input directory")
    resolved = [Path(p).expanduser().resolve() for p in directories]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same run directory was supplied more than once")
    rows = [summarize_run(path, label) for path, label in zip(resolved, labels)]
    warnings = []
    hashes = {r["task_set_hash"] for r in rows if r["task_set_hash"]}
    if len(hashes) > 1 or len({r["task_count"] for r in rows}) > 1:
        warnings.append("输入运行的 task 集合或数量不同，比较 accuracy 和资源均值时请考虑数据集差异。")
    if any(r["task_set_hash"] is None for r in rows):
        warnings.append("部分运行缺少 task 清单，无法确认输入运行使用相同的 task 集合。")
    report = dict(title=title, runs=rows, warnings=warnings,
                  definitions=dict(accuracy="日志 summary.pass_rate；passed / scored leaf trajectories。",
                                   cost="summary 的已知总费用（含 evaluator），按币种分别统计。",
                                   tokens="calls/ 中唯一调用的 input_tokens + output_tokens；含缓存输入与已计入 output 的 reasoning，不重复相加。",
                                   steps="calls/ 中唯一逻辑 API 调用数，包含失败及被替代的调用；HTTP 重试不另计 step。",
                                   average="资源总量 / total_tasks（唯一 task 数）；多轮与分支的资源消耗一起计入。"))
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir in resolved:
        raise ValueError("Output directory must differ from input directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    # Script data must not terminate its enclosing script, even with user labels.
    embedded = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (ROOT / "dashboard.html").read_text(encoding="utf-8")
    (output_dir / "index.html").write_text(template.replace("__REPORT_DATA__", embedded), encoding="utf-8")
    (output_dir / "metrics.json").write_text(serialized + "\n", encoding="utf-8")
    fields = ["label", "directory", "profiles", "mode", "task_count", "pass_rate", "passed", "scored_traces",
              "currency", "total_cost", "avg_cost", "cost_complete", "input_tokens", "output_tokens",
              "total_tokens", "avg_tokens", "tokens_complete", "missing_usage_calls", "step_count", "avg_steps"]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for currency, cost in row["total_cost_by_currency"].items():
                values = {k: row.get(k) for k in fields}
                values.update(profiles=" / ".join(row["profiles"]), currency=currency,
                              total_cost=cost, avg_cost=row["avg_cost_by_currency"][currency])
                writer.writerow(values)
    return output_dir, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path, help="Inference log directories (summary.json + calls/)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output", help="Output directory (default: visualized/output)")
    parser.add_argument("--label", action="append", help="Custom point label; repeat in the same order as inputs")
    parser.add_argument("--title", default="Profile 推理 · Accuracy / Cost")
    parser.add_argument("--open", action="store_true", help="Open the generated page in your default browser")
    args = parser.parse_args(argv)
    try:
        output_dir, report = build_report(args.directories, args.output_dir, args.label, args.title)
    except ValueError as exc:
        parser.error(str(exc))
    for row in report["runs"]:
        rate = f"{row['pass_rate']:.2%}" if row["pass_rate"] is not None else "N/A"
        cost = row["avg_cost_by_currency"].get(row["currency"])
        cost_text = f"{cost:.6f}" if cost is not None else "N/A"
        print(f"{row['label']}: pass_rate={rate}, cost/task={cost_text} {row['currency']}, "
              f"tokens/task={row['avg_tokens']:.2f}, steps/task={row['avg_steps']:.2f}")
        for note in row["notes"]:
            print(f"  {note}", file=sys.stderr)
    print(f"Dashboard: {output_dir / 'index.html'}")
    print(f"Metrics: {output_dir / 'metrics.json'} / {output_dir / 'metrics.csv'}")
    if args.open:
        webbrowser.open((output_dir / "index.html").as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
