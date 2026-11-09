#!/usr/bin/env python3
"""Summarize the frozen four-model experiment, including incomplete groups."""
import csv
import argparse
import json
from pathlib import Path


def summarize(root, benchmark=None):
    plan = json.loads((root / "inputs/experiment.json").read_text())
    groups = []
    for model in plan["models"]:
        for dataset in ([benchmark] if benchmark else ("scienceworld", "hle")):
            tasks = json.loads((root / f"inputs/{dataset}.tasks.json").read_text())["tasks"]
            planned = len(tasks) * plan["runs_per_task"]
            paths = list((root / dataset / model).glob("*.summary.json"))
            if len(paths) > 1:
                raise ValueError(f"Multiple experiment summaries in {dataset}/{model}")
            s = json.loads(paths[0].read_text()) if paths else {}
            valid = s.get("valid_traces", 0)
            successes = round((s.get("success_rate") or 0) * valid)
            complete = (s.get("planned_traces") == planned and valid == planned
                        and s.get("cost_complete") is True)
            groups.append(dict(model=model, benchmark=dataset, planned=planned,
                recorded=s.get("recorded_traces", 0), valid=valid, successes=successes,
                success_rate=successes / planned if complete else None,
                observed_success_rate=successes / valid if valid else None,
                complete=complete, cost_complete=s.get("cost_complete") if s else None,
                agent_cost_cny=s.get("agent_cost", 0),
                judge_cost_cny=s.get("evaluation_cost", 0),
                known_cost_cny=s.get("known_cost", 0)))
    totals = []
    for model in plan["models"]:
        rows = [r for r in groups if r["model"] == model]
        total = {k: sum(r[k] for r in rows) for k in (
            "planned", "recorded", "valid", "successes", "agent_cost_cny",
            "judge_cost_cny", "known_cost_cny")}
        complete = all(r["complete"] for r in rows)
        total.update(model=model, benchmark="ALL", complete=complete,
            cost_complete=all(r["cost_complete"] is True for r in rows),
            success_rate=total["successes"] / total["planned"] if complete else None,
            observed_success_rate=total["successes"] / total["valid"] if total["valid"] else None)
        totals.append(total)
    result = {"currency": "CNY", "cost_basis": "returned-token list-price estimate, not invoice",
              "groups": groups, "models": totals, "complete": all(r["complete"] for r in totals)}
    suffix = f"-{benchmark}" if benchmark else ""
    (root / f"batch-summary{suffix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (root / f"batch-summary{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups + totals)
    lines = ["# Four-model evaluation" + (f" — {benchmark}" if benchmark else ""), "", "Cost: CNY estimates from returned usage; includes separate judge cost.",
             "Incomplete groups have no final pass rate; inspect JSON/CSV for partial counts.", "",
             "| Model | Dataset | Valid / planned | Passed | Pass rate | Agent cost | Judge cost | Total known cost |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in groups + totals:
        rate = f"{r['success_rate']:.2%}" if r["complete"] else "INCOMPLETE"
        lines.append(f"| {r['model']} | {r['benchmark']} | {r['valid']}/{r['planned']} | "
                     f"{r['successes']} | {rate} | {r['agent_cost_cny']:.6f} | "
                     f"{r['judge_cost_cny']:.6f} | {r['known_cost_cny']:.6f} |")
    (root / f"REPORT{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nReports:", root / f"REPORT{suffix}.md", root / f"batch-summary{suffix}.csv")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--benchmark", choices=("scienceworld", "hle"))
    args = parser.parse_args()
    summarize(args.root.resolve(), args.benchmark)
