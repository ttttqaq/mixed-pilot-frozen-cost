#!/usr/bin/env python3
"""Archive a complete run, then remove selected tasks and rebuild subset statistics.

Filters by task identity, never by model outcome. No inference or re-evaluation.
macOS clone copies retain an inexpensive complete recovery copy of the run.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".filter-tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def filter_lines(path, keep):
    count = valid = 0
    calls = set()
    tmp = path.with_suffix(path.suffix + ".filter-tmp")
    with path.open() as source, tmp.open("w") as target:
        for line in source:
            row = json.loads(line)
            if keep(row):
                target.write(line)
                count += 1
                valid += bool(row.get("labels_valid"))
                if row.get("call_id"):
                    calls.add(row["call_id"])
    tmp.replace(path)
    return count, valid, len(calls)


def filter_run(root, removed):
    with root.with_name("." + root.name + ".infer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _filter_run(root, removed)


def _filter_run(root, removed):
    if (root / "FILTER_REPORT.json").exists():
        raise ValueError("Run has already been filtered; inspect FILTER_REPORT.json")
    plan, tree, old = (read(root / name) for name in ("plan.json", "tree.json", "summary.json"))
    if old["mode"] != "single-model":
        raise ValueError("This archive filter supports single-model runs only")
    original_ids = {t["task_id"] for t in plan["tasks"]}
    if not removed or not removed < original_ids:
        raise ValueError("Exclusions must be a nonempty proper subset of run tasks")
    kept = original_ids - removed
    nodes = [n for n in tree["nodes"] if n["task_id"] in kept]
    node_ids = {n["id"] for n in nodes}
    removed_nodes = {n["id"] for n in tree["nodes"] if n["task_id"] in removed}
    for node in nodes:
        if not set(node["children"]) <= node_ids or (node.get("parent_id") and node["parent_id"] not in node_ids):
            raise ValueError("Task removal would leave dangling tree links")
    backup = root.with_name(root.name + ".backup-before-filter-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    subprocess.run(["cp", "-cR", str(root), str(backup)], check=True)
    for name in ("plan.json", "tree.json", "summary.json"):
        if (backup/name).read_bytes() != (root/name).read_bytes():
            raise ValueError("Backup metadata does not match original")
    ledger, profiles = {}, defaultdict(float)
    for path in (root / "calls").glob("*.json"):
        value = read(path)
        if value["node_id"] in node_ids:
            ledger[value["call_id"]] = {key:value.get(key) for key in ("cost", "selected_profile", "status")}
            ledger[value["call_id"]]["has_response"] = "response" in value
            profiles[value["selected_profile"]] += value.get("cost") or 0
        elif value["node_id"] in removed_nodes:
            path.unlink()
        else:
            raise ValueError("Unrecognized node in call ledger; original run remains in backup")
    for node in nodes:
        if not set(node["prefix_call_ids"] + node["own_call_ids"]) <= set(ledger):
            raise ValueError("Retained node references a missing call")
    for node_id in removed_nodes:
        (root / "traces" / (node_id + ".traj.json")).unlink(missing_ok=True)
        workspace = root / "workspaces" / node_id
        if workspace.exists():
            shutil.rmtree(workspace)
    plan["tasks"] = [t for t in plan["tasks"] if t["task_id"] in kept]
    tree["nodes"] = nodes
    tree["jobs"] = [j for j in tree["jobs"] if j["task_id"] in kept]
    write(root / "plan.json", plan)
    write(root / "tree.json", tree)
    counts = {}
    for filename in ("samples.jsonl", "steps.jsonl", "leaf-data/traces.jsonl", "leaf-data/steps.jsonl"):
        counts[filename] = filter_lines(root / filename, lambda row: row["task_id"] in kept)
    filter_lines(root / "events.jsonl", lambda row: row.get("call_id") in ledger)
    leaf = read(root / "leaf-data/summary.json")
    leaf.update(traces=counts["leaf-data/traces.jsonl"][0], steps=counts["leaf-data/steps.jsonl"][0],
                valid_steps=counts["leaf-data/steps.jsonl"][1], unique_exported_calls=counts["leaf-data/steps.jsonl"][2])
    write(root / "leaf-data/summary.json", leaf)
    leaves = [n for n in nodes if not n["children"] and not n.get("superseded")]
    scored = [n for n in leaves if n.get("labels_valid")]
    passed = sum(bool(n.get("success")) for n in scored)
    agent_cost = sum(c["cost"] or 0 for c in ledger.values())
    evaluation_cost = sum(n.get("evaluation", {}).get("cost", 0) for n in nodes)
    unpriced = sum(c["cost"] is None for c in ledger.values())
    summary = {**old, "jobs":[j for j in old["jobs"] if j["task_id"] in kept],
        "nodes":len(nodes), "superseded_nodes":sum(bool(n.get("superseded")) for n in nodes),
        "failed_nodes":sum(not n.get("superseded") and n["status"] in {"initialization_error", "replay_failed", "environment_error", "api_or_accounting_error", "provider_error"} for n in nodes),
        "total_api_cost":agent_cost+evaluation_cost, "agent_cost":agent_cost, "evaluation_cost":evaluation_cost,
        "cost_complete":unpriced==0 and all(n.get("evaluation",{}).get("cost_complete",True) for n in nodes),
        "unpriced_calls":unpriced, "api_attempts":len(ledger), "api_responses":sum(c["has_response"] for c in ledger.values()),
        "passed":passed, "scored_traces":len(scored), "total_traces":len(leaves), "unscored_traces":len(leaves)-len(scored),
        "pass_rate":passed/len(scored) if scored else None, "samples":counts["samples.jsonl"][0],
        "valid_samples":counts["samples.jsonl"][1], "planned_jobs":len(kept), "started_jobs":len(tree["jobs"]),
        "unstarted_jobs":len(kept)-len(tree["jobs"]), "cost_by_profile":dict(profiles),
        "original_run_known_api_cost":old["total_api_cost"], "filtered_task_subset":True}
    write(root / "summary.json", summary)
    report = dict(filtered_at=datetime.datetime.now().astimezone().isoformat(), backup=str(backup),
                  removed_task_ids=sorted(removed), retained_task_ids=sorted(kept), removed_node_count=len(removed_nodes),
                  retained_node_count=len(nodes), original_known_api_cost=old["total_api_cost"],
                  retained_known_api_cost=summary["total_api_cost"],
                  note="Historical billing is preserved in backup; filtered summary counts retained calls only. Event cumulative costs retain original historical values. No rescoring occurred. Do not resume this filtered run; use a new output for added tasks.")
    write(root / "FILTER_REPORT.json", report)
    rate = f"{passed/len(scored):.2%}" if scored else "N/A"
    (root / "SUMMARY.md").write_text(
        f"# Filtered run: {len(kept)} retained tasks\n\n"
        f"Pass rate: {rate} ({passed}/{len(scored)} scored); {len(leaves)-len(scored)} unscored.\n\n"
        f"Retained-call known cost: {summary['total_api_cost']:.6f} {old['currency']}. "
        f"Original full-run known cost: {old['total_api_cost']:.6f} {old['currency']}; filtering does not erase incurred charges.\n\n"
        f"Removed {len(removed)} replaced tasks, including {len(removed_nodes)} historical/final nodes. "
        f"Original run backup: {backup}\n\n"
        "No re-evaluation occurred. Environment readiness is separate from old prediction labels. "
        "Do not resume this filtered checkpoint; run added tasks in a new output directory.\n")
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("task_ids")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True, help="JSON list of task IDs")
    args = parser.parse_args()
    filter_run(args.run.resolve(), set(read(args.exclude)))
