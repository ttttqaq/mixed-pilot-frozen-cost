#!/usr/bin/env python3
"""Replace unready train tasks with verified additions, stopping at source quotas.

No model calls. Original dataset and its 20 test tasks are backed up/preserved.
Only gold-passed/empty-failed candidates with a local amd64 image are admitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mixed_pilot_harness.inference import swe_runtime as native


def read(p):
    return json.loads(Path(p).read_text())


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def image_exists(ref):
    p = subprocess.run(["docker", "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", ref],
                       capture_output=True, text=True, timeout=60)
    return p.returncode == 0 and p.stdout.strip() == "linux/amd64"


def baseline_ready(result):
    checks = result.get("reclassified_checks") or result.get("prepared_checks") or result.get("original_checks") or {}
    return (result.get("status") == "ready" and checks.get("gold", {}).get("status") == "passed"
            and checks.get("empty", {}).get("status") == "failed")


def make_plan(entries, root):
    return {"tasks": [dict(task_id="train--" + e["id"], task_dir=str(root / e["path"]),
                           record_path=str(root / e["path"] / "reference/source_record.json")) for e in entries]}


def publish(state, args):
    """Publish just the currently verified selection, retaining the original test set."""
    backup = Path(state["backup"])
    if not backup.exists():
        temp = backup.with_name(backup.name + ".copying")
        if temp.exists():
            shutil.rmtree(temp)
        shutil.copytree(args.dataset, temp)
        temp.rename(backup)
    selected = state["selected"]
    if len({e["id"] for e in selected}) != len(selected):
        raise ValueError("Duplicate selected task")
    for entry in selected:
        dest = args.dataset / entry["path"]
        if not dest.exists():
            source = args.candidates / entry["path"]
            if not source.exists():
                source = backup / entry["path"]
            temporary = dest.with_name(dest.name + ".copying")
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(source, temporary)
            temporary.rename(dest)
    # Every removed directory remains intact in the verified full backup.
    for entry in state["excluded"]:
        target = args.dataset / entry["path"]
        if target.exists():
            if not (backup / entry["path"] / "reference/source_record.json").exists():
                raise ValueError("Refusing to remove a task without its backup")
            shutil.rmtree(target)
    test = read(backup / "test.json")["tasks"] if (backup / "test.json").exists() else []
    native.write_json(args.dataset / "train.json", dict(schema_version="1.0", split="train", task_count=len(selected), tasks=selected))
    native.write_json(args.dataset / "train/catalog.json", dict(schema_version="1.0", name="SWE-neo-data/train", task_count=len(selected), tasks=selected))
    native.write_json(args.dataset / "catalog.json", dict(schema_version="1.0", name="SWE-neo-data", task_count=len(selected)+len(test),
                      split_counts=dict(train=len(selected), test=len(test)), tasks=selected+test))
    images = {entry["upstream_id"]: state["images"][entry["upstream_id"]] for entry in selected}
    native.write_json(args.output / "ready-images.json", dict(require_prepared=True, images=images))
    native.write_json(args.dataset / "environment-images.json", dict(require_prepared=True, images=images))
    native.write_json(args.output / "selected-plan.json", make_plan(selected, args.dataset))
    selection = dict(updated_at=stamp(), train_count=len(selected), target_train_count=100,
                     source_counts=dict(Counter(e["dataset"] for e in selected)), original_backup=str(backup),
                     excluded_original_ids=[e["id"] for e in state["excluded"]],
                     selected_ids=[e["id"] for e in selected], manifest=str(args.dataset / "environment-images.json"),
                     baseline="gold passed and empty failed offline", selection_bias="environment availability; not an unbiased benchmark sample")
    native.write_json(args.dataset / "environment-selection.json", selection)
    sampling = read(backup / "sampling.json")
    sampling.update(train_per_source=50 if len(selected)==100 else None,
                    assignment="Environment-filtered train selection; original test tasks unchanged",
                    environment_selection=selection, original_sampling_backup=str(backup / "sampling.json"))
    native.write_json(args.dataset / "sampling.json", sampling)
    native.write_json(args.dataset / "validation_report.json", dict(status="valid", task_count=len(selected)+len(test),
                      split_counts=dict(train=len(selected), test=len(test)),
                      source_counts={"train":dict(Counter(e["dataset"] for e in selected)), "test":dict(Counter(e["dataset"] for e in test))},
                      environment_ready_train=len(selected), target_train_count=100, test_environment_validated=False))
    (args.dataset / "README.md").write_text(
        f"# SWE-neo-data\n\nCurrent verified train tasks: {len(selected)}/100. Original test tasks: {len(test)}.\n\n"
        "Train tasks are filtered by offline gold/empty baseline checks. Task package manifests retain source-conversion metadata; "
        "environment-selection.json and environment-images.json record the environment readiness evidence and image selection.\n\n"
        f"Original dataset backup: {backup}\n\nUse train.json for inference and set SWE_NEO_PREPARED_IMAGES to "
        f"{args.dataset / 'environment-images.json'}. Docker images remain in Docker Desktop storage.\n")
    hashes = {p.relative_to(args.dataset).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.dataset.rglob("*") if p.is_file() and p.name != "checksums.json"}
    native.write_json(args.dataset / "checksums.json", hashes)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.output / "status.json"
    original_report = read(args.original_report)["results"]
    if state_path.exists():
        state = read(state_path)
    else:
        original = read(args.dataset / "train.json")["tasks"]
        if len(original) != 100:
            raise ValueError("Expected the original 100 train tasks")
        selected, excluded, images = [], [], {}
        for entry in original:
            result = original_report.get("train--" + entry["id"], {})
            row = read(args.dataset / entry["path"] / "reference/source_record.json")
            if baseline_ready(result) and result.get("base_commit") == row["base_commit"] and image_exists(result["image"]):
                selected.append(entry)
                images[row["instance_id"]] = {k:result[k] for k in ("image", "source_image", "base_commit")}
            else:
                excluded.append(entry)
        state = dict(started_at=stamp(), selected=selected, excluded=excluded, images=images, attempts={},
                     original_ids=[e["id"] for e in original], quotas=dict(Counter(e["dataset"] for e in original)),
                     backup=str(args.dataset.with_name(args.dataset.name + ".backup-environments-" + time.strftime("%Y%m%d-%H%M%S"))))
    def save():
        state.update(updated_at=stamp(), pid=os.getpid(), ready=len(state["selected"]), missing=100-len(state["selected"]), new_api_cost=0)
        native.write_json(state_path, state)
    state["status"] = "running"
    save()
    child = None
    def stop(signum, frame):
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                pass
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        publish(state, args)
        original_ids = set(state["original_ids"])
        all_entries = read(args.candidates / "train.json")["tasks"]
        candidates = [e for e in all_entries if e["id"] not in original_ids]
        if len(candidates) != 900:
            raise ValueError("Expected exactly 900 new candidates")
        # Prefer Python environments to limit large compiler/image preparation; stable within each language group.
        candidates.sort(key=lambda e: 0 if str(e.get("language", "")).lower() in ("python", "py") else 1)
        native.write_json(args.output / "candidates-plan.json", make_plan(candidates, args.candidates))
        for entry in candidates:
            counts = Counter(e["dataset"] for e in state["selected"])
            if len(state["selected"]) == 100:
                break
            if counts[entry["dataset"]] >= state["quotas"][entry["dataset"]]:
                continue
            if entry["id"] in state["attempts"]:
                continue
            if shutil.disk_usage(ROOT).free < 30 * 1024**3:
                raise RuntimeError("Less than 30 GiB host space remains; stopped before pulling another image")
            state["active_task"] = entry["id"]
            save()
            print(f"[{len(state['selected'])}/100] candidate {entry['id']}", flush=True)
            plan = args.output / "current-plan.json"
            native.write_json(plan, make_plan([entry], args.candidates))
            pull_out = args.output / "pulls" / entry["id"]
            cmd = [sys.executable, "-u", str(ROOT / "scripts/pull_swe_images.py"), "--plan", str(plan),
                   "--output", str(pull_out), "--attempts", "2", "--timeout", "600"]
            state["active_stage"] = "pulling"
            save()
            child = subprocess.Popen(cmd, cwd=ROOT)
            rc = child.wait()
            if rc:
                state["attempts"][entry["id"]] = {"status":"pull_failed", "logs":str(pull_out)}
                save()
                continue
            cmd = [sys.executable, "-u", str(ROOT / "scripts/prepare_swe_dependencies.py"),
                   "--plan", str(args.output / "candidates-plan.json"), "--output", str(args.output / "dependencies"),
                   "--only", "train--" + entry["id"], "--proxy", args.proxy,
                   "--prepare-timeout", "900", "--eval-timeout", "600"]
            state["active_stage"] = "offline_validation"
            save()
            child = subprocess.Popen(cmd, cwd=ROOT)
            child.wait()
            results = read(args.output / "dependencies/preparation.json")["results"]
            result = results.get("train--" + entry["id"], {})
            state["attempts"][entry["id"]] = {"status":result.get("status", "error"), "logs":result.get("logs")}
            if baseline_ready(result) and image_exists(result["image"]):
                state["selected"].append(entry)
                state["images"][entry["upstream_id"]] = {k:result[k] for k in ("image", "source_image", "base_commit")}
                save()
                publish(state, args)
                print(f"READY {len(state['selected'])}/100", flush=True)
            save()
        state.update(status="complete" if len(state["selected"]) == 100 else "candidates_exhausted", finished_at=stamp())
        state.pop("active_task", None)
        state.pop("active_stage", None)
        save()
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", default="http://host.docker.internal:8902")
    args = parser.parse_args()
    for key in ("dataset", "candidates", "original_report", "output"):
        setattr(args, key, getattr(args, key).resolve())
    run(args)
