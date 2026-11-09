"""Incrementally expand an existing SWE collection; preserve its original train tasks.

Uses immutable upstream files and the existing convert_swe_tasksets converter.
No Docker operations or model API calls. Publication keeps a complete sibling backup.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

from convert_swe_tasksets import SOURCES, convert_row, validate_task, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def source_file(existing, cache, source):
    provenance = read(existing / "raw" / (source + ".json"))["source"]
    upstream = ("data/train-00000-of-00001.parquet" if source == "swe-rebench-v2"
                else "swebench_pro_verified.jsonl")
    filename = "rebench-original.parquet" if source == "swe-rebench-v2" else "pro-original.jsonl"
    target = cache / filename
    if not target.exists():
        url = f"https://huggingface.co/datasets/{provenance['dataset']}/resolve/{provenance['revision']}/{upstream}"
        print(f"Downloading pinned source: {url}", flush=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        temporary.replace(target)
    return target


def records(path, source):
    if source == "swe-rebench-v2":
        import pyarrow.parquet as parquet
        index = 0
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                yield index, row
                index += 1
    else:
        with path.open(encoding="utf-8") as stream:
            index = 0
            for line in stream:
                if line.strip():
                    yield index, json.loads(line)
                    index += 1


def selection(existing, source, target_count, seed):
    old = read(existing / "raw" / (source + ".json"))
    entries = [r for r in read(existing / "train.json")["tasks"] if r["dataset"] == source]
    if target_count < len(entries):
        raise ValueError("Target count must not remove existing train tasks")
    excluded = {r["source_row_index"] for r in read(existing / "catalog.json")["tasks"]
                if r["dataset"] == source}
    population = old["sampling"]["population_size"]
    candidates = [i for i in range(population) if i not in excluded]
    additional = random.Random(f"{seed}:{source}:extension").sample(candidates, target_count - len(entries))
    return old, entries, excluded, additional


def prepare_source(existing, cache, stage, source, target_count, seed):
    old, entries, excluded, additional = selection(existing, source, target_count, seed)
    raw_path = stage / "raw" / (source + ".json")
    config = {"seed": seed, "target_per_source": target_count,
              "original_train_indices": [e["source_row_index"] for e in entries],
              "added_indices": additional, "excluded_existing_indices": sorted(excluded)}
    if raw_path.exists():
        payload = read(raw_path)
        if payload["extension"] != config or payload["source"] != old["source"]:
            raise ValueError("Staged sample does not match the requested extension")
    else:
        path = source_file(existing, cache, source)
        wanted = set(config["original_train_indices"] + additional)
        fetched = {}
        population = 0
        for index, row in records(path, source):
            population = index + 1
            if index in wanted:
                fetched[index] = row
        if population != old["sampling"]["population_size"] or set(fetched) != wanted:
            raise ValueError("Pinned source population or selected indices differ")
        rows = []
        for entry in entries:
            preserved = read(existing / entry["path"] / "reference/source_record.json")
            pinned = fetched[entry["source_row_index"]]
            for key in ("instance_id", "repo", "base_commit", "patch", "test_patch", "problem_statement"):
                if pinned[key] != preserved[key]:
                    raise ValueError(f"Existing task disagrees with pinned source: {entry['id']}, {key}")
            rows.append(preserved)
        rows.extend(fetched[i] for i in additional)
        previous_ids = {r["upstream_id"] for r in read(existing / "catalog.json")["tasks"]
                        if r["dataset"] == source}
        if any(row["instance_id"] in previous_ids for row in rows[len(entries):]):
            raise ValueError("An added upstream ID already exists in the original collection")
        if len({r["instance_id"] for r in rows}) != target_count:
            raise ValueError("Duplicate upstream instance IDs")
        payload = {"source": old["source"], "extension": config,
                   "upstream_file_sha256": digest(path), "rows": rows}
        write_json(raw_path, payload)
    indices = config["original_train_indices"] + additional
    result = []
    for rank, (index, row) in enumerate(zip(indices, payload["rows"])):
        if rank < len(entries):
            entry = entries[rank]
            target = stage / entry["path"]
            if not target.exists():
                shutil.copytree(existing / entry["path"], target)
            # Original 100 packages are kept byte-for-byte, including task IDs.
        else:
            from convert_swe_tasksets import task_id
            target = stage / "train/tasks" / task_id(source, row)
            if not target.exists():
                convert_row(source, payload["source"], row, stage / "train", materialize=False)
            manifest_path = target / "manifest.json"
            manifest = read(manifest_path)
            manifest["source"]["row_index"] = index
            manifest["routing_metadata"] = {"router_split": "train", "sampling_seed": seed,
                "sampling_rank": rank, "source_dataset": source, "extension": True}
            write_json(manifest_path, manifest)
        manifest = read(target / "manifest.json")
        validate_task(target)
        result.append({"id": manifest["id"], "path": target.relative_to(stage).as_posix(),
                       "dataset": source, "upstream_id": row["instance_id"], "repo": row["repo"],
                       "base_commit": row["base_commit"],
                       "language": row.get("language", row.get("repo_language")), "source_row_index": index})
    print(f"{source}: {len(entries)} retained + {len(additional)} added = {len(result)} train tasks", flush=True)
    return result


def validate(stage, existing, target_count):
    entries = read(stage / "catalog.json")["tasks"]
    expected = Counter({source: target_count for source in SOURCES})
    if Counter(e["dataset"] for e in entries) != expected:
        raise ValueError("Unexpected per-source counts")
    if len({e["id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate task IDs")
    if len({(e["dataset"], e["upstream_id"]) for e in entries}) != len(entries):
        raise ValueError("Duplicate upstream IDs")
    if len(list((stage / "train/tasks").glob("*/manifest.json"))) != len(entries):
        raise ValueError("Unexpected task directories")
    if (stage / "test").exists() or (stage / "test.json").exists():
        raise ValueError("Train-only collection must not contain a test split")
    for entry in entries:
        target = stage / entry["path"]
        validate_task(target)
        manifest = read(target / "manifest.json")
        if manifest["routing_metadata"]["router_split"] != "train":
            raise ValueError("Non-train task")
        raw = read(target / "reference/source_record.json")
        for field, name in [("patch", "gold.patch"), ("test_patch", "test.patch")]:
            if (target / "reference" / name).read_bytes() != raw[field].encode():
                raise ValueError("Patch content was altered")
        native = read(target / "reference/native_record.json")
        for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if native[field] != (raw.get(field, raw.get(field.lower())) or []):
                raise ValueError("Native test metadata differs from source")
    originals = read(existing / "train.json")["tasks"]
    for entry in originals:
        if snapshot(existing / entry["path"]) != snapshot(stage / entry["path"]):
            raise ValueError(f"Original task was changed: {entry['id']}")
    for path in [stage / "train.json", stage / "train/catalog.json"]:
        if read(path)["tasks"] != entries:
            raise ValueError("Catalog mismatch")
    report = {"status": "valid", "task_count": len(entries), "split_counts": {"train": len(entries)},
              "source_counts": dict(expected), "original_train_preserved": len(originals),
              "new_tasks": len(entries) - len(originals), "original_packages_byte_identical": True,
              "duplicate_ids": 0, "evaluation_ready": False, "docker_preparation_performed": False}
    write_json(stage / "validation_report.json", report)
    return report


def run(args):
    existing, stage, cache = args.dataset_root.resolve(), args.staging.resolve(), args.cache.resolve()
    if existing == stage or existing in stage.parents or stage in existing.parents:
        raise ValueError("Staging and dataset root must be separate directories")
    if args.target_per_source < 1:
        raise ValueError("Target count must be positive")
    cache.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "raw").mkdir(exist_ok=True)
    before = snapshot(existing)
    baseline = cache / "original-checksums.json"
    if baseline.exists() and read(baseline) != before:
        raise ValueError("Original collection changed since this extension was started")
    write_json(baseline, before)
    sources = [args.prepare_source] if args.prepare_source else list(SOURCES)
    entries = []
    for source in sources:
        entries.extend(prepare_source(existing, cache, stage, source, args.target_per_source, args.seed))
    if args.prepare_source:
        print("Source checkpoint saved; full collection has not been published.", flush=True)
        return
    write_json(stage / "catalog.json", {"schema_version": "1.0", "name": existing.name,
               "task_count": len(entries), "split_counts": {"train": len(entries)}, "tasks": entries})
    for path in [stage / "train.json", stage / "train/catalog.json"]:
        write_json(path, {"schema_version": "1.0", "split": "train", "task_count": len(entries), "tasks": entries})
    write_json(stage / "sampling.json", {"seed": args.seed, "train_per_source": args.target_per_source,
        "assignment": "All retained and added tasks belong to train; no test split.",
        "algorithm": "Retain original train records; seeded sample without replacement excluding all original train/test indices.",
        "sources": {s: {k: v for k, v in read(stage / "raw" / (s + ".json")).items() if k != "rows"} for s in SOURCES}})
    workflow = stage / "workflow"
    workflow.mkdir(exist_ok=True)
    for name in ("convert_swe_tasksets.py", "extend_swe_train.py"):
        shutil.copy2(Path(__file__).parent / name, workflow / name)
    (stage / "README.md").write_text(
        f"# {existing.name}\n\n{len(entries)} tasks, all assigned to train; {args.target_per_source} per source.\n\n"
        "Use train.json or catalog.json. There is no test split. Existing train packages are preserved byte-for-byte. "
        "The old collection is backed up before publication. Original test records are excluded from new sampling.\n\n"
        "Upstream source.split remains unchanged for provenance (Pro Verified originates from upstream test); "
        "routing_metadata.router_split is train for every task.\n\n"
        "New workspaces are descriptors. No repository source or Docker environment was prepared. "
        "Gold/test patches and source records remain in reference/ and raw/, never in agent-visible workspaces.\n",
        encoding="utf-8")
    print(json.dumps(validate(stage, existing, args.target_per_source), ensure_ascii=False, indent=2), flush=True)
    hashes = snapshot(stage)
    hashes.pop("checksums.json", None)
    write_json(stage / "checksums.json", hashes)
    if not args.publish:
        print(f"Validated staging collection: {stage}; pass --publish to install with a backup.", flush=True)
        return
    if snapshot(existing) != before:
        raise ValueError("Original collection changed during preparation; publication cancelled")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = existing.with_name(existing.name + ".backup-" + stamp)
    existing.rename(backup)
    try:
        stage.rename(existing)
    except BaseException:
        backup.rename(existing)
        raise
    print(f"Published: {existing}\nOriginal backup: {backup}", flush=True)
