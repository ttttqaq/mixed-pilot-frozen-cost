#!/usr/bin/env python3
"""Sample both full SWE source splits and package a reproducible train/test set."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

from convert_swe_tasksets import SOURCES, convert_row, request_json, validate_task, write_json


def sample_source(source: str, output: Path, seed: int, count: int, workers: int) -> None:
    repo, config, split = SOURCES[source]
    metadata = request_json(f"https://huggingface.co/api/datasets/{repo}")
    revision = metadata["sha"]
    size_query = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split})
    size = request_json("https://datasets-server.huggingface.co/size?" + size_query)
    if size.get("partial"):
        raise ValueError("dataset row count is only partially available")
    population = next(item["num_rows"] for item in size["size"]["splits"] if item["split"] == split)
    indices = random.Random(f"{seed}:{source}").sample(range(population), count)
    provenance = {
        "dataset": repo, "revision": revision, "config": config, "split": split,
        "license": (metadata.get("cardData") or {}).get("license") or "unknown",
        "url": f"https://huggingface.co/datasets/{repo}",
    }
    sampling = {
        "seed": seed, "rng_seed": f"{seed}:{source}", "population_size": population,
        "algorithm": "Python random.Random(str_seed).sample(range(population_size), count)",
        "without_replacement": True, "indices_in_random_order": indices,
        "revision_verification": (
            "Immutable Hub JSONL revision streamed; Hub revision unchanged before and after download"
            if source == "swe-bench-pro-verified" else
            "Hub revision checked unchanged before and after viewer API sampling"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["source"] != provenance or saved["sampling"] != sampling:
            raise ValueError("existing sample has different source revision or sampling parameters")
        print(f"{source}: reusing complete {len(saved['rows'])}-row sample", flush=True)
        return
    checkpoint = output.with_suffix(".partial.json")
    fetched = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["source"] != provenance or saved["sampling"] != sampling:
            raise ValueError("partial sample has different source revision or sampling parameters")
        fetched = {int(key): row for key, row in saved["fetched"].items()}

    def get_row(index: int) -> dict:
        query = urllib.parse.urlencode({
            "dataset": repo, "config": config, "split": split, "offset": index, "length": 1,
        })
        last_error = None
        for attempt in range(4):
            try:
                time.sleep(1.5)
                page = request_json("https://datasets-server.huggingface.co/rows?" + query)
                if len(page["rows"]) != 1 or page["rows"][0]["row_idx"] != index:
                    raise ValueError("viewer returned an unexpected row index")
                item = page["rows"][0]
                if item.get("truncated_cells"):
                    raise ValueError(f"row {index} contains truncated fields: {item['truncated_cells']}")
                return item["row"]
            except ValueError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(min(30, 15 * (attempt + 1)) if getattr(exc, "code", None) == 429 else 2 ** attempt)
        raise RuntimeError(f"failed to fetch {source} row {index}: {last_error}")

    pending = [index for index in indices if index not in fetched]
    print(f"{source}: sampling {count}/{population} rows, revision={revision}", flush=True)
    failures = []
    if source == "swe-bench-pro-verified":
        # This source publishes a small JSONL file. Stream a pinned revision and
        # retain only the preselected records, avoiding the viewer's row quota.
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/swebench_pro_verified.jsonl"
        wanted = set(pending)
        record_index = 0
        with urllib.request.urlopen(url, timeout=120) as stream:
            for line in stream:
                if not line.strip():
                    continue
                if record_index in wanted:
                    fetched[record_index] = json.loads(line)
                    write_json(checkpoint, {"source": provenance, "sampling": sampling, "fetched": fetched})
                    if len(fetched) % 10 == 0:
                        print(f"{source}: downloaded {len(fetched)}/{count}", flush=True)
                record_index += 1
        if record_index != population or len(fetched) != count:
            raise ValueError("JSONL source and published population size differ")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(get_row, index): index for index in pending}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    fetched[index] = future.result()
                    write_json(checkpoint, {"source": provenance, "sampling": sampling, "fetched": fetched})
                    if len(fetched) % 10 == 0:
                        print(f"{source}: downloaded {len(fetched)}/{count}", flush=True)
                except Exception as exc:
                    failures.append(f"row {index}: {exc}")
    if failures:
        raise RuntimeError("\n".join(failures))
    after = request_json(f"https://huggingface.co/api/datasets/{repo}")
    if after["sha"] != revision:
        raise ValueError("dataset changed during sampling; refusing to mix revisions")
    rows = [fetched[index] for index in indices]
    if len({row["instance_id"] for row in rows}) != count:
        raise ValueError("sample contains duplicate upstream instance IDs")
    write_json(output, {"source": provenance, "sampling": sampling, "rows": rows})
    checkpoint.unlink(missing_ok=True)
    print(f"{source}: saved {count} complete, untruncated rows", flush=True)


def build(raw_dir: Path, output: Path, train_count: int, test_count: int, seed: int) -> None:
    count = train_count + test_count
    if (output / "train").exists() or (output / "test").exists():
        raise FileExistsError("train/test output already exists; choose a fresh staging directory")
    entries = {"train": [], "test": []}
    selections = {}
    (output / "raw").mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        input_path = raw_dir / (source + ".json")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if len(payload["rows"]) != count or payload["sampling"]["seed"] != seed:
            raise ValueError("raw sample size or seed differs from requested split")
        selections[source] = {**payload["source"], **payload["sampling"],
                              "raw_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
        shutil.copy2(input_path, output / "raw" / input_path.name)
        for rank, (index, row) in enumerate(zip(payload["sampling"]["indices_in_random_order"], payload["rows"])):
            split = "train" if rank < train_count else "test"
            task_dir = convert_row(source, payload["source"], row, output / split, materialize=False)
            manifest_path = task_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"]["row_index"] = index
            manifest["routing_metadata"] = {
                "router_split": split, "sampling_seed": seed,
                "sampling_rank": rank, "source_dataset": source,
            }
            write_json(manifest_path, manifest)
            validate_task(task_dir)
            entries[split].append({
                "id": manifest["id"], "path": task_dir.relative_to(output).as_posix(),
                "dataset": source, "upstream_id": row["instance_id"],
                "repo": row["repo"], "base_commit": row["base_commit"],
                "language": row.get("language", row.get("repo_language")),
                "source_row_index": index,
            })
    output.mkdir(parents=True, exist_ok=True)
    for split in entries:
        write_json(output / split / "catalog.json", {
            "schema_version": "1.0", "name": f"SWE-neo-data/{split}",
            "task_count": len(entries[split]), "tasks": entries[split],
        })
        write_json(output / (split + ".json"), {
            "schema_version": "1.0", "split": split,
            "task_count": len(entries[split]), "tasks": entries[split],
        })
    write_json(output / "sampling.json", {
        "seed": seed, "train_per_source": train_count, "test_per_source": test_count,
        "assignment": "first train_per_source sampled rows -> train; remaining rows -> test",
        "sources": selections,
    })
    write_json(output / "catalog.json", {
        "schema_version": "1.0", "name": "SWE-neo-data", "task_count": 2 * count,
        "split_counts": {split: len(items) for split, items in entries.items()},
        "tasks": entries["train"] + entries["test"],
    })
    (output / "README.md").write_text(
        "# SWE-neo-data\n\n"
        f"Seed: {seed}. Each source contributes {train_count} train and {test_count} test tasks.\n\n"
        "`train/tasks/` and `test/tasks/` contain the normalized task directories. "
        "`train.json`, `test.json`, and `catalog.json` index these directories. "
        "`sampling.json` records the source revisions, random indices, and sampling algorithm.\n\n"
        "`source.split` preserves the upstream split; `routing_metadata.router_split` is the new "
        "experiment split. Sampling is uniform without replacement within each full source split, "
        "followed by assignment in random order. This is a task split, not a repository holdout.\n\n"
        "Every task has manifest.json, prompt.md, prompt.txt, rubrics.json, workspace/, and reference/. "
        "The complete source record, solution patch, test patch, and test metadata are grader-only. "
        "The raw sample files also contain grading information and must not be given to the solving agent.\n\n"
        "Workspaces are repository descriptors. Source repositories are downloaded on demand with "
        "workflow/convert_swe_tasksets.py materialize --task-dir <path>. "
        "Native container evaluation is not configured; evaluation.ready=false.\n",
        encoding="utf-8",
    )
    (output / "workflow").mkdir()
    for name in ("build_swe_neo_data.py", "convert_swe_tasksets.py"):
        shutil.copy2(Path(__file__).parent / name, output / "workflow" / name)
    validate_dataset(output)
    hashes = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    write_json(output / "checksums.json", hashes)


def validate_dataset(output: Path) -> None:
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    sampling = json.loads((output / "sampling.json").read_text(encoding="utf-8"))
    all_rows = []
    split_rows = {}
    for split, expected_per_source in (("train", sampling["train_per_source"]), ("test", sampling["test_per_source"])):
        rows = json.loads((output / (split + ".json")).read_text(encoding="utf-8"))["tasks"]
        if Counter(row["dataset"] for row in rows) != Counter({source: expected_per_source for source in SOURCES}):
            raise ValueError(f"{split} has wrong per-source counts")
        if len(list((output / split / "tasks").glob("*/manifest.json"))) != len(rows):
            raise ValueError(f"{split} directory and index counts differ")
        for row in rows:
            task_dir = output / row["path"]
            validate_task(task_dir)
            manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest["routing_metadata"]["router_split"] != split:
                raise ValueError("manifest experiment split mismatch")
            raw = json.loads((task_dir / "reference/source_record.json").read_text(encoding="utf-8"))
            if (task_dir / "reference/gold.patch").read_bytes() != raw["patch"].encode("utf-8"):
                raise ValueError("gold patch is not preserved verbatim")
            if (task_dir / "reference/test.patch").read_bytes() != raw["test_patch"].encode("utf-8"):
                raise ValueError("test patch is not preserved verbatim")
            native = json.loads((task_dir / "reference/native_record.json").read_text(encoding="utf-8"))
            if native["FAIL_TO_PASS"] != raw.get("FAIL_TO_PASS", raw.get("fail_to_pass")):
                raise ValueError("FAIL_TO_PASS does not match source")
            if native["PASS_TO_PASS"] != raw.get("PASS_TO_PASS", raw.get("pass_to_pass")):
                raise ValueError("PASS_TO_PASS does not match source")
        split_rows[split] = rows
        all_rows.extend(rows)
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("duplicate converted task ID")
    if len({(row["dataset"], row["upstream_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("upstream instance overlaps across splits")
    if catalog["task_count"] != len(all_rows) or catalog["tasks"] != all_rows:
        raise ValueError("root catalog does not match split catalogs")
    train_instances = {(row["repo"].lower(), row["base_commit"]) for row in split_rows["train"]}
    test_instances = {(row["repo"].lower(), row["base_commit"]) for row in split_rows["test"]}
    if train_instances & test_instances:
        raise ValueError("same repository/base-commit instance occurs in both splits")
    report = {
        "status": "valid", "task_count": len(all_rows),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "source_counts": {split: dict(Counter(row["dataset"] for row in rows)) for split, rows in split_rows.items()},
        "language_counts": {split: dict(Counter(row["language"] for row in rows)) for split, rows in split_rows.items()},
        "train_test_instance_overlap": 0,
        "workspace_modes": {"environment_descriptor": len(all_rows)},
        "evaluation_ready": False,
    }
    write_json(output / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sample", "build", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-source", type=int, default=50)
    parser.add_argument("--test-per-source", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.train_per_source < 1 or args.test_per_source < 1 or not 1 <= args.workers <= 8:
        parser.error("positive split sizes and 1..8 workers are required")
    if args.mode == "sample":
        for source in SOURCES:
            sample_source(source, args.output / (source + ".json"), args.seed,
                          args.train_per_source + args.test_per_source, args.workers)
    elif args.mode == "build":
        if args.raw_dir is None:
            parser.error("build requires --raw-dir")
        build(args.raw_dir, args.output, args.train_per_source, args.test_per_source, args.seed)
    else:
        validate_dataset(args.output)


if __name__ == "__main__":
    main()
