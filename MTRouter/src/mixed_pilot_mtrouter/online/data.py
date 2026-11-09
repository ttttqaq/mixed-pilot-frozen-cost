"""Read only explicit train/evaluation indexes and enforce native-instance isolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def native_identity(task: dict) -> tuple:
    if task.get("benchmark") == "scienceworld":
        return "scienceworld", task["task_name"], int(task["variation_idx"])
    if task.get("benchmark") == "alfworld":
        if not isinstance(task.get("source_id"), str) or not task["source_id"]:
            raise ValueError("ALFWorld tasks require a nonempty native source_id")
        return "alfworld", task["source_id"]
    raise ValueError(f"Online router supports only ScienceWorld/ALFWorld: {task.get('benchmark')!r}")


def assert_training_rows(rows: list[dict]) -> None:
    """Fail closed before any teacher/router update if evaluation data leaked in."""
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Online training rows must be dictionaries")
        state = row.get("state", row.get("before_state", row))
        if not isinstance(state, dict):
            raise ValueError("Online training row state must be a dictionary")
        split = state.get("split", row.get("split"))
        task_id = str(state.get("task_id", row.get("task_id", "")))
        if split != "train" or task_id.startswith(("valid--", "test--", "dev--")):
            raise ValueError(f"Only train rows may update online models: split={split!r}, task_id={task_id!r}")


def _read_index(root: Path, split: str) -> tuple[dict, list[dict], str]:
    path = root / f"{split}.json"
    if not path.is_file():
        # Never fall back to globbing every manifest (which could touch test).
        raise ValueError(f"Explicit {split}.json index required: {path}")
    if path.is_symlink():
        raise ValueError(f"Online split indexes must be ordinary files, not symlinks: {path}")
    payload = path.read_bytes()
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get("split") != split:
        raise ValueError(f"Invalid {split} index split")
    rows = data.get("tasks")
    if (not isinstance(rows, list) or not rows or type(data.get("task_count")) is not int
            or data["task_count"] != len(rows)):
        raise ValueError(f"Invalid/empty {split} task index")
    seen_ids, seen_paths = set(), set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("path"), str):
            raise ValueError(f"Invalid task entry in {split}.json")
        directory = (root / row["path"]).resolve()
        if not directory.is_relative_to(root) or row["id"] in seen_ids or directory in seen_paths:
            raise ValueError(f"Duplicate or escaping task entry in {split}.json: {row!r}")
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError(f"Online task manifests must not be symlinks: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Task manifest must be an object: {manifest_path}")
        routed = manifest.get("routing_metadata", {}).get("router_split") or manifest.get("source", {}).get("split")
        if routed != split or manifest.get("id") != row["id"]:
            raise ValueError(f"Index/manifest split or ID mismatch: {manifest_path}")
        if manifest.get("environment", {}).get("benchmark") not in {"scienceworld", "alfworld"}:
            raise ValueError(f"Unsupported online benchmark: {manifest_path}")
        if row.get("benchmark", manifest["environment"]["benchmark"]) != manifest["environment"]["benchmark"]:
            raise ValueError(f"Index/manifest benchmark mismatch: {manifest_path}")
        seen_ids.add(row["id"])
        seen_paths.add(directory)
    return data, rows, hashlib.sha256(payload).hexdigest()


def load_train_valid(config: dict) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Return online train tasks, validation tasks and a frozen split manifest.

    Only train.json and the configured evaluation index are opened; no task globbing.
    Original source_split is provenance only; routing_metadata.router_split is
    the experiment split and is checked against each selected index.
    """
    dataset = config["dataset"]
    evaluation_split = dataset.get("validation_split")
    if dataset.get("train_split") != "train" or evaluation_split not in {"valid", "test"}:
        raise ValueError("Require train and an explicit valid/test evaluation split")
    root = Path(dataset["root"]).expanduser().resolve()
    selected = {split: _read_index(root, split) for split in ("train", evaluation_split)}
    # Import lazily so static config/preflight checks need no Java or Torch.
    from mixed_pilot_harness.inference.dataset import load_dataset
    train = load_dataset(root, "train")
    valid = load_dataset(root, evaluation_split)
    for split, tasks in (("train", train), (evaluation_split, valid)):
        if len(tasks) != len(selected[split][1]):
            raise ValueError(f"Loader did not preserve all {split} index rows")
        if any(task.get("split") != split for task in tasks):
            raise ValueError(f"Unexpected task split in {split} collection")
        for task in tasks:
            native_identity(task)
    train_native = {native_identity(t) for t in train}
    valid_native = {native_identity(t) for t in valid}
    train_ids = {t.get("source_task_id", t["task_id"]) for t in train}
    valid_ids = {t.get("source_task_id", t["task_id"]) for t in valid}
    if len(train_native) != len(train) or len(valid_native) != len(valid):
        raise ValueError("Duplicate native task instance within an online split")
    if train_native & valid_native or train_ids & valid_ids:
        raise ValueError("Train/evaluation native identities or task IDs overlap")
    snapshots = {}
    for split, tasks in (("train", train), (evaluation_split, valid)):
        snapshots[split] = {
            "index": str(root / f"{split}.json"), "index_sha256": selected[split][2],
            "count": len(tasks), "task_ids": [t["task_id"] for t in tasks],
            "native_identities": [list(native_identity(t)) for t in tasks],
            "manifest_hashes": {
                row["id"]: hashlib.sha256((root / row["path"] / "manifest.json").read_bytes()).hexdigest()
                for row in selected[split][1]
            },
            "content_hashes": {t["task_id"]: t.get("content_hashes", {}) for t in tasks},
        }
    digest = hashlib.sha256(json.dumps(snapshots, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "schema": "reference-router-train-eval-v2", "root": str(root),
        "training_split": "train", "evaluation_split": evaluation_split,
        "test_access": "evaluation only" if evaluation_split == "test" else "excluded",
        "split_fingerprint": digest, "splits": snapshots,
    }
    return train, valid, manifest
