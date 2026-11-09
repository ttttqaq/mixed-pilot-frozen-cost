"""Train-only references, held outside actor tasks/history/features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_references(config, train, manifest):
    root = Path(config["dataset"]["root"]).resolve()
    index = json.loads((root / "train.json").read_text())
    tasks = {t["source_task_id"]: t for t in train}
    result, hashes = {}, {}
    for row in index["tasks"]:
        task = tasks[row["id"]]
        if task["split"] != "train":
            raise ValueError("Only train references may supervise Router")
        path = root / row["path"] / "reference/gold_path.json"
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Reference path escapes dataset or is a symlink")
        payload = path.read_bytes()
        value = json.loads(payload)
        actions = value.get("actions")
        if not isinstance(actions, list) or not actions or any(not isinstance(x, str) or not x.strip() for x in actions):
            raise ValueError("Reference must contain a nonempty action list: " + task["task_id"])
        if value.get("completed") is not True or value.get("final_score") != 100:
            raise ValueError("Reference is not a verified success path")
        digest = hashlib.sha256(payload).hexdigest()
        # Validation's final score describes the reference, never this attempt.
        result[task["task_id"]] = {"actions": actions, "sha256": digest,
                                  "semantics": "one verified successful path from the initial state; alternative solutions are allowed"}
        hashes[task["task_id"]] = digest
    if len(result) != len(train):
        raise ValueError("Missing train reference")
    manifest["train_reference_hashes"] = hashes
    manifest["reference_fingerprint"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return result
