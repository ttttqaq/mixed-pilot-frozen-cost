"""Read task lists or manifests without exposing reference answers to the Agent."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def validate_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    instances = set()
    for row in rows:
        task_id = row.get("task_id", "")
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id):
            raise ValueError("task_id must be a safe, nonempty filename: %r" % task_id)
        if task_id in seen:
            raise ValueError("duplicate task_id: " + task_id)
        seen.add(task_id)
        if row.get("benchmark") == "hle":
            _validate_hle(row)
            instance = ("hle", row["source_id"])
            if instance in instances:
                raise ValueError("duplicate HLE instance (possibly across splits): " + row["source_id"])
            instances.add(instance)
            continue
        if row.get("benchmark") != "scienceworld":
            raise ValueError("unsupported collection benchmark")
        if not isinstance(row.get("task_name"), str) or not row["task_name"]:
            raise ValueError("missing ScienceWorld task_name: " + task_id)
        variation = row.get("variation_idx")
        if type(variation) is not int or variation < 0:
            raise ValueError("variation_idx must be a nonnegative integer: " + task_id)
        if row.get("split") not in {"train", "valid", "dev", "test", "all"}:
            raise ValueError("task requires explicit split train/valid/dev/test/all: " + task_id)
        instance = (row["task_name"], variation)
        if instance in instances:
            raise ValueError("duplicate ScienceWorld instance (possibly across splits): %r" % (instance,))
        instances.add(instance)
        row.setdefault("simplification_str", "")
    return rows


def load_tasks(path: Path, benchmark: str = "scienceworld") -> list[dict[str, Any]]:
    """Accept JSON list/{tasks: [...]}, JSONL, one manifest, or a tasks directory.

    ScienceWorld manifests use environment={benchmark, task_name, variation_idx,
    simplification_str}. HLE manifests identify prompt/workspace/reference paths;
    router_split is the experiment split and source_split retains provenance.
    Reference answers are never included in returned task records.
    """
    if path.is_dir():
        files = [path / "manifest.json"] if (path / "manifest.json").is_file() else sorted(path.glob("*/manifest.json"))
        if not files and (path / "tasks").is_dir():
            files = sorted((path / "tasks").glob("*/manifest.json"))
        rows = []
        for file in files:
            manifest = json.loads(file.read_text())
            if benchmark == "hle" and _is_hle(manifest):
                rows.append(_hle_manifest(manifest, file.parent))
            elif benchmark == "scienceworld" and manifest.get("environment", {}).get("benchmark") == "scienceworld":
                rows.append(_manifest_task(manifest))
    else:
        if path.suffix == ".jsonl":
            data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            data = json.loads(path.read_text())
        if isinstance(data, dict) and benchmark == "hle" and _is_hle(data):
            rows = [_hle_manifest(data, path.parent)]
        elif isinstance(data, dict) and "environment" in data:
            rows = [_manifest_task(data)]
        else:
            rows = data.get("tasks", []) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("task file must contain a list of task objects")
    if not rows:
        raise ValueError("no %s tasks found in %s" % (benchmark, path))
    if any(row.get("benchmark") != benchmark for row in rows):
        raise ValueError("task benchmark does not match --benchmark " + benchmark)
    return validate_tasks(rows)


def _is_hle(manifest: dict) -> bool:
    return str(manifest.get("source", {}).get("dataset", "")).lower().startswith("hle")


def _task_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("task path escapes task directory: " + value)
    return path


def _hle_manifest(manifest: dict, root: Path) -> dict:
    root = root.resolve()
    source = manifest.get("source", {})
    routing = manifest.get("routing_metadata", {})
    reference = _task_path(root, manifest.get("evaluation", {}).get("reference_dir", "reference"))
    return {
        "task_id": manifest["id"], "benchmark": "hle",
        "task_name": routing.get("hle_category") or routing.get("raw_subject") or "HLE",
        "task_dir": str(root),
        "prompt_path": str(_task_path(root, manifest.get("prompt", "prompt.md"))),
        "workspace_path": str(_task_path(root, manifest.get("workspace", {}).get("path", "workspace"))),
        "reference_path": str(reference / "gold.json"),
        "split": routing.get("router_split") or source.get("split"),
        "source_split": source.get("split"), "source_id": source.get("upstream_id") or manifest["id"],
        "source_revision": source.get("revision"),
        "allowed_tools": manifest.get("agent", {}).get("tools", []),
    }


def _validate_hle(row: dict) -> None:
    if row.get("split") not in {"train", "dev", "test", "all"}:
        raise ValueError("HLE task requires an explicit experiment split")
    for key in ("task_name", "source_id", "task_dir", "prompt_path", "workspace_path", "reference_path"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError("missing HLE " + key)
    root = Path(row["task_dir"]).resolve()
    paths = {key: _task_path(root, row[key]) for key in ("prompt_path", "workspace_path", "reference_path")}
    if not paths["workspace_path"].is_dir():
        raise ValueError("missing HLE workspace: " + str(paths["workspace_path"]))
    if paths["reference_path"].is_relative_to(paths["workspace_path"]):
        raise ValueError("HLE reference must be outside the agent workspace")
    tools = row.get("allowed_tools", [])
    supported = {"read_file", "list_files", "search_files", "write_file", "replace_in_file", "shell", "git_diff"}
    if not isinstance(tools, list) or any(t not in supported for t in tools):
        raise ValueError("unsupported HLE agent tools")
    # Content fingerprints invalidate resume when the prompt, reference or workspace changes.
    # Neither answers nor reference text are returned to the model or task list.
    hashes = {key: hashlib.sha256(paths[key].read_bytes()).hexdigest()
              for key in ("prompt_path", "reference_path")}
    files = []
    for path in sorted(paths["workspace_path"].rglob("*")):
        if path.is_symlink():
            raise ValueError("HLE workspace symlinks are not supported: " + str(path))
        if path.is_file():
            files.append((str(path.relative_to(paths["workspace_path"])), hashlib.sha256(path.read_bytes()).hexdigest()))
    hashes["workspace_path"] = hashlib.sha256(json.dumps(files).encode()).hexdigest()
    row["content_hashes"] = hashes


def _manifest_task(manifest: dict) -> dict:
    env = manifest["environment"]
    return {
        "task_id": manifest["id"], "benchmark": env.get("benchmark"),
        "task_name": env.get("task_name"), "variation_idx": env.get("variation_idx"),
        "simplification_str": env.get("simplification_str", ""),
        "split": manifest.get("source", {}).get("split"),
        "source_revision": manifest.get("source", {}).get("revision"),
    }


def select_tasks(rows: list[dict], *, split: str, pattern: str = "*",
                 limit: int | None = None, seed: int = 42) -> list[dict]:
    """Deterministic round-robin over shuffled types; limit counts instances."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if (split == "all" or row["split"] == split) and fnmatch.fnmatch(row["task_name"], pattern):
            groups.setdefault(row["task_name"], []).append(row)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    for name in names:
        groups[name].sort(key=lambda x: (x.get("variation_idx", 0), x["task_id"]))
        rng.shuffle(groups[name])
    result = []
    while any(groups.values()):
        for name in names:
            if groups[name]:
                result.append(groups[name].pop())
    return result if limit is None else result[:limit]
