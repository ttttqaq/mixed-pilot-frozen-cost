#!/usr/bin/env python3
"""Fetch a bounded Hugging Face sample and package SWE tasks in the local task layout.

Only public issue text enters prompt.md/workspace. Gold patches and native test
metadata are kept in reference/. This script does not configure native evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


SOURCES = {
    "swe-rebench-v2": ("nebius/SWE-rebench-V2", "default", "train"),
    "swe-bench-pro-verified": ("opencompass/SWEBench-Pro-Verified", "default", "test"),
}
USER_AGENT = "mixed-pilot-task-converter/1.0"
MAX_ROWS = 20
MAX_ARCHIVE_BYTES = 150_000_000


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def fetch(source: str, offset: int, limit: int, output: Path) -> None:
    if offset < 0 or not 1 <= limit <= MAX_ROWS:
        raise ValueError(f"offset must be nonnegative and limit must be 1..{MAX_ROWS}")
    repo, config, split = SOURCES[source]
    metadata = request_json(f"https://huggingface.co/api/datasets/{repo}")
    revision = metadata["sha"]
    params = urllib.parse.urlencode(
        {"dataset": repo, "config": config, "split": split, "offset": offset, "length": limit}
    )
    page = request_json(f"https://datasets-server.huggingface.co/rows?{params}")
    rows = [item["row"] for item in page["rows"]]
    if len(rows) != limit:
        raise ValueError(f"requested {limit} rows but received {len(rows)}")
    payload = {
        "source": {
            "dataset": repo,
            "revision": revision,
            "config": config,
            "split": split,
            "license": metadata.get("cardData", {}).get("license") or "unknown",
            "url": f"https://huggingface.co/datasets/{repo}",
            "offset": offset,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"downloaded {len(rows)} rows from {repo}@{revision} to {output}")


def task_id(source: str, row: dict) -> str:
    upstream = str(row["instance_id"])
    slug = re.sub(r"[^a-z0-9]+", "-", upstream.lower()).strip("-")[:45].rstrip("-")
    digest = hashlib.sha256(upstream.encode()).hexdigest()[:10]
    return f"{source}-{slug}-{digest}"


def require_row(row: dict) -> None:
    for field in ("instance_id", "repo", "base_commit", "problem_statement", "patch", "test_patch"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"missing required field: {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
        raise ValueError("base_commit is not a 40-character SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repo"]):
        raise ValueError("repo is not an owner/repo identifier")


def download_archive(row: dict, destination: Path) -> str:
    """Materialize source at base commit without exposing later Git history."""
    repo, commit = row["repo"], row["base_commit"]
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("repository archive exceeds the 150 MB pilot limit")
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError("empty repository archive")
        top = members[0].name.split("/", 1)[0]
        for member in members:
            parts = Path(member.name).parts
            if not parts or parts[0] != top or any(part in (".", "..") for part in parts):
                raise ValueError("unsafe archive path")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("archive contains links or device files")
            if not member.isfile() and not member.isdir():
                continue
            relative = Path(*parts[1:])
            if not relative.parts:
                continue
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError("archive member could not be read")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if member.mode & 0o111:
                    target.chmod(target.stat().st_mode | 0o111)
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "-c", "user.name=Task Import",
         "-c", "user.email=task-import@localhost", "commit", "-qm", "Import benchmark base snapshot"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_row(source: str, provenance: dict, row: dict, root: Path, materialize: bool) -> Path:
    require_row(row)
    identifier = task_id(source, row)
    target = root / "tasks" / identifier
    if target.exists():
        raise FileExistsError(target)
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task-convert-", dir=root) as temporary:
        staging = Path(temporary) / identifier
        staging.mkdir()
        reference = staging / "reference"
        reference.mkdir()
        workspace = staging / "workspace"
        git_commit = download_archive(row, workspace) if materialize else None
        if not materialize:
            workspace.mkdir()
            (workspace / "README.md").write_text(
                f"# Source repository\n\n{row['repo']} at {row['base_commit']}.\n"
                "This pilot package contains a repository descriptor; source files have not been materialized.\n",
                encoding="utf-8",
            )

        # Upstream solution and tests remain outside the agent-visible workspace.
        (reference / "gold.patch").write_text(row["patch"], encoding="utf-8")
        (reference / "test.patch").write_text(row["test_patch"], encoding="utf-8")
        # Preserve every original field privately, including fields not mapped
        # into the common manifest (creation dates, native counts, annotations).
        write_json(reference / "source_record.json", row)
        if source == "swe-rebench-v2":
            fail, passed = row.get("FAIL_TO_PASS") or [], row.get("PASS_TO_PASS") or []
            runtime = {"image_name": row.get("image_name"), "install_config": row.get("install_config")}
            language = row.get("language")
            extra_prompt = row.get("interface")
            native = {"FAIL_TO_PASS": fail, "PASS_TO_PASS": passed, "runtime": runtime,
                      "pr_description": row.get("pr_description"), "meta": row.get("meta")}
            repository_license = row.get("license")
        elif source == "swe-bench":
            fail = row["FAIL_TO_PASS"]
            passed = row["PASS_TO_PASS"]
            fail = json.loads(fail) if isinstance(fail, str) else fail
            passed = json.loads(passed) if isinstance(passed, str) else passed
            if not isinstance(fail, list) or not fail or not isinstance(passed, list):
                raise ValueError("Invalid original SWE-bench test lists")
            runtime = {"version": row["version"],
                       "environment_setup_commit": row.get("environment_setup_commit")}
            language, extra_prompt, repository_license = "python", None, None
            native = {"FAIL_TO_PASS": fail, "PASS_TO_PASS": passed, "runtime": runtime}
        else:
            fail, passed = row.get("fail_to_pass") or [], row.get("pass_to_pass") or []
            runtime = {"dockerhub_tag": row.get("dockerhub_tag"),
                       "before_repo_set_cmd": row.get("before_repo_set_cmd"),
                       "selected_test_files_to_run": row.get("selected_test_files_to_run")}
            language = row.get("repo_language")
            extra_prompt = "\n\n".join(
                f"## {name}\n\n{row[key]}" for name, key in
                (("Requirements", "requirements"), ("Interface", "interface")) if row.get(key)
            )
            native = {"FAIL_TO_PASS": fail, "PASS_TO_PASS": passed, "runtime": runtime,
                      "issue_specificity": row.get("issue_specificity"),
                      "issue_categories": row.get("issue_categories")}
            repository_license = None
        write_json(reference / "native_record.json", native)

        prompt = f"# Task\n\n{row['problem_statement'].strip()}\n"
        if extra_prompt:
            prompt += f"\n{extra_prompt.strip()}\n"
        prompt += (
            "\n# Workspace\n\n"
            + ("The repository snapshot at the benchmark base commit is in the current directory.\n"
               if materialize else "The repository is described in `README.md` but is not present in this pilot workspace.\n")
            + "\n# Deliverables\n\nModify the repository to resolve the issue, then summarize the change and tests run.\n"
            + "\n# Constraints\n\nDo not use the hidden reference patch or test patch while solving the task.\n"
        )
        (staging / "prompt.md").write_text(prompt, encoding="utf-8")
        (staging / "prompt.txt").write_text(prompt, encoding="utf-8")
        workspace_info = {
            "path": "workspace", "mode": "materialized_git_repository" if materialize else "environment_descriptor",
            "agent_visible": True, "original_payload_preserved": materialize,
            "limitations": [] if materialize else ["Repository source files are not materialized in this pilot package."],
        }
        if git_commit:
            workspace_info["git_commit"] = git_commit
            workspace_info["upstream_base_commit"] = row["base_commit"]
        manifest = {
            "schema_version": "1.0", "id": identifier,
            "title": row["problem_statement"].strip().splitlines()[0].lstrip("# ")[:120],
            "category": "coding", "difficulty": "unrated",
            "agent_environment": {"kind": "process"},
            "source": {"dataset": provenance["dataset"], "upstream_id": row["instance_id"],
                       "split": provenance["split"], "revision": provenance["revision"],
                       "url": provenance["url"], "license": provenance["license"],
                       "repository_license": repository_license, "repo": row["repo"],
                       "base_commit": row["base_commit"], "language": language},
            "prompt": "prompt.md", "workspace": workspace_info,
            "agent": {"tools": ["read_file", "list_files", "search_files", "write_file",
                                 "replace_in_file", "shell", "git_diff"],
                      "recommended_max_turns": 50,
                      "output_contract": {"type": "workspace_patch", "final_response": "summarize changes and tests"}},
            "evaluation": {"rubrics": "rubrics.json", "reference_dir": "reference",
                           "reference_agent_visible": False,
                           "verification_level": "native_evaluator_pending",
                           "ready": False,
                           "limitations": ["Native container evaluator is not configured by this conversion."]},
            "privacy": {"variant": "clean", "contains_injected_private_data": False, "injected_data": []},
        }
        write_json(staging / "manifest.json", manifest)
        criteria = [{"id": "fail_to_pass", "description": "Previously failing tests pass in the native evaluator.",
                     "points": 100 if not passed else 70, "method": "native_test",
                     "expected_ref": "reference/native_record.json#/FAIL_TO_PASS"}]
        if passed:
            criteria.append({"id": "pass_to_pass", "description": "Existing passing tests remain passing.",
                             "points": 30, "method": "native_test",
                             "expected_ref": "reference/native_record.json#/PASS_TO_PASS"})
        write_json(staging / "rubrics.json", {
            "schema_version": "1.0", "task_id": identifier, "max_score": 100,
            "pass_threshold": 100, "verification_level": "native_evaluator_pending",
            "criteria": criteria,
        })
        staging.rename(target)
    return target


def convert(input_path: Path, output: Path, materialize_indices: set[int]) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    provenance, rows = payload["source"], payload["rows"]
    source = next((key for key, value in SOURCES.items() if value[0] == provenance["dataset"]), None)
    if source is None:
        raise ValueError("unsupported dataset: " + str(provenance["dataset"]))
    if not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError(f"input must contain 1..{MAX_ROWS} rows")
    if any(index < 0 or index >= len(rows) for index in materialize_indices):
        raise ValueError("materialize index outside downloaded sample")
    paths = [convert_row(source, provenance, row, output, index in materialize_indices)
             for index, row in enumerate(rows)]
    print(f"converted {len(paths)} {source} tasks into {output / 'tasks'}")
    for path in paths:
        print(path)


def materialize_task(task_dir: Path) -> None:
    """Replace one descriptor workspace with a clean, history-free base snapshot."""
    manifest_path = task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["workspace"]["mode"] != "environment_descriptor":
        raise ValueError("task is not an unmaterialized repository descriptor")
    source = manifest["source"]
    row = {"repo": source["repo"], "base_commit": source["base_commit"]}
    with tempfile.TemporaryDirectory(prefix="task-materialize-", dir=task_dir.parent) as temporary:
        new_workspace = Path(temporary) / "workspace"
        git_commit = download_archive(row, new_workspace)
        old_workspace = task_dir / "workspace"
        backup = Path(temporary) / "descriptor-workspace"
        old_workspace.rename(backup)
        try:
            new_workspace.rename(old_workspace)
        except Exception:
            backup.rename(old_workspace)
            raise
    manifest["workspace"].update(
        mode="materialized_git_repository", original_payload_preserved=True,
        limitations=[], git_commit=git_commit, upstream_base_commit=source["base_commit"]
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt_path = task_dir / "prompt.md"
    prompt = prompt_path.read_text(encoding="utf-8").replace(
        "The repository is described in `README.md` but is not present in this pilot workspace.",
        "The repository snapshot at the benchmark base commit is in the current directory.",
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"materialized {task_dir} at {source['base_commit']} (local snapshot {git_commit})")


def validate_task(task_dir: Path) -> None:
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    rubrics = json.loads((task_dir / "rubrics.json").read_text(encoding="utf-8"))
    if manifest["id"] != task_dir.name or rubrics["task_id"] != task_dir.name:
        raise ValueError(f"task ID mismatch: {task_dir}")
    for name in ("prompt.md", "prompt.txt", "workspace", "reference", "rubrics.json"):
        if not (task_dir / name).exists():
            raise ValueError(f"missing {name}: {task_dir}")
    if manifest["evaluation"]["reference_agent_visible"] is not False:
        raise ValueError(f"reference is agent-visible: {task_dir}")
    if sum(item["points"] for item in rubrics["criteria"]) != rubrics["max_score"]:
        raise ValueError(f"rubric points do not sum to max_score: {task_dir}")
    if manifest["evaluation"]["verification_level"] != "native_evaluator_pending":
        raise ValueError(f"evaluation status not marked pending: {task_dir}")
    reference = task_dir / "reference"
    for name in ("gold.patch", "test.patch", "native_record.json"):
        if not (reference / name).is_file():
            raise ValueError(f"missing reference/{name}: {task_dir}")
    prompt = (task_dir / "prompt.md").read_text(encoding="utf-8")
    if (task_dir / "prompt.txt").read_text(encoding="utf-8") != prompt:
        raise ValueError(f"prompt variants differ: {task_dir}")
    gold = (reference / "gold.patch").read_text(encoding="utf-8")
    if gold in prompt:
        raise ValueError(f"gold patch leaked into prompt: {task_dir}")
    workspace = task_dir / "workspace"
    if reference.resolve().is_relative_to(workspace.resolve()):
        raise ValueError(f"reference inside workspace: {task_dir}")
    if manifest["workspace"]["mode"] == "materialized_git_repository":
        head = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
        if head != manifest["workspace"]["git_commit"]:
            raise ValueError(f"workspace commit mismatch: {task_dir}")
        status = subprocess.check_output(["git", "-C", str(workspace), "status", "--porcelain"], text=True)
        if status:
            raise ValueError(f"workspace is not clean: {task_dir}")
    elif manifest["workspace"]["mode"] != "environment_descriptor":
        raise ValueError(f"unsupported workspace mode: {task_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch", help="Download at most 20 public dataset rows")
    get.add_argument("--source", required=True, choices=sorted(SOURCES))
    get.add_argument("--offset", type=int, default=0)
    get.add_argument("--limit", type=int, default=2)
    get.add_argument("--output", type=Path, required=True)
    make = commands.add_parser("convert", help="Package downloaded rows into task directories")
    make.add_argument("--input", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--materialize-indices", default="",
                      help="Comma-separated zero-based sample indices to fetch as source snapshots")
    checkout = commands.add_parser("materialize", help="Fetch source snapshot for one converted task")
    checkout.add_argument("--task-dir", type=Path, required=True)
    verify = commands.add_parser("validate", help="Check pilot task packages")
    verify.add_argument("--output", type=Path, required=True)
    extend = commands.add_parser("extend-train", help="Preserve existing train tasks and extend both sources without a test split")
    extend.add_argument("--dataset-root", type=Path, required=True)
    extend.add_argument("--staging", type=Path, required=True)
    extend.add_argument("--cache", type=Path, required=True)
    extend.add_argument("--target-per-source", type=int, default=500)
    extend.add_argument("--seed", type=int, default=20261002)
    extend.add_argument("--prepare-source", choices=sorted(SOURCES), help="Prepare only one source checkpoint without publication")
    extend.add_argument("--publish", action="store_true", help="Replace dataset root after validation; retain a complete sibling backup")
    args = parser.parse_args()
    if args.command == "extend-train":
        from extend_swe_train import run
        run(args)
    elif args.command == "fetch":
        fetch(args.source, args.offset, args.limit, args.output)
    elif args.command == "convert":
        indices = {int(item) for item in args.materialize_indices.split(",") if item.strip()}
        convert(args.input, args.output, indices)
    elif args.command == "materialize":
        materialize_task(args.task_dir)
    else:
        tasks = sorted((args.output / "tasks").glob("*/manifest.json"))
        if not tasks:
            raise ValueError("no converted tasks found")
        for manifest_path in tasks:
            validate_task(manifest_path.parent)
        print(f"validated {len(tasks)} converted tasks")


if __name__ == "__main__":
    main()
