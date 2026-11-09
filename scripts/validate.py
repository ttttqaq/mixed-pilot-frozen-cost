#!/usr/bin/env python3
import hashlib, json, os, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_root = Path(
    os.environ.get("MIXED_PILOT_DATA_ROOT", str(Path.home() / "Data" / "mixed-pilot-data"))
).expanduser().resolve()
tasks_root = data_root / "tasks"
errors = []
catalog = json.loads((root / "catalog.json").read_text())
ids = []
if not tasks_root.is_dir():
    print("INVALID")
    print(f"- data root has no tasks directory: {tasks_root}")
    sys.exit(1)
for task_dir in sorted(tasks_root.iterdir()):
    if not task_dir.is_dir():
        continue
    try:
        manifest = json.loads((task_dir / "manifest.json").read_text())
        rubrics = json.loads((task_dir / "rubrics.json").read_text())
    except Exception as exc:
        errors.append(f"{task_dir.name}: invalid JSON: {exc}")
        continue
    ids.append(manifest.get("id"))
    if manifest.get("id") != task_dir.name:
        errors.append(f"{task_dir.name}: id mismatch")
    for name in ("prompt.md", "workspace", "reference", "rubrics.json"):
        if not (task_dir / name).exists(): errors.append(f"{task_dir.name}: missing {name}")
    if manifest.get("evaluation", {}).get("reference_agent_visible") is not False:
        errors.append(f"{task_dir.name}: reference must be hidden")
    expected_commit = manifest.get("workspace", {}).get("git_commit")
    if expected_commit:
        proc = subprocess.run(["git", "-C", str(task_dir / "workspace"), "rev-parse", "HEAD"], text=True, capture_output=True)
        if proc.returncode or proc.stdout.strip() != expected_commit:
            errors.append(f"{task_dir.name}: workspace git commit mismatch")
    points = sum(item.get("points", 0) for item in rubrics.get("criteria", []))
    if points != rubrics.get("max_score"):
        errors.append(f"{task_dir.name}: rubric points {points} != max_score")
    if rubrics.get("task_id") != task_dir.name:
        errors.append(f"{task_dir.name}: rubric task_id mismatch")
catalog_ids = [item["id"] for item in catalog["tasks"]]
if ids != catalog_ids: errors.append("catalog task order/content does not match tasks directory")
inventory = root / "SHA256SUMS"
for line in inventory.read_text().splitlines():
    expected, rel = line.split("  ", 1)
    path = data_root / rel if rel.startswith("tasks/") else root / rel
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected: errors.append(f"checksum mismatch: {rel}")
if errors:
    print("INVALID")
    print("\n".join(f"- {e}" for e in errors))
    sys.exit(1)
print(f"VALID: {len(ids)} tasks; all normalized rubrics sum to 100; checksums match")
