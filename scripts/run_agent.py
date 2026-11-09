#!/usr/bin/env python3
import argparse, datetime, json, os, shutil, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Prepare an isolated mixed-pilot attempt and optionally run an agent command.")
parser.add_argument("--task", required=True)
parser.add_argument(
    "--data-root",
    default=os.environ.get("MIXED_PILOT_DATA_ROOT", str(Path.home() / "Data" / "mixed-pilot-data")),
    help="External data root containing tasks/",
)
parser.add_argument("--attempt-root", default=str(root / "attempts"))
parser.add_argument("--prepare-only", action="store_true")
parser.add_argument("command", nargs=argparse.REMAINDER, help="agent command after --; prompt is sent on stdin")
args = parser.parse_args()
task = Path(args.data_root).expanduser().resolve() / "tasks" / args.task
if not task.is_dir(): parser.error(f"unknown task: {args.task}")
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
attempt = Path(args.attempt_root).resolve() / f"{stamp}-{args.task}"
workspace = attempt / "workspace"
output = attempt / "output"
trace = attempt / "trace"
shutil.copytree(task / "workspace", workspace, symlinks=True)
output.mkdir(parents=True); trace.mkdir(parents=True)
for name in ("manifest.json", "prompt.md", "rubrics.json"):
    shutil.copy2(task / name, attempt / name)
request = {"task_id": args.task, "prompt": (task / "prompt.md").read_text(), "workspace": str(workspace), "output_dir": str(output)}
(attempt / "request.json").write_text(json.dumps(request, indent=2) + "\n")
print(attempt)
if args.prepare_only or not args.command: sys.exit(0)
command = args.command[1:] if args.command and args.command[0] == "--" else args.command
env = os.environ.copy()
env.update({"MIXED_PILOT_TASK_ID": args.task, "MIXED_PILOT_TASK_DIR": str(attempt), "MIXED_PILOT_WORKSPACE": str(workspace), "MIXED_PILOT_OUTPUT_DIR": str(output), "MIXED_PILOT_PROMPT": str(attempt / "prompt.md")})
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
proc = subprocess.run(command, cwd=workspace, env=env, input=request["prompt"], text=True, capture_output=True)
(trace / "stdout.txt").write_text(proc.stdout)
(trace / "stderr.txt").write_text(proc.stderr)
event = {"event": "agent_process", "started_at": started, "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command, "returncode": proc.returncode, "stdout": "trace/stdout.txt", "stderr": "trace/stderr.txt"}
(trace / "events.jsonl").write_text(json.dumps(event) + "\n")
sys.stdout.write(proc.stdout); sys.stderr.write(proc.stderr); sys.exit(proc.returncode)
