"""Optional, paused-training benchmark evaluation using the existing harness."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .service import make_server


@dataclass
class OnlineEvaluationConfig:
    dataset: Path
    harness_config: Path
    pricing: Path
    every_steps: int = 500
    split: str = "test"
    step_limit: int = 50
    timeout: float = 7200
    python: str = sys.executable
    profiles: list[str] | None = None


def _task_key(task):
    task_id = task["task_id"]
    prefix, sep, rest = task_id.partition("--")
    return task["benchmark"], rest if sep and prefix in {"train", "test", "dev"} else task_id


class OnlineEvaluator:
    def __init__(self, config, output_dir, model_names, *, process_runner=None):
        if config.every_steps <= 0 or config.step_limit <= 0:
            raise ValueError("Online evaluation step intervals/limits must be positive")
        if not math.isfinite(config.timeout) or config.timeout <= 0:
            raise ValueError("Online evaluation timeout must be finite and positive")
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.root = self.output_dir / "online_eval"
        self.model_names = list(model_names)
        self.profiles = list(config.profiles or model_names)
        if len(self.profiles) != len(self.model_names) or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("Online evaluation profiles must map one-to-one to trained models")
        self.last_step = None
        self.tasks = []
        self.process_runner = process_runner or self._run_process

    def arguments(self, output, service_url):
        return [
            "--dataset", str(self.config.dataset.resolve()),
            "--split", self.config.split, "--mode", "mtrouter",
            "--models", *self.profiles,
            "--router-model-names", *self.model_names,
            "--router-service-url", service_url,
            "--config", str(self.config.harness_config.resolve()),
            "--pricing", str(self.config.pricing.resolve()),
            "--step-limit", str(self.config.step_limit),
            "--stop-mode", "run-mode", "--runs", "1", "--seed", "42",
            "--output-dir", str(output),
        ]

    def environment(self):
        env = dict(os.environ)
        env["PATH"] = str(Path(self.config.python).resolve().parent) + os.pathsep + env.get("PATH", os.defpath)
        for name in ("NO_PROXY", "no_proxy"):
            env[name] = ",".join(filter(None, [env.get(name, ""), "127.0.0.1", "localhost"]))
        return env

    def preflight(self):
        # Optional harness imports do not affect standalone MTRouter training.
        from mixed_pilot_harness.inference.cli import parser, plan

        if self.root.exists():
            raise ValueError(f"Online evaluation output already exists; use a new --output-dir: {self.root}")
        args = self.arguments(self.root / "preflight", "http://127.0.0.1:1")
        self.tasks, harness, _, _ = plan(parser().parse_args([*args, "--dry-run"]))
        unsupported = {t["benchmark"] for t in self.tasks} - {"scienceworld", "alfworld"}
        if unsupported:
            raise ValueError(f"Online training evaluation supports ScienceWorld/ALFWorld, got {sorted(unsupported)}")
        missing = []
        for name in self.profiles:
            profile = harness.profile(name)
            if profile.kind != "scripted" and not profile.api_key_optional and not os.environ.get(profile.api_key_env):
                missing.append(profile.api_key_env or name)
        if missing:
            raise ValueError("Online validation requires credential variables: " + ", ".join(sorted(set(missing))))
        command = [self.config.python, "-m", "mixed_pilot_harness.inference.cli", *args, "--dry-run"]
        result = subprocess.run(command, capture_output=True, text=True, env=self.environment(), timeout=60)
        if result.returncode:
            raise ValueError("Online evaluation interpreter/preflight failed: " + (result.stderr or result.stdout)[-2000:])
        print(f"Online validation: {len(self.tasks)} tasks, every {self.config.every_steps} optimizer steps + final; "
              "LLM API calls are billed separately.", flush=True)

    def check_training_tasks(self, samples):
        overlap = {_task_key(t) for t in self.tasks} & {_task_key(s) for s in samples}
        if overlap:
            raise ValueError(f"Online validation tasks overlap training tasks: {sorted(overlap)[:5]}")

    def _run_process(self, command, log_path, timeout):
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=self.environment(), start_new_session=True)
            try:
                return process.wait(timeout=timeout)
            except BaseException:
                # Also stop Java/PDDL children on timeout or user interruption.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise

    def maybe_run(self, scorer, step, epoch, *, final=False, writer=None):
        if step <= 0 or step == self.last_step or (not final and step % self.config.every_steps):
            return None
        self.last_step = step
        directory = self.root / f"step-{step:08d}"
        started = time.monotonic()
        report = {"global_step": step, "epoch": epoch, "final": final,
                  "started_at": datetime.now(timezone.utc).isoformat(), "status": "error",
                  "checkpoint": str(directory / "q_function.pt"),
                  "inference_dir": str(directory / "inference")}
        was_training = scorer.training
        server = thread = None
        try:
            directory.mkdir(parents=True, exist_ok=False)
            scorer.save(directory / "q_function.pt")
            scorer.eval()
            # Reuse the resident encoder/Q network while training is paused.
            # The inference subprocess loads environments/providers, not a second encoder.
            server = make_server(scorer, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            args = self.arguments(directory / "inference", f"http://127.0.0.1:{server.server_port}")
            command = [self.config.python, "-m", "mixed_pilot_harness.inference.cli", *args]
            print(f"\nOnline validation step={step}: training paused; log={directory / 'inference.log'}", flush=True)
            report["returncode"] = self.process_runner(command, directory / "inference.log", self.config.timeout)
            summary = json.loads((directory / "inference/summary.json").read_text())
            report.update({k: summary.get(k) for k in (
                "total_tasks", "total_traces", "scored_traces", "unscored_traces",
                "passed", "pass_rate", "total_api_cost", "currency", "cost_complete")})
            expected = len(self.tasks) or summary["total_tasks"]
            complete = (report["returncode"] == 0 and expected > 0 and
                        report["total_tasks"] == expected and report["total_traces"] == expected and
                        report["scored_traces"] == expected)
            report["status"] = "scored" if complete else "partial" if (report["scored_traces"] or 0) else "error"
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if server is not None:
                if thread is not None:
                    server.shutdown()
                    thread.join()
                server.server_close()
            scorer.train(was_training)
        report["duration_seconds"] = time.monotonic() - started
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        with (self.output_dir / "online_validation.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        if writer is not None and report["status"] == "scored":
            writer.add_scalar("OnlineValidation/pass_rate", report["pass_rate"], step)
            writer.add_scalar("OnlineValidation/known_api_cost", report["total_api_cost"], step)
        print(f"Online validation step={step}: {report['status']}, "
              f"passed={report.get('passed')}/{report.get('scored_traces')}; training continues.", flush=True)
        return report
