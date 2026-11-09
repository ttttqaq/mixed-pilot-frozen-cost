"""Offline end-to-end batch selection, resume and failure preservation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORE = {"score": 55, "success_probability": 0.6, "estimated_remaining_steps": 4,
         "confidence": 0.7, "reason": "Still needs comparison.", "evidence": ["Two planes observed."]}


class BatchJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "traces"
        self.source.mkdir()
        self.out = self.root / "output"
        self.config = self.root / "config.json"
        self.responses = self.root / "responses.json"
        self.responses.write_text(json.dumps([{"text": json.dumps(SCORE)}]))
        self.config.write_text(json.dumps({"profiles": {"offline": {
            "kind": "scripted", "model": "offline", "responses_file": "responses.json"}}}))
        for n in range(3):
            first = {"role": "user", "content": "Find the shallower plane."}
            steps, history = [], [first]
            for step in range(n + 1):
                steps.append({"step_idx": step, "state": {"history": list(history), "system": "rules"},
                              "response": "look around", "observation": "Two planes observed."})
                history = history + [{"role": "assistant", "content": "look around"},
                                     {"role": "user", "content": "Two planes observed."}]
            (self.source / f"task{n}.traj.json").write_text(json.dumps({
                "info": {"task_id": f"task{n}", "run_id": 0, "final_score": -100, "success": False},
                "steps": steps}))

    def invoke(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts/judge_trace_batch.py"),
                               "--trace-dir", str(self.source), "--config", str(self.config),
                               "--profile", "offline", "--count", "2", "--workers", "2",
                               "--output-dir", str(self.out), *args],
                              capture_output=True, text=True, timeout=30)

    def summary(self):
        return json.loads((self.out / "summary.json").read_text())

    def test_dry_run_real_cli_and_resume(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads((self.out / "selection.json").read_text())
        self.assertEqual(len(plan["traces"]), 2)
        self.assertEqual(self.summary()["scored_steps"], 0)
        self.assertFalse(list(self.out.glob("results/**/*.started.json")))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = self.summary()
        self.assertEqual(summary["scored_steps"], plan["planned_steps"])
        self.assertEqual(summary["usage_all_recorded_attempts"]["recorded_responses"], plan["planned_steps"])
        self.assertTrue(all(r["score"] == 55 for t in summary["traces"] for r in t["steps"]))
        self.assertIn("55", (self.out / "scores.md").read_text())
        # Make new calls fail: a successful resume must not touch the provider.
        self.responses.write_text("[]")
        self.assertEqual(self.invoke().returncode, 0)
        self.assertEqual(len(list(self.out.glob("results/**/*.started.json"))), plan["planned_steps"])
        self.assertEqual(self.invoke("--seed", "999").returncode, 2)

    def test_invalid_attempts_retained_and_only_explicitly_retried(self):
        self.responses.write_text(json.dumps([{"text": '{"score": 999}'}]))
        result = self.invoke()
        self.assertEqual(result.returncode, 1, result.stderr)
        summary = self.summary()
        self.assertEqual(summary["scored_steps"], 0)
        self.assertEqual(summary["failed_steps"], summary["planned_steps"])
        self.assertTrue(all(r["score"] is None for t in summary["traces"] for r in t["steps"]))
        self.responses.write_text(json.dumps([{"text": json.dumps(SCORE)}]))
        self.assertEqual(self.invoke().returncode, 1)
        self.assertEqual(self.invoke("--retry-failed").returncode, 0)
        summary = self.summary()
        self.assertEqual(summary["usage_all_recorded_attempts"]["recorded_responses"], summary["planned_steps"] * 2)
        self.assertTrue(all(r["attempts"] == 2 for t in summary["traces"] for r in t["steps"]))
