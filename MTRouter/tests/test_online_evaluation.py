"""Offline scheduling, HTTP bridge and failure tests; no paid provider calls."""
from dataclasses import replace
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request, build_opener, ProxyHandler

from mixed_pilot_mtrouter.online_evaluation import OnlineEvaluationConfig, OnlineEvaluator


class Scorer:
    model_names = ["a", "b"]

    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)

    def save(self, path):
        path.write_text("checkpoint")

    def predict(self, messages, models):
        assert not self.training
        return [0.2, 0.8]


class Writer:
    def __init__(self): self.values = []
    def add_scalar(self, *args): self.values.append(args)


class OnlineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = OnlineEvaluationConfig(self.root / "dataset", self.root / "harness.json",
                                             self.root / "prices.json", every_steps=500)
        self.calls = []
        self.scorer = Scorer()

    def runner(self, command, log_path, timeout, *, returncode=0, scored=2):
        self.calls.append(command)
        service = command[command.index("--router-service-url") + 1]
        request = Request(service + "/route", data=json.dumps({"models": ["a", "b"],
                          "messages": [{"role": "user", "content": "task"}]}).encode(),
                          headers={"Content-Type": "application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            self.assertEqual(json.load(response)["model"], "b")
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({
            "total_tasks": 2, "total_traces": 2, "scored_traces": scored,
            "unscored_traces": 2 - scored, "passed": 1, "pass_rate": 1 / scored,
            "total_api_cost": 0.2, "currency": "CNY", "cost_complete": returncode == 0,
        }))
        log_path.write_text("offline fixture")
        return returncode

    def evaluator(self, **kwargs):
        return OnlineEvaluator(self.config, self.root / "output", ["a", "b"],
                               process_runner=kwargs.get("runner", self.runner))

    def test_optimizer_interval_final_and_duplicate_suppression(self):
        evaluator = self.evaluator()
        writer = Writer()
        self.assertIsNone(evaluator.maybe_run(self.scorer, 0, 1))
        self.assertIsNone(evaluator.maybe_run(self.scorer, 499, 3))
        first = evaluator.maybe_run(self.scorer, 500, 3, writer=writer)
        self.assertEqual(first["status"], "scored")
        self.assertTrue(self.scorer.training)
        self.assertIsNone(evaluator.maybe_run(self.scorer, 500, 3, final=True))
        last = evaluator.maybe_run(self.scorer, 620, 4, final=True, writer=writer)
        self.assertTrue(last["final"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(writer.values), 4)
        self.assertTrue(Path(last["checkpoint"]).exists())
        rows = [json.loads(line) for line in (self.root / "output/online_validation.jsonl").read_text().splitlines()]
        self.assertEqual([r["global_step"] for r in rows], [500, 620])

    def test_partial_results_do_not_become_complete_metrics(self):
        evaluator = self.evaluator(runner=lambda *args: self.runner(*args, returncode=1, scored=1))
        writer = Writer()
        report = evaluator.maybe_run(self.scorer, 500, 3, writer=writer)
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["cost_complete"])
        self.assertEqual(writer.values, [])
        self.assertTrue(self.scorer.training)

    def test_environment_failure_preserves_checkpoint_and_training_mode(self):
        def fail(*args): raise RuntimeError("environment failed")
        report = self.evaluator(runner=fail).maybe_run(self.scorer, 500, 3)
        self.assertEqual(report["status"], "error")
        self.assertIn("environment failed", report["error"])
        self.assertTrue(Path(report["checkpoint"]).exists())
        self.assertTrue(self.scorer.training)

    def test_eval_mode_remains_eval_and_timeout_stops_child(self):
        evaluator = self.evaluator()
        self.scorer.eval()
        evaluator.maybe_run(self.scorer, 1, 1, final=True)
        self.assertFalse(self.scorer.training)
        with self.assertRaises(subprocess.TimeoutExpired):
            evaluator._run_process([sys.executable, "-c", "import time; time.sleep(30)"],
                                   self.root / "timeout.log", 0.1)

    def test_invalid_controls_and_task_overlap_are_rejected(self):
        for config in [replace(self.config, every_steps=0), replace(self.config, step_limit=0),
                       replace(self.config, timeout=float("nan"))]:
            with self.assertRaises(ValueError): OnlineEvaluator(config, self.root, ["a", "b"])
        with self.assertRaises(ValueError):
            OnlineEvaluator(replace(self.config, profiles=["a"]), self.root, ["a", "b"])
        evaluator = self.evaluator()
        evaluator.tasks = [{"benchmark": "scienceworld", "task_id": "test--same"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluator.check_training_tasks([{"benchmark": "scienceworld", "task_id": "train--same"}])
        evaluator.check_training_tasks([{"benchmark": "scienceworld", "task_id": "train--different"}])

    def fixture(self, kind="scripted"):
        task = self.config.dataset / "tasks/sw-task"
        task.mkdir(parents=True)
        (task / "manifest.json").write_text(json.dumps({
            "id": "sw-task", "source": {"split": "test"},
            "environment": {"benchmark": "scienceworld", "task_name": "boil", "variation_idx": 0},
        }))
        (self.root / "responses.json").write_text("[]")
        self.config.harness_config.write_text(json.dumps({"profiles": {
            name: {"kind": kind, "model": name, "responses_file": "responses.json",
                   "api_key_env": "MTR_ONLINE_TEST_KEY", "base_url": "https://example.invalid/v1"}
            for name in ["a", "b"]}}))
        self.config.pricing.write_text(json.dumps({"currency": "CNY", "as_of": "test", "profiles": {
            name: {"input_per_million": 0, "output_per_million": 0} for name in ["a", "b"]}}))

    def test_preflight_uses_actual_harness_dry_run_without_providers(self):
        self.fixture()
        evaluator = self.evaluator()
        evaluator.preflight()
        self.assertEqual(len(evaluator.tasks), 1)
        self.assertFalse(evaluator.root.exists())

    def test_missing_credentials_fail_before_training_or_subprocess(self):
        self.fixture(kind="openai")
        with patch.dict(os.environ, {"MTR_ONLINE_TEST_KEY": ""}), patch("subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "credential variables"):
                self.evaluator().preflight()
            run.assert_not_called()

    def parsed_training_args(self, config, overrides=()):
        from mixed_pilot_mtrouter.cli import train_q_function

        with patch.object(sys, "argv", ["train", "--config", str(config), *overrides]), \
                patch.object(train_q_function, "train") as train, redirect_stdout(io.StringIO()):
            train_q_function.main()
        return train.call_args.args[0]

    def test_online_defaults_remain_disabled_without_yaml_section(self):
        import yaml

        config = self.root / "training.yaml"
        for section in [{}, {"online_evaluation": None}]:
            config.write_text(yaml.safe_dump({"model_pool": ["a", "b"], **section}))
            args = self.parsed_training_args(config)
            self.assertIsNone(args.eval_dataset)
            self.assertEqual(args.eval_every_steps, 500)
            self.assertEqual(args.eval_step_limit, 50)
            self.assertIsNone(args.eval_model_profiles)

    def test_online_yaml_defaults_and_command_line_precedence(self):
        import yaml

        config = self.root / "training.yaml"
        config.write_text(yaml.safe_dump({"model_pool": ["a", "b"], "online_evaluation": {
            "dataset": str(self.config.dataset), "every_steps": 123,
            "harness_config": str(self.config.harness_config), "pricing": str(self.config.pricing),
            "split": "train", "step_limit": 17, "timeout": 120.5,
            "python": sys.executable, "model_profiles": ["profile-a", "profile-b"],
        }}))
        args = self.parsed_training_args(config)
        self.assertEqual((args.eval_dataset, args.eval_config, args.eval_pricing, args.eval_python),
                         (self.config.dataset, self.config.harness_config, self.config.pricing, Path(sys.executable)))
        self.assertEqual((args.eval_every_steps, args.eval_split, args.eval_step_limit, args.eval_timeout),
                         (123, "train", 17, 120.5))
        self.assertEqual(args.eval_model_profiles, ["profile-a", "profile-b"])

        args = self.parsed_training_args(config, [
            "--eval-dataset", "/cli/dataset", "--eval-config", "/cli/harness.json",
            "--eval-pricing", "/cli/prices.json", "--eval-python", "/cli/python",
            "--eval-every-steps", "456", "--eval-split", "all", "--eval-step-limit", "23",
            "--eval-timeout", "300", "--eval-model-profiles", "cli-a", "cli-b",
        ])
        self.assertEqual((args.eval_dataset, args.eval_config, args.eval_pricing, args.eval_python),
                         (Path("/cli/dataset"), Path("/cli/harness.json"), Path("/cli/prices.json"), Path("/cli/python")))
        self.assertEqual((args.eval_every_steps, args.eval_split, args.eval_step_limit, args.eval_timeout),
                         (456, "all", 23, 300.0))
        self.assertEqual(args.eval_model_profiles, ["cli-a", "cli-b"])

    def test_training_hooks_reuse_cache_and_run_interval_plus_final(self):
        import torch
        import yaml
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from mixed_pilot_mtrouter.cli.precompute_embeddings import main as precompute
        from mixed_pilot_mtrouter.cli.train_q_function import main as train

        self.fixture()
        encoder = self.root / "encoder"
        encoder.mkdir()
        (encoder / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntask\n")
        BertTokenizerFast(vocab_file=str(encoder / "vocab.txt")).save_pretrained(encoder)
        BertModel(BertConfig(vocab_size=6, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16,
                            max_position_embeddings=128)).save_pretrained(encoder)
        data = self.root / "train-data"
        data.mkdir()
        rows = [{"schema": "mtrouter-step-v1", "sample_id": f"{task}-{name}",
                 "task_id": f"train--training-{task}", "benchmark": "scienceworld",
                 "step_idx": 0, "run_id": 0, "state": {"history": [{"role": "user", "content": "task"}]},
                 "action_model": name, "y": float(name == "b"), "cost": 0.01,
                 "remaining_cost": 0.01, "cost_currency": "CNY", "labels_valid": True,
                 "cost_complete": True} for task in range(4) for name in ["a", "b"]]
        (data / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        metadata = self.root / "models.yaml"
        metadata.write_text(yaml.safe_dump({"models": {"a": {}, "b": {}}}))
        config = self.root / "training.yaml"
        config.write_text(yaml.safe_dump({
            "model_pool": ["a", "b"],
            "history_encoder": {"model_name": str(encoder), "max_tokens": 64,
                                "pooling_mode": "cls", "backend": "hf"},
            "model_encoder": {"custom_models_path": str(metadata)},
            "q_network": {"hidden_dims": [8], "dropout": 0.1},
            "precompute": {"enabled": True, "batch_size": 2, "backend": "hf"},
            "online_evaluation": {
                "dataset": str(self.config.dataset), "harness_config": str(self.config.harness_config),
                "pricing": str(self.config.pricing), "every_steps": 2,
            },
        }))
        cache = self.root / "embeddings.pt"
        with patch.object(sys, "argv", ["precompute", "--config", str(config), "--data-dir", str(data),
                                        "--output", str(cache), "--device", "cpu"]), redirect_stdout(io.StringIO()):
            precompute()
        before = hashlib.sha256(cache.read_bytes()).hexdigest()

        def runner(evaluator, command, log_path, timeout):
            self.calls.append(command)
            service = command[command.index("--router-service-url") + 1]
            request = Request(service + "/route", data=json.dumps({"models": ["a", "b"],
                              "messages": [{"role": "user", "content": "task"}]}).encode(),
                              headers={"Content-Type": "application/json"})
            with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
                self.assertIn(json.load(response)["model"], ["a", "b"])
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({
                "total_tasks": 1, "total_traces": 1, "scored_traces": 1, "unscored_traces": 0,
                "passed": 1, "pass_rate": 1.0, "total_api_cost": 0.0,
                "currency": "CNY", "cost_complete": True}))
            return 0

        output = self.root / "trained"
        args = ["train", "--config", str(config), "--data-dir", str(data),
                "--precomputed-path", str(cache), "--output-dir", str(output),
                "--device", "cpu", "--epochs", "1", "--batch-size", "2", "--val-ratio", "0.25"]
        with patch.object(sys, "argv", args), patch.object(OnlineEvaluator, "_run_process", runner), \
                redirect_stdout(io.StringIO()):
            train()
        records = [json.loads(line) for line in (output / "online_validation.jsonl").read_text().splitlines()]
        self.assertEqual([r["global_step"] for r in records], [2, 3])
        self.assertEqual([r["final"] for r in records], [False, True])
        self.assertTrue(all(r["status"] == "scored" for r in records))
        self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(), before)
        self.assertTrue((output / "q_function_best.pt").exists())
        # Monitoring must not change the optimizer trajectory or model weights.
        baseline = self.root / "baseline"
        config_values = yaml.safe_load(config.read_text())
        config_values["online_evaluation"]["dataset"] = None
        config.write_text(yaml.safe_dump(config_values))
        plain_args = args.copy()
        plain_args[plain_args.index("--output-dir") + 1] = str(baseline)
        with patch.object(sys, "argv", plain_args), redirect_stdout(io.StringIO()):
            train()
        monitored = torch.load(output / "q_function.pt", weights_only=True)["state_dict"]
        original = torch.load(baseline / "q_function.pt", weights_only=True)["state_dict"]
        self.assertEqual(set(monitored), set(original))
        self.assertTrue(all(torch.equal(monitored[k], original[k]) for k in original))


if __name__ == "__main__": unittest.main()
