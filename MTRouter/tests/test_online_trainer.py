"""End-to-end online iterations with real torch updates and offline fixtures.

The tasks are synthetic metadata, not benchmark evaluation. These tests check
policy-versioned teacher grounding, paid-call accounting, train/valid isolation,
and paired rollout semantics without API credentials, Java, or GPU models.
"""
import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_mtrouter.online.config import DEFAULTS
from mixed_pilot_mtrouter.online.learning import state_key
from mixed_pilot_mtrouter.online.smoke import smoke_config, smoke_factories
from mixed_pilot_mtrouter.online.trainer import OnlineTrainer


def rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


class OnlineTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = smoke_config(deepcopy(DEFAULTS))
        self.config.update(project_root=str(self.root), output_dir=str(self.root / "run"))
        self.config["encoder"].update(hash_dim=8, device="cpu")
        self.config["model"].update(hidden_dims=[8], model_metadata=None,
                                    meta_embed_dim=8, residual_embed_dim=4, model_dim=8)
        self.config["teacher"].update(hidden_dims=[8], dropout=0., min_real_samples=2, epochs=1)
        self.config["teacher"]["llm"].update(max_calls_per_iteration=100)
        self.config["optimization"].update(epochs=1, batch_size=8)
        self.config["online"].update(iterations=2, tasks_per_iteration=2,
                                      states_per_iteration=2, exploration_epsilon=0.)
        self.config["audit"].update(enabled=False, random_fraction=0.)
        self.config["selection"].update(enabled=False)
        self.config["rollout"].update(step_limit=6)
        self.config["evaluation"].update(every_iterations=1, max_tasks=2)
        self.train = [self.task("train", i) for i in range(2)]
        self.valid = [self.task("test", i + 20) for i in range(2)]
        self.manifest = {"schema": "synthetic-train-valid-v1", "split_fingerprint": "fixture-fingerprint",
                         "splits": {"train": {"task_ids": [t["task_id"] for t in self.train]},
                                    "valid": {"task_ids": [t["task_id"] for t in self.valid]}},
                         "test_access": "excluded"}
        self.references = {t["task_id"]: {"actions": ["PRIVATE_REFERENCE"], "sha256": "fixture-ref"} for t in self.train}
        self.harness = HarnessConfig(self.root / "harness.json", self.config["model_pool"][0], {
            name: ProviderProfile(kind="scripted", model=name, name=name)
            for name in self.config["model_pool"]})
        teacher = self.config["teacher"]["llm"]["profile"]
        self.harness.profiles[teacher] = ProviderProfile(kind="scripted", name=teacher, model=teacher)
        self.pricing = {"currency": "CNY", "profiles": {name: {"input_per_million": 1.,
                       "output_per_million": 1.} for name in [*self.config["model_pool"], teacher]}}

    @staticmethod
    def task(split, index):
        return {"task_id": f"{split}--synthetic-{index}", "source_task_id": f"synthetic-{index}",
                "benchmark": "scienceworld", "task_name": f"synthetic-{index}",
                "variation_idx": index, "split": split,
                "content_hashes": {"manifest": f"fixture-{split}-{index}"}}

    def tearDown(self):
        self.temp.cleanup()

    def trainer(self, config=None, **kwargs):
        config = config or self.config
        episode, provider = smoke_factories(config)
        return OnlineTrainer(config, self.train, self.valid, self.manifest,
                             self.harness, self.pricing, simulation=True,
                             episode_factory=episode, provider_factory=provider, references=self.references, **kwargs)

    @staticmethod
    def run_trainer(trainer):
        # Deterministic first states deliberately force nonterminal short forks;
        # acquisition-function behavior is tested separately from the loop.
        with redirect_stdout(io.StringIO()), patch.object(trainer, "_select", side_effect=lambda visits, count: visits[:count]):
            return asyncio.run(trainer.run())

    def test_two_iterations_ground_teacher_update_router_and_keep_valid_out_of_replay(self):
        trainer = self.trainer()
        before = {key: value.clone() for key, value in trainer.learner.router.state_dict().items()}
        summary = self.run_trainer(trainer)
        self.assertEqual(summary["stop_reason"], "iterations_completed")
        self.assertEqual(summary["completed_iterations"], 2)
        self.assertEqual(summary["test_tasks_used"], 2)
        self.assertEqual(summary["evaluation_split"], "test")
        self.assertTrue(summary["simulation"])
        cold=next(json.loads(f.read_text()) for f in (trainer.out/"trajectories").glob("*.json") if json.loads(f.read_text())["scope"]=="cold_start")
        decisions=rows(trainer.out/"policy_decisions.jsonl")
        self.assertTrue(all(set(d["probabilities"].values())=={.5} for d in decisions if d["policy_version"]==cold["policy_version"]))
        logs = rows(trainer.out / "training_log.jsonl")
        self.assertEqual([row["iteration"] for row in logs], [1, 2])
        self.assertTrue(all(row["router"]["updates"] > 0 for row in logs))
        self.assertTrue(all(row["teacher"] == "fixed_reference_llm_no_updates" for row in logs))
        self.assertFalse(hasattr(trainer.learner, "teacher"))
        self.assertGreater(trainer.learner.router_updates, 0)
        self.assertTrue(any(not torch.equal(value, trainer.learner.router.state_dict()[key]) for key, value in before.items()))
        self.assertGreater(len(trainer.pairs), 0)
        self.assertTrue(any(pair["source"] == "llm_teacher" for pair in trainer.pairs.values()))
        self.assertTrue(all(row["state"]["split"] == "train" for row in trainer.router_rows.values()))
        self.assertFalse(any(row["state"]["task_id"].startswith("valid--") for row in trainer.router_rows.values()))
        validation = rows(trainer.out / "evaluation.jsonl")
        self.assertEqual(len(validation), 2)
        self.assertTrue(all(v["split"] == "test" and v["training_updates"] is False for v in validation))
        self.assertTrue(all(v["planned_tasks"] == 2 and v["complete"] for v in validation))
        self.assertTrue((trainer.out / "latest.pt").exists())
        self.assertTrue((trainer.out / "best_eval.pt").exists())

    def test_pairs_share_exact_before_state_and_assessor_cost_is_separate(self):
        trainer = self.trainer()
        self.run_trainer(trainer)
        trajectories = {path.stem: json.loads(path.read_text()) for path in (trainer.out / "trajectories").glob("*.json")}
        for pair in trainer.pairs.values():
            prefix_length = pair["state"]["step_idx"]
            branches = [trajectories[name] for name in pair["branch_trajectory_ids"]]
            first = [branch["all_steps"][prefix_length] for branch in branches]
            self.assertEqual([step["profile"] for step in first], self.config["model_pool"])
            self.assertEqual(state_key(first[0]["before_state"]), state_key(first[1]["before_state"]))
            self.assertEqual(first[0]["before_state"]["snapshot_id"], first[1]["before_state"]["snapshot_id"])
            self.assertEqual(pair["intervention_call_ids"], [step["call_id"] for step in first])
            # Actual suffix cost excludes inherited prefixes and assessor calls.
            for model, branch in zip(self.config["model_pool"], branches):
                known_suffix = sum(step["cost"] for step in branch["all_steps"][prefix_length:])
                self.assertGreaterEqual(pair["targets"][model]["cost"] + 1e-12, known_suffix)
        records = list(trainer.ledger.records.values())
        assessor = [r for r in records if r["scope"] == "teacher"]
        self.assertTrue(assessor)
        self.assertTrue(all(r["profile"] == self.config["teacher"]["llm"]["profile"] for r in assessor))
        self.assertTrue(all("training-only assessor" in r["request"]["system"] for r in assessor))
        self.assertTrue(all("test--" not in json.dumps(r["request"]) for r in assessor))
        actors = [r for r in records if r["scope"] != "teacher"]
        self.assertTrue(all("PRIVATE_REFERENCE" not in json.dumps(r["request"]) for r in actors))
        self.assertTrue(all("PRIVATE_REFERENCE" not in json.dumps(r["state"]) for r in trainer.router_rows.values()))
        report = trainer.ledger.report()
        self.assertAlmostEqual(report["by_scope"]["teacher"]["known_cost"], sum(r["cost"] for r in assessor))
        self.assertAlmostEqual(report["total_cost"], sum(r["cost"] for r in records))
        self.assertEqual(len({r["call_id"] for r in records}), len(records))
        self.assertTrue(report["cost_complete"])

    def test_completed_resume_restores_models_and_does_not_buy_duplicate_validation(self):
        first = self.trainer()
        self.run_trainer(first)
        count = len(first.ledger.records)
        cost = first.ledger.cost
        evaluations = len(rows(first.out / "evaluation.jsonl"))
        expected = first.learner.predict_actions(next(iter(first.router_rows.values()))["state"])
        restored = self.trainer(resume=True)
        self.assertEqual(expected, restored.learner.predict_actions(next(iter(restored.router_rows.values()))["state"]))
        summary = self.run_trainer(restored)
        self.assertEqual(summary["completed_iterations"], 2)
        self.assertEqual(len(restored.ledger.records), count)
        self.assertAlmostEqual(restored.ledger.cost, cost)
        self.assertEqual(len(rows(first.out / "evaluation.jsonl")), evaluations)

    def test_validation_resume_does_not_repeat_training_or_completed_tasks(self):
        from mixed_pilot_mtrouter.online.runtime import RecoverableRequestError
        trainer = self.trainer()
        original = trainer.runner.rollout
        async def interrupted(task, policy, **kwargs):
            if kwargs.get("scope") == "test" and task["task_id"] == self.valid[1]["task_id"]:
                partial = await original(task, policy, horizon=1, **kwargs)
                error = RecoverableRequestError("injected mid-validation failure")
                error.rollout_result = partial
                raise error
            return await original(task, policy, **kwargs)
        trainer.runner.rollout = interrupted
        summary = self.run_trainer(trainer)
        self.assertEqual(summary["completed_iterations"], 1)
        self.assertEqual(summary["stop_reason"], "request_retries_exhausted")
        self.assertEqual(len(rows(trainer.out / "training_log.jsonl")), 1)
        restored = self.trainer(resume=True)
        summary = self.run_trainer(restored)
        self.assertEqual(summary["completed_iterations"], 2)
        self.assertEqual([r["iteration"] for r in rows(restored.out / "training_log.jsonl")], [1, 2])
        self.assertEqual(restored.ledger.report()["by_scope"]["test"]["calls"], 16)
        self.assertEqual([r["iteration"] for r in rows(restored.out / "evaluation.jsonl")], [1, 2])

    def test_validation_keeps_native_success_separate_from_uncertain_cost(self):
        trainer = self.trainer()
        async def outcome(task, policy, **kwargs):
            return {"task_id":task["task_id"],"labels_valid":False,"outcome_valid":True,
                    "outcome_reward":1.,"reward":None,"success":True,"cost_complete":False,
                    "trajectory_id":task["task_id"]}
        trainer.runner.rollout = outcome
        with redirect_stdout(io.StringIO()):
            report = asyncio.run(trainer._evaluate(0))
        self.assertTrue(report["complete"])
        self.assertEqual(report["pass_rate"], 1.)
        self.assertEqual(report["mean_reward"], 1.)
        self.assertIsNone(report["mean_cost"])
        self.assertFalse(report["cost_complete"])
        self.assertFalse(trainer.router_rows)

    def test_validation_resume_keeps_uncertain_attempt_fees_in_cost_report(self):
        from mixed_pilot_mtrouter.online.runtime import RecoverableRequestError
        self.config["online"]["iterations"] = 1
        trainer = self.trainer()
        original = trainer.runner.rollout
        async def interrupted(task, policy, **kwargs):
            if kwargs.get("scope") == "test" and task["task_id"] == self.valid[1]["task_id"]:
                partial = await original(task, policy, horizon=1, **kwargs)
                cid = trainer.ledger.begin("test", self.config["strong_model"])
                trainer.ledger.finish(cid, cost=None, complete=False, status="request_error")
                trainer.ledger.reserve(cid, .25, "injected timed-out request")
                error = RecoverableRequestError("injected mid-validation timeout")
                error.rollout_result = partial
                raise error
            return await original(task, policy, **kwargs)
        trainer.runner.rollout = interrupted
        self.assertEqual(self.run_trainer(trainer)["stop_reason"], "request_retries_exhausted")
        restored = self.trainer(resume=True)
        self.assertEqual(self.run_trainer(restored)["stop_reason"], "iterations_completed")
        report = rows(restored.out / "evaluation.jsonl")[-1]
        completed = json.loads((restored.out / "evaluation-progress.json").read_text())["results"].values()
        self.assertTrue(all(result["cost_complete"] for result in completed))
        self.assertTrue(report["complete"])
        self.assertAlmostEqual(report["reserved_cost"], .25)
        self.assertFalse(report["cost_complete"])
        self.assertIsNone(report["mean_cost"])
        self.assertAlmostEqual(report["mean_budget_cost"], (report["cost"] + .25) / 2)

    def test_full_rollout_teacher_ablation_uses_only_real_labels(self):
        config = deepcopy(self.config)
        config["output_dir"] = str(self.root / "full")
        config["teacher"]["enabled"] = False
        config["teacher"]["enabled"] = False
        config["short_rollout"]["enabled"] = False
        config["online"].update(iterations=1, tasks_per_iteration=1, states_per_iteration=1)
        config["cold_start"]["enabled"] = False
        trainer = self.trainer(config)
        summary = self.run_trainer(trainer)
        self.assertEqual(summary["completed_iterations"], 1)
        self.assertFalse(hasattr(trainer.learner, "teacher_optimizer"))
        self.assertTrue(trainer.pairs)
        self.assertTrue(all(p["source"] == "real" for p in trainer.pairs.values()))
        self.assertTrue(all(r["source"] == "real" for r in trainer.router_rows.values()))
        self.assertNotIn("teacher", trainer.ledger.report()["by_scope"])
        decisions = rows(trainer.out / "policy_decisions.jsonl")
        logged = {(d["snapshot_id"], d["selected_model"]) for d in decisions}
        for pair in trainer.pairs.values():
            start = pair["state"]["step_idx"]
            for name in pair["branch_trajectory_ids"]:
                branch = json.loads((trainer.out / "trajectories" / (name + ".json")).read_text())
                self.assertTrue(branch["terminal"])
                self.assertTrue(branch["labels_valid"])
                # All calls after the one forced intervention came through the
                # frozen policy, with their decision provenance independently logged.
                for step in branch["all_steps"][start + 1:]:
                    self.assertIn((step["before_state"]["snapshot_id"], step["profile"]), logged)



    def test_greedy_validation_changes_only_validation_cost_not_training_or_sampling_rng(self):
        enabled_config = deepcopy(self.config)
        enabled_config["output_dir"] = str(self.root / "with-validation")
        enabled_config["online"]["exploration_epsilon"] = .25
        enabled_config["budget"]["total"] = 30.
        enabled = self.trainer(enabled_config)
        self.run_trainer(enabled)
        expected_router = {k: v.detach().clone() for k, v in enabled.learner.router.state_dict().items()}
        expected_sampling_rng = enabled.rng.getstate()
        expected_torch_rng = torch.get_rng_state().clone()
        enabled_ledger = enabled.ledger.report()

        disabled_config = deepcopy(enabled_config)
        disabled_config["output_dir"] = str(self.root / "without-validation")
        disabled_config["evaluation"]["enabled"] = False
        disabled = self.trainer(disabled_config)
        self.run_trainer(disabled)
        for name, value in disabled.learner.router.state_dict().items():
            self.assertTrue(torch.equal(expected_router[name], value), "Validation changed router tensor " + name)
        self.assertEqual(expected_sampling_rng, disabled.rng.getstate())
        self.assertTrue(torch.equal(expected_torch_rng, torch.get_rng_state()))
        disabled_ledger = disabled.ledger.report()
        self.assertIn("test", enabled_ledger["by_scope"])
        self.assertNotIn("test", disabled_ledger["by_scope"])
        enabled_training_scopes = {k: v for k, v in enabled_ledger["by_scope"].items() if k != "test"}
        self.assertEqual(enabled_training_scopes, disabled_ledger["by_scope"])
        self.assertAlmostEqual(enabled_ledger["total_cost"] - disabled_ledger["total_cost"],
                               enabled_ledger["by_scope"]["test"]["known_cost"])
        with_logs = rows(enabled.out / "training_log.jsonl")
        without_logs = rows(disabled.out / "training_log.jsonl")
        self.assertEqual([(r["policy_version"], r["next_policy_version"]) for r in with_logs],
                         [(r["policy_version"], r["next_policy_version"]) for r in without_logs])




if __name__ == "__main__":
    unittest.main()
