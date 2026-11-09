"""Online configuration and split guards; no paid calls or native environments."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mixed_pilot_mtrouter.online.config import load_config
from mixed_pilot_mtrouter.online.data import assert_training_rows, load_train_valid


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "online.yaml"
        self.write({"project_root": str(self.root)})

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, data):
        self.path.write_text(json.dumps(data))  # JSON is a strict subset of YAML.

    def test_paths_resolve_against_declared_project_and_override_types(self):
        config = load_config(self.path, ["online.iterations=2", "teacher.enabled=false",
                                         "model.hidden_dims=[16, 8]", "output_dir=outputs/new"])
        self.assertEqual(config["online"]["iterations"], 2)
        self.assertIs(config["teacher"]["enabled"], False)
        self.assertEqual(config["model"]["hidden_dims"], [16, 8])
        self.assertEqual(config["output_dir"], str(self.root / "outputs/new"))
        self.assertEqual(config["dataset"]["root"], str(self.root / "data/VirtualWorld-subset"))
        self.assertEqual(config["teacher"]["llm"]["profile"], "gpt-6.1-terra")

    def test_unknown_fields_and_malformed_overrides_fail(self):
        for data in [{"model": {"hiden_dims": [4]}}, {"datasets": {}}, {"teacher": None}]:
            self.write(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                load_config(self.path)
        self.write({})
        for override in ["online.unknown=3", "online=3", "online.iterations", "online.iterations.x=4"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(self.path, [override])

    def test_duplicate_yaml_keys_do_not_silently_shadow_configuration(self):
        self.path.write_text("seed: 1\nseed: 2\n")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML"):
            load_config(self.path)


    def test_test_is_evaluation_only_and_training_is_train(self):
        config = load_config(self.path)
        self.assertEqual(config["dataset"]["validation_split"], "test")
        for field, split in [("train_split", "test"), ("validation_split", "train"), ("validation_split", "all")]:
            self.write({"dataset": {field: split}})
            with self.assertRaises(ValueError): load_config(self.path)

    def test_invalid_numerics_and_booleans_are_refused(self):
        for override in ["budget.total=.nan", "budget.lambda_=-1", "online.iterations=0",
                         "online.iterations=true", "selection.enabled=1", "audit.random_fraction=1.1",
                         "rollout.step_limit=2.5", "model.dropout=1", "model.hidden_dims=[]",
                         "short_rollout.horizon=10", "encoder.device=cuda:garbage"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(self.path, [override])

    def test_teacher_is_the_strong_member_of_exactly_two_profiles(self):
        config = load_config(self.path, ["model_pool=[a, b]", "strong_model=a"])
        self.assertEqual(config["teacher"]["llm"]["profile"], "gpt-6.1-terra")
        for overrides in [["strong_model=other"], ["model_pool=[a, a]"],
                          ["model_pool=[a, b, c]"], ["teacher.llm.profile=qwen-flash-2025-07-28"]]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(self.path, overrides)

    def test_teacher_ablation_is_one_switch_and_policy_replay_cannot_be_mixed(self):
        config = load_config(self.path, ["teacher.enabled=false"])
        self.assertFalse(config["teacher"]["enabled"])
        self.assertFalse(config["teacher"]["enabled"])
        self.assertEqual(config["teacher"]["llm"]["max_history_chars"], 16000)
        for override in ["teacher.real_replay_iterations=2", "teacher.llm.max_history_chars=0",
                         "teacher.llm.anchor_fraction=1.1", "teacher.evidence_dropout=1"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(self.path, [override])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for split, task_id, variation in [("train", "one", 1), ("valid", "two", 2)]:
            self.add(split, task_id, variation)
        # Malformed deliberately: a loader which touches test fails this test.
        (self.root / "test.json").write_text("DO NOT READ THE TEST INDEX")
        self.config = {"dataset": {"root": str(self.root), "train_split": "train", "validation_split": "valid"}}

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, split, task_id, variation):
        path = self.root / "tasks" / task_id
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": task_id,
            "environment": {"benchmark": "scienceworld", "task_name": "boil", "variation_idx": variation},
            "source": {"split": "train", "revision": "frozen"},
            "routing_metadata": {"router_split": split},
        }
        (path / "manifest.json").write_text(json.dumps(manifest))
        (self.root / f"{split}.json").write_text(json.dumps({
            "split": split, "task_count": 1, "tasks": [{"id": task_id, "path": f"tasks/{task_id}"}],
        }))

    def test_only_selected_indexes_are_read_and_valid_native_tasks_keep_valid_split(self):
        real_read = Path.read_text
        real_bytes = Path.read_bytes
        touched = []
        def checked_read(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.json")
            touched.append(path.name)
            return real_read(path, *args, **kwargs)
        def checked_bytes(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.json")
            return real_bytes(path, *args, **kwargs)
        with patch.object(Path, "read_text", checked_read), patch.object(Path, "read_bytes", checked_bytes):
            train, valid, manifest = load_train_valid(self.config)
        self.assertEqual([t["split"] for t in train], ["train"])
        self.assertEqual([t["split"] for t in valid], ["valid"])
        self.assertEqual(valid[0]["task_id"], "valid--two")
        self.assertEqual(manifest["splits"]["valid"]["count"], 1)
        self.assertEqual(len(manifest["split_fingerprint"]), 64)
        self.assertEqual("excluded", manifest["test_access"])

    def test_selected_test_is_loaded_without_reading_valid(self):
        self.add("test", "three", 3)
        (self.root/"valid.json").write_text("MUST NOT READ VALID")
        self.config["dataset"]["validation_split"] = "test"
        train, evaluation, manifest = load_train_valid(self.config)
        self.assertEqual([t["task_id"] for t in evaluation], ["test--three"])
        self.assertEqual(manifest["evaluation_split"], "test")
        self.assertEqual(set(manifest["splits"]), {"train", "test"})
        with self.assertRaises(ValueError): assert_training_rows(evaluation)

    def test_same_native_instance_in_train_and_valid_is_refused(self):
        self.add("valid", "two", 1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_train_valid(self.config)

    def test_missing_valid_index_never_falls_back_to_globbing(self):
        (self.root / "valid.json").unlink()
        with self.assertRaisesRegex(ValueError, "Explicit valid.json"):
            load_train_valid(self.config)

    def test_index_manifest_split_mismatch_and_escaping_path_are_refused(self):
        index = self.root / "valid.json"
        data = json.loads(index.read_text())
        data["tasks"][0]["path"] = "../outside"
        index.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "escaping"):
            load_train_valid(self.config)
        self.add("valid", "two", 2)
        path = self.root / "tasks/two/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["routing_metadata"]["router_split"] = "test"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_train_valid(self.config)

    def test_training_update_guard_rejects_evaluation_or_ambiguous_rows(self):
        assert_training_rows([{"state": {"split": "train", "task_id": "train--one"}}])
        for row in [{"state": {"split": "valid", "task_id": "valid--two"}},
                    {"split": "test", "task_id": "test--three"},
                    {"state": {"split": "train", "task_id": "test--three"}},
                    {"state": {"task_id": "one"}}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                assert_training_rows([row])


if __name__ == "__main__":
    unittest.main()
