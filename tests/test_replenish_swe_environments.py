import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replenish_swe_environments import baseline_ready, publish


class ReplenishTests(unittest.TestCase):
    def test_only_gold_pass_empty_fail_is_admitted(self):
        for gold, empty, expected in [("passed", "failed", True), ("passed", "error", False),
                                      ("passed", "passed", False), ("failed", "failed", False)]:
            self.assertEqual(baseline_ready({"status":"ready", "prepared_checks":
                             {"gold":{"status":gold}, "empty":{"status":empty}}}), expected)
        self.assertFalse(baseline_ready({"status":"ready"}))

    def test_publication_backs_up_removed_tasks_and_preserves_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            args = SimpleNamespace(dataset=p/"dataset", candidates=p/"candidates", output=p/"report")
            args.output.mkdir()
            entries = []
            for name, split in [("ready", "train"), ("bad", "train"), ("oldtest", "test")]:
                e = dict(id=name, path=f"{split}/tasks/{name}", dataset="source", upstream_id=name)
                task = args.dataset/e["path"]/"reference"
                task.mkdir(parents=True)
                (task/"source_record.json").write_text(json.dumps({"instance_id":name}))
                entries.append(e)
            (args.dataset/"test.json").write_text(json.dumps({"tasks":[entries[2]]}))
            (args.dataset/"sampling.json").write_text('{}')
            state = dict(backup=str(p/"backup"), selected=[entries[0]], excluded=[entries[1]],
                         images={"ready":{"image":"fixture", "source_image":"fixture", "base_commit":"abc"}})
            publish(state, args)
            self.assertFalse((args.dataset/entries[1]["path"]).exists())
            self.assertTrue((p/"backup"/entries[1]["path"] / "reference/source_record.json").exists())
            self.assertTrue((args.dataset/entries[2]["path"]).exists())
            self.assertEqual(json.loads((args.dataset/"train.json").read_text())["task_count"], 1)
            self.assertEqual(json.loads((args.dataset/"environment-images.json").read_text())["images"], state["images"])
            publish(state, args)  # Resume publication is idempotent.
            self.assertTrue((p/"backup"/entries[1]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
