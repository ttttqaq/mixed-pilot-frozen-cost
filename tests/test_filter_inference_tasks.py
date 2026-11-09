import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from filter_inference_tasks import filter_run, read, write


class FilterRunTests(unittest.TestCase):
    def test_whole_task_filter_preserves_backup_and_kept_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            for folder in ["calls", "traces", "workspaces", "leaf-data"]:
                (root / folder).mkdir(parents=True)
            nodes = []
            jobs = [{"task_id":t} for t in ["keep", "remove"]]
            for i, task in enumerate(["keep", "remove", "remove"]):
                node_id, call_id = f"n{i}", f"c{i}"
                nodes.append(dict(id=node_id, task_id=task, children=[], parent_id=None,
                                  prefix_call_ids=[], own_call_ids=[call_id], superseded=i==2,
                                  labels_valid=True, success=True, status="finished"))
                write(root / "calls" / f"{call_id}.json", dict(call_id=call_id, node_id=node_id,
                      selected_profile="model", cost=i+1, response="original", status="completed"))
                write(root / "traces" / f"{node_id}.traj.json", {"task":task})
                (root / "workspaces" / node_id).mkdir()
            write(root / "plan.json", {"tasks":jobs})
            write(root / "tree.json", {"nodes":nodes,"jobs":jobs})
            write(root / "summary.json", dict(mode="single-model", jobs=jobs, total_api_cost=6, currency="CNY"))
            write(root / "leaf-data/summary.json", {})
            for name in ["samples.jsonl", "steps.jsonl", "leaf-data/traces.jsonl", "leaf-data/steps.jsonl"]:
                (root/name).write_text(''.join(json.dumps(dict(task_id=t, call_id=f"c{i}", labels_valid=True))+'\n'
                                             for i,t in enumerate(["keep","remove"])))
            (root/"events.jsonl").write_text(''.join(json.dumps({"call_id":f"c{i}"})+'\n' for i in range(3)))
            original_bytes = (root/"calls/c0.json").read_bytes()
            filter_run(root, {"remove"})
            summary = read(root/"summary.json")
            self.assertEqual(summary["total_api_cost"], 1)
            self.assertEqual(summary["original_run_known_api_cost"], 6)
            self.assertEqual(summary["total_traces"], 1)
            self.assertEqual((root/"calls/c0.json").read_bytes(), original_bytes)
            self.assertFalse((root/"traces/n2.traj.json").exists())
            self.assertFalse((root/"workspaces/n1").exists())
            backup = Path(read(root/"FILTER_REPORT.json")["backup"])
            self.assertTrue((backup/"calls/c2.json").exists())
            self.assertEqual(read(backup/"summary.json")["total_api_cost"], 6)
            with self.assertRaisesRegex(ValueError, "already been filtered"):
                filter_run(root, {"remove"})


if __name__ == "__main__":
    unittest.main()
