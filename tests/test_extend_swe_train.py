import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extend_swe_train as extension
from convert_swe_tasksets import SOURCES, convert_row, write_json


class ExtendTrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.existing = root / "dataset"
        self.stage = root / "stage"
        self.cache = root / "cache"
        (self.existing / "raw").mkdir(parents=True)
        self.cache.mkdir()
        self.rows = {}
        entries = []
        for source, (dataset, config, upstream_split) in SOURCES.items():
            provenance = dict(dataset=dataset, revision="a" * 40, config=config,
                              split=upstream_split, url="https://example.org", license="test")
            rows = [dict(instance_id=f"{source}-{i}", repo="owner/repo", base_commit=f"{i:040x}",
                         problem_statement=f"Fix task {i}", patch=f"gold-{i}", test_patch=f"test-{i}",
                         FAIL_TO_PASS=["test"], PASS_TO_PASS=[], fail_to_pass=["test"], pass_to_pass=[])
                    for i in range(6)]
            self.rows[source] = rows
            (self.cache / (source + ".json")).write_text(json.dumps(rows))
            write_json(self.existing / "raw" / (source + ".json"),
                       dict(source=provenance, sampling={"population_size": 6}, rows=rows[:2]))
            for index, split in [(0, "train"), (1, "test")]:
                target = convert_row(source, provenance, rows[index], self.existing / split, False)
                manifest = extension.read(target / "manifest.json")
                manifest["routing_metadata"] = {"router_split": split}
                write_json(target / "manifest.json", manifest)
                entries.append(dict(id=target.name, path=target.relative_to(self.existing).as_posix(),
                                    dataset=source, upstream_id=rows[index]["instance_id"], source_row_index=index))
        write_json(self.existing / "catalog.json", {"tasks": entries})
        write_json(self.existing / "train.json", {"tasks": [e for e in entries if e["source_row_index"] == 0]})
        self.args = SimpleNamespace(dataset_root=self.existing, staging=self.stage, cache=self.cache,
                                    target_per_source=3, seed=42, prepare_source=None, publish=False)

    def run_extension(self):
        with patch.object(extension, "source_file", side_effect=lambda old, cache, source: cache / (source + ".json")), \
             patch.object(extension, "records", side_effect=lambda path, source: enumerate(self.rows[source])):
            extension.run(self.args)

    def test_preserves_originals_excludes_test_and_resumes_without_duplicates(self):
        before = extension.snapshot(self.existing)
        self.run_extension()
        self.run_extension()
        self.assertEqual(before, extension.snapshot(self.existing))
        report = extension.validate(self.stage, self.existing, 3)
        self.assertEqual(report["task_count"], 6)
        self.assertEqual(report["new_tasks"], 4)
        for source in SOURCES:
            raw = extension.read(self.stage / "raw" / (source + ".json"))
            self.assertNotIn(1, raw["extension"]["added_indices"])
            self.assertNotIn(0, raw["extension"]["added_indices"])
        hashes = extension.read(self.stage / "checksums.json")
        self.assertNotIn("checksums.json", hashes)
        self.assertTrue(all(extension.digest(self.stage / p) == h for p, h in hashes.items()))

    def test_publish_retains_complete_original_backup(self):
        before = extension.snapshot(self.existing)
        self.args.publish = True
        self.run_extension()
        backups = list(self.existing.parent.glob("dataset.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(extension.snapshot(backups[0]), before)
        self.assertEqual(extension.read(self.existing / "catalog.json")["task_count"], 6)
        self.assertFalse((self.existing / "test").exists())

    def test_original_changes_block_resume(self):
        self.run_extension()
        (self.existing / "changed.txt").write_text("concurrent change")
        with self.assertRaisesRegex(ValueError, "Original collection changed"):
            self.run_extension()


if __name__ == "__main__":
    unittest.main()
