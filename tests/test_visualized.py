"""Verify resource accounting and portable report output without provider calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("visualized_generate", Path(__file__).resolve().parents[1] / "visualized/generate.py")
visualized = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualized)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_run(self, name="run", summary=None, calls=None):
        path = self.root / name
        (path / "calls").mkdir(parents=True)
        s = dict(total_tasks=2, pass_rate=.5, passed=1, scored_traces=2,
                 total_api_cost=3, currency="CNY", cost_complete=True, api_attempts=2,
                 mode="single-model", stop_reason="tasks_finished")
        s.update(summary or {})
        (path / "summary.json").write_text(json.dumps(s))
        (path / "plan.json").write_text(json.dumps({"tasks": [{"task_id": "a"}, {"task_id": "b"}]}))
        calls = calls if calls is not None else [
            dict(call_id="c0", selected_profile="p", cost=1, usage={"input_tokens":100,"output_tokens":20,"raw":{"prompt_tokens_details":{"cached_tokens":50},"completion_tokens_details":{"reasoning_tokens":10}}},api_attempts=[{"status":"error"},{"status":"response"}]),
            dict(call_id="c1", selected_profile="p", cost=2, usage={"input_tokens":200,"output_tokens":30})]
        for i,c in enumerate(calls):
            (path / "calls" / f"{i}.json").write_text(json.dumps(c))
        return path

    def test_unique_ledger_with_shared_prefix_and_transport_retry(self):
        path=self.make_run()
        # These copies must never enter aggregation.
        (path / "traces").mkdir()
        for i in range(3):
            (path / "traces" / f"{i}.traj.json").write_text(json.dumps({"steps":[{"call_id":"c0","cost":1},{"call_id":"c1","cost":2}]}))
        row=visualized.summarize_run(path)
        self.assertEqual((row["pass_rate"],row["avg_cost_by_currency"]["CNY"]),(.5,1.5))
        self.assertEqual((row["total_tokens"],row["avg_tokens"],row["avg_steps"]),(350,175,1))

    def test_missing_usage_unscored_and_unstarted_do_not_change_denominator(self):
        path=self.make_run(summary={"cost_complete":False,"pass_rate":None,"unscored_traces":1,"unstarted_jobs":1},calls=[
            dict(call_id="c0",cost=1,usage={"input_tokens":100,"output_tokens":20}),
            dict(call_id="c1",cost=None,status="error")])
        row=visualized.summarize_run(path)
        self.assertEqual((row["avg_tokens"],row["avg_steps"]),(60,1))
        self.assertEqual(row["missing_usage_calls"],1)
        self.assertIsNone(row["pass_rate"])
        self.assertFalse(row["tokens_complete"])
        self.assertGreaterEqual(len(row["notes"]),4)

    def test_legacy_task_count_and_token_usage(self):
        path=self.make_run(summary={"total_tasks":None},calls=[
            dict(call_id="c0",cost=1,usage={"raw":{"prompt_tokens":100,"completion_tokens":20}}),
            dict(call_id="c1",cost=2,usage={"prompt_tokens":200,"completion_tokens":30})])
        row=visualized.summarize_run(path)
        self.assertEqual(row["task_count"],2)
        self.assertEqual(row["avg_tokens"],175)

    def test_multi_currency_and_evaluation_cost_use_summary(self):
        path=self.make_run(summary={"cost_by_currency":{"CNY":5,"USD":.2}})
        row=visualized.summarize_run(path)
        self.assertEqual(row["avg_cost_by_currency"],{"CNY":2.5,"USD":.1})

    def test_report_safe_embedding_and_multiple_runs(self):
        a=self.make_run("a")
        b=self.make_run("b",summary={"pass_rate":.75})
        label='</script><img src=x onerror=alert(1)>'
        out,report=visualized.build_report([a,b],self.root / "report",[label,"B"])
        page=(out / "index.html").read_text()
        self.assertNotIn(label,page)
        data=page.split('<script id="report-data" type="application/json">')[1].split('</script>')[0]
        self.assertEqual(json.loads(data)["runs"][0]["label"],label)
        self.assertEqual(len(report["runs"]),2)
        self.assertNotIn("__REPORT_DATA__",page)
        self.assertTrue((out / "metrics.csv").read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_mismatched_ledger_and_duplicate_call_ids_are_errors(self):
        path=self.make_run(summary={"api_attempts":3})
        with self.assertRaisesRegex(ValueError,"logical calls"):
            visualized.summarize_run(path)
        path=self.make_run("duplicate",calls=[dict(call_id="same",cost=1),dict(call_id="same",cost=2)])
        with self.assertRaisesRegex(ValueError,"duplicate call_id"):
            visualized.summarize_run(path)

    def test_invalid_metrics_and_missing_ledger_are_errors(self):
        path=self.make_run(summary={"pass_rate":float("nan")})
        with self.assertRaisesRegex(ValueError,"pass_rate"):
            visualized.summarize_run(path)
        path=self.make_run("missing")
        for file in (path / "calls").iterdir(): file.unlink()
        (path / "calls").rmdir()
        with self.assertRaisesRegex(ValueError,"missing calls/"):
            visualized.summarize_run(path)

    def test_dataset_difference_and_cli_errors(self):
        a=self.make_run("a")
        b=self.make_run("b",summary={"total_tasks":3})
        _,report=visualized.build_report([a,b],self.root / "report")
        self.assertTrue(report["warnings"])
        with self.assertRaisesRegex(ValueError,"--label"):
            visualized.build_report([a,b],self.root / "report",["a"])
        with self.assertRaisesRegex(ValueError,"more than once"):
            visualized.build_report([a,a],self.root / "report")


if __name__ == "__main__":
    unittest.main()
