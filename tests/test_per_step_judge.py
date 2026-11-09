"""Per-step judging: evidence boundaries, schema and CLI; no paid API calls."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mixed_pilot_harness.cli import main
from mixed_pilot_harness.config import load_config
from mixed_pilot_harness.per_step_judge import judge_step, prepare_judgment
from mixed_pilot_harness.per_step_judge.core import parse_judgment
from mixed_pilot_harness.types import ProviderResponse, ToolCall, Usage


SCORE = {"score": 65, "success_probability": 0.8, "estimated_remaining_steps": 3,
         "confidence": 0.7, "reason": "已找到目标，但尚未测量。", "evidence": ["环境中发现斜面 B 和 C。"]}


class JudgeFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trace = self.root / "example.traj.json"
        first = {"role": "user", "content": "<task_description>找出最缓斜面。</task_description>\n初始环境：走廊"}
        history = [first, {"role": "assistant", "content": "go to workshop"},
                   {"role": "user", "content": "You move to the workshop."}]
        self.data = {"info": {"trace_id": "example.traj", "task_id": "slope", "run_id": 0,
                              "benchmark": "scienceworld", "task_name": "inclined-plane-determine-angle",
                              "variation_idx": 0, "final_score": -100, "success": False,
                              "evaluation": {"candidate_answer": "FUTURE_ANSWER"}},
                     "messages": [{"role": "assistant", "content": "FUTURE_TOP_LEVEL_MESSAGE"}],
                     "steps": [
                         {"step_idx": 0, "state": {"history": [first], "system": "simulation rules"},
                          "response": "go to workshop", "command": "go to workshop",
                          "observation": "You move to the workshop."},
                         {"step_idx": 1, "state": {"history": history, "system": "simulation rules"},
                          "response": "THOUGHT: inspect\n```text\nlook around\n```", "command": "look around",
                          "observation": "inclined plane B; inclined plane C", "selected_profile": "candidate-secret-name",
                          "requested_model": "candidate-secret-name", "returned_model": "candidate-secret-name"},
                         {"step_idx": 2, "response": "FUTURE_STEP_ANSWER"},
                     ]}
        self.trace.write_text(json.dumps(self.data))
        self.config_file = self.root / "harness.json"
        self.config_file.write_text(json.dumps({"default_profile": "judge", "agent": {"provider_timeout": 37},
            "profiles": {"judge": {"kind": "openai", "model": "judge-model", "max_tokens": 1024}}}))
        self.config = load_config(self.config_file)

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["per-step-judge", "--config", str(self.config_file), "--profile", "judge",
                         "--trace", str(self.trace), *args])
        return code, stdout.getvalue(), stderr.getvalue()


class JudgeTests(JudgeFixture, unittest.IsolatedAsyncioTestCase):
    def test_target_step_and_no_future_outcome_or_model_bias(self):
        result = prepare_judgment(self.trace, 2)
        prompt = result["request"]["prompt"]
        self.assertIn("inclined plane B; inclined plane C", prompt)
        self.assertIn("You move to the workshop.", prompt)
        self.assertIn("找出最缓斜面", prompt)
        for secret in ["FUTURE_", "final_score", "candidate-secret-name", "-100"]:
            self.assertNotIn(secret, prompt)
        self.assertEqual((result["step"], result["step_idx"], result["reference_source"]), (2, 1, "task_goal_only"))
        changed = copy.deepcopy(self.data)
        changed["info"]["final_score"] = 100
        changed["info"]["success"] = True
        changed["steps"][2]["response"] = "CHANGED_FUTURE"
        self.trace.write_text(json.dumps(changed))
        self.assertEqual(prepare_judgment(self.trace, 2)["request"], result["request"])

    def test_explicit_answer_and_hle_tool_results(self):
        step = self.data["steps"][1]
        step.pop("observation")
        step["tool_calls"] = [{"id": "a", "name": "read_file", "arguments": {"path": "data.txt"}}]
        step["tool_results"] = [{"tool_call_id": "a", "name": "read_file", "output": "measurement 25 degrees"}]
        step["state"]["tools"] = [{"name": "read_file"}]
        self.trace.write_text(json.dumps(self.data))
        result = prepare_judgment(self.trace, 2, reference_answer="标准答案：斜面 B")
        self.assertEqual(result["reference_source"], "provided_reference")
        for value in ["measurement 25 degrees", "标准答案：斜面 B", "read_file"]:
            self.assertIn(value, result["request"]["prompt"])

    def test_no_feedback_is_explicit_for_submission(self):
        self.data["steps"][1].pop("observation")
        self.data["steps"][1]["command"] = "task completed"
        self.trace.write_text(json.dumps(self.data))
        prompt = prepare_judgment(self.trace, 2)["request"]["prompt"]
        self.assertIn('"recorded": false', prompt)
        self.assertIn('"observation": null', prompt)

    def test_invalid_inputs_fail_before_provider(self):
        for step in (0, -1, 4, True, 1.5):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, "one-based"):
                prepare_judgment(self.trace, step)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            prepare_judgment(self.trace, 1, reference_answer="  ")
        self.data["steps"][1]["state"]["history"].append({"role": "assistant", "content": "future"})
        self.trace.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "wrong number"):
            prepare_judgment(self.trace, 2)
        self.trace.write_text("{}")
        with self.assertRaisesRegex(ValueError, "collection"):
            prepare_judgment(self.trace, 1)

    def test_score_validation(self):
        self.assertEqual(parse_judgment("```json\n" + json.dumps(SCORE) + "\n```"), SCORE)
        cases = [dict(SCORE, score=-1), dict(SCORE, score=101), dict(SCORE, score=True),
                 dict(SCORE, score=float("nan")), dict(SCORE, success_probability=2),
                 dict(SCORE, confidence="high"), dict(SCORE, estimated_remaining_steps=-1),
                 dict(SCORE, estimated_remaining_steps=False), dict(SCORE, reason=""),
                 dict(SCORE, evidence=[]), dict(SCORE, score=100)]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_judgment(json.dumps(value))
        for text in ['{"score": 50}', '[]', 'prefix ' + json.dumps(SCORE), '{"score":']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_judgment(text)

    async def test_one_shared_chat_call_no_tools_and_usage_preserved(self):
        response = ProviderResponse(text=json.dumps(SCORE), model="actual-judge", stop_reason="stop",
                                    usage=Usage(42, 10, raw={"prompt_cache_hit_tokens": 20}))
        with patch("mixed_pilot_harness.per_step_judge.core.chat_once", new_callable=AsyncMock,
                   return_value=response) as chat:
            result = await judge_step(self.trace, 2, config=self.config, profile="judge")
        chat.assert_awaited_once()
        self.assertEqual(chat.call_args.kwargs["profile_name"], "judge")
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["judgment"], SCORE)
        self.assertEqual(result["response"]["usage"]["raw"], {"prompt_cache_hit_tokens": 20})
        self.assertEqual(result["response"]["model"], "actual-judge")
        self.assertEqual(self.trace.read_text(), json.dumps(self.data))

    async def test_bad_response_is_retained_without_retry(self):
        cases = [ProviderResponse(text="not JSON"),
                 ProviderResponse(text=json.dumps(SCORE), stop_reason="length"),
                 ProviderResponse(text=json.dumps(SCORE), tool_calls=[ToolCall("a", "shell", {})])]
        for response in cases:
            with patch("mixed_pilot_harness.per_step_judge.core.chat_once", new_callable=AsyncMock,
                       return_value=response) as chat:
                result = await judge_step(self.trace, 2, config=self.config, profile="judge")
            self.assertEqual(result["status"], "invalid_response")
            self.assertIsNone(result["judgment"])
            self.assertEqual(result["response"]["text"], response.text)
            chat.assert_awaited_once()


class JudgeCliTests(JudgeFixture, unittest.TestCase):
    # CLI tests are synchronous because main() owns its asyncio event loop.
    def test_dry_run_never_calls_provider_and_saves_prompt(self):
        result = self.root / "prepared.json"
        with patch("mixed_pilot_harness.per_step_judge.core.chat_once", new_callable=AsyncMock) as chat:
            code, stdout, stderr = self.invoke("--step", "2", "--dry-run", "--result", str(result))
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout), json.loads(result.read_text()))
        self.assertEqual(json.loads(stdout)["status"], "prepared")
        chat.assert_not_awaited()

    def test_refuses_overwriting_trace(self):
        code, _, stderr = self.invoke("--step", "2", "--result", str(self.trace))
        self.assertEqual(code, 2)
        self.assertIn("must not overwrite", stderr)
        self.assertEqual(self.trace.read_text(), json.dumps(self.data))

    def test_offline_cli_uses_actual_scripted_chat_adapter(self):
        answers = self.root / "answers.json"
        self.config_file.write_text(json.dumps({"profiles": {
            "judge": {"kind": "scripted", "model": "offline-judge", "responses_file": "answers.json"}}}))
        reference = self.root / "gold.txt"
        reference.write_text("B is the shallowest plane", encoding="utf-8")
        result = self.root / "out" / "judgment.json"
        for response, expected in [(json.dumps(SCORE), 0), ('{"score": 999}', 2)]:
            answers.write_text(json.dumps([{"text": response}]))
            code, stdout, stderr = self.invoke("--step", "2", "--reference-answer-file", str(reference),
                                              "--result", str(result))
            self.assertEqual(code, expected, stderr)
            saved = json.loads(result.read_text())
            self.assertEqual(saved, json.loads(stdout))
            self.assertEqual(saved["reference_source"], "provided_reference")
            self.assertIn("B is the shallowest plane", saved["request"]["prompt"])
            self.assertEqual(saved["response"]["text"], response)
