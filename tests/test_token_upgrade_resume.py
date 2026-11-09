"""Token-limit resume must not re-bill completed tasks or discard failed attempts."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mixed_pilot_harness.collection.collector import CollectionSettings, collect
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.types import ProviderError, ProviderResponse, Usage


TASK = {"task_id": "resume-test", "benchmark": "scienceworld", "task_name": "boil",
        "variation_idx": 0, "simplification_str": "", "split": "train"}
PRICES = {"currency": "CNY", "profiles": {"a": {"input_per_million": 1, "output_per_million": 2}}}


class Environment:
    def __init__(self, **kwargs): pass
    def load_task(self, *args):
        return {"observation": "kitchen", "task_description": "Boil water", "valid_actions": []}
    def step(self, action): return {"observation": "done", "score": 100, "done": True}
    def close(self): pass


class Provider:
    def __init__(self, texts): self.texts = iter(texts)
    async def complete(self, *args):
        item = next(self.texts)
        if isinstance(item, Exception):
            raise item
        return ProviderResponse(text=item, model="a", usage=Usage(10, 10))


class TokenUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def test_hle_same_limit_retry_archives_workspace_and_starts_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "public"
            public.mkdir()
            (public / "README.md").write_text("public input")
            prompt = root / "prompt.md"
            prompt.write_text("What is six times seven?")
            reference = root / "gold.json"
            reference.write_text('{"answer": "42"}')
            task = {"task_id": "hle-retry", "benchmark": "hle", "task_name": "test",
                    "workspace_path": str(public), "prompt_path": str(prompt),
                    "reference_path": str(reference), "allowed_tools": [], "split": "train"}
            cfg = HarnessConfig(root / "config.json", "a",
                                {"a": ProviderProfile("scripted", "a", max_tokens=16384)})
            settings = CollectionSettings("baseline", ["a"], "a", benchmark="hle", hle_evaluator="exact")
            output = root / "out"
            with redirect_stdout(io.StringIO()):
                await collect([task], cfg, settings, PRICES, output,
                              provider_factory=lambda *a, **kw: Provider(["", "", ProviderError("timeout")]))
                trace_path = next(output.glob("*.traj.json"))
                original_trace = trace_path.read_bytes()
                workspace = output / "workspaces" / trace_path.stem
                (workspace / "old-attempt.txt").write_text("preserve this")
                summary = await collect([task], cfg, settings, PRICES, output,
                                        provider_factory=lambda *a, **kw: Provider([
                                            "Explanation: multiplication\nAnswer: 42\nConfidence: 100%"]),
                                        resume_token_upgrade=True, retry_errors=True)
            self.assertEqual(summary["valid_traces"], 1)
            self.assertEqual(summary["success_rate"], 1)
            self.assertEqual(next((output / "failed-attempts").glob("*.traj.json")).read_bytes(), original_trace)
            archived = next((output / "failed-attempts").glob("*.workspace"))
            self.assertEqual((archived / "old-attempt.txt").read_text(), "preserve this")
            self.assertFalse((workspace / "old-attempt.txt").exists())
            self.assertEqual((workspace / "README.md").read_text(), "public input")

    async def test_reuse_preserves_original_trace_and_archives_old_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = HarnessConfig(root / "config.json", "a", {"a": ProviderProfile("scripted", "a")})
            settings = CollectionSettings("baseline", ["a"], "a")
            with redirect_stdout(io.StringIO()):
                await collect([TASK], cfg, settings, PRICES, root, env_factory=Environment,
                              provider_factory=lambda *a, **kw: Provider(["```text\nlook around\n``` "]))
                trace = next(root.glob("*.traj.json"))
                original = trace.read_bytes()
                cfg.profiles["a"].max_tokens = 16384
                def fail(*args, **kwargs): raise AssertionError("completed task must not call API")
                summary = await collect([TASK], cfg, settings, PRICES, root, env_factory=Environment,
                                        provider_factory=fail, resume_token_upgrade=True, retry_errors=True)
            self.assertEqual(summary["skipped_existing"], 1)
            self.assertEqual(trace.read_bytes(), original)
            self.assertEqual(len(list(root.glob("*.summary.json"))), 1)
            self.assertEqual(len(list(root.glob("history/*/*.summary.json"))), 1)
            self.assertEqual(len(summary["reused_trace_ids"]), 1)
            cfg.profiles["a"].temperature = 0.5
            with self.assertRaisesRegex(ValueError, "incompatible"):
                await collect([TASK], cfg, settings, PRICES, root, env_factory=Environment,
                              provider_factory=fail, resume_token_upgrade=True)

    async def test_failed_trace_is_preserved_and_retried_under_new_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = HarnessConfig(root / "config.json", "a", {"a": ProviderProfile("scripted", "a")})
            settings = CollectionSettings("baseline", ["a"], "a")
            with redirect_stdout(io.StringIO()):
                await collect([TASK], cfg, settings, PRICES, root, env_factory=Environment,
                              provider_factory=lambda *a, **kw: Provider(["", "", ProviderError("timeout")]))
                original = next(root.glob("*.traj.json")).read_bytes()
                cfg.profiles["a"].max_tokens = 16384
                summary = await collect([TASK], cfg, settings, PRICES, root, env_factory=Environment,
                                        provider_factory=lambda *a, **kw: Provider(["```text\nlook around\n```"]),
                                        resume_token_upgrade=True, retry_errors=True)
            self.assertEqual(summary["valid_traces"], 1)
            self.assertEqual(summary["archived_attempts"], 1)
            self.assertAlmostEqual(summary["known_cost"], 0.00009)
            self.assertEqual(next((root / "failed-attempts").glob("*.traj.json")).read_bytes(), original)
            trace = json.loads(next(root.glob("*.traj.json")).read_text())
            self.assertEqual(trace["info"]["config"]["profiles"]["a"]["max_tokens"], 16384)
