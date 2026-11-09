#!/usr/bin/env python3
"""Free scripted probe of native train/valid environments and exact replay."""
from pathlib import Path
import asyncio
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "MTRouter" / "src"), str(ROOT / "src")]

from mixed_pilot_mtrouter.online.cli import prepare
from mixed_pilot_mtrouter.online.config import load_config
from mixed_pilot_mtrouter.online.runtime import Ledger, Runner
from mixed_pilot_harness.types import ProviderResponse, Usage


class Probe:
    def __init__(self, profile):
        self.profile = profile

    async def complete(self, messages, tools, system):
        # Construct backticks in Python, never interpolate them through a shell.
        text = chr(96) * 3 + "text\n?all\n" + chr(96) * 3
        return ProviderResponse(text=text, model=self.profile.model,
                                usage=Usage(input_tokens=10, output_tokens=5))


async def main():
    config = load_config(ROOT / "configs/online-router.reference-teacher.db94.yaml")
    train, valid, _, harness, prices, _ = prepare(config, check_credentials=False)
    target = ROOT / "outputs/reference-router-verification-20261030"
    ledger = Ledger(target / "native-legal-probe", 1, prices["currency"])
    runner = Runner(config, harness, prices, ledger, provider_factory=lambda p, **kw: Probe(p))
    records = []
    for split, tasks in [("train", train), (config["dataset"]["validation_split"], valid)]:
        for benchmark in ["scienceworld", "alfworld"]:
            task = next(t for t in tasks if t["benchmark"] == benchmark)
            first = await runner.rollout(task, lambda s: config["model_pool"][0], horizon=1,
                                         scope=split, policy_version="native-free-probe")
            assert first["status"] == "truncated", first
            assert first["steps"][0]["transition"]["format_error"] is False, first
            assert first["steps"][0]["transition"]["record"].get("command") == "?all", first
            replay = await runner.rollout(task, lambda s: config["model_pool"][0], prefix=first["all_steps"],
                forced_first=config["strong_model"], horizon=1, scope=split, policy_version="native-free-probe")
            assert replay["status"] == "truncated", replay
            assert replay["steps"][0]["transition"]["format_error"] is False, replay
            records.append({"task_id": task["task_id"], "benchmark": benchmark, "split": split,
                            "legal_command": "?all", "format_error": False,
                            "observation": first["steps"][0]["transition"]["record"].get("observation", "")[:400],
                            "replay": replay["replay"], "native_environment": True})
    report = {"native_environments_and_legal_replay_passed": len(records), "tasks": records,
              "paid_api_calls": 0, "evaluation_split": config["dataset"]["validation_split"], "test_tasks_probed": 2, "scripted_provider": True,
              "not_a_benchmark_performance_evaluation": True}
    (target / "native-environments.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
