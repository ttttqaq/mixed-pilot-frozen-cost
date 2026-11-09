#!/usr/bin/env python3
"""Verify real frozen Qwen embeddings and online-checkpoint reload without API."""
from pathlib import Path
import asyncio
import gc
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "MTRouter" / "src"), str(ROOT / "src")]

import torch
from mixed_pilot_mtrouter.online.cli import prepare
from mixed_pilot_mtrouter.online.config import load_config
from mixed_pilot_mtrouter.online.learning import LearningSystem
from mixed_pilot_mtrouter.online.runtime import Ledger, Runner
from check_online_native import Probe


async def main():
    config = load_config(ROOT / "configs/online-router.reference-teacher.db94.yaml",
                         ["encoder.max_tokens=512", "encoder.device=cuda:0", "model.hidden_dims=[16,16]",
                          "model.ensemble_size=2"])
    train, _, _, harness, prices, _ = prepare(config, check_credentials=False)
    output = ROOT / "outputs/reference-router-verification-20261030"
    runner = Runner(config, harness, prices, Ledger(output / "hf-scripted-probe", 1, prices["currency"]),
                    provider_factory=lambda p, **kw: Probe(p))
    task = next(t for t in train if t["benchmark"] == "scienceworld")
    result = await runner.rollout(task, lambda s: config["model_pool"][0], horizon=1,
                                 scope="collection", policy_version="encoder-verification")
    assert result["status"] == "truncated"
    state = result["state"]
    learner = LearningSystem(config)
    before = learner.predict_actions(state)
    dimension = learner.features.feature_dim
    assert dimension == 1027, dimension
    assert all(not p.requires_grad for p in learner.features.encoder.parameters())
    fingerprint = learner.policy_fingerprint()
    learner.save(output / "hf-online-checkpoint.pt")
    del learner
    gc.collect(); torch.cuda.empty_cache()
    restored = LearningSystem.load(output / "hf-online-checkpoint.pt")
    restored.features.cache.clear()
    after = restored.predict_actions(state)
    error = max(abs(before[name][key] - after[name][key]) for name in before for key in before[name])
    assert error < 1e-6, error
    assert fingerprint == restored.policy_fingerprint()
    report = {"real_encoder": config["encoder"]["model_name"], "feature_dimension": dimension,
              "encoder_frozen": True, "fresh_reload_max_prediction_error": error,
              "policy_fingerprint": fingerprint, "predictions": after, "paid_api_calls": 0,
              "evaluation_split": config["dataset"]["validation_split"], "test_tasks_probed": 0, "not_a_benchmark_performance_evaluation": True}
    (output / "hf-encoder.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
