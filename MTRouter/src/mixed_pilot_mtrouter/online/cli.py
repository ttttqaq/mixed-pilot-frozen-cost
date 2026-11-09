"""Online router training entry point; evaluation uses the selected valid/test split without training leakage."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import load_config, validate_config
from .data import load_train_valid


@contextmanager
def api_credentials(config, args):
    """Consume runtime secrets before any resolved config/checkpoint is saved.

    Explicit CLI values override YAML, which overrides the existing environment.
    Provider adapters retain their credential interface within this process only.
    """
    configured = config.pop("api_keys", {})
    updates = {}
    for name, env in (("dashscope", "DASHSCOPE_API_KEY"), ("zhipu", "ZHIPU_API_KEY"), ("newcli", "NEWCLI_API_KEY")):
        value = getattr(args, name + "_api_key", None)
        explicit = value is not None
        if not explicit:
            value = configured.get(name)
        if value is None or (value == "" and not explicit):
            continue
        if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value.strip()):
            raise ValueError(f"API key for {name} must be a nonempty token")
        updates[env] = value.strip()
    previous = {env: os.environ.get(env) for env in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for env, value in previous.items():
            if value is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = value


def prepare(config, *, check_credentials=True, simulation=False):
    from mixed_pilot_harness.config import load_config as load_harness
    from mixed_pilot_harness.collection.pricing import load_prices
    train, valid, manifest = load_train_valid(config)
    harness = load_harness(Path(config["harness_config"]))
    # Keep the external teacher private to this online method; no harness edits.
    from mixed_pilot_harness.config import ProviderProfile
    settings = config["teacher"]["llm"]
    name = settings["profile"]
    harness.profiles[name] = ProviderProfile(kind="online-responses-teacher", name=name,
        model=settings["model"], base_url=settings["base_url"], api_key_env="NEWCLI_API_KEY",
        max_tokens=settings["max_tokens"], timeout=config["resilience"]["teacher_timeout"])
    needed = list(config["model_pool"])
    if config["teacher"]["enabled"]:
        needed.append(name)
    pricing = load_prices(Path(config["pricing"]), config["model_pool"])
    if pricing["currency"] != "CNY":
        raise ValueError("Reference teacher rates are declared in CNY; pricing.currency must be CNY")
    pricing["profiles"][name] = {k: settings[k] for k in
        ("input_per_million", "output_per_million", "cached_input_per_million")}
    missing = []
    for name in set(needed):
        profile = harness.profile(name)
        if profile.kind == "router":
            raise ValueError("Online candidates/assessor must be concrete provider profiles")
        if check_credentials and not simulation and profile.kind != "scripted" and not profile.api_key_optional:
            if not profile.api_key_env or not os.environ.get(profile.api_key_env):
                missing.append(profile.api_key_env or name)
    if missing:
        raise ValueError("Required API credentials are missing: " + ", ".join(sorted(set(missing)))
                         + ". Set api_keys in the training YAML or pass --dashscope-api-key/--zhipu-api-key/--newcli-api-key")
    hashes = {}
    for key in ("harness_config", "pricing"):
        hashes[key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    metadata = Path(config["model"]["model_metadata"])
    if not metadata.is_file():
        raise ValueError("Missing model metadata: " + str(metadata))
    hashes["model_metadata"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    sources = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        sources.update(source.name.encode()); sources.update(source.read_bytes())
    native_adapter = Path(config["project_root"]) / "src/mixed_pilot_harness/collection/episodes.py"
    if native_adapter.is_file():
        sources.update(b"native_episode_adapter"); sources.update(native_adapter.read_bytes())
    hashes["online_method_sources"] = sources.hexdigest()
    manifest["runtime_artifact_hashes"] = hashes
    manifest["runtime_fingerprint"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config["project_root"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest["code_commit"] = commit
    from .references import load_references
    references = load_references(config, train, manifest) if config["teacher"]["enabled"] else {}
    return train, valid, manifest, harness, pricing, references


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dashscope-api-key", metavar="KEY", help="Qwen API key; overrides api_keys.dashscope in YAML")
    p.add_argument("--zhipu-api-key", metavar="KEY", help="GLM actor API key; overrides api_keys.zhipu in YAML")
    p.add_argument("--newcli-api-key", metavar="KEY", help="Fixed Teacher key; overrides api_keys.newcli in YAML")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="Strict dotted YAML override; repeat for ablations")
    p.add_argument("--output-dir", type=Path, help="New directory; existing runs require --resume")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume last committed round; preserves all incurred costs")
    mode.add_argument("--evaluate-checkpoint", type=Path,
                      help="Evaluate an ONLINE checkpoint on configured evaluation split only, in a new output directory; no updates")
    p.add_argument("--dry-run", action="store_true", help="Check train/evaluation/config/prices only; no encoder, environment, API or output creation")
    p.add_argument("--smoke", action="store_true", help="CPU/hash + SYNTHETIC environments/LLM replies; no network and no benchmark score")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, args.set)
        with api_credentials(config, args):
            return _run(config, args)
    except (ValueError, FileNotFoundError) as exc:
        print("online-router: " + str(exc), file=sys.stderr)
        return 2


def _run(config, args):
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.expanduser().resolve())
    factories = (None, None)
    if args.smoke:
        from .smoke import smoke_config, smoke_factories
        config = smoke_config(config)
        validate_config(config)
        factories = smoke_factories(config)
    elif config["encoder"]["backend"] == "hash":
        raise ValueError("The hash encoder is a plumbing fixture; use --smoke or encoder.backend=hf")
    if args.evaluate_checkpoint is not None:
        config["teacher"]["enabled"] = False
        if not args.evaluate_checkpoint.is_file():
            raise ValueError("Online evaluation checkpoint does not exist")
        config["evaluation"]["enabled"] = True
    train, valid, manifest, harness, pricing, references = prepare(config, check_credentials=not args.dry_run,
                                                     simulation=args.smoke)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "train_tasks": len(train), "evaluation_tasks": len(valid), "evaluation_split": config["dataset"]["validation_split"],
            "test_tasks_used": len(valid) if config["dataset"]["validation_split"] == "test" else 0, "dataset": config["dataset"], "model_pool": config["model_pool"],
            "teacher_llm": config["teacher"]["llm"], "budget": config["budget"],
            "output_dir": config["output_dir"], "split_fingerprint": manifest["split_fingerprint"],
            "runtime_fingerprint": manifest["runtime_fingerprint"], "simulation": args.smoke},
            indent=2, ensure_ascii=False))
        return 0
    # Heavy learning imports occur only after static preflight/credentials.
    import torch
    if args.smoke:
        torch.set_num_threads(1)
    from .trainer import OnlineTrainer
    learner = None
    if args.evaluate_checkpoint is not None:
        from .learning import LearningSystem
        learner = LearningSystem.load(args.evaluate_checkpoint,
                                      {"encoder": {"device": config["encoder"]["device"]}})
        if learner.features.backend == "hash" and not args.smoke:
            raise ValueError("A hash smoke checkpoint cannot evaluate real benchmark performance")
        if learner.model_names != config["model_pool"]:
            raise ValueError("Evaluation pool names/order differ from checkpoint")
        if learner.features.task_budget != config["rollout"]["task_budget"] or learner.features.step_limit != config["rollout"]["step_limit"]:
            raise ValueError("Evaluation rollout budget/step normalization must match checkpoint")
    trainer = OnlineTrainer(config, train, valid, manifest, harness, pricing,
        resume=args.resume, references=references,
        episode_factory=factories[0], provider_factory=factories[1], simulation=args.smoke, learner=learner)
    if args.evaluate_checkpoint is not None:
        asyncio.run(trainer._evaluate(learner.metadata.get("completed_iterations", learner.metadata.get("iteration", 0))))
        print(json.dumps(trainer._summary("evaluation_only"), indent=2, ensure_ascii=False))
        return 0
    report = asyncio.run(trainer.run())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["stop_reason"] in {"iterations_completed", "global_budget"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
