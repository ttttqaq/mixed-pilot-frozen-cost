"""Strict, reproducible configuration for online router training.

Training is restricted to train. Evaluation selects exactly valid or test;
evaluation rows and references cannot enter Router supervision.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject YAML keys silently shadowing an earlier experiment setting."""


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration mapping keys must be strings")
        if key in mapping:
            raise ValueError(f"Duplicate YAML configuration key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


DEFAULTS: dict[str, Any] = {
    "project_root": "/DB/data/yujietang/Agent/code/mixed-pilot",
    "harness_config": "harness.json",
    "pricing": "configs/inference.prices.json",
    "api_keys": {"dashscope": "", "zhipu": "", "newcli": ""},
    "dataset": {"root": "data/VirtualWorld-subset", "train_split": "train", "validation_split": "test"},
    "model_pool": ["qwen-flash-2025-07-28", "glm-5.3"],
    "strong_model": "glm-5.3",
    "seed": 42,
    "output_dir": "outputs/reference-router",
    "resilience": {"enabled": True, "stop_after_retries": True, "max_retries": 3, "retry_delay": 5.0,
                   "max_retry_delay": 60.0, "request_timeout": 300.0, "teacher_timeout": 600.0,
                   "rate_limit_wait": 30.0, "rate_limit_retries": 5, "qwen_min_interval": 2.0},
    "budget": {"total": 30.0, "cost_reference": 0.1, "lambda_": 0.05},
    "encoder": {
        "backend": "hf", "model_name": "/DB/data/yujietang/huggingface_models/Qwen3-Embedding-0.6B",
        "max_tokens": 8192, "device": "auto", "hash_dim": 32,
    },
    "model": {"hidden_dims": [256, 256], "dropout": 0.1, "ensemble_size": 3,
              "model_metadata": "MTRouter/configs/models.ksr-db94.yaml"},
    "rollout": {"task_budget": 1.0, "step_limit": 50, "replay_attempts": 2,
                "retry_wait": 0.0, "trace_delay": 0.0, "reward": "success"},
    "cold_start": {"enabled": True, "tasks": 4, "strategy": "uniform", "queries": 4},
    "online": {"iterations": 5, "tasks_per_iteration": 4, "states_per_iteration": 4,
               "exploration_epsilon": 0.1},
    "selection": {"enabled": True, "random_fraction": 0.25, "impact_weight": 1.0,
                  "uncertainty_weight": 1.0, "cost_weight": 1.0},
    "short_rollout": {"enabled": True, "horizon": 3, "adaptive": True,
                      "max_horizon": 9, "extension": 3},
    "teacher": {
        "enabled": True,
        "llm": {"profile": "gpt-6.1-terra", "model": "gpt-6.1-terra",
                "base_url": "https://code1.newcli.com/v1", "max_calls_per_iteration": 24,
                "max_tokens": 8192, "max_history_chars": 16000,
                "reasoning_effort": "low", "json_mode": False, "min_interval": 15.0,
                "input_per_million": 10.95, "output_per_million": 54.75,
                "cached_input_per_million": 0.5475},
    },
    "audit": {"enabled": True, "uncertainty_threshold": 0.1, "random_fraction": 0.2,
              "max_pairs_per_iteration": 2},
    "optimization": {"epochs": 10, "batch_size": 64, "learning_rate": 0.001,
                     "weight_decay": 0.01, "grad_clip": 1.0, "anchor_weight": 0.1,
                     "pairwise_weight": 0.2, "cost_loss_weight": 1.0,
                     "replay_limit": 50000, "pseudo_weight": 0.25},
    "evaluation": {"enabled": True, "every_iterations": 2, "max_tasks": None},
    "checkpoint": {"every_iterations": 1},
}


def _merge(default: dict, provided: dict, prefix: str = "") -> dict:
    if not isinstance(provided, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a mapping")
    result = deepcopy(default)
    for key, value in provided.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if not isinstance(key, str) or key not in default:
            raise ValueError(f"Unknown configuration field: {name}")
        if isinstance(default[key], dict):
            result[key] = _merge(default[key], value, name)
        elif isinstance(value, dict):
            raise ValueError(f"{name} must not be a mapping")
        else:
            result[key] = deepcopy(value)
    return result


def _override(config: dict, specification: str) -> None:
    if "=" not in specification:
        raise ValueError(f"Override must be dotted.key=value: {specification!r}")
    path, raw = specification.split("=", 1)
    keys = path.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise ValueError(f"Unknown configuration override: {path}")
        cursor = cursor[key]
    if not keys[-1] or keys[-1] not in cursor or isinstance(cursor[keys[-1]], dict):
        raise ValueError(f"Unknown/scalar-only configuration override: {path}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid override value for {path}") from exc
    if isinstance(value, dict):
        raise ValueError(f"Override {path} must be a scalar or list")
    cursor[keys[-1]] = value


def _number(value: Any, path: str, *, minimum: float = 0.0, strict: bool = False,
            maximum: float | None = None, integer: bool = False) -> None:
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f"{path} must be {'an integer' if integer else 'a finite number'}")
    if not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{path} must be finite and {relation} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be <= {maximum}")


def _string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")


def validate_config(config: dict) -> None:
    """Validate a merged config without inspecting models, credentials or data."""
    for name, value in config.get("api_keys", {}).items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"api_keys.{name} must be a string or null")
    for path in ("project_root", "harness_config", "pricing", "output_dir", "strong_model"):
        _string(config[path], path)
    dataset = config["dataset"]
    _string(dataset["root"], "dataset.root")
    if dataset["train_split"] != "train" or dataset["validation_split"] not in {"valid", "test"}:
        raise ValueError("Online training requires train_split=train and validation_split=valid/test (evaluation only)")
    pool = config["model_pool"]
    if (not isinstance(pool, list) or len(pool) != 2 or
            any(not isinstance(p, str) or not p.strip() for p in pool) or len(set(pool)) != 2):
        raise ValueError("model_pool must contain exactly two distinct profile names")
    if config["strong_model"] not in pool:
        raise ValueError("strong_model must be in model_pool")
    if config["teacher"]["llm"]["profile"] in pool:
        raise ValueError("Teacher profile must be separate from actor model_pool")
    if not config["teacher"]["llm"]["base_url"].startswith("https://"):
        raise ValueError("Teacher base_url must use HTTPS")
    _number(config["seed"], "seed", integer=True)

    bool_paths = ["cold_start.enabled", "selection.enabled", "short_rollout.enabled",
                  "short_rollout.adaptive", "teacher.enabled", "audit.enabled", "evaluation.enabled"]
    for path in bool_paths:
        value = _get(config, path)
        if type(value) is not bool:
            raise ValueError(f"{path} must be a boolean")
    if "resilience" in config:
        if type(config["resilience"]["enabled"]) is not bool:
            raise ValueError("resilience.enabled must be a boolean")
        if type(config["resilience"]["stop_after_retries"]) is not bool:
            raise ValueError("resilience.stop_after_retries must be a boolean")
        _number(config["resilience"]["max_retries"], "resilience.max_retries", integer=True, maximum=10)
        _number(config["resilience"]["rate_limit_retries"], "resilience.rate_limit_retries", integer=True, maximum=10)
        for key in ("rate_limit_wait", "qwen_min_interval"):
            _number(config["resilience"][key], "resilience." + key)
        for key in ("retry_delay", "max_retry_delay", "request_timeout", "teacher_timeout"):
            _number(config["resilience"][key], "resilience." + key, strict=True)
        if config["resilience"]["max_retry_delay"] < config["resilience"]["retry_delay"]:
            raise ValueError("resilience.max_retry_delay must be >= retry_delay")

    positive_ints = [
        "encoder.max_tokens", "encoder.hash_dim", "model.ensemble_size", "rollout.step_limit",
        "rollout.replay_attempts", "cold_start.tasks",
        "online.iterations", "online.tasks_per_iteration", "online.states_per_iteration",
        "short_rollout.horizon", "short_rollout.max_horizon", "short_rollout.extension",
        "teacher.llm.max_tokens", "teacher.llm.max_history_chars", "optimization.epochs", "optimization.batch_size",
        "optimization.replay_limit", "evaluation.every_iterations", "checkpoint.every_iterations",
    ]
    for path in positive_ints:
        _number(_get(config, path), path, strict=True, integer=True)
    for path in ["cold_start.queries", "teacher.llm.max_calls_per_iteration", "audit.max_pairs_per_iteration"]:
        _number(_get(config, path), path, integer=True)
    for path in ["budget.total", "budget.cost_reference", "rollout.task_budget", "optimization.learning_rate"]:
        _number(_get(config, path), path, strict=True)
    for path in ["budget.lambda_", "rollout.retry_wait", "rollout.trace_delay", "selection.impact_weight",
                 "selection.uncertainty_weight", "selection.cost_weight", "audit.uncertainty_threshold",
                 "optimization.weight_decay", "optimization.grad_clip", "optimization.anchor_weight",
                 "optimization.pairwise_weight", "optimization.cost_loss_weight", "optimization.pseudo_weight"]:
        _number(_get(config, path), path)
    for path in ["online.exploration_epsilon", "selection.random_fraction", "audit.random_fraction"]:
        _number(_get(config, path), path, maximum=1.0)
    _number(config["model"]["dropout"], "model.dropout", maximum=1.0)
    if config["model"]["dropout"] == 1:
        raise ValueError("model.dropout must be < 1")
    _number(config["teacher"]["llm"]["min_interval"], "teacher.llm.min_interval")
    for key in ("input_per_million", "output_per_million", "cached_input_per_million"):
        _number(config["teacher"]["llm"][key], "teacher.llm." + key, strict=True)
    for key in ("profile", "model", "base_url"):
        _string(config["teacher"]["llm"][key], "teacher.llm." + key)
    if config["teacher"]["llm"]["reasoning_effort"] not in {"low", "medium", "high"}:
        raise ValueError("teacher.llm.reasoning_effort must be low/medium/high")
    if type(config["teacher"]["llm"]["json_mode"]) is not bool:
        raise ValueError("teacher.llm.json_mode must be a boolean")
    for section in ("model",):
        dimensions = config[section]["hidden_dims"]
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"{section}.hidden_dims must be a nonempty list")
        for dimension in dimensions:
            _number(dimension, f"{section}.hidden_dims[]", strict=True, integer=True)
    if config["short_rollout"]["max_horizon"] < config["short_rollout"]["horizon"]:
        raise ValueError("short_rollout.max_horizon must be >= horizon")
    if config["encoder"]["backend"] not in {"hf", "hash"}:
        raise ValueError("encoder.backend must be hf or hash (hash is only for smoke tests)")
    _string(config["encoder"]["model_name"], "encoder.model_name")
    device = config["encoder"]["device"]
    if not isinstance(device, str) or not (device in {"auto", "cpu", "cuda", "mps"}
            or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("encoder.device must be auto/cpu/cuda/cuda:N/mps")
    _string(config["model"]["model_metadata"], "model.model_metadata")
    if config["rollout"]["reward"] not in {"success", "normalized_score"}:
        raise ValueError("rollout.reward must be success or normalized_score")
    if config["cold_start"]["strategy"] not in {"uniform"}:
        raise ValueError("cold_start.strategy must be uniform")
    if config["evaluation"]["max_tasks"] is not None:
        _number(config["evaluation"]["max_tasks"], "evaluation.max_tasks", strict=True, integer=True)


def _get(config: dict, dotted: str) -> Any:
    value = config
    for key in dotted.split("."):
        value = value[key]
    return value


def _infer_project_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "setup.cfg").is_file() and (parent / "MTRouter").is_dir():
            return parent
    return path.parent.parent if path.parent.name == "configs" else path.parent


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load strict YAML and apply ``section.key=value`` command-line overrides.

    Relative file paths are based on project_root, independent of the shell cwd.
    Missing project_root is inferred from the config location for portable tests.
    An explicitly relative project_root is based on the config's own directory.
    """
    source = Path(path).expanduser().resolve()
    try:
        provided = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {source}") from exc
    if provided is None:
        provided = {}
    config = _merge(DEFAULTS, provided)
    if "project_root" not in provided:
        config["project_root"] = str(_infer_project_root(source))
    for item in overrides or []:
        _override(config, item)
    _string(config["project_root"], "project_root")
    root = Path(config["project_root"]).expanduser()
    if not root.is_absolute():
        root = source.parent / root
    root = root.resolve()
    config["project_root"] = str(root)
    validate_config(config)
    for dotted in ["harness_config", "pricing", "output_dir", "dataset.root", "model.model_metadata"]:
        keys = dotted.split(".")
        cursor = config
        for key in keys[:-1]:
            cursor = cursor[key]
        value = Path(cursor[keys[-1]]).expanduser()
        cursor[keys[-1]] = str((root / value).resolve())
    encoder_name = config["encoder"]["model_name"]
    if encoder_name.startswith(("/", "./", "../", "~")):
        config["encoder"]["model_name"] = str((root / Path(encoder_name).expanduser()).resolve())
    return config
