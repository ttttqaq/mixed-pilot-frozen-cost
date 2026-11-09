"""Budget-conditioned online router.

The frozen text model is only a feature extractor.  Candidate LLM calls belong
to the rollout runtime, not this module.  The explicit ``hash`` backend exists
for deterministic, offline plumbing tests; it is not a language embedding.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..research.training.model_encoder import ModelEncoder, ModelEncoderConfig


FORMAT = "mixed-pilot-reference-router-v2"
FEATURE_SCHEMA = "complete-role-content-tool-history+remaining-budget-steps-v1"


def _plain(value):
    return json.loads(json.dumps(value, default=str))


def _merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _finite(value, name, *, low=None, high=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def serialize_history(state: dict) -> str:
    """Include actual tool arguments/results, never responses from the future.

    The first user message is separated so HF truncation can retain the task
    together with the most recent context.  Metadata such as final outcomes,
    policy version, task ID and the replay prefix is deliberately not encoded.
    """
    history = state.get("history", [])
    if not isinstance(history, list):
        raise ValueError("state.history must be a list")
    parts = []
    if state.get("system"):
        parts.append("[System]\n" + str(state["system"]))
    for message in history:
        if not isinstance(message, dict):
            raise ValueError("history messages must be dictionaries")
        role = str(message.get("role", "unknown"))
        text = str(message.get("content") or "")
        extras = {key: message[key] for key in ("tool_calls", "tool_call_id", "name")
                  if message.get(key)}
        if extras:
            text += "\n[Tool structure]\n" + json.dumps(extras, ensure_ascii=False, sort_keys=True)
        # Reasoning is included only when it is actually present in the saved
        # provider-visible history, not from a grader or a future call.
        if message.get("reasoning_content"):
            text += "\n[Recorded reasoning]\n" + str(message["reasoning_content"])
        parts.append(f"[{role}]\n{text}")
    return "\n\n".join(parts)


def state_key(state: dict) -> str:
    content = {"text": serialize_history(state),
               "remaining_budget": state.get("remaining_budget"),
               "remaining_steps": state.get("remaining_steps"),
               "step_idx": state.get("step_idx", 0)}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class FrozenFeatures:
    def __init__(self, config, device):
        self.config = dict(config.get("encoder", {}))
        self.backend = self.config.get("backend", "hf")
        if self.backend not in {"hf", "hash"}:
            raise ValueError("online encoder.backend must be hf or explicit smoke-only hash")
        self.task_budget = _finite(config.get("rollout", {}).get("task_budget", 1.0), "task_budget", low=1e-12)
        self.step_limit = int(config.get("rollout", {}).get("step_limit", 50))
        if self.step_limit <= 0:
            raise ValueError("step_limit must be positive")
        self.cache = OrderedDict()
        self.cache_limit = int(self.config.get("cache_limit", 50000))
        self.encoder = None
        if self.backend == "hash":
            self.output_dim = int(self.config.get("hash_dim", 32))
            if self.output_dim < 4:
                raise ValueError("hash_dim must be at least 4")
        else:
            from ..research.training.encoders import HistoryEncoder, HistoryEncoderConfig
            self.encoder = HistoryEncoder(HistoryEncoderConfig(
                model_name=self.config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
                freeze_encoder=True,
                max_tokens=int(self.config.get("max_tokens", 8192)),
                pooling_mode="last_token", backend="hf",
            )).to(device)
            self.encoder.eval()
            self.output_dim = self.encoder.output_dim
        self.feature_dim = self.output_dim + 3

    def _trim_hf(self, text):
        # Preserve the first user task and latest interactions.  Token IDs are
        # sliced before encoding; no char-based guesses or hidden answer input.
        maximum = int(self.config.get("max_tokens", 8192))
        tokenizer = self.encoder.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= maximum - 8:
            return text
        first = text.find("[user]\n")
        end = text.find("\n\n[", first + 1) if first >= 0 else -1
        task = text[:end] if end > 0 else text
        task_ids = tokenizer.encode(task, add_special_tokens=False)[:max(1, maximum // 2)]
        tail_size = max(1, maximum - len(task_ids) - 24)
        return (tokenizer.decode(task_ids, skip_special_tokens=True)
                + "\n[Recent context]\n"
                + tokenizer.decode(ids[-tail_size:], skip_special_tokens=True))

    def _text_embedding(self, text):
        semantic_config = {key: self.config.get(key) for key in
                           ("backend", "model_name", "max_tokens", "hash_dim")}
        digest = hashlib.sha256((FEATURE_SCHEMA + json.dumps(semantic_config, sort_keys=True)
                                 + text).encode()).hexdigest()
        if digest in self.cache:
            self.cache.move_to_end(digest)
            return self.cache[digest]
        if self.backend == "hash":
            vector = torch.zeros(self.output_dim, dtype=torch.float32)
            for token in re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE):
                raw = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(raw[:4], "little") % self.output_dim] += (1 if raw[4] & 1 else -1)
            result = F.normalize(vector, dim=0)
        else:
            with torch.no_grad():
                result = self.encoder(text=self._trim_hf(text)).detach().cpu().float()
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite frozen history embedding")
        self.cache[digest] = result
        if self.cache_limit > 0:
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        return result

    def encode(self, state):
        remaining = _finite(state.get("remaining_budget", self.task_budget), "remaining_budget", low=0)
        steps = _finite(state.get("remaining_steps", self.step_limit), "remaining_steps", low=0)
        step_idx = _finite(state.get("step_idx", 0), "step_idx", low=0)
        numeric = torch.tensor([remaining / self.task_budget,
                                steps / self.step_limit,
                                step_idx / self.step_limit], dtype=torch.float32)
        return torch.cat([self._text_embedding(serialize_history(state)), numeric])


def _mlp(input_dim, hidden_dims, dropout):
    layers = []
    for size in hidden_dims:
        layers.extend([nn.Linear(input_dim, int(size)), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(float(dropout)))
        input_dim = int(size)
    return nn.Sequential(*layers), input_dim


class ActionValueNet(nn.Module):
    def __init__(self, state_dim, model_names, metadata, cfg):
        super().__init__()
        self.models = ModelEncoder(model_names, ModelEncoderConfig(
            meta_embed_dim=int(cfg.get("meta_embed_dim", 32)),
            residual_embed_dim=int(cfg.get("residual_embed_dim", 16)),
            output_dim=int(cfg.get("model_dim", 64)),
            residual_l2_weight=float(cfg.get("residual_l2_weight", .001)),
        ), custom_models_config=metadata)
        self.backbone, size = _mlp(state_dim + self.models.output_dim,
                                   cfg.get("hidden_dims", [256, 256]), cfg.get("dropout", .1))
        self.reward = nn.Linear(size, 1)
        self.cost = nn.Linear(size, 1)

    def forward(self, state, actions):
        model = self.models.forward_batch(actions)
        hidden = self.backbone(torch.cat([state, model], dim=-1))
        return torch.sigmoid(self.reward(hidden)).squeeze(-1), F.softplus(self.cost(hidden)).squeeze(-1)


class LearningSystem:
    """Trainable Router ensemble plus an immutable history encoder.

    Costs entering/leaving this API are in the recorded currency.  Internally
    only, cost targets use ``budget.cost_reference``. Fixed LLM estimates are
    explicitly downweighted targets; no Teacher parameters are trained.
    """
    def __init__(self, config: dict, *, _model_metadata=None):
        self.config = _plain(config)
        self.model_names = list(config.get("model_pool", []))
        if not self.model_names or len(set(self.model_names)) != len(self.model_names):
            raise ValueError("model_pool must contain distinct model names")
        self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}
        self.set_seed(int(config.get("seed", 42)))
        device = config.get("encoder", {}).get("device", config.get("device", "auto"))
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.features = FrozenFeatures(config, self.device)
        budget = config.get("budget", {})
        objective = config.get("objective", {})
        self.cost_reference = _finite(budget.get("cost_reference", objective.get("cost_reference", self.features.task_budget)),
                                      "cost_reference", low=1e-12)
        self.lambda_ = _finite(budget.get("lambda_", objective.get("lambda_", .1)), "lambda_", low=0)
        self.optimization = config.get("optimization", {})
        model_cfg = config.get("model", {})
        if _model_metadata is not None:
            self.model_metadata = _model_metadata
        elif model_cfg.get("model_metadata"):
            import yaml
            self.model_metadata = yaml.safe_load(Path(model_cfg["model_metadata"]).read_text())["models"]
        elif self.features.backend == "hash":
            # Explicit test backend: identity residuals still distinguish models.
            self.model_metadata = {name: dict(ModelEncoder.META_DEFAULTS) for name in self.model_names}
        else:
            raise ValueError("model.model_metadata is required for production HF training")
        unknown = set(self.model_names) - self.model_metadata.keys()
        if unknown:
            raise ValueError(f"Model metadata missing for {sorted(unknown)}")
        count = int(model_cfg.get("ensemble_size", 3))
        if count < 1:
            raise ValueError("Ensemble size must be positive")
        self.router = nn.ModuleList([ActionValueNet(self.features.feature_dim, self.model_names,
                                                    self.model_metadata, model_cfg) for _ in range(count)]).to(self.device)
        lr = _finite(self.optimization.get("learning_rate", .001), "learning_rate", low=1e-12)
        wd = _finite(self.optimization.get("weight_decay", .01), "weight_decay", low=0)
        self.router_optimizer = torch.optim.AdamW(self.router.parameters(), lr=lr, weight_decay=wd)
        self.router_updates = 0
        self.metadata = {}

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def policy_fingerprint(self, epsilon=None):
        """Identify the actual continuation policy, not its training round.

        The digest covers action-network tensors and the features/objective
        used to turn their predictions into epsilon-greedy choices.  Teacher
        weights, optimizer/RNG state, caches and train/eval mode are excluded:
        none changes the frozen policy's conditional action distribution.
        """
        if epsilon is None:
            epsilon = self.config.get("online", {}).get("exploration_epsilon", 0.)
        epsilon = _finite(epsilon, "policy epsilon", low=0, high=1)
        encoder = self.config.get("encoder", {})
        semantic = {
            "fingerprint_schema": "online-epsilon-greedy-policy-v1",
            "feature_schema": FEATURE_SCHEMA,
            "models": self.model_names,
            "encoder": {"backend": self.features.backend,
                        "model_name": encoder.get("model_name", "Qwen/Qwen3-Embedding-0.6B") if self.features.backend == "hf" else None,
                        "max_tokens": int(encoder.get("max_tokens", 8192)) if self.features.backend == "hf" else None,
                        "embedding_dim": self.features.output_dim,
                        "pooling": "last_token+l2" if self.features.backend == "hf" else "deterministic-token-hash+l2"},
            "task_budget_feature_scale": self.features.task_budget,
            "step_limit_feature_scale": self.features.step_limit,
            "cost_reference": self.cost_reference,
            "lambda_": self.lambda_, "epsilon": epsilon,
        }
        digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode())
        for name, tensor in sorted(self.router.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            description = json.dumps({"name": name, "dtype": str(value.dtype),
                                      "shape": list(value.shape)}, sort_keys=True).encode()
            digest.update(len(description).to_bytes(8, "big")); digest.update(description)
            # Byte view supports all Torch dtypes, including BF16, without
            # depending on NumPy's numeric dtype support or tensor device.
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
        return "policy-" + digest.hexdigest()

    def predict_actions(self, state):
        self.router.eval()
        with torch.no_grad():
            x = self.features.encode(state).to(self.device).unsqueeze(0).expand(len(self.model_names), -1)
            outputs = [net(x, list(range(len(self.model_names)))) for net in self.router]
            reward = torch.stack([o[0] for o in outputs])
            cost = torch.stack([o[1] for o in outputs]) * self.cost_reference
            result = {}
            for i, name in enumerate(self.model_names):
                result[name] = {"reward": float(reward[:, i].mean()), "cost": float(cost[:, i].mean()),
                                "reward_std": float(reward[:, i].std(unbiased=False)),
                                "cost_std": float(cost[:, i].std(unbiased=False))}
            return result

    def _rows(self, rows, *, policy_version=None):
        valid = []
        for row in rows:
            if row.get("labels_valid") is False or row.get("cost_complete") is False:
                continue
            version = row.get("policy_version", row.get("state", {}).get("policy_version"))
            if policy_version is not None and version != policy_version:
                continue
            if row.get("model") not in self.model_to_idx:
                raise ValueError(f"Unknown router action {row.get('model')!r}")
            item = dict(row)
            item["reward"] = _finite(row["reward"], "reward target", low=0, high=1)
            item["cost"] = _finite(row["cost"], "cost target", low=0)
            item["weight"] = _finite(row.get("weight", 1.0), "sample weight", low=0)
            if item["weight"]:
                valid.append(item)
        limit = int(self.optimization.get("replay_limit", 50000))
        return valid[-limit:] if limit > 0 else valid

    @staticmethod
    def _weighted_loss(pred, target, weights):
        return ((pred - target).pow(2) * weights).sum() / weights.sum().clamp_min(1e-12)

    def train_router(self, rows, pairs=None):
        valid = self._rows(rows)
        if not valid:
            return {"samples": 0, "pairs": 0, "updates": 0, "loss": None}
        xs = torch.stack([self.features.encode(r["state"]) for r in valid]).to(self.device)
        actions = [self.model_to_idx[r["model"]] for r in valid]
        rewards = torch.tensor([r["reward"] for r in valid], device=self.device)
        costs = torch.tensor([r["cost"] / self.cost_reference for r in valid], device=self.device)
        weights = torch.tensor([r["weight"] for r in valid], device=self.device)
        pair_data = []
        for pair in pairs or []:
            if pair.get("a") not in self.model_to_idx or pair.get("b") not in self.model_to_idx:
                raise ValueError("Pair actions must belong to model_pool")
            weight = _finite(pair.get("weight", 1.0), "pair weight", low=0)
            delta = _finite(pair["delta"], "pair delta")
            if weight:
                pair_data.append((self.features.encode(pair["state"]), self.model_to_idx[pair["a"]],
                                  self.model_to_idx[pair["b"]], delta, weight))
        if pair_data:
            pair_x = torch.stack([p[0] for p in pair_data]).to(self.device)
            pair_a, pair_b = [p[1] for p in pair_data], [p[2] for p in pair_data]
            pair_delta = torch.tensor([p[3] for p in pair_data], device=self.device)
            pair_weights = torch.tensor([p[4] for p in pair_data], device=self.device)
        self.router.eval()
        with torch.no_grad():
            anchors = [(r.detach(), c.detach()) for r, c in (net(xs, actions) for net in self.router)]
        epochs = int(self.optimization.get("epochs", 10))
        batch_size = int(self.optimization.get("batch_size", 64))
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        anchor_weight = float(self.optimization.get("anchor_weight", .1))
        pairwise_weight = float(self.optimization.get("pairwise_weight", .2))
        cost_weight = float(self.optimization.get("cost_loss_weight", 1.0))
        sums = {name: 0. for name in ("loss", "reward_loss", "cost_loss", "anchor_loss", "pairwise_loss", "regularization")}
        updates = 0
        self.router.train()
        for _ in range(epochs):
            for offset in range(0, len(valid), batch_size):
                losses, metrics = [], []
                for j, net in enumerate(self.router):
                    idx = torch.randint(len(valid), (min(batch_size, len(valid) - offset),), device=self.device)
                    selected_actions = [actions[i] for i in idx.tolist()]
                    pr, pc = net(xs[idx], selected_actions)
                    rloss = self._weighted_loss(pr, rewards[idx], weights[idx])
                    closs = self._weighted_loss(pc, costs[idx], weights[idx])
                    ar, ac = anchors[j]
                    anchor = F.mse_loss(pr, ar[idx]) + F.mse_loss(pc, ac[idx])
                    ploss = torch.zeros((), device=self.device)
                    if pair_data and pairwise_weight:
                        pi = torch.randint(len(pair_data), (min(batch_size, len(pair_data)),), device=self.device)
                        ra, ca = net(pair_x[pi], [pair_a[i] for i in pi.tolist()])
                        rb, cb = net(pair_x[pi], [pair_b[i] for i in pi.tolist()])
                        delta = (ra - self.lambda_ * ca) - (rb - self.lambda_ * cb)
                        ploss = self._weighted_loss(delta, pair_delta[pi], pair_weights[pi])
                    reg = net.models.get_residual_l2_loss()
                    losses.append(rloss + cost_weight * closs + anchor_weight * anchor + pairwise_weight * ploss + reg)
                    metrics.append((rloss, closs, anchor, ploss, reg))
                loss = torch.stack(losses).mean()
                self.router_optimizer.zero_grad(); loss.backward()
                clip = float(self.optimization.get("grad_clip", 1.0))
                if clip > 0:
                    nn.utils.clip_grad_norm_(self.router.parameters(), clip)
                self.router_optimizer.step()
                sums["loss"] += float(loss.detach())
                for index, key in enumerate(("reward_loss", "cost_loss", "anchor_loss", "pairwise_loss", "regularization")):
                    sums[key] += float(torch.stack([m[index] for m in metrics]).mean().detach())
                updates += 1
        self.router_updates += updates
        self.router.eval()
        return {**{k: v / updates for k, v in sums.items()}, "samples": len(valid),
                "pairs": len(pair_data), "updates": updates}

    def _checkpoint_extra(self):
        return {}

    def _restore_checkpoint_extra(self, data):
        if 'frozen_cost_schema' in data:
            raise ValueError('Use FrozenCostLearner.load for a frozen-cost checkpoint')

    def save(self, path, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": FORMAT, "feature_schema": FEATURE_SCHEMA, "config": self.config,
                "model_names": self.model_names, "model_metadata": _plain(self.model_metadata),
                "router": self.router.state_dict(),
                "router_optimizer": self.router_optimizer.state_dict(),
                "router_updates": self.router_updates,
                "metadata": _plain(metadata if metadata is not None else self.metadata),
                "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "embedding_cache": dict(self.features.cache) if self.config.get("encoder", {}).get("persist_cache", True) else {}}
        data.update(self._checkpoint_extra())
        temporary = path.with_name(path.name + ".tmp")
        torch.save(data, temporary); temporary.replace(path)

    @classmethod
    def load(cls, path, config_override=None):
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        if data.get("format") != FORMAT or data.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Not a compatible online-router checkpoint")
        config = _merge(data["config"], config_override)
        if config.get("model_pool") != data["model_names"]:
            raise ValueError("Checkpoint model_pool names/order cannot be changed")
        # Shapes and text semantics must be stable.  Device/paths can move.
        for key in ("backend", "hash_dim", "max_tokens"):
            if config.get("encoder", {}).get(key) != data["config"].get("encoder", {}).get(key):
                raise ValueError(f"Checkpoint encoder {key} cannot be changed")
        # These are learned feature/target units, not merely run-time budgets.
        # Changing them while loading would silently reinterpret saved heads.
        old = data["config"]
        for key in ("task_budget", "step_limit"):
            if config.get("rollout", {}).get(key) != old.get("rollout", {}).get(key):
                raise ValueError(f"Checkpoint rollout {key} feature scale cannot be changed")
        if config.get("budget", {}).get("cost_reference") != old.get("budget", {}).get("cost_reference"):
            raise ValueError("Checkpoint budget.cost_reference target scale cannot be changed")
        obj = cls(config, _model_metadata=data["model_metadata"])
        obj.router.load_state_dict(data["router"])
        obj.router_optimizer.load_state_dict(data["router_optimizer"])
        for optimizer in (obj.router_optimizer,):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(obj.device)
        obj.router_updates = data.get("router_updates", 0)
        obj.metadata = data.get("metadata", {})
        obj.features.cache.update(data.get("embedding_cache", {}))
        obj._restore_checkpoint_extra(data)
        torch.set_rng_state(data["torch_rng_state"])
        random.setstate(data["python_rng_state"])
        if torch.cuda.is_available() and data.get("cuda_rng_states"):
            if len(data["cuda_rng_states"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(data["cuda_rng_states"])
        obj.router.eval()
        return obj
