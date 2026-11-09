"""Thin adapter over an OpenSquilla V4 Phase 3 inference bundle.

The bundle remains the source of truth for feature extraction, LightGBM,
ONNX MLP/BGE inference, probability fusion, and the optional legacy
post-processing decision. This module supplies the API-step input projection,
selects the configured decision policy, and maps R0-R3 to harness tiers.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from .config import RouterConfig
from .decision import RouteSelectionError, select_route
from .input_adapter import project_opensquilla_v4_input
from .types import RouterDecision, RouterRequest


class RouterRuntimeError(RuntimeError):
    """The inherited OpenSquilla runtime cannot be loaded or executed."""


@contextmanager
def _runtime_import_path(bundle_dir: Path) -> Iterator[None]:
    old_path = list(sys.path)
    sys.path.insert(0, str(bundle_dir / "runtime_src"))
    try:
        yield
    finally:
        sys.path[:] = old_path


class OpenSquillaV4Engine:
    """Reuse OpenSquilla's complete V4 Phase 3 inference core."""

    def __init__(self, config: RouterConfig) -> None:
        if config.bundle_dir is None:
            raise RouterRuntimeError("OpenSquilla V4 bundle_dir is not configured")
        self.config = config
        self.bundle_dir = config.bundle_dir
        runtime_root = self.bundle_dir / "runtime_src"
        runtime_config = self.bundle_dir / "router.runtime.yaml"
        if not runtime_root.is_dir() or not runtime_config.is_file():
            raise RouterRuntimeError(
                "incomplete OpenSquilla V4 bundle at %s" % self.bundle_dir
            )
        try:
            import yaml
        except ImportError as exc:
            raise RouterRuntimeError(
                "PyYAML is required; install mixed-pilot-harness[router]"
            ) from exc
        try:
            raw_config = yaml.safe_load(runtime_config.read_text(encoding="utf-8")) or {}
            with _runtime_import_path(self.bundle_dir):
                from src.router.inference.core import InferenceCore
                from src.router.inference.types import InferenceRequest

                use_aux_head = (
                    bool(raw_config.get("v4", {}).get("aux_head_inference", False))
                    if config.use_aux_head is None
                    else config.use_aux_head
                )
                self._request_type = InferenceRequest
                self._core = InferenceCore.from_model_dir(
                    self.bundle_dir,
                    raw_config,
                    use_aux_head=use_aux_head,
                )
        except RouterRuntimeError:
            raise
        except Exception as exc:
            raise RouterRuntimeError(
                "failed to load OpenSquilla V4 bundle %s: %s" % (self.bundle_dir, exc)
            ) from exc
        self.model_version = self._model_version()

    def _model_version(self) -> str:
        path = self.bundle_dir / "version.json"
        if not path.is_file():
            return "v4"
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("version") or "v4")
        except (OSError, ValueError, TypeError):
            return "v4"

    def route(self, request: RouterRequest) -> RouterDecision:
        projected = project_opensquilla_v4_input(request, self.config.input)
        decisions = [
            SimpleNamespace(
                route_class=item.route_class,
                difficulty=item.difficulty,
                margin=item.margin,
            )
            for item in projected.previous_route_decisions
        ]
        inference_request = self._request_type(
            current_user_text=projected.current_user_text,
            history_user_texts=projected.history_texts,
            prev_assistant_text=projected.previous_assistant_text,
            # The compact harness does not persist per-message usage. Keeping
            # this unset is preferable to fabricating token/duration values.
            prev_assistant_usage=None,
            prev_route_decisions=decisions,
            flags_text_override=projected.flags_text,
            context_metadata=projected.context_metadata,
        )
        try:
            result = self._core.predict(inference_request)
        except Exception as exc:
            raise RouterRuntimeError("OpenSquilla V4 inference failed: %s" % exc) from exc
        decision = result.decision
        legacy_route_class = str(getattr(decision, "route_class", "R1"))
        probabilities = {
            str(key): float(value)
            for key, value in dict(getattr(result, "probabilities", {}) or {}).items()
        }
        flags = dict(getattr(decision, "flags", {}) or {})
        try:
            selection = select_route(
                probabilities,
                legacy_route_class,
                flags,
                self.config.decision,
            )
        except RouteSelectionError as exc:
            raise RouterRuntimeError("OpenSquilla V4 route selection failed: %s" % exc) from exc
        route_class = selection.selected_route_class
        tier = self.config.route_class_mapping.get(route_class, self.config.fallback_tier)
        legacy_thinking_mode = getattr(decision, "thinking_mode", None)
        legacy_prompt_policy = getattr(decision, "prompt_policy", None)
        uses_legacy_controls = self.config.decision.mode == "legacy"
        return RouterDecision(
            tier=tier,
            route_class=route_class,
            confidence=float(probabilities.get(route_class, 0.0)),
            source="opensquilla_v4",
            probabilities=probabilities,
            margin=float(getattr(decision, "margin", 0.0) or 0.0),
            difficulty=float(getattr(decision, "difficulty_score", 0.0) or 0.0),
            metadata={
                "model_version": self.model_version,
                "thinking_mode": legacy_thinking_mode if uses_legacy_controls else None,
                "prompt_policy": legacy_prompt_policy if uses_legacy_controls else None,
                "legacy_thinking_mode": legacy_thinking_mode,
                "legacy_prompt_policy": legacy_prompt_policy,
                "flags": flags,
                "api_step_index": projected.api_step_index,
                "decision_mode": self.config.decision.mode,
                "raw_route_class": selection.raw_route_class,
                "legacy_route_class": selection.legacy_route_class,
                "selected_route_class": selection.selected_route_class,
                "raw_confidence": float(
                    probabilities.get(selection.raw_route_class, 0.0)
                ),
                "legacy_confidence": float(
                    probabilities.get(selection.legacy_route_class, 0.0)
                ),
                "selected_confidence": float(probabilities.get(route_class, 0.0)),
                "applied_rules": selection.applied_rules,
                "legacy_aux_downgrade_applied": bool(
                    getattr(decision, "aux_downgrade_applied", False)
                ),
                "legacy_sticky_applied": bool(
                    getattr(decision, "sticky_applied", False)
                ),
            },
        )


class FixedRouterEngine:
    """Explicit deterministic engine for offline smoke tests and baselines."""

    _ROUTE_CLASS = {"c0": "R0", "c1": "R1", "c2": "R2", "c3": "R3"}

    def __init__(self, tier: str, source: str = "fixed") -> None:
        self.tier = tier
        self.source = source

    def route(self, request: RouterRequest) -> RouterDecision:
        route_class = self._ROUTE_CLASS[self.tier]
        return RouterDecision(
            tier=self.tier,
            route_class=route_class,
            confidence=1.0 if self.source == "fixed" else 0.0,
            source=self.source,
            probabilities={route_class: 1.0},
            metadata={"api_step_index": request.api_step_index},
        )
