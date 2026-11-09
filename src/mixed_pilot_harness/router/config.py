"""TOML configuration for the step-level router provider."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..config import ProviderProfile

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python 3.9/3.10 package extra
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ImportError:  # pragma: no cover - exercised only in incomplete installs
        tomllib = None  # type: ignore[assignment]


TEXT_TIERS = ("c0", "c1", "c2", "c3")
DECISION_MODES = ("legacy", "raw", "api_step")
DEFAULT_ROUTE_CLASS_MAPPING = {
    "R0": "c0",
    "R1": "c1",
    "R2": "c2",
    "R3": "c3",
}


class RouterConfigError(ValueError):
    """The router TOML or one of its tier targets is invalid."""


@dataclass(frozen=True)
class RouterInputConfig:
    max_history_messages: int = 4
    max_message_chars: int = 6000
    max_flags_chars: int = 12000


@dataclass(frozen=True)
class RouterDecisionConfig:
    # legacy: use OpenSquilla's turn-level post-processing decision
    # raw: use the fused model probability argmax
    # api_step: use raw argmax plus explicitly enabled step-safe rules
    mode: str = "legacy"
    high_risk_floor: bool = True


@dataclass(frozen=True)
class RouterConfig:
    path: Path
    engine: str
    tiers: Dict[str, ProviderProfile]
    bundle_dir: Optional[Path] = None
    require_runtime: bool = True
    fallback_tier: str = "c2"
    fixed_tier: str = "c1"
    use_aux_head: Optional[bool] = None
    route_class_mapping: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ROUTE_CLASS_MAPPING)
    )
    input: RouterInputConfig = field(default_factory=RouterInputConfig)
    decision: RouterDecisionConfig = field(default_factory=RouterDecisionConfig)
    service_url: str = ""
    service_timeout: float = 120.0
    checkpoint: Optional[Path] = None
    device: str = 'cpu'
    seed: int = 42
    model_names: tuple[str, ...] = ()
    embedding_url: str = ''
    model_metadata: Optional[Path] = None
    possibility: Dict[str, float] = field(default_factory=dict)


_PROFILE_FIELDS = {item.name for item in fields(ProviderProfile)} - {"name"}


def _read_toml(path: Path) -> Dict[str, Any]:
    if tomllib is None:
        raise RouterConfigError(
            "TOML support is unavailable; install mixed-pilot-harness[router] "
            "or tomli on Python 3.9/3.10"
        )
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except OSError as exc:
        raise RouterConfigError("cannot read router config %s: %s" % (path, exc)) from exc
    except Exception as exc:
        raise RouterConfigError("invalid router TOML %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise RouterConfigError("router config must contain a TOML table")
    return value


def _table(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RouterConfigError("%s must be a TOML table" % label)
    return dict(value)


def _positive_int(value: Any, label: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RouterConfigError("%s must be a positive integer" % label)
    return value


def _boolean(value: Any, label: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RouterConfigError("%s must be true or false" % label)
    return value


def _resolve_target(
    tier: str,
    raw: Mapping[str, Any],
    profiles: Mapping[str, ProviderProfile],
) -> ProviderProfile:
    values = dict(raw)
    profile_name = str(values.pop("profile", "") or "").strip()
    if profile_name:
        if profile_name not in profiles:
            raise RouterConfigError(
                "tiers.%s references unknown harness profile %r" % (tier, profile_name)
            )
        target = replace(profiles[profile_name])
        if not target.name:
            target = replace(target, name=profile_name)
    else:
        kind = str(values.get("kind") or "").strip()
        model = str(values.get("model") or "").strip()
        if not kind or not model:
            raise RouterConfigError(
                "tiers.%s must define profile=... or both kind and model" % tier
            )
        target = ProviderProfile(name="inline:%s" % tier, kind=kind, model=model)

    unknown = sorted(set(values) - _PROFILE_FIELDS)
    if unknown:
        raise RouterConfigError(
            "tiers.%s contains unsupported provider fields: %s"
            % (tier, ", ".join(unknown))
        )
    if values:
        target = replace(target, **values)
    if target.kind.strip().lower() == "router":
        raise RouterConfigError("tiers.%s cannot target another router provider" % tier)
    if not target.model.strip():
        raise RouterConfigError("tiers.%s target model must not be empty" % tier)
    return target


def load_router_config(
    path: Path,
    profiles: Optional[Mapping[str, ProviderProfile]] = None,
) -> RouterConfig:
    """Load and validate a router configuration.

    Tier tables may either reference a profile from ``harness.json`` or define
    an inline ``ProviderProfile``. Relative bundle paths are resolved against
    the router TOML rather than the process working directory.
    """

    resolved = path.expanduser().resolve()
    data = _read_toml(resolved)
    router = _table(data.get("router"), "router")
    raw_tiers = _table(data.get("tiers"), "tiers")
    available_profiles = profiles or {}

    engine = str(router.get("engine") or "opensquilla_v4").strip().lower()
    if engine not in {"opensquilla_v4", "fixed", "mtrouter", "random"}:
        raise RouterConfigError("unsupported router engine: %s" % engine)
    tier_names = tuple(raw_tiers) if engine in {"mtrouter","random"} else TEXT_TIERS
    if not tier_names:
        raise RouterConfigError("at least one tier is required")

    missing = [tier for tier in tier_names if tier not in raw_tiers]
    if missing:
        raise RouterConfigError("router config is missing tiers: %s" % ", ".join(missing))
    extra = sorted(set(raw_tiers) - set(tier_names))
    if extra:
        raise RouterConfigError("router config contains unknown tiers: %s" % ", ".join(extra))
    tiers = {
        tier: _resolve_target(tier, _table(raw_tiers[tier], "tiers.%s" % tier), available_profiles)
        for tier in tier_names
    }

    if engine in {'random','mtrouter'}:
        from .profiles import inline_config
        dynamic = inline_config({**router, 'models':[p.name for p in tiers.values()]},
                                {p.name:p for p in tiers.values()}, resolved.parent)
        return replace(dynamic, path=resolved, tiers=tiers)

    service_url = str(router.get("service_url") or "").rstrip("/")
    service_timeout = router.get("service_timeout", 120.0)
    fallback_tier = str(router.get("fallback_tier") or "c2").strip().lower()
    fixed_tier = str(router.get("fixed_tier") or "c1").strip().lower()
    for label, tier in (("fallback_tier", fallback_tier), ("fixed_tier", fixed_tier)):
        if tier not in TEXT_TIERS:
            raise RouterConfigError("router.%s must be one of %s" % (label, TEXT_TIERS))

    raw_bundle = str(router.get("bundle_dir") or "").strip()
    bundle_dir = None
    if raw_bundle:
        bundle_dir = Path(raw_bundle).expanduser()
        if not bundle_dir.is_absolute():
            bundle_dir = (resolved.parent / bundle_dir).resolve()
        else:
            bundle_dir = bundle_dir.resolve()
    if engine == "opensquilla_v4" and bundle_dir is None:
        raise RouterConfigError("router.bundle_dir is required for engine=opensquilla_v4")

    raw_aux = router.get("use_aux_head")
    if raw_aux is not None and not isinstance(raw_aux, bool):
        raise RouterConfigError("router.use_aux_head must be true or false")

    mapping = dict(DEFAULT_ROUTE_CLASS_MAPPING)
    mapping.update(
        {
            str(key): str(value).strip().lower()
            for key, value in _table(data.get("route_classes"), "route_classes").items()
        }
    )
    if set(mapping) != set(DEFAULT_ROUTE_CLASS_MAPPING):
        raise RouterConfigError("route_classes may only define R0, R1, R2 and R3")
    if any(tier not in TEXT_TIERS for tier in mapping.values()):
        raise RouterConfigError("route_classes values must reference c0, c1, c2 or c3")

    raw_input = _table(data.get("input"), "input")
    input_config = RouterInputConfig(
        max_history_messages=_positive_int(
            raw_input.get("max_history_messages"),
            "input.max_history_messages",
            RouterInputConfig.max_history_messages,
        ),
        max_message_chars=_positive_int(
            raw_input.get("max_message_chars"),
            "input.max_message_chars",
            RouterInputConfig.max_message_chars,
        ),
        max_flags_chars=_positive_int(
            raw_input.get("max_flags_chars"),
            "input.max_flags_chars",
            RouterInputConfig.max_flags_chars,
        ),
    )
    raw_decision = _table(data.get("decision"), "decision")
    unknown_decision = sorted(set(raw_decision) - {"mode", "high_risk_floor"})
    if unknown_decision:
        raise RouterConfigError(
            "decision contains unsupported fields: %s"
            % ", ".join(unknown_decision)
        )
    decision_mode = str(raw_decision.get("mode") or "legacy").strip().lower()
    if decision_mode not in DECISION_MODES:
        raise RouterConfigError(
            "decision.mode must be one of %s" % (DECISION_MODES,)
        )
    decision_config = RouterDecisionConfig(
        mode=decision_mode,
        high_risk_floor=_boolean(
            raw_decision.get("high_risk_floor"),
            "decision.high_risk_floor",
            RouterDecisionConfig.high_risk_floor,
        ),
    )
    return RouterConfig(
        path=resolved,
        engine=engine,
        tiers=tiers,
        service_url=service_url,
        service_timeout=service_timeout,
        bundle_dir=bundle_dir,
        require_runtime=bool(router.get("require_runtime", True)),
        fallback_tier=fallback_tier,
        fixed_tier=fixed_tier,
        use_aux_head=raw_aux,
        route_class_mapping=mapping,
        input=input_config,
        decision=decision_config,
    )
