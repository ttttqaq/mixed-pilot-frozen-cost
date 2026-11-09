"""JSON configuration for provider profiles and harness defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ProviderProfile:
    kind: str
    model: str
    base_url: str = ""
    endpoint: str = ""
    api_key_env: str = ""
    api_key_optional: bool = False
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0
    max_tokens: int = 8192
    temperature: Optional[float] = None
    responses_file: str = ""
    # Used only when kind="router". Relative paths are resolved against the
    # harness JSON file, just like scripted response files.
    router_config: str = ""
    router: Dict[str, Any] = field(default_factory=dict)
    # The harness profile key is retained for tracing and router diagnostics.
    # It is metadata only and is never sent to an upstream provider.
    name: str = ""
    # Provider-specific generation settings (e.g. thinking); core request fields
    # cannot be overridden through this mapping.
    extra_body: Dict[str, Any] = field(default_factory=dict)
    # Optional explicit window; no guesses based on provider/model aliases.
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None

    def __post_init__(self):
        for name in ('context_window', 'max_input_tokens'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f'{name} must be a positive integer')


@dataclass
class HarnessConfig:
    path: Path
    default_profile: str
    profiles: Dict[str, ProviderProfile]
    agent: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def profile(self, name: Optional[str]) -> ProviderProfile:
        selected = name or self.default_profile
        if selected not in self.profiles:
            available = ", ".join(sorted(self.profiles)) or "<none>"
            raise ValueError("unknown provider profile %r; available: %s" % (selected, available))
        return replace(self.profiles[selected])


def load_config(path: Path) -> HarnessConfig:
    resolved = path.resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    profiles = {
        name: ProviderProfile(**{**value, "name": name})
        for name, value in data.get("profiles", {}).items()
    }
    return HarnessConfig(
        path=resolved,
        default_profile=data.get("default_profile", ""),
        profiles=profiles,
        agent=dict(data.get("agent", {})),
        tasks=dict(data.get("tasks", {})),
    )


def resolve_config_path(value: Optional[str], cwd: Path) -> Path:
    if value:
        return Path(value)
    candidate = cwd / "harness.json"
    if candidate.exists():
        return candidate
    raise ValueError("no harness config found; pass --config PATH")


def resolve_config_relative(config: HarnessConfig, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config.path.parent / path).resolve()


def resolve_data_root(value: Optional[str] = None) -> Path:
    """Resolve the external benchmark-data root without linking it into the repo."""
    configured = value or os.environ.get("MIXED_PILOT_DATA_ROOT")
    path = Path(configured) if configured else Path.home() / "Data" / "mixed-pilot-data"
    return path.expanduser().resolve()
