"""Shared, single-request chat path. No tools or agent loop."""

from __future__ import annotations

from typing import Callable, Optional

from .config import HarnessConfig
from .providers import create_provider
from .types import Message, ProviderResponse


async def chat_once(
    config: HarnessConfig,
    prompt: str,
    *,
    profile_name: Optional[str] = None,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_factory: Callable = create_provider,
) -> ProviderResponse:
    """Use the same profile resolution and timeout as ``mixed-pilot chat``."""
    profile = config.profile(profile_name)
    if model:
        profile.model = model
    if base_url:
        profile.base_url = base_url
    if config.agent.get("provider_timeout") is not None:
        profile.timeout = float(config.agent["provider_timeout"])
    provider = provider_factory(profile, config_dir=config.path.parent, profiles=config.profiles)
    return await provider.complete([Message("user", prompt)], [], system)
