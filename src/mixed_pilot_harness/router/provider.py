"""Composite Provider that routes each logical API step to a real provider."""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from ..config import ProviderProfile
from ..types import Message, Provider, ProviderEvent, ProviderResponse, ToolDefinition
from .config import RouterConfig, load_router_config
from .input_adapter import build_router_request
from .opensquilla_v4 import FixedRouterEngine, OpenSquillaV4Engine, RouterRuntimeError
from .types import RouterDecision, RouterEngine


ProviderFactory = Callable[[ProviderProfile], Provider]
AgentTrailSupplier = Callable[[List[Message]], Iterable[str]]


async def route_request(engine, request):
    # Use the same transient transport retry policy as concrete providers.
    # Invalid scores, pool mismatches and local model errors are never retried.
    from ..providers import HTTP_RETRY_POLICY, _is_retryable_transport_error
    policy = HTTP_RETRY_POLICY.get()
    retries = policy[0] if policy is not None else 2
    for attempt in range(retries + 1):
        try:
            return await asyncio.to_thread(engine.route, request)
        except Exception as exc:
            if attempt==retries or not _is_retryable_transport_error(exc): raise
            await asyncio.sleep(policy[1] if policy is not None else .5 * 2**attempt)


def _build_engine(config: RouterConfig) -> RouterEngine:
    if config.engine == 'random':
        from .random import RandomRouterEngine
        return RandomRouterEngine(config)
    if config.engine == "mtrouter":
        from .mtrouter import MTRouterEngine
        return MTRouterEngine(config)
    if config.engine == "fixed":
        return FixedRouterEngine(config.fixed_tier)
    try:
        return OpenSquillaV4Engine(config)
    except Exception as exc:
        if config.require_runtime:
            if isinstance(exc, RouterRuntimeError):
                raise
            raise RouterRuntimeError(str(exc)) from exc
        warnings.warn(
            "OpenSquilla V4 router unavailable; using configured fallback tier %s: %s"
            % (config.fallback_tier, exc),
            RuntimeWarning,
            stacklevel=2,
        )
        return FixedRouterEngine(config.fallback_tier, source="v4_unavailable")


class RouterProvider:
    """Provider-compatible step router.

    The runner sends the full accumulated message list on every ``complete``
    call. This wrapper classifies that logical request, then delegates the
    unchanged messages/tools/system values to exactly one concrete provider.
    """

    provider_name = "router"

    def __init__(
        self,
        config: RouterConfig,
        provider_factory: ProviderFactory,
        *,
        engine: Optional[RouterEngine] = None,
        agent_trail_supplier: Optional[AgentTrailSupplier] = None,
    ) -> None:
        self.config = config
        self._engine = engine or _build_engine(config)
        self._agent_trail_supplier = agent_trail_supplier
        self._providers: Dict[str, Provider] = {
            tier: provider_factory(profile) for tier, profile in config.tiers.items()
        }
        self._decisions: List[RouterDecision] = []
        self._events: List[ProviderEvent] = []
        self.last_decision: Optional[RouterDecision] = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        provider_factory: ProviderFactory,
        *,
        profiles: Optional[Mapping[str, ProviderProfile]] = None,
        agent_trail_supplier: Optional[AgentTrailSupplier] = None,
    ) -> "RouterProvider":
        return cls(
            load_router_config(path, profiles=profiles),
            provider_factory,
            agent_trail_supplier=agent_trail_supplier,
        )

    async def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        system: str,
    ) -> ProviderResponse:
        # A fresh one-message task resets step history when a provider instance
        # is deliberately reused by an embedding application.
        if len(messages) == 1 and messages[0].role == "user":
            self._decisions = []
            self._events = []
            self.last_decision = None
        request = build_router_request(
            messages,
            tools,
            system,
            self._decisions,
            other_agent_trails=(
                self._agent_trail_supplier(messages)
                if self._agent_trail_supplier is not None
                else None
            ),
        )
        decision = await route_request(self._engine, request)
        if decision.tier not in self._providers:
            raise RouterRuntimeError(
                "router returned unconfigured tier %r" % decision.tier
            )
        self.last_decision = decision
        self._events.append(self._decision_event(decision))
        response = await self._providers[decision.tier].complete(messages, tools, system)
        self._decisions.append(decision)
        self._decisions = self._decisions[-5:]
        return response

    def drain_events(self) -> List[ProviderEvent]:
        """Return pending provider events once, including pre-request decisions."""

        events, self._events = self._events, []
        return events

    def _decision_event(self, decision: RouterDecision) -> ProviderEvent:
        metadata = dict(decision.metadata)
        target_profile = self.config.tiers[decision.tier]
        target_provider = self._providers[decision.tier]
        route_class = decision.route_class
        data = {
            "decision_mode": metadata.get("decision_mode", self.config.decision.mode),
            "raw_route_class": metadata.get("raw_route_class", route_class),
            "legacy_route_class": metadata.get("legacy_route_class", route_class),
            "selected_route_class": metadata.get("selected_route_class", route_class),
            "tier": decision.tier,
            "confidence": decision.confidence,
            "probabilities": dict(decision.probabilities),
            "margin": decision.margin,
            "difficulty": decision.difficulty,
            "applied_rules": list(metadata.get("applied_rules", [])),
            "flags": dict(metadata.get("flags", {})),
            "router_source": decision.source,
            "api_step_index": metadata.get("api_step_index"),
            "target_profile": target_profile.name or decision.tier,
            "target_provider": target_provider.provider_name,
            "target_model": target_profile.model,
            "decision_metadata": metadata,
        }
        return ProviderEvent(event="router_decision", data=data)
