"""Provider-neutral contracts used by the router utility package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from ..types import Message, ToolDefinition


@dataclass(frozen=True)
class RouterDecision:
    """One four-class route decision for one logical provider request."""

    tier: str
    route_class: str
    confidence: float
    source: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    difficulty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouterRequest:
    """Provider-neutral input for one logical API request step.

    The complete request is retained here. Model-specific clipping and feature
    projection belong to a RouterEngine adapter, not to this public contract.
    """

    api_step_index: int
    messages: List[Message]
    tools: List[ToolDefinition]
    system: str
    other_agent_trails: List[str] = field(default_factory=list)
    previous_route_decisions: List[RouterDecision] = field(default_factory=list)


class RouterEngine(Protocol):
    """Classify one logical provider request into a configured tier."""

    def route(self, request: RouterRequest) -> RouterDecision:
        ...
