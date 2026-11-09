"""Small provider-neutral runtime types.

The event and provider boundaries follow the same separation used by
OpenSquilla 0.5.2: providers return model events/results, while the runner
owns tool execution and emits tool-result events. The implementation here is
intentionally reduced to the fields needed by this benchmark harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: Optional[str] = None
    parse_error: Optional[str] = None


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning_owner: Optional[str] = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    reasoning_content: Optional[str] = None
    reasoning_owner: Optional[str] = None
    context_management: Dict[str, Any] = field(default_factory=dict)
    cost_complete: bool = True
    api_attempts: List[Dict[str, Any]] = field(default_factory=list)

    def assistant_message(self):
        return Message('assistant', self.text, tool_calls=self.tool_calls,
                       reasoning_content=self.reasoning_content, reasoning_owner=self.reasoning_owner)


@dataclass(frozen=True)
class ProviderEvent:
    """A provider-owned trace event drained by the provider-neutral runner."""

    event: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    turn_id: str
    text: str
    steps: int
    provider_calls: int
    tool_calls: int
    usage: Usage

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Provider(Protocol):
    provider_name: str

    async def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        system: str,
    ) -> ProviderResponse:
        """Complete one provider step in a single agent turn."""


class HarnessError(RuntimeError):
    """Base error for expected harness failures."""


class ProviderError(HarnessError):
    """A provider request or response failed."""

    def __init__(self, message, *, cost_known_zero=False):
        super().__init__(message)
        self.cost_known_zero = cost_known_zero


class TurnLimitError(HarnessError):
    """The provider kept requesting tools after the configured step limit."""


class WorkspaceError(HarnessError):
    """A workspace could not be prepared or accessed safely."""
