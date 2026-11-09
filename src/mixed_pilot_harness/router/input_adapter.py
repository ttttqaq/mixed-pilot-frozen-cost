"""Adapt a complete harness message trail to the OpenSquilla V4 schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..types import Message, ToolDefinition
from .config import RouterInputConfig
from .types import RouterDecision, RouterRequest

_FILE_REFERENCE = re.compile(r"(?:^|[\s\"'`(])(?:[A-Za-z_][\w.-]*/)+[\w.-]+", re.MULTILINE)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass(frozen=True)
class OpenSquillaV4Input:
    """Bounded compatibility projection for the inherited V4 feature schema."""

    api_step_index: int
    current_user_text: str
    history_texts: List[str]
    previous_assistant_text: Optional[str]
    flags_text: str
    context_metadata: dict
    previous_route_decisions: List[RouterDecision]


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    # Keep both the beginning (tool/error identity) and the end (usually the
    # most recent diagnostic lines) without pretending the projection is full.
    head = max(1, limit // 3)
    tail = max(1, limit - head - len("\n[TRUNCATED]\n"))
    return value[:head] + "\n[TRUNCATED]\n" + value[-tail:]


def _tool_calls(message: Message) -> str:
    if not message.tool_calls:
        return ""
    calls = [
        {
            "name": call.name,
            "arguments": call.arguments,
        }
        for call in message.tool_calls
    ]
    return json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def message_text(message: Message, max_chars: int) -> str:
    """Render one message with stable role/tool provenance for routing only."""

    if message.role == "tool":
        prefix = "[TOOL_RESULT name=%s]" % (message.name or "unknown")
        return _clip(prefix + "\n" + (message.content or ""), max_chars)
    rendered = "[%s]" % message.role.upper()
    if message.content:
        rendered += "\n" + message.content
    calls = _tool_calls(message)
    if calls:
        rendered += "\n[TOOL_CALLS]\n" + calls
    return _clip(rendered, max_chars)


def _first_user_text(messages: Iterable[Message]) -> str:
    for message in messages:
        if message.role == "user" and message.content:
            return message.content
    return ""


def _latest_assistant(messages: List[Message], max_chars: int) -> Optional[str]:
    for message in reversed(messages):
        if message.role == "assistant":
            return message_text(message, max_chars)
    return None


def build_router_request(
    messages: List[Message],
    tools: List[ToolDefinition],
    system: str,
    previous_decisions: List[RouterDecision],
    *,
    other_agent_trails: Optional[Iterable[str]] = None,
) -> RouterRequest:
    """Retain the complete harness request as the router's public input."""

    return RouterRequest(
        api_step_index=sum(1 for message in messages if message.role == "assistant") + 1,
        messages=list(messages),
        tools=list(tools),
        system=system,
        other_agent_trails=[str(trail) for trail in (other_agent_trails or [])],
        previous_route_decisions=list(previous_decisions),
    )


def project_opensquilla_v4_input(
    request: RouterRequest,
    config: RouterInputConfig,
) -> OpenSquillaV4Input:
    """Build the compatibility projection consumed by the inherited V4 model.

    The task objective remains the legacy ``current_user_text``. Recent
    user/assistant/tool progress occupies the legacy history channel, while the
    latest assistant response retains its dedicated channel. This preserves
    the 390/1536-dimensional model contract, but it does not claim that tool
    messages have the same distribution as the original user-history data.
    """

    messages = request.messages
    tools = request.tools
    current_user = _clip(_first_user_text(messages), config.max_message_chars)
    previous_assistant = _latest_assistant(messages, config.max_message_chars)
    recent = [
        message_text(message, config.max_message_chars)
        for message in messages[1:]
        if message.role in {"user", "assistant", "tool"}
    ]
    external_trails = [
        _clip("[AGENT_TRAIL]\n" + str(trail), config.max_message_chars)
        for trail in request.other_agent_trails
        if str(trail).strip()
    ]
    # External trails are treated as older context so they cannot evict the
    # task's most recent tool progress from the bounded legacy history channel.
    history = [*external_trails, *recent][-config.max_history_messages :]
    serialized = [message_text(message, config.max_message_chars) for message in messages]
    serialized.extend(external_trails)
    all_text = "\n\n".join(serialized)
    tool_results = [message for message in messages if message.role == "tool"]
    assistant_count = request.api_step_index - 1
    flags_parts = [current_user, *history]
    flags_text = _clip("\n\n".join(part for part in flags_parts if part), config.max_flags_chars)
    return OpenSquillaV4Input(
        api_step_index=request.api_step_index,
        current_user_text=current_user,
        history_texts=history,
        previous_assistant_text=previous_assistant,
        flags_text=flags_text,
        context_metadata={
            "turn_index": assistant_count,
            "context_tokens_est": max(0, len(all_text) // 4),
            "n_tools": len(tools),
            "tool_result_length": sum(len(message.content or "") for message in tool_results),
            "has_code_block": "```" in all_text,
            "has_file_reference": bool(_FILE_REFERENCE.search(all_text)),
            "has_url": bool(_URL.search(all_text)),
            "has_tool_results": bool(tool_results),
        },
        previous_route_decisions=list(request.previous_route_decisions),
    )
