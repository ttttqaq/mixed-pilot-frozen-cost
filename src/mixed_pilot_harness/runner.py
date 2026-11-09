"""One-turn, multi-step agent runner."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from .runtime import RuntimeTracer
from .tools import WorkspaceTools
from .trace import JsonlTrace
from .types import Message, Provider, TurnLimitError, TurnResult, Usage


DEFAULT_SYSTEM_PROMPT = """You are a coding and research agent operating in one prepared workspace.
Use tools when needed, keep all file operations inside the workspace, run relevant checks, and
finish with a concise result. The shell tool is rooted at the workspace but is not an OS security
boundary; do not read or write outside the supplied workspace."""


class TurnRunner:
    """Run exactly one turn; provider tool requests create steps inside it."""

    def __init__(
        self,
        provider: Provider,
        workspace: Path,
        max_steps: int = 32,
        tool_timeout: float = 120,
        trace: Optional[object] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        secret_env_name: str = "",
        task_id: str = "",
    ) -> None:
        self.provider = provider
        self.workspace = workspace.resolve()
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.task_id = task_id or uuid.uuid4().hex
        self.tools = WorkspaceTools(
            self.workspace,
            timeout=tool_timeout,
            secret_env_names=[secret_env_name] if secret_env_name else [],
        )
        self.trace = (
            trace if isinstance(trace, RuntimeTracer) else RuntimeTracer(trace or JsonlTrace(None))
        )
        self.trace.configure(
            task_id=self.task_id,
            workspace=self.workspace,
            provider=self.provider.provider_name,
            max_steps=self.max_steps,
            system_prompt=self.system_prompt,
            tools=self.tools.definitions(),
        )

    def set_shell_env(self, env) -> None:
        self.tools.shell_env = env

    async def run(self, prompt: str) -> TurnResult:
        turn_id = uuid.uuid4().hex
        messages = [Message(role="user", content=prompt)]
        provider_calls = 0
        tool_calls = 0
        usage = Usage()
        self.trace.task_started(prompt, turn_id)
        for step in range(1, self.max_steps + 1):
            iteration = self.trace.iteration_started(turn_id, step, messages)
            provider_calls += 1
            self.trace.provider_started(iteration)
            try:
                response = await self.provider.complete(
                    messages,
                    self.tools.definitions(),
                    self.system_prompt,
                )
            except Exception as exc:
                self._flush_provider_events(iteration)
                self.trace.provider_failed(iteration, exc)
                raise
            self._flush_provider_events(iteration)
            usage.input_tokens += response.usage.input_tokens
            usage.output_tokens += response.usage.output_tokens
            self.trace.provider_completed(iteration, response)
            messages.append(
                response.assistant_message()
            )
            if not response.tool_calls:
                result = TurnResult(
                    turn_id=turn_id,
                    text=response.text,
                    steps=step,
                    provider_calls=provider_calls,
                    tool_calls=tool_calls,
                    usage=usage,
                )
                self.trace.iteration_completed(iteration)
                self.trace.task_completed(turn_id, result)
                return result
            for call in response.tool_calls:
                tool_calls += 1
                self.trace.tool_use(iteration, call)
                output = (json.dumps({"ok": False, "error": "Invalid tool arguments: " + call.parse_error})
                          if call.parse_error else await self.tools.execute(call.name, call.arguments))
                is_error = False
                try:
                    parsed = json.loads(output)
                    is_error = isinstance(parsed, dict) and parsed.get("ok") is False
                except json.JSONDecodeError:
                    pass
                self.trace.tool_result(iteration, call, output, is_error)
                messages.append(
                    Message(
                        role="tool",
                        content=output,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
            self.trace.iteration_completed(iteration)
        error = TurnLimitError("turn reached max_steps=%d" % self.max_steps)
        self.trace.task_failed(turn_id, self.max_steps, error)
        raise error

    def _flush_provider_events(self, iteration) -> None:
        drain = getattr(self.provider, "drain_events", None)
        if not callable(drain):
            return
        for event in drain():
            self.trace.provider_event(iteration, event)
