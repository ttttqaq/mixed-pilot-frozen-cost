"""Provider adapters and their provider-compatible factory.

Normal profiles map to one adapter and model. The factory treats ``router`` as
the single composite-provider exception; concrete HTTP adapters remain unaware
of routing. OpenAI-compatible endpoints cover OpenAI, OpenRouter, DeepSeek,
DashScope, Ollama, and similar services; Anthropic and Gemini use their native
message formats.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import errno
import http.client
import json
import os
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .config import ProviderProfile
from .context import fit_context, encoded_size
from .types import (
    Message,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)


def _join_url(base: str, suffix: str) -> str:
    return base.rstrip("/") + "/" + suffix.lstrip("/")


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


_RETRYABLE_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.EPIPE,
    errno.ETIMEDOUT,
}

# Transient upstream failures: request timeout, rate limit, gateway/origin errors,
# and Cloudflare's 520-524 edge range (524 is an origin read timeout wearing an
# HTTP status). Every other status fails the same way on a retry.
_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524})

# Scoped by the inference entry point, including evaluator calls. Other entry
# points retain their existing retry policy.
HTTP_RETRY_POLICY = ContextVar('http_retry_policy', default=None)


def _error_chain(error: BaseException):
    current: Optional[BaseException] = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = (
            current.reason
            if isinstance(current, urllib.error.URLError)
            and isinstance(current.reason, BaseException)
            else current.__cause__ or current.__context__
        )


def _is_retryable_transport_error(error: BaseException) -> bool:
    chain = list(_error_chain(error))
    for item in chain:
        if isinstance(item, urllib.error.HTTPError):
            return item.code in _RETRYABLE_HTTP_STATUS
    for item in chain:
        if isinstance(
            item,
            (
                ssl.SSLEOFError,
                ConnectionError,
                TimeoutError,
                socket.timeout,
                http.client.IncompleteRead,
            ),
        ):
            return True
        if isinstance(item, socket.gaierror) and item.errno == socket.EAI_AGAIN:
            return True
        if isinstance(item, OSError) and item.errno in _RETRYABLE_ERRNOS:
            return True
        if isinstance(item, ssl.SSLError) and (
            "unexpected_eof_while_reading" in str(item).lower()
            or "eof occurred in violation of protocol" in str(item).lower()
        ):
            return True
    return False


class HttpResult(dict):
    """Local transport metadata never shares keys with provider JSON."""
    def __init__(self, data, attempts):
        super().__init__(data)
        self.api_attempts = attempts
        self.cost_complete = all(a.get('cost_known_zero', True) for a in attempts if a['status'] == 'error')


def response_accounting(data):
    return dict(cost_complete=getattr(data, 'cost_complete', True),
                api_attempts=getattr(data, 'api_attempts', []))


class JsonHttpProvider:
    provider_name = "http"

    def __init__(self, profile: ProviderProfile) -> None:
        self.profile = profile

    def _api_key(self) -> str:
        value = os.environ.get(self.profile.api_key_env, "") if self.profile.api_key_env else ""
        if not value and not self.profile.api_key_optional:
            label = self.profile.api_key_env or "the configured API key environment variable"
            raise ProviderError("missing provider credential: %s" % label, cost_known_zero=True)
        return value

    async def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        uncertain = False
        attempts = []
        policy = HTTP_RETRY_POLICY.get()
        retries = policy[0] if policy is not None else 2
        for attempt in range(retries + 1):
            try:
                data = await asyncio.to_thread(self._post_json_sync, url, payload, headers)
                attempts.append({'status': 'response'})
                return HttpResult(data, attempts)
            except ProviderError as exc:
                attempts.append({'status': 'error', 'cost_known_zero': exc.cost_known_zero,
                                 'error_type': type(exc.__cause__ or exc).__name__})
                exc.api_attempts = list(attempts)
                uncertain = uncertain or not exc.cost_known_zero
                exc.cost_known_zero = not uncertain
                if attempt == retries or not _is_retryable_transport_error(exc):
                    raise
                await asyncio.sleep(policy[1] if policy is not None else 0.5 * (2**attempt))
        raise AssertionError("provider retry loop exhausted")

    def _fit_context(self, payload, window=None):
        return fit_context(payload, window or self.profile.context_window,
                           self.profile.max_tokens, self.profile.max_input_tokens)

    async def _request(self, url, payload, headers, context):
        attempts = []
        # A relay may expose a smaller window. Only explicit context rejections
        # get this bounded retry; network retries keep their original policy.
        for retry in range(4):
            try:
                data = await self._post_json(url, payload, headers)
                attempts.extend(getattr(data, 'api_attempts', []))
                return HttpResult(data, attempts)
            except ProviderError as exc:
                attempts.extend(getattr(exc, 'api_attempts', []))
                exc.api_attempts = list(attempts)
                exc.context_management = dict(context)
                exc.cost_known_zero = exc.cost_known_zero and all(
                    a.get('cost_known_zero', True) for a in attempts if a['status'] == 'error')
                detail = str(exc).lower()
                rejected = any(isinstance(e, urllib.error.HTTPError) and e.code in {400, 413, 422}
                               for e in _error_chain(exc)) and any(x in detail for x in (
                    'context_length_exceeded', 'maximum context length', 'context window',
                    'input tokens exceed', 'prompt is too long', '上下文长度', '输入长度超过'))
                if retry == 3 or not rejected or not self.profile.context_window:
                    raise
                target = int(encoded_size(payload) * .6) + self.profile.max_tokens + 1024
                try:
                    compact = self._fit_context(payload, min(target, self.profile.context_window))
                except ProviderError as compact_error:
                    compact_error.cost_known_zero = exc.cost_known_zero
                    compact_error.api_attempts = list(attempts)
                    compact_error.context_management = dict(context)
                    raise compact_error from exc
                context.setdefault('server_recompressions', []).append(compact)
                context['lossy'] = context.get('lossy', False) or compact['lossy']
                context['conservative_total_tokens'] = compact['conservative_total_tokens']
                context['request_sha256'] = compact['request_sha256']

    def _post_json_sync(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        safe_headers = {"Content-Type": "application/json", **headers, **self.profile.headers}
        request = urllib.request.Request(url, data=body, headers=safe_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.profile.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            key = self._api_key()
            if key:
                detail = detail.replace(key, "[REDACTED]")
            raise ProviderError("provider HTTP %s: %s" % (exc.code, detail),
                                cost_known_zero=exc.code in {400, 401, 403, 404, 413, 422, 429}) from exc
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as exc:
            detail = str(exc)
            key = self._api_key()
            if key:
                detail = detail.replace(key, "[REDACTED]")
            raise ProviderError("provider transport error: %s" % detail) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError("provider returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ProviderError("provider returned a non-object JSON response")
        return parsed


class OpenAICompatibleProvider(JsonHttpProvider):
    provider_name = "openai-compatible"

    def _endpoint(self) -> str:
        if self.profile.endpoint:
            return self.profile.endpoint
        return _join_url(self.profile.base_url, "chat/completions")

    def _message(self, message: Message) -> Dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        result: Dict[str, Any] = {"role": message.role, "content": message.content or None}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments if call.parse_error else json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        if message.role == 'assistant':
            owner = self._endpoint() + '|' + self.profile.model
            thinking = self.profile.extra_body.get('thinking')
            if message.reasoning_content is not None and message.reasoning_owner == owner:
                result['reasoning_content'] = message.reasoning_content
            elif message.tool_calls and ((isinstance(thinking, dict) and thinking.get('type') == 'enabled')
                                         or self.profile.extra_body.get('enable_thinking') is True):
                result['reasoning_content'] = ''
        return result

    async def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        system: str,
    ) -> ProviderResponse:
        wire_messages = []
        if system:
            wire_messages.append({"role": "system", "content": system})
        # Keep empty responses in the caller's trace/history for accounting, but
        # never resend an assistant entry with neither text nor tool calls.
        wire_messages.extend(
            self._message(message) for message in messages
            if not (message.role == "assistant" and not message.content.strip()
                    and not message.tool_calls)
        )
        payload: Dict[str, Any] = {
            "model": self.profile.model,
            "messages": wire_messages,
            "max_tokens": self.profile.max_tokens,
        }
        if self.profile.temperature is not None:
            payload["temperature"] = self.profile.temperature
        reserved = {"model", "messages", "tools", "tool_choice", "stream", "max_tokens", "temperature"}
        if reserved.intersection(self.profile.extra_body):
            raise ProviderError("extra_body cannot override core request fields", cost_known_zero=True)
        payload.update(self.profile.extra_body)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"
        context = self._fit_context(payload)
        key = self._api_key()
        headers = {"Authorization": "Bearer %s" % key} if key else {}
        data = await self._request(self._endpoint(), payload, headers, context)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI-compatible response has no assistant choice") from exc
        calls = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            raw_arguments = function.get("arguments", "{}")
            error = None
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                if not isinstance(arguments, dict):
                    raise ValueError('tool arguments must be a JSON object')
            except (ValueError, TypeError) as exc:
                arguments, error = {}, str(exc)
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or uuid.uuid4().hex),
                    name=str(function.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    raw_arguments=raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments),
                    parse_error=error,
                )
            )
        usage = data.get("usage") or {}
        return ProviderResponse(
            **response_accounting(data),
            text=message.get("content") or "",
            tool_calls=calls,
            stop_reason=choice.get("finish_reason") or "end_turn",
            model=data.get("model") or self.profile.model,
            reasoning_content=message.get('reasoning_content'),
            reasoning_owner=self._endpoint() + '|' + self.profile.model,
            context_management=context,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                raw=dict(usage),
            ),
        )



class AnthropicProvider(JsonHttpProvider):
    provider_name = "anthropic"

    def _endpoint(self) -> str:
        return self.profile.endpoint or _join_url(self.profile.base_url, "messages")

    @staticmethod
    def _messages(messages: List[Message]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                    result[-1]["content"].append(block)
                else:
                    result.append({"role": "user", "content": [block]})
                continue
            content: Any = message.content
            if message.tool_calls:
                blocks = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                content = blocks
            result.append({"role": message.role, "content": content})
        return result

    async def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        system: str,
    ) -> ProviderResponse:
        payload: Dict[str, Any] = {
            "model": self.profile.model,
            "messages": self._messages(messages),
            "max_tokens": self.profile.max_tokens,
        }
        if system:
            payload["system"] = system
        if self.profile.temperature is not None:
            payload["temperature"] = self.profile.temperature
        reserved = {"model", "messages", "system", "tools", "tool_choice", "stream", "max_tokens", "temperature"}
        if reserved.intersection(self.profile.extra_body):
            raise ProviderError("extra_body cannot override core request fields", cost_known_zero=True)
        payload.update(self.profile.extra_body)
        if tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
            ]
        key = self._api_key()
        context = self._fit_context(payload)
        data = await self._request(
            self._endpoint(),
            payload,
            {"x-api-key": key, "anthropic-version": "2023-06-01"}, context,
        )
        text_parts = []
        calls = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or uuid.uuid4().hex),
                        name=str(block.get("name") or ""),
                        arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
                    )
                )
        usage = data.get("usage") or {}
        return ProviderResponse(
            **response_accounting(data),
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason=data.get("stop_reason") or "end_turn",
            model=data.get("model") or self.profile.model,
            context_management=context,
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                raw=dict(usage),
            ),
        )


class GeminiProvider(JsonHttpProvider):
    provider_name = "gemini"

    def _endpoint(self, key: str) -> str:
        if self.profile.endpoint:
            base = self.profile.endpoint
        else:
            model = urllib.parse.quote(self.profile.model, safe="")
            base = _join_url(self.profile.base_url, "models/%s:generateContent" % model)
        separator = "&" if "?" in base else "?"
        return base + separator + "key=" + urllib.parse.quote(key, safe="")

    @staticmethod
    def _contents(messages: List[Message]) -> List[Dict[str, Any]]:
        contents = []
        for message in messages:
            if message.role == "tool":
                try:
                    response = json.loads(message.content)
                except json.JSONDecodeError:
                    response = {"content": message.content}
                if not isinstance(response, dict):
                    response = {"content": response}
                part = {"functionResponse": {"name": message.name, "response": response}}
                if contents and contents[-1]["role"] == "user" and all(
                    "functionResponse" in item for item in contents[-1]["parts"]
                ):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
                continue
            parts = []
            if message.content:
                parts.append({"text": message.content})
            parts.extend(
                {"functionCall": {"name": call.name, "args": call.arguments}}
                for call in message.tool_calls
            )
            contents.append({"role": "model" if message.role == "assistant" else "user", "parts": parts})
        return contents

    async def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        system: str,
    ) -> ProviderResponse:
        payload: Dict[str, Any] = {"contents": self._contents(messages)}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        generation: Dict[str, Any] = {"maxOutputTokens": self.profile.max_tokens}
        if self.profile.temperature is not None:
            generation["temperature"] = self.profile.temperature
        payload["generationConfig"] = generation
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                        for tool in tools
                    ]
                }
            ]
        key = self._api_key()
        context = self._fit_context(payload)
        data = await self._request(self._endpoint(key), payload, {}, context)
        try:
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response has no candidate content") from exc
        text_parts = []
        calls = []
        for part in parts:
            if "text" in part:
                text_parts.append(part.get("text") or "")
            if "functionCall" in part:
                call = part["functionCall"] or {}
                calls.append(
                    ToolCall(
                        id=uuid.uuid4().hex,
                        name=str(call.get("name") or ""),
                        arguments=call.get("args") if isinstance(call.get("args"), dict) else {},
                    )
                )
        usage = data.get("usageMetadata") or {}
        return ProviderResponse(
            **response_accounting(data),
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason=candidate.get("finishReason") or "end_turn",
            model=self.profile.model,
            context_management=context,
            usage=Usage(
                input_tokens=int(usage.get("promptTokenCount") or 0),
                output_tokens=int(usage.get("candidatesTokenCount") or 0),
                raw=dict(usage),
            ),
        )


class ScriptedProvider:
    """Offline deterministic provider used by tests and harness smoke runs."""

    provider_name = "scripted"

    def __init__(self, responses: List[Dict[str, Any]], model: str = "scripted") -> None:
        self.responses = list(responses)
        self.model = model
        self.calls = 0

    async def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        system: str,
    ) -> ProviderResponse:
        del messages, tools, system
        self.calls += 1
        if not self.responses:
            raise ProviderError("scripted provider response sequence exhausted")
        raw = self.responses.pop(0)
        calls = [
            ToolCall(
                id=str(item.get("id") or uuid.uuid4().hex),
                name=str(item.get("name") or ""),
                arguments=dict(item.get("arguments") or {}),
            )
            for item in raw.get("tool_calls", [])
        ]
        usage = raw.get("usage") or {}
        return ProviderResponse(
            text=str(raw.get("text") or ""),
            tool_calls=calls,
            stop_reason=str(raw.get("stop_reason") or "end_turn"),
            model=str(raw.get("model") or self.model),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                raw=dict(usage),
            ),
        )


def create_provider(
    profile: ProviderProfile,
    config_dir: Optional[Path] = None,
    profiles: Optional[Mapping[str, ProviderProfile]] = None,
) -> Provider:
    kind = profile.kind.strip().lower()
    if kind == "router":
        from .router.provider import RouterProvider

        from .router.profiles import profile_config

        def target_factory(target: ProviderProfile) -> Provider:
            return create_provider(
                target,
                config_dir=config_dir,
                profiles=profiles,
            )

        return RouterProvider(profile_config(profile, profiles or {}, config_dir or Path.cwd()), target_factory)
    if kind in {"openai", "openai-compatible", "ollama", "openrouter"}:
        return OpenAICompatibleProvider(profile)
    if kind == "anthropic":
        return AnthropicProvider(profile)
    if kind == "gemini":
        return GeminiProvider(profile)
    if kind == "scripted":
        path = Path(profile.responses_file)
        if not path.is_absolute() and config_dir is not None:
            path = config_dir / path
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("scripted provider file must contain a JSON array")
        return ScriptedProvider(data, model=profile.model)
    raise ValueError("unsupported provider kind: %s" % profile.kind)
