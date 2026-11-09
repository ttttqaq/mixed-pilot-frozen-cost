"""Training-only NEW CLI Responses transport; actor adapters are unchanged."""
from __future__ import annotations

import json
import asyncio
import http.client
import time
import urllib.error
import urllib.request

from mixed_pilot_harness.providers import JsonHttpProvider, response_accounting, HTTP_RETRY_POLICY
from mixed_pilot_harness.types import ProviderError, ProviderResponse, Usage


def read_response(stream):
    """Require a terminal event; partial deltas never imply successful completion."""
    if "text/event-stream" not in stream.headers.get("Content-Type", "").lower():
        value = json.loads(stream.read().decode("utf-8-sig"))
        if not isinstance(value, dict) or value.get("status") not in {"completed", "incomplete", "failed"}:
            raise ProviderError("Responses request has no terminal status")
        return value
    chunks = []
    for data in sse_events(stream):
        if data.strip() == "[DONE]":
            break
        event = json.loads(data)
        kind = event.get("type")
        if kind == "response.output_text.delta":
            chunks.append(event.get("delta", ""))
        elif kind in {"response.completed", "response.incomplete", "response.failed"}:
            result = event.get("response")
            if not isinstance(result, dict) or result.get("status") != kind.split(".")[1]:
                raise ProviderError("Responses terminal event/status mismatch")
            if not result.get("output") and chunks:
                result["output"] = [{"type":"message","content":[{"type":"output_text","text":"".join(chunks)}]}]
            return result
        elif kind == "error" or event.get("error"):
            raise ProviderError("Responses stream reported an error")
    raise ProviderError("Responses stream ended without a terminal event")


def sse_events(stream):
    data = []
    for raw in stream:
        line = raw.decode("utf-8-sig").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


class ResponsesTeacherProvider(JsonHttpProvider):
    provider_name = "newcli-responses-teacher"

    def __init__(self, profile, min_interval=15.):
        super().__init__(profile)
        self.min_interval = min_interval
        self.last_finished = None
        self.request_lock = asyncio.Lock()

    async def _post_json(self, url, payload, headers):
        # No nested transport retries. Runner assigns a ledger ID per attempt.
        async with self.request_lock:
            delay = self.min_interval if self.last_finished is None else max(0., self.min_interval-(time.monotonic()-self.last_finished))
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError as exc:
                exc.cost_known_zero = True  # No dispatch happened during cooldown.
                raise
            token = HTTP_RETRY_POLICY.set((0, 0))
            try:
                return await super()._post_json(url, payload, headers)
            finally:
                self.last_finished = time.monotonic()
                HTTP_RETRY_POLICY.reset(token)

    def _post_json_sync(self, url, payload, headers):
        key = self._api_key()
        req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.profile.timeout) as reply:
                return read_response(reply)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            if key:
                detail = detail.replace(key, "[REDACTED]")
            raise ProviderError(f"Teacher HTTP {exc.code}: {detail}",
                                cost_known_zero=exc.code in {400, 401, 403, 404, 413, 422, 429}) from exc
        except (OSError, http.client.IncompleteRead, UnicodeError, ValueError) as exc:
            raise ProviderError("Teacher response transport or decoding failed") from exc

    async def complete(self, messages, tools, system):
        if tools:
            raise ProviderError("Fixed Teacher cannot use tools", cost_known_zero=True)
        key = self._api_key()
        payload = {"model": self.profile.model, "instructions": system,
                   "input": [{"role": m.role, "content": [{"type": "input_text", "text": str(m.content or "")}]} for m in messages],
                   "store": False, "stream": True, "max_output_tokens": self.profile.max_tokens,
                   "reasoning": {"effort": self.profile.extra_body.get("reasoning_effort", "low")}}
        if self.profile.extra_body.get("response_format"):
            payload["text"] = {"format": {"type": "json_object"}}
        data = await self._post_json(self.profile.base_url.rstrip("/") + "/responses", payload,
                                     {"Authorization": "Bearer " + key, "Accept": "text/event-stream, application/json"})
        output = data.get("output", [])
        text = "".join(part.get("text", "") for item in output if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")
        usage = data.get("usage") or {}
        raw = dict(usage)
        raw["prompt_tokens_details"] = usage.get("input_tokens_details") or {}
        invalid_output = bool(data.get("error")) or any(item.get("type") not in {"message", "reasoning"} for item in output)
        status = data.get("status")
        return ProviderResponse(text=text, model=data.get("model", self.profile.model),
            stop_reason="end_turn" if status == "completed" and not invalid_output else "length" if status == "incomplete" else "error",
            usage=Usage(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0), raw=raw),
            **response_accounting(data))
