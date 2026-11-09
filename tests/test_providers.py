from __future__ import annotations

import os
import unittest

from mixed_pilot_harness.config import ProviderProfile
from mixed_pilot_harness.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderError,
)
from mixed_pilot_harness.types import Message, ToolCall, ToolDefinition


TOOL = ToolDefinition(
    name="read_file",
    description="read",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)


class _CaptureMixin:
    response = {}

    async def _post_json(self, url, payload, headers):
        self.captured = {"url": url, "payload": payload, "headers": headers}
        return self.response


class CaptureOpenAI(_CaptureMixin, OpenAICompatibleProvider):
    pass


class CaptureAnthropic(_CaptureMixin, AnthropicProvider):
    pass


class CaptureGemini(_CaptureMixin, GeminiProvider):
    pass


class ProviderAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_anthropic_thinking_disabled_is_sent(self):
        provider = CaptureAnthropic(ProviderProfile(
            kind="anthropic", model="test", api_key_optional=True,
            extra_body={"thinking": {"type": "disabled"}}))
        provider.response = {"content": [{"type": "text", "text": "done"}]}
        await provider.complete([Message("user", "task")], [TOOL], "system")
        self.assertEqual(provider.captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(provider.captured["payload"]["system"], "system")
        self.assertEqual(provider.captured["payload"]["tools"][0]["name"], "read_file")

    async def test_anthropic_extra_body_cannot_override_core_fields(self):
        for key in ("model", "messages", "system", "tools", "tool_choice", "stream", "max_tokens", "temperature"):
            with self.subTest(key=key):
                provider = CaptureAnthropic(ProviderProfile(
                    kind="anthropic", model="test", api_key_optional=True,
                    extra_body={key: "override"}))
                with self.assertRaises(ProviderError):
                    await provider.complete([Message("user", "task")], [], "system")
                self.assertFalse(hasattr(provider, "captured"))

    async def test_empty_assistant_history_is_filtered_without_mutation(self):
        provider = CaptureOpenAI(ProviderProfile(kind="openai", model="test", api_key_optional=True))
        provider.response = {"choices": [{"message": {"content": "recovered"}}]}
        messages = [Message("user", "task"), Message("assistant", ""),
                    Message("user", "Please provide a command"), Message("assistant", " \n"),
                    Message("assistant", "", tool_calls=[ToolCall("c1", "read_file", {"path": "a"})]),
                    Message("tool", "", tool_call_id="c1"), Message("assistant", "useful text")]
        await provider.complete(messages, [TOOL], "system")
        wire = provider.captured["payload"]["messages"]
        self.assertEqual(len(wire), 6)
        self.assertEqual(wire[2]["content"], "Please provide a command")
        self.assertEqual(wire[3]["tool_calls"][0]["id"], "c1")
        self.assertEqual(wire[4]["role"], "tool")
        self.assertEqual(wire[5]["content"], "useful text")
        self.assertEqual(len(messages), 7)
        self.assertEqual(messages[1].content, "")

    async def test_collection_generation_settings_and_raw_usage(self):
        provider = CaptureOpenAI(ProviderProfile(
            kind="openai", model="test", api_key_optional=True,
            extra_body={"thinking": {"type": "disabled"}}))
        provider.response = {"choices": [{"message": {"content": "done"}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                       "prompt_cache_hit_tokens": 8}}
        result = await provider.complete([Message("user", "test")], [], "system")
        self.assertEqual(provider.captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(result.usage.raw["prompt_cache_hit_tokens"], 8)

    async def test_openai_compatible_tool_round_trip(self):
        provider = CaptureOpenAI(
            ProviderProfile(
                kind="openai",
                model="model-a",
                base_url="https://example.test/v1",
                api_key_optional=True,
            )
        )
        provider.response = {
            "model": "model-a",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        response = await provider.complete([Message("user", "inspect")], [TOOL], "system")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "a.py"})
        self.assertEqual(provider.captured["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(provider.captured["payload"]["tools"][0]["function"]["name"], "read_file")

    async def test_anthropic_maps_tool_results_to_user_blocks(self):
        os.environ["HARNESS_TEST_ANTHROPIC_KEY"] = "dummy"
        self.addCleanup(os.environ.pop, "HARNESS_TEST_ANTHROPIC_KEY", None)
        provider = CaptureAnthropic(
            ProviderProfile(
                kind="anthropic",
                model="claude-test",
                base_url="https://example.test/v1",
                api_key_env="HARNESS_TEST_ANTHROPIC_KEY",
            )
        )
        provider.response = {
            "model": "claude-test",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        messages = [
            Message("user", "inspect"),
            Message("assistant", tool_calls=[ToolCall("call-1", "read_file", {"path": "a.py"})]),
            Message("tool", "contents", tool_call_id="call-1", name="read_file"),
        ]
        response = await provider.complete(messages, [TOOL], "system")
        block = provider.captured["payload"]["messages"][-1]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(response.text, "done")

    async def test_gemini_maps_function_call(self):
        os.environ["HARNESS_TEST_GEMINI_KEY"] = "dummy"
        self.addCleanup(os.environ.pop, "HARNESS_TEST_GEMINI_KEY", None)
        provider = CaptureGemini(
            ProviderProfile(
                kind="gemini",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                api_key_env="HARNESS_TEST_GEMINI_KEY",
            )
        )
        provider.response = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "read_file", "args": {"path": "a.py"}}}
                        ]
                    },
                }
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3},
        }
        response = await provider.complete([Message("user", "inspect")], [TOOL], "system")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertIn("models/gemini-test:generateContent?key=dummy", provider.captured["url"])

    async def test_gemini_groups_parallel_function_results(self):
        os.environ["HARNESS_TEST_GEMINI_KEY"] = "dummy"
        self.addCleanup(os.environ.pop, "HARNESS_TEST_GEMINI_KEY", None)
        provider = CaptureGemini(
            ProviderProfile(
                kind="gemini",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                api_key_env="HARNESS_TEST_GEMINI_KEY",
            )
        )
        provider.response = {
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "done"}]}}]
        }
        messages = [
            Message("user", "inspect"),
            Message(
                "assistant",
                tool_calls=[ToolCall("a", "read_file", {}), ToolCall("b", "read_file", {})],
            ),
            Message("tool", "one", tool_call_id="a", name="read_file"),
            Message("tool", "two", tool_call_id="b", name="read_file"),
        ]
        await provider.complete(messages, [TOOL], "system")
        tool_turn = provider.captured["payload"]["contents"][-1]
        self.assertEqual(tool_turn["role"], "user")
        self.assertEqual(len(tool_turn["parts"]), 2)


if __name__ == "__main__":
    unittest.main()
