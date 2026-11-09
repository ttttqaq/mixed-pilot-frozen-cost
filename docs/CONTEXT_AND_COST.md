# Context windows and retry cost accounting

## Configured limits

Verified on 2026-10-11 against official model documentation. Values are tokens.
The `context_window` in `harness.json` covers input + reserved output. Profile names
are aliases: `gpt-5.6` currently requests **gpt-5.5**, so its limit follows GPT-5.5.

| Profile | context_window | Official source |
| --- | ---: | --- |
| gpt-5.6 (model: gpt-5.5) | 1,050,000 | [OpenAI](https://developers.openai.com/api/docs/models/gpt-5.5) |
| claude-opus-4-8, claude-sonnet-4-6 | 1,000,000 | [Anthropic](https://platform.claude.com/docs/en/build-with-claude/context-windows) |
| deepseek-flash, deepseek-v4-pro | 1,000,000 | [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/) |
| glm-5 | 200,000 | [Z.ai](https://docs.z.ai/guides/llm/glm-5) |
| glm-4.7-flash | 200,000 | [Z.ai, GLM-4.7 family](https://docs.z.ai/guides/llm/glm-4.7) |
| glm-5.3 | 1,000,000 | [Z.ai](https://docs.z.ai/guides/llm/glm-5.3) |
| qwen-flash-2025-07-28 | 1,000,000 | [Alibaba, explicitly equivalent snapshot](https://help.aliyun.com/zh/model-studio/qwen-flash) |
| qwen3.6-35b-a3b | 262,144 | [Alibaba](https://help.aliyun.com/zh/model-studio/qwen3-6-35b-a3b) |
| qwen3-8b | 131,072 | [Alibaba](https://help.aliyun.com/en/model-studio/qwen3-8b) |

Qwen3-8B also sets `max_input_tokens: 98304`, its separate official API input cap.
For GLM, the domestic documentation could not be reliably retrieved; the defaults
use the manufacturer's Z.ai model specifications. Published 200K/1M values are
interpreted conservatively as 200,000/1,000,000.

Routers have no generation window of their own: each selected concrete profile
applies its own window **after routing**, including when switching to a smaller
model. The MTRouter encoder's own truncation is separate and unchanged.
Third-party relays and dedicated deployments may enforce smaller limits; override
`context_window` / `max_input_tokens` in that concrete profile if necessary.
Unknown/new profiles do not acquire a guessed limit: configure them explicitly.

## Local compaction

OpenAI-compatible, Anthropic and Gemini adapters share `context.py`:

1. Estimate serialized request UTF-8 bytes conservatively as tokens; reserve
   `max_tokens` plus 1,024 framing tokens. This is not an exact vendor tokenizer
   and can compact substantially earlier than the real token limit.
2. Replace old complete assistant/tool-result groups with bounded historical
   excerpts, retaining the initial task and latest group.
3. If still too large, shorten tool results first, then assistant text, then user
   text. Excerpts retain the beginning/end and an explicit omission marker.
4. Never cut tool IDs, call arguments, or reasoning fields in place. If a latest
   completed tool group itself cannot fit, replace the entire group with a user
   historical excerpt, so no orphaned tool results remain.
5. Preserve the system instruction and tool schemas. If those plus minimum text
   and output reservation cannot fit, report a known-zero-cost configuration
   error. Reduce system/tools/`max_tokens` in this exceptional case.

Explicit HTTP context-length rejections trigger up to three further compression
attempts (60% of the current serialized input per attempt). This is separate from
transport retries and never retries arbitrary HTTP 400 errors. Compression is
local and makes no paid summarization call. It is **lossy**, so it can affect task
quality; it does not guarantee successful inference for arbitrary prompts/relays.

Full canonical messages, tool outputs, usage and cost remain in the trace.
`context_management` records the strategy, sizes, omission counts, request hashes,
and any server-triggered recompressions. Only the outgoing request is compacted.
A later step selecting a larger-window model can use more of the original history.

## Retry accounting and live reports

A response carries `api_attempts` and `cost_complete`. For example, a timeout then
success retains the successful usage/cost, records both attempts, and has
`cost_complete=false`: the timed-out request may also have been billed.
A known-zero rejection (e.g. HTTP 429) then success does not introduce uncertainty.
No missing usage/cost is fabricated as zero.

This flag propagates through agent/evaluator calls, checkpoints, resumed runs,
trace exports and summaries. Unknown-cost traces are excluded from the main
`steps.jsonl` complete-cost training samples; diagnostic exports retain the
`cost_complete` flag. The infer/collect_steps CLIs default to continuing with known costs only, even in
budget mode. Unknown portions contribute zero to the budget counter, but remain
unknown in the ledger and invalid for complete-cost training. Actual charges may
exceed the budget. Use `--stop-on-unknown-cost` to stop before the next call.
Exhausted retries end the affected path; other jobs continue. If an entire round
makes no known-cost progress, the existing no_cost_progress guard still stops it.

Updated on every saved call, including before tool execution:

- `total_api_cost`: known agent + evaluator amount in the current currency.
- `cost_by_profile`: backwards-compatible known **agent** costs in current currency.
- `profile_costs[profile][currency]`: agent/evaluator/total known amounts,
  completeness, uncertain calls and physical API attempt counts.
- `cost_by_currency`: historical amounts, never combined across currencies.
- `SUMMARY.md`: live per-profile table and an explicit lower-bound notice when
  cost is incomplete. `api_call_cost` events also include the profile breakdown.

Shared prefixes count once in summaries; superseded attempts remain billed.
Costs remain estimates from configured pricing and returned usage, not reconciled
provider invoices. Router service/encoder compute is not priced by these LLM rates.

## Resume

The newly configured context windows change provider settings. Use
`--resume-partial` for an existing output directory: completed tasks remain,
unfinished tasks restart under the new configuration and old costs stay recorded.
Adding the optional `max_input_tokens` field with no value alone does not invalidate
old checkpoints. Unknown retry costs are retained after resume; default resume permits an incomplete ledger. The strict `--stop-on-unknown-cost`
option requires reconciliation first. This option may change on resume; use
`--resume-partial` to restart jobs previously aborted for unknown cost.
