# Hosted provider compatibility

Provider selection, upstream protocol, destination, and model are independent. Choose `endpoint` explicitly when the provider exposes a different protocol from the public method. Molto does not infer capabilities from a hostname or silently switch endpoints. Configuration support is not certification of every hosted model.

## Destinations and credentials

An explicit `api_key` overrides the configured provider key; an explicit empty key fails locally. Missing keys cannot fall back to OpenAI or Anthropic credentials. An explicit `base_url` overrides the destinations below. Caller authentication and account headers cannot override the selected upstream identity.

| Provider | Upstream | Default SDK base URL | Credential |
| --- | --- | --- | --- |
| `togetherai` | Chat Completions | `https://api.together.ai/v1` | `TOGETHER_AI_API_KEY` |
| `gemini` | Chat Completions | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` |
| `openrouter` | Responses, Chat Completions | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `openrouter` | Messages | `https://openrouter.ai/api` | `OPENROUTER_API_KEY` (Bearer) |
| `fireworks` | Responses, Chat Completions | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
| `fireworks` | Messages | `https://api.fireworks.ai/inference` | `FIREWORKS_API_KEY` |
| `bedrock` | Responses, Chat Completions, Messages | Explicit regional URL required | `AWS_BEARER_TOKEN_BEDROCK` (Bearer) |

OpenAI SDK paths append `/responses` or `/chat/completions`; Anthropic SDK paths append `/v1/messages`. Bedrock examples reported in issue #78 are `https://bedrock-runtime.ap-southeast-1.amazonaws.com/openai/v1` and `https://bedrock-runtime.us-east-1.amazonaws.com/anthropic`. Specify the source region and model/inference profile yourself. This supports API-key authentication; it does not add SigV4, Converse, InvokeModel, or a cross-region routing policy. The source hostname does not establish execution region or residency.

Gemini rejects other upstream protocols. Together requires an explicit URL for alternative upstream protocols; such custom endpoint probes do not claim that Together hosts Responses or Messages.

The destination references are [Together's OpenAI compatibility guide](https://docs.together.ai/docs/inference/openai-compatibility), [Gemini's OpenAI guide](https://ai.google.dev/gemini-api/docs/openai), [OpenRouter's API reference](https://openrouter.ai/docs/api/reference/overview), [OpenRouter's Messages configuration](https://github.com/OpenRouterTeam/docs/blob/main/cookbook/coding-agents/claude-code-integration.mdx), [Fireworks' Anthropic guide](https://docs.fireworks.ai/tools-sdks/anthropic-compatibility), and [Bedrock API-key authentication](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html). These documents establish configuration, not replay or live compatibility.

## History and reasoning

Native Chat accepts assistant `reasoning: str | list[str]` and the string-only `reasoning_content` alias. It preserves exact strings, whitespace, array order, empty entries, explicit null content, and valid function/custom tool histories. Reasoning extensions reject null, non-string members, and unrelated fields. Stream-only tool `index` is not accepted in submitted history; remove SDK bookkeeping before sending the collected message.

Translated input rejects two populated reasoning aliases. An empty string or empty array does not mask the populated alternate; an array containing an empty string remains populated. Ordered array entries become separate reasoning parts. When unsigned parts become a Chat string, their separator is a newline, including leading or trailing empty parts. The existing reasoning-before-text-before-calls history order remains enforced. Output streams finish active parts before starting subsequent text or reasoning; noncontiguous resumed calls reject explicitly.

Together translates its documented `reasoning` dialect as unsigned raw reasoning. A model emitting `reasoning_content` remains native-compatible but is explicitly rejected on translated output rather than silently losing reasoning. The fresh Qwen capture below demonstrates this limitation. Fireworks Chat uses raw `reasoning_content`; its Responses and Messages profiles use summary semantics. Bedrock and OpenRouter use summary profiles. Native routes retain provider output; translated routes still reject signatures or summaries they cannot represent. No tag parsing or automatic model-specific dialect fallback is provided.

Native Gemini history supports `extra_content.google` on messages, individual content parts, and function calls. The Google object is opaque, including tool-result `parts[].file_data`, signatures, and null values. Attachments are not moved or combined. Translated input/output containing populated Google state rejects explicitly. OpenRouter Responses reasoning `format` similarly survives native continuation and rejects unsupported translation.

Gemini missing tool indices are repaired only when identity is unambiguous: stable call IDs select stable indices; an anonymous continuation requires a single known call. Parallel calls never share an inferred index. Conflicting identities and ambiguous fragments fail. Repeated identical IDs on repaired fragments are treated as identity snapshots so SDK accumulation does not concatenate IDs. Otherwise indexed native chunks are preserved. Translated streams additionally require contiguous fragments per call.

Request transport remains separate from history. For Google's literal wire `extra_body.google` envelope, pass `extra_body={"extra_body": {"google": {...}}}`. Ordinary transport extras merge at the top level under SDK precedence. They do not replace `extra_content` history. The separate Anthropic `extra_content.anthropic` continuation work belongs to [PR #92](https://github.com/EmbeddedLLM/molto/pull/92), targeting PR #74; this change does not implement that dependency.

## Evidence matrix

The fresh captures are under `tests/cassettes/provider_recordings/2026-11-03/`. Each group has a manifest with original ordered exchanges, requested/returned model, real destination, native mode, settings, SDK/Python versions, and completion or failure status. They were recorded through the native SDK with zero retries, outside pytest, and copied without modifying their bytes. Incomplete groups are failure evidence, not complete conversation support.

| Provider/configuration | Fresh live evidence | Offline evidence | Remaining limitations |
| --- | --- | --- | --- |
| Together, `Qwen/Qwen3.5-9B`, Chat at `/v1/chat/completions` | Two-turn text and three-turn sequential tools in both native modes; explicit parallel tools in native non-streaming mode | Exact native SDK output; accepted non-streaming tool/parallel histories replay exactly; strict unsupported-history rejection for recorded null reasoning/output-only placeholders | Emits `reasoning_content`, so translated reasoning is unsupported. Recorded text history includes null reasoning; streamed collection includes output-only null annotations. These histories reject locally. Streamed parallel runs returned only one call; no live streaming parallel claim |
| Together, `meta-llama/Llama-3.3-70B-Instruct-Turbo`, Chat | HTTP 400 `model_not_available` | Original error body retained | Account cannot use this model as a serverless endpoint |
| Gemini, `gemini-3.8-flash`, Chat | HTTP 400 location restriction | Native extension round trips, index repair/ambiguity, translation rejection, exact nested Google wire envelope (synthetic) | No successful live model conversation from this environment |
| OpenRouter, each protocol | None; credential unavailable | Destination/authentication/native SDK replay and Responses format continuation (synthetic) | Model routing, required backend state, live text/tools, translated provider semantics unknown |
| Fireworks, each protocol independently | None; credential unavailable | Exact per-protocol destinations/authentication and native SDK replay | Model-specific reasoning/history, usage placement, stored/manual Responses continuation and live text/tools unknown |
| Bedrock, explicit reported regional URLs | None; credential/profile unavailable | URL joining, Bearer authentication, native SDK replay | Exact live model/profile mapping, reasoning, full tools and both native modes unknown |

`tests/cassettes/reference_pr76/` retains deliberately matched historical success fixtures from [PR #76, commit 42b9040](https://github.com/EmbeddedLLM/molto/pull/76/commits/42b9040cb2c74b3676d1749b2c4473c16b3bda9f). They cover text, raw reasoning, remote/data images, sequential tools, named/disabled tool choice, and parallel calls. Their manifest records original paths and hashes. These are independent of fresh provider availability. Historical Together error fixtures retain original response bodies/status/headers; aligned or shortened requests do not establish a live error reproduction.

Synthetic SDK parity probes and replayed OpenAI/Anthropic fixtures establish local routing and serialization only. They do not turn an unrecorded provider cell into live support. Full issue #78 completion still requires the unknown provider cells, broader model-specific continuation evidence, and the separate Anthropic dependency.
