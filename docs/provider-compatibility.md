# Provider compatibility

Each supported provider/protocol pair has its own explicit upstream client. Each class implements one protocol, and thin subclasses share the protocol implementation. Prefer Responses when it supports the desired model and capabilities, then Messages, then Chat Completions. `molto.upstream.registry.CLIENTS` selects the client; unsupported provider/endpoint pairs fail locally instead of silently switching protocols.

Existing OpenAI Responses and Chat Completions, Anthropic Messages, and vLLM routes remain available. Cross-vendor configurations such as OpenAI Messages and Anthropic Responses are unsupported.

## Selected protocols and configuration

| Provider | Client | Default SDK base URL | Credential |
| --- | --- | --- | --- |
| Together AI | `TogetherAIResponsesClient`, `TogetherAIChatCompletionsClient` | `https://api.together.ai/v1` | `TOGETHER_AI_API_KEY` |
| Fireworks | `FireworksResponsesClient`, `FireworksChatCompletionsClient` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
| OpenRouter | `OpenRouterResponsesClient` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| Bedrock | `BedrockResponsesClient`, `BedrockChatCompletionsClient` | Explicit regional URL required | `AWS_BEARER_TOKEN_BEDROCK` or `BEDROCK_API_KEY` |
| Gemini | `GeminiChatCompletionsClient` | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` |

An explicit `api_key` overrides the configured provider key; an empty key fails locally. Missing provider keys cannot fall back to OpenAI credentials. An explicit `base_url` overrides the default without changing the provider or protocol. Authentication and account headers supplied through `extra_headers` cannot override the selected upstream identity.

Bedrock accepts either credential name from the environment or `.env`. `AWS_BEARER_TOKEN_BEDROCK` takes precedence when both names are configured, including across those sources. For the same name, the environment overrides `.env`.

Select the upstream protocol explicitly when it differs from the public method. For example, `Molto.chat_completions(..., provider="openrouter", endpoint="responses")` translates Chat input through the Responses client. The existing method defaults remain unchanged.

Protocol availability is documented by [Together](https://docs.together.ai/docs/inference/responses-api), [Fireworks](https://docs.fireworks.ai/api-reference/post-responses), [OpenRouter](https://openrouter.ai/docs/api_reference/responses/overview), [Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-responses-api.html), and [Gemini](https://ai.google.dev/gemini-api/docs/openai). Availability does not establish support for every model or feature. Together's Responses documentation lists GLM-5.2 and GLM-5.3 families; its Chat Completions recordings do not validate Responses for those Chat models. Together, Fireworks, and Bedrock expose separate Chat Completions clients to reach models and capabilities available through that protocol. Bedrock's [Chat Completions guide](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html) documents the regional runtime and Mantle destinations; callers must choose a destination supported by their model.

## Provider differences

OpenRouter Responses identifies raw text and summaries explicitly. Its client accepts unsigned raw reasoning, including the recorded `format: "unknown"`, or a summary with its required signature. Named formats, formatted summaries, mixed representations, and mid-item representation changes reject. Native passthrough preserves the original fields. Raw reasoning projects to Chat and Messages; signed summaries can project to Messages. Incoming unsigned Messages thinking still follows the summary-input contract and rejects, so this output support does not establish a complete Messages reasoning round trip.

Fireworks Responses uses the summary profile. Its recorded low-reasoning output contains unsigned summaries, which remain valid native evidence but reject during normalization. Together uses the raw reasoning profile. The Chat Completions recordings disable reasoning; they do not establish reasoning-enabled compatibility for Bedrock or Fireworks.

Gemini Chat preserves opaque `extra_content.google` attachments on messages, content parts, and function calls, including nested nulls. Translation rejects populated opaque state. Missing streamed tool indices are repaired only when the call identity is unambiguous; ambiguous output rejects rather than losing calls.

## Replay evidence

Provider conversations share `tests/upstream/test_multi_turn.py`. Cassettes use the existing scenario-first layout, for example `tests/cassettes/upstream_ir/tool_use/togetherai-responses-stream-00.yaml`. Each provider uses its own recordings. Canonical OpenAI/Anthropic fixtures are not reused as evidence of hosted-provider behavior.

The suite checks native SDK parity, actual-output history, normalized lifecycle and semantics, exact expected subsequent requests, and focused public translation or rejection cases. Independent SDK responses supply the comparison values. Expected normalized request copies account for documented wire representations and omitted optional metadata; recorded fixtures are not rewritten to match Molto output.

| Provider/model | Protocol | Retained conversation evidence | Remaining limits |
| --- | --- | --- | --- |
| Together `zai-org/GLM-5.2` | Responses | Two-turn text and three-turn sequential tools, streaming and non-streaming, with minimal reasoning | Other models, images, parallel calls, and reasoning-enabled continuation are unverified |
| Fireworks `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` | Responses | Text, sequential tools, and parallel tools in both native modes with reasoning disabled; streamed unsigned-summary tool history | Unsigned summaries cannot translate; other models and images are unverified |
| Together `Qwen/Qwen3.5-9B` | Chat Completions | Two-turn text and three-turn sequential tools, streaming and non-streaming, with `reasoning.enabled=false` | Other models, images, parallel calls, and reasoning-enabled continuation are unverified |
| Fireworks `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` | Chat Completions | Two-turn text and three-turn sequential tools, streaming and non-streaming, with `reasoning_effort="none"` | Other models, images, parallel calls, and reasoning-enabled continuation are unverified |
| OpenRouter `nvidia/nemotron-3.5-lightning:free` | Responses | Text in both native modes; non-streaming sequential tools; streamed raw-reasoning sequential tools | Parallel Responses calls and public translated Messages reasoning round trips are unverified |
| Gemini | Chat Completions | Configuration, Google attachment, and missing-index regressions | Live recording was blocked by the provider's location restriction |
| Bedrock `global.openai.gpt-5.6-luna` | Responses and Chat Completions | Two-turn text and three-turn sequential tools, streaming and non-streaming, with reasoning disabled | Verified only through the runtime endpoint in `ap-southeast-1`; other models, regions, images, parallel calls, and reasoning-enabled continuation are unverified |

Bedrock captures use `https://bedrock-runtime.ap-southeast-1.amazonaws.com/openai/v1`, manual actual-output history, and zero retries. Responses requests use `store=false`; Chat Completions requests use `reasoning_effort="none"`. The model is an explicit system-defined inference profile. An earlier Mantle `openai.gpt-oss-20b` attempt returned HTTP 200 but stalled after `response.created` and `response.in_progress`; it was stopped after more than five minutes and remains in staging, outside the successful replay corpus. A Chat Completions attempt with `mistral.ministral-3-3b-instruct` returned an invalid-model HTTP 400 on runtime and timed out after 75 seconds on Mantle; neither attempt is successful replay evidence. The Singapore endpoint and global inference profile do not establish that inference occurred in Singapore.

Chat Completions continuation retains the actual SDK output. Together repeats `delta.role`; the SDK accumulator concatenates these values, so the recording caller and replay helper restore the single `assistant` role before reuse. Fireworks rejects null output-only fields in assistant history, so its recording caller and replay helper omit those nulls. These are history-construction choices in the evidence workflow; native production output remains unchanged. Native parity sends input copies without empty output-only annotations, null `reasoning_content`/`tool_calls`, or stream-only tool indices, which are absent from the Chat request schema. The recorded source history is still checked against actual SDK output, and response bytes remain unchanged.

The retained OpenRouter captures use an explicit free model, disabled provider fallbacks, zero price caps, and no plugins. Reported costs were zero. Responses history is manual with `store=false`. Captures establish only the recorded model, settings, and date; replay does not prove that translated request bodies were executed live.

The duplicate PR #76 corpus, per-run manifests, source snapshots, and unsupported-protocol captures are omitted. Full issue #78 coverage still requires the remaining provider/model cells.
