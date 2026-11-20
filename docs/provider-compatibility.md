# Provider compatibility

Each supported provider/protocol pair has its own explicit upstream client. Each class implements one protocol, and thin subclasses share the protocol implementation. The shared Responses and Chat Completions clients ignore output extensions they do not model; a provider client validates only its own extensions. Prefer Responses when it supports the desired model and capabilities, then Messages, then Chat Completions. `molto.upstream.registry.CLIENTS` selects the client for both native and translated routes. Both require a registered provider/endpoint pair, including when an explicit `base_url` is supplied. Unsupported pairs fail before HTTP. Neither path switches protocols.

Choose a registered pair explicitly: OpenRouter uses `endpoint="responses"`, Gemini uses `endpoint="chat/completions"`, and Anthropic uses `endpoint="messages"`. A custom URL changes the destination of that registered client; it does not register another protocol. SDK resource paths are appended to the base URL, and Bedrock requires an explicit regional destination.

## Selected protocols and configuration

| Provider | Client | Default SDK base URL | Credential |
| --- | --- | --- | --- |
| Together AI | `TogetherAIResponsesClient`, `TogetherAIChatCompletionsClient` | `https://api.together.ai/v1` | `TOGETHER_AI_API_KEY` |
| Fireworks | `FireworksResponsesClient`, `FireworksChatCompletionsClient` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
| OpenRouter | `OpenRouterResponsesClient` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| Bedrock | `BedrockResponsesClient`, `BedrockMessagesClient`, `BedrockChatCompletionsClient` | Explicit regional URL required | `AWS_BEARER_TOKEN_BEDROCK` or `BEDROCK_API_KEY` |
| Gemini | `GeminiChatCompletionsClient` | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` |

An explicit `api_key` overrides the configured provider key; hosted clients reject an empty key locally. Missing provider keys cannot fall back to OpenAI credentials. An explicit `base_url` overrides the default without changing the provider or protocol. Authentication and account headers supplied through `extra_headers` cannot override the selected upstream identity.

Bedrock accepts either credential name from the environment or `.env`. `AWS_BEARER_TOKEN_BEDROCK` takes precedence when both names are configured, including across those sources. For the same name, the environment overrides `.env`.

For Claude on Bedrock, select `endpoint="messages"`. The [Anthropic-native Messages API](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html) accepts the regular Anthropic SDK wire format on runtime and Mantle. Supply a base URL ending in `/anthropic`, such as `https://bedrock-runtime.us-east-1.amazonaws.com/anthropic` or `https://bedrock-mantle.us-east-1.api.aws/anthropic`; the SDK appends `/v1/messages`. This client uses the selected Bedrock key in `x-api-key`, suppresses ambient Anthropic bearer credentials, and reuses Messages streaming and translation. It does not use the InvokeModel/Converse transports or sign requests with AWS SigV4. Keep the model or inference-profile ID expected by the selected destination; Molto does not switch protocols based on model names. Native calls still require a registered provider/endpoint pair.

```python
response = await molto.responses(
    {"model": "global.anthropic.claude-sonnet-5", "input": "Hello!", "max_output_tokens": 1024},
    provider="bedrock",
    endpoint="messages",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com/anthropic",
)
```

`Molto.messages(..., provider="bedrock", base_url=...)` passes Messages through natively. Responses and Chat callers select `endpoint="messages"` for translation; both streaming and non-streaming results are supported. Model access and region availability are owned by Bedrock.

Select the upstream protocol explicitly when it differs from the public method. For example, `Molto.chat_completions(..., provider="openrouter", endpoint="responses")` translates Chat input through the Responses client. The existing method defaults remain unchanged.

Protocol availability is documented by [Together](https://docs.together.ai/docs/inference/responses-api), [Fireworks](https://docs.fireworks.ai/api-reference/post-responses), [OpenRouter](https://openrouter.ai/docs/api_reference/responses/overview), [Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-responses-api.html), and [Gemini](https://ai.google.dev/gemini-api/docs/openai). Availability does not establish support for every model or feature. Together's Responses documentation lists GLM-5.2 and GLM-5.3 families; its Chat Completions recordings do not validate Responses for those Chat models. Together, Fireworks, and Bedrock expose separate Chat Completions clients to reach models and capabilities available through that protocol. Bedrock's [Chat Completions guide](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html) documents the regional runtime and Mantle destinations; callers must choose a destination supported by their model.

## Provider differences

The [reasoning guide](reasoning.md) describes profile selection, wire labels, signature routing, and the limits of collected Chat output.

Messages thinking-capability lookup recognizes the `claude-...` portion regardless of provider, route, or inference-profile prefixes. For example, `us.anthropic.claude-haiku-4-5-20251001-v1:0` uses the manual thinking budget, while `global.anthropic.claude-sonnet-5` uses adaptive thinking. The exact supplied model ID stays unchanged on the wire.

OpenRouter accepts `format="unknown"` on raw-only Responses reasoning. Translated history omits this encoding label. Named formats and formatted summary or signature state reject. Fireworks and OpenRouter both follow the explicit Responses labels. Their raw profiles classify reasoning only when it arrives through Chat or Messages.

Bedrock Chat uses raw classification and the `reasoning` field for GPT-OSS 20B on Oregon Mantle. A runtime probe returned literal `<reasoning>` tags in ordinary content, which remains visible text. Signed Bedrock Messages thinking uses its summary profile. The selected model and endpoint determine which features are available.

Gemini uses the summary profile and the `google` Chat signature namespace. Its `extra_content.google.thought` marker identifies summary text. A streamed tool signature can accompany the first arguments. A message signature can arrive after the final text chunk. Continuation restores those attachments. Google describes the exposed text as a thought summary. [Google OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)

Gemini repairs missing streamed tool indices from first-seen call IDs or the sole known call. Repaired fragments omit repeated IDs so SDK accumulation retains the original call ID. Native chunks with indices pass through directly.

A [bounded live probe on 2026-11-16](https://github.com/EmbeddedLLM/molto/actions/runs/43849083274) tested signature placement with `gemini-3.1-flash-lite`, minimal reasoning, and non-streaming native OpenAI SDK requests. Native tool generation and unchanged continuation succeeded. The probe then combined the earlier tool call with a genuine message-level signature from its successful continuation, removing the call's original signature. These comparison histories were synthetic; they were not emitted by Gemini in that form.

| Synthetic history      | Signature on message only                              | Same signature moved onto tool call    |
| ---------------------- | ------------------------------------------------------ | -------------------------------------- |
| Text and one tool call | HTTP 400; error identified a missing thought signature | HTTP 200; returned the expected result |
| One tool call, no text | HTTP 200; returned the expected result                 | HTTP 200; returned the expected result |

The no-text result establishes request acceptance for the tested configuration. Equivalent model use and other configurations remain unverified.

Gemini and Fireworks Chat clients omit `stream_options.include_obfuscation` and null top-level message fields, which their tested endpoints reject, on native and translated requests. Other streaming options and nested message values remain intact. Gemini tool-bearing successful `finish_reason="stop"` becomes normalized tool termination on translated routes, producing Messages `stop_reason="tool_use"`; native output retains the provider's original finish reason.

Translated Responses function tools default omitted or `null` `strict` to `true`; explicit `false` remains non-strict. Translated Chat Completions tools retain their non-strict default. Molto forwards the supplied parameter schema without adding strict-mode constraints and does not retry with non-strict tools. This is a deliberate approximation of [native Responses](https://developers.openai.com/api/docs/guides/function-calling#strict-mode), which attempts schema normalization and can fall back to non-strict mode. Supply a strict-compatible schema or set `strict: false` explicitly; an upstream may reject an incompatible schema or unsupported strict mode. Native passthrough preserves omission and explicit values for the provider to interpret.

## Migration notes

- Native and translated calls now require a registered provider/endpoint pair. An explicit `base_url` no longer enables unregistered native fallback.
- Compatible-server callers using `provider="openai"` must select `provider="openrouter"` or `provider="gemini"` for those providers' translated extensions, and select their registered endpoint.
- Translated Responses function-tool `strict: null` previously became upstream `false` and now becomes `true`. Omitted `strict` also defaults to `true`; explicit `false` and supplied schemas remain unchanged. Molto neither normalizes strict schemas nor retries with non-strict tools.
- Responses now preserves Fireworks summary labels. Chat and Messages continuation through Fireworks or OpenRouter can return that unlabeled text as raw reasoning.
- Shared Chat history now groups interleaved text and calls into separate fields. It preserves text order and call order but cannot preserve their interleaving. Explicit assistant-message boundaries remain separate.
- Gemini streams text before a trailing message signature and restores the token on the assistant message. Tool signatures stay with their calls. vLLM and Bedrock Chat use `reasoning` by default. For older vLLM deployments, set `chat_reasoning_field="reasoning_content"`.

## Replay evidence

Bedrock Messages has independent native text/tool and signed-thinking recordings for the model below, plus offline transport and translation coverage using the existing matched Messages/OpenAI fixtures. Matched tests compare native Anthropic SDK output and translated OpenAI SDK output, and check the runtime/Mantle URLs, credential isolation, and request bodies. Live provider evidence comes from the Bedrock-owned recordings; matched fixtures alone do not establish server acceptance.

Provider conversations share `tests/upstream/test_multi_turn.py`. Cassettes use the existing scenario-first layout, for example `tests/cassettes/upstream_ir/tool_use/togetherai-responses-stream-00.yaml`. Each provider uses its own recordings. Canonical OpenAI/Anthropic fixtures are not reused as evidence of hosted-provider behavior.

The suite checks native SDK parity, actual-output history, normalized lifecycle and semantics, exact expected subsequent requests, and focused public translation or rejection cases. Independent SDK responses supply the comparison values. Expected normalized request copies account for documented wire representations, downstream strictness defaults, parsed tool-argument JSON, and omitted optional metadata; recorded fixtures are not rewritten to match Molto output.

The reasoning conversation tests carry actual Molto output through two tool results to a final answer. OpenRouter, Fireworks, and Together Responses recordings cover Chat and Messages callers in both downstream response modes. Fireworks and Together Chat recordings cover Responses and Messages callers in both modes. Bedrock Messages signed-thinking recordings cover Responses callers in both modes, preserving reasoning text and signatures. Translated calls consume the streaming upstream recordings even when the downstream response is non-streaming. Non-streaming upstream recordings are independently replayed for native SDK parity and native continuation.

The recordings below were imported unchanged. The later [cassette storage cleanup](../tests/cassettes/README.md) removed duplicate response byte copies while preserving every parsed response value and SSE line. Non-streaming JSON replay now serializes the retained `body` rather than preserving its original whitespace.

| Provider/model | Protocol | Retained conversation evidence | Remaining limits |
| --- | --- | --- | --- |
| Together `zai-org/GLM-5.2` | Responses | Two-turn text and three-turn sequential tools in both native modes; additional three-turn reasoning/tool continuation with `reasoning.effort="low"`, including public Chat/Messages continuation in both downstream modes | Other models, images, and parallel calls are unverified |
| Together `zai-org/GLM-5.2` | Chat Completions | Three-turn native reasoning/tool continuation in both modes with `reasoning_effort="low"`; public Responses/Messages continuation in both downstream modes | Other models, images, and parallel calls are unverified |
| Fireworks `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` | Responses | Text, sequential and parallel tools, and low-effort reasoning/tool continuation. Missing cache-write counts remain unknown. | Chat or Messages continuation can return a recorded summary as raw reasoning. Other models and images are unverified. |
| Together `Qwen/Qwen3.5-9B` | Chat Completions | Two-turn text and three-turn sequential tools, streaming and non-streaming, with `reasoning.enabled=false` | Other models, images, parallel calls, and reasoning-enabled continuation are unverified |
| Fireworks `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` | Chat Completions | Two-turn text and three-turn sequential tools in both modes; additional three-turn native reasoning/tool continuation with `reasoning_effort="low"`, including public Responses/Messages continuation in both downstream modes | Other models, images, and parallel calls are unverified |
| OpenRouter `nvidia/nemotron-3.5-lightning:free` | Responses | Text in both native modes; non-streaming sequential tools; streamed raw-reasoning sequential tools with public Chat/Messages continuation in both downstream modes | Non-streaming native reasoning and parallel Responses calls are unverified |
| Gemini `gemini-3.1-flash-lite` | Chat Completions | Text and sequential tool history, including message and tool signatures. Medium-effort captures with `include_thoughts=true` exercise the Google thought marker. | Other models, images, and live parallel calls are unverified. |
| Bedrock `global.openai.gpt-5.6-luna` | Responses and Chat Completions | Two-turn text and three-turn sequential tools, streaming and non-streaming, with reasoning disabled | Verified only through the runtime endpoint in `ap-southeast-1`; other models, regions, images, parallel calls, and reasoning-enabled continuation are unverified |
| Bedrock `openai.gpt-oss-20b` | Chat Completions | Three-turn raw reasoning and tool continuation on Oregon Mantle in both native modes | Runtime and other model or region combinations require separate evidence. |
| Bedrock `global.anthropic.claude-haiku-4-5-20251001-v1:0` | Messages | Text/tool conversations in both modes; additional three-turn signed-thinking/tool continuation in both native modes with a 1,024-token thinking budget; normalized and public Responses continuation preserve signatures in both downstream modes | Verified through runtime in `ap-southeast-1`; Mantle, other models/regions, images, and parallel tools are unverified |

The first 24 recordings in `upstream_ir/tool_use_reasoning_low/` were captured on November 17, 2026: 12 Together GLM-5.2 exchanges across Responses/Chat and both modes, six Fireworks Nemotron Chat exchanges, and six Bedrock Haiku 4.5 Messages exchanges across both modes. Each three-turn native SDK conversation carries actual reasoning and tool output through two tool results to a final answer. Together and Fireworks used low reasoning; Bedrock Messages used enabled thinking with a 1,024-token budget. All used a 2,048-token output cap, zero SDK retries, and the recorder proxy outside pytest. These captures establish the tested models/settings, not other models or equivalent reasoning use across independent generations. Six additional Bedrock Mantle Chat captures use `openai.gpt-oss-20b` in `us-west-2`, low effort, a 2,048-token output cap, and zero retries. Two three-turn tool conversations include raw `reasoning`; summary-bearing Bedrock Chat output remains unverified. Six Fireworks Responses captures omit the SDK-required cache-write count. The scoped compatibility model preserves that count as unknown in both native modes.

The Bedrock GPT-5.6 Luna captures use `https://bedrock-runtime.ap-southeast-1.amazonaws.com/openai/v1`, manual actual-output history, and zero retries. Responses requests use `store=false`; Chat Completions requests use `reasoning_effort="none"`. The model is an explicit system-defined inference profile. An earlier Mantle `openai.gpt-oss-20b` attempt returned HTTP 200 but stalled after `response.created` and `response.in_progress`; it was stopped after more than five minutes and remains in staging, outside the successful replay corpus. A Chat Completions attempt with `mistral.ministral-3-3b-instruct` returned an invalid-model HTTP 400 on runtime and timed out after 75 seconds on Mantle; neither attempt is successful replay evidence. The Singapore endpoint and global inference profile do not establish that inference occurred in Singapore.

Bedrock Messages captures were recorded on November 16, 2026 with native Anthropic SDK 0.121.0 through the recorder proxy, outside pytest, at `https://bedrock-runtime.ap-southeast-1.amazonaws.com/anthropic/v1/messages`. All 10 requests returned HTTP 200, with a 512-token output cap and zero retries; all five streams completed. Caller history contains the actual SDK content blocks, and the two sequential calls calculate 2 + 3 and then 5 + 7 before returning 12. The recorded request and response values are unchanged. Six additional native Messages requests recorded on November 17 enabled thinking with a 1,024-token budget and a 2,048-token output cap; both three-turn conversations carried actual signed thinking and tool results through 2 + 3 and then 5 + 7 to a final answer of 12. Native SDK continuation replay passed for all six exchanges. Normalized replay and public Responses continuation also passed, comparing actual returned thinking and signatures with the Anthropic SDK and checking exact subsequent upstream requests in both downstream modes. The global inference profile does not establish the execution region.

Chat Completions continuation retains the actual SDK output. Together and Gemini repeat `delta.role`; the SDK accumulator concatenates these values, so the recording caller and replay helper restore the single `assistant` role before reuse. Fireworks rejects null output-only fields in assistant history, so its recording caller and replay helper omit those nulls. Gemini also rejects null optional message fields; its caller omits top-level nulls while retaining nested Google state unchanged. These are history-construction choices in the evidence workflow; native production output remains unchanged. The Fireworks and Gemini clients also omit null top-level message fields from every request they send, so history echoed from SDK output is accepted; nested values, including Google state, are unchanged. Every other client forwards message nulls as sent, which the Bedrock and Together recorded requests show those providers accept. Native parity sends input copies without empty output-only annotations, null `reasoning_content`/`tool_calls`, or stream-only tool indices, which are absent from the Chat request schema. The recorded source history is still checked against actual SDK output, and recorded response values remain unchanged.

Gemini captures came from the [successful GitHub-hosted recording run](https://github.com/EmbeddedLLM/molto/actions/runs/42160882628), using the existing Actions secret, native OpenAI SDK calls through the recorder proxy, zero retries, and 90-second request deadlines. The temporary driver and workflow remain on a separate recording branch. Artifact YAML was imported unchanged before the storage cleanup described above. Streamed tool calls omit `index`; independent SDK references assign index zero only for these recorded single-call responses before comparison with the Gemini client. Native parity otherwise retains full provider output, including streamed `finish_reason="stop"` on tool calls and Google thought signatures on text and tool output.

A [GitHub-hosted Gemini probe](https://github.com/EmbeddedLLM/molto/actions/runs/44128342414) captured text and tool generation in both native modes with medium effort and `include_thoughts=true`. Its complete replies used the Google thought marker. Its streams returned unmarked text or calls. The probe stopped after the first turns because it expected a `reasoning_content` field.

A [second Gemini recording run](https://github.com/EmbeddedLLM/molto/actions/runs/44276479795) retained ten original exchanges with medium thinking, `include_thoughts=true`, and a 2,048-token output cap. The four native conversations used actual assistant output for text and tool continuation in both modes. A [mixed-mode follow-up](https://github.com/EmbeddedLLM/molto/actions/runs/44277431390) completed another five requests. Complete responses marked the entire content field as thought, including prose resembling an answer. There was no separate visible answer in those messages. The streams in both runs carried unmarked text or calls and genuine signatures. Molto follows the marker received in each mode.

A four-call [combined-history comparison](https://github.com/EmbeddedLLM/molto/actions/runs/44280209744) kept the streamed answer and its genuine message signature together. The comparison added `reasoning_content` from an independent marked reply to that same prompt. The control and combined messages both returned the expected answer, 36, in streaming and non-streaming mode. This validates acceptance of the combined shape for this model and setting.

The retained OpenRouter captures use an explicit free model, disabled provider fallbacks, zero price caps, and no plugins. Reported costs were zero. Responses history is manual with `store=false`. Captures establish only the recorded model, settings, and date; replay does not prove that translated request bodies were executed live.

The duplicate PR #76 corpus, per-run manifests, source snapshots, and unsupported-protocol captures are omitted. Full issue #78 coverage still requires the remaining provider/model cells.

## Opt-in live smoke checks

Replay verifies saved responses and request construction; it cannot establish that a hosted endpoint accepts today's translated requests. Run the standalone smoke command explicitly with one provider, model, and upstream protocol:

```bash
uv run --extra test python -m scripts.smoke_provider \
  --provider gemini --model gemini-3.1-flash-lite \
  --upstream chat/completions \
  --extra-body '{"reasoning_effort":"minimal"}'
```

The command uses configured provider credentials and makes live, potentially billable requests. It first checks Molto's native route, then the selected downstream protocols (all three by default), with text and a forced function call followed by its actual tool-result history in streaming and non-streaming modes. It checks tool identity, arguments, termination, and final output. The native route is a diagnostic control, not an independent SDK comparison; offline replay supplies SDK parity coverage.

Defaults bound the run to 18 HTTP attempts, 512 output tokens per request, a 45-second HTTP timeout, and a 180-second total deadline. SDK retries count against the HTTP budget. When the budget runs out, the run stops and reports the exhausted budget rather than a connection error. Use `--downstream`, `--mode`, `--max-requests`, and `--deadline` to narrow a run, and `--base-url` for an explicit protocol destination. Provider-specific `--extra-body` options cannot replace the smoke inputs, tools, modes, or token limits. Results report pass/fail and error class/status without printing provider response bodies or credentials. A failing check exits nonzero.

These checks create no cassettes and never run as part of pytest or default CI. A pass establishes only the selected model, settings, and scenarios at that time. Known unsupported reasoning/signature translations may fail; choose explicit reasoning controls where appropriate rather than treating all provider/model combinations as supported.
