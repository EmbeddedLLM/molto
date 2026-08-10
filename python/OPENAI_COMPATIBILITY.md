# OpenAI compatibility

This document describes the supported OpenAI-compatible boundary. Every route uses `InputIR`, an OpenAI-compatible SDK endpoint, semantic `OutputEvent` values, and `OutputIR` before public composition.

## Route matrix

| Public method and body | Model prefix | `endpoint` | Public output | Status |
| --- | --- | --- | --- | --- |
| `chat_completions(OpenAIChatCompletionRequest)` | `openai/` | `chat/completions` | Chat response/chunks | Supported subset |
| `chat_completions(OpenAIChatCompletionRequest)` | `openai/` | `responses` | Chat response/chunks | Supported semantic subset; reasoning output is omitted |
| `responses(OpenAIResponsesRequest)` | `openai/` | `responses` | Responses response/events | Supported subset |
| `responses(OpenAIResponsesRequest)` | `openai/` | `chat/completions` | Responses response/events | Supported semantic subset |
| Either matching public method/body | `vllm/` | Either OpenAI-compatible endpoint | Method-selected OpenAI output | Explicit `base_url`; unverified subset |
| `chat_completions(OpenAIChatCompletionRequest)` | `anthropic/` | `chat/completions` (default) | Chat response/chunks | Conservative text/function subset; unverified |
| `responses(OpenAIResponsesRequest)` | `anthropic/` | `chat/completions` (default) | Responses response/events | Supported semantic subset; unverified |
| Either matching public method/body | `anthropic/` | `responses` | None | Unsupported |

The public method selects both the accepted OpenAI request schema and returned output protocol. The keyword-only `endpoint` argument selects the execution endpoint, ahead of the provider-aware default. It is separate from the OpenAI request body and never reaches `InputIR` conversion or provider translation. Anthropic models default to Chat Completions for both public methods because Anthropic does not provide an OpenAI-compatible Responses endpoint. The model prefix selects the provider configuration.

Anthropic's OpenAI-compatible endpoint fully supports common text, token-limit, streaming, and function-calling fields. It accepts but ignores OpenAI `response_format`, function-tool `strict`, `reasoning_effort`, `store`, penalties, and logprob controls; those fields are recorded separately from effective provider capabilities. Claude Sonnet 5 also rejects non-default sampling parameters, so callers should omit `temperature` and `top_p` for that model.

## Supported portable capabilities

- text history and representable image input;
- function definitions, function-call history, function results, and supported function choices;
- text, JSON object, and JSON Schema output formats;
- common sampling, token-limit, reasoning-effort, and parallel-call controls;
- generated Chat audio and previous assistant-audio references on the Chat Completions route;
- standard OpenAI Responses hosted tools, hosted-tool choices, output items, citations, and include selections on the matching Responses route;
- stateless Responses continuation for assistant output, reasoning, web-search-call, code-interpreter-call, and image-generation-call history, including `store=False` and image-generation ID-only references;
- reasoning items when the selected output contract supports them;
- Responses reasoning effort when `Molto.chat_completions()` selects Chat output;
- Responses text verbosity (`low`, `medium`, or `high`) on the native Responses route;
- ordinary completion, incomplete, failed, refusal, and portable usage data.

When `Molto.chat_completions(..., endpoint="responses")` executes an `OpenAIChatCompletionRequest`, Chat Completions has no reasoning-item or reasoning-event shape. Molto therefore passes representable reasoning configuration through to the provider, retains standard Chat reasoning-token usage, and intentionally omits returned Responses reasoning items and events from the Chat response or stream.

The Chat request adapter removes `stream_options` when the selected endpoint is Responses because it is not a Responses SDK argument. In the other direction, the Responses request adapter removes `include` and reasoning `summary` when the selected endpoint is Chat Completions because Chat has no corresponding request fields. `store` remains a Chat field; Anthropic accepts and ignores it as documented by its OpenAI compatibility layer. The Responses request builder omits neutral `frequency_penalty=0` and `presence_penalty=0` defaults because the Responses SDK does not accept those arguments; non-zero penalties are dropped in non-strict mode and rejected before provider construction when `strict=True`.

## Client configuration and strict mode

Transport configuration is split by lifetime: `Molto` accepts an optional shared `http_client` and default `max_retries`, while `Molto.responses()` and `Molto.chat_completions()` require `api_key` and accept `base_url` plus an optional per-call `max_retries` override. A per-call retry value takes priority over the gateway default.

`strict` controls how unsupported request features are handled. Without it (the default), a feature the selected provider/endpoint contract does not honor is dropped and ignored, and the request proceeds with the supported subset. With `strict=True`, the same feature raises `ModelCapabilityError` before provider construction. The Anthropic `ignored_features` carve-out follows the same rule: fields the compatibility endpoint accepts but ignores are silent in non-strict mode and rejected in strict mode.

## Rejected capabilities

Molto rejects a feature when the OpenAI-shaped IR or selected public output cannot preserve it completely. This includes Responses hosted tools routed to Chat Completions, MCP where the selected route cannot preserve it, generated Chat audio when the destination is not Chat Completions, provider-native events, unsupported annotations/logprobs, unknown `extra_body` fields, background execution, and provider-affine state on a different provider.

Stateless replay of MCP, computer-use, file-search, and other hosted output-item types remains unsupported and is rejected rather than partially projected.

Rejection happens at public validation, capability preflight, or output composition. In non-strict mode, request features the selected contract does not honor are dropped and ignored before provider construction instead of rejected; strict mode turns the same gaps into capability-preflight errors. No unsupported provider feature is projected into ordinary text, except for the explicitly documented Responses reasoning omission when the destination is Chat Completions.

## Event contract

The internal event vocabulary contains semantic lifecycle events:

- `response.started`;
- output item added/done;
- content part added/delta/done;
- function-call argument delta/done;
- `response.completed`, `response.incomplete`, and `response.failed`.

It also contains two OpenAI-compatible fidelity events:

- `openai.stream_frame`, containing one canonical SDK-validated Chat chunk or Responses stream event;
- `openai.response_snapshot`, containing one complete SDK-validated response.

These are typed IR events, not extension envelopes. Matching-format adapters use them for exact parsed-SDK parity. Cross-format adapters ignore them and compose only semantics they can represent. Provider-specific events and output items have no public passthrough path.

## Usage mapping

Responses `input_tokens`/`output_tokens` and Chat `prompt_tokens`/`completion_tokens` map to `Usage.input_tokens` and `Usage.output_tokens`. Cached input and reasoning output counts are retained when the OpenAI payload provides them. Supported endpoint detail fields, including Chat audio token details, are typed in `Usage`; unknown detail fields are rejected rather than stored as passthrough metadata.

## Verification

Parity tests use official OpenAI SDK parsing and approved cassettes. They compare the complete supported public response or ordered event sequence after the request adapter, IR boundary, provider translation, and output adapter have all run. Local implementation work does not record or replay cassettes; those checks remain owned by the user or CI.
