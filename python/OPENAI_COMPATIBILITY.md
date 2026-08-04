# OpenAI compatibility

This document describes the supported OpenAI-compatible boundary. Every route uses `InputIR`, an OpenAI-compatible SDK endpoint, semantic `OutputEvent` values, and `OutputIR` before public composition.

## Route matrix

| Public method and body | Model prefix | `upstream` | Public output | Status |
| --- | --- | --- | --- | --- |
| `chat_completions(OpenAIChatCompletionRequest)` | `openai/` | `chat_completions` | Chat response/chunks | Supported subset |
| `chat_completions(OpenAIChatCompletionRequest)` | `openai/` | `responses` | Chat response/chunks | Supported semantic subset; reasoning output is omitted |
| `responses(OpenAIResponsesRequest)` | `openai/` | `responses` | Responses response/events | Supported subset |
| `responses(OpenAIResponsesRequest)` | `openai/` | `chat_completions` | Responses response/events | Supported semantic subset |
| Either matching public method/body | `vllm/` | Either OpenAI-compatible endpoint | Method-selected OpenAI output | Explicit `base_url`; unverified subset |
| Either matching public method/body | `anthropic/` | `chat_completions` | Method-selected OpenAI output | Conservative text/function subset; unverified |
| Any public method/body | `anthropic/` | `responses` | None | Unsupported |

The public method selects both the accepted OpenAI request schema and returned output protocol. The keyword-only `upstream` parameter selects the execution endpoint and defaults to the matching protocol. The model prefix selects the provider configuration.

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
- ordinary completion, incomplete, failed, refusal, and portable usage data.

When `Molto.chat_completions()` uses `upstream="responses"`, Chat Completions has no reasoning-item or reasoning-event shape. Molto therefore passes representable reasoning configuration through to the provider, retains standard Chat reasoning-token usage, and intentionally omits returned Responses reasoning items and events from the Chat response or stream.

The Chat request adapter removes `stream_options` when the selected upstream is Responses because it is not a Responses SDK argument. The Responses request builder omits neutral `frequency_penalty=0` and `presence_penalty=0` defaults because the Responses SDK does not accept those arguments; non-zero penalties are rejected before provider construction.

## Rejected capabilities

Molto rejects a feature when the OpenAI-shaped IR or selected public output cannot preserve it completely. This includes Responses hosted tools on Chat output, MCP where the selected route cannot preserve it, generated Chat audio when the destination is not Chat Completions, provider-native events, unsupported annotations/logprobs, unknown `extra_body` fields, background execution, and provider-affine state on a different provider.

Stateless replay of MCP, computer-use, file-search, and other hosted output-item types remains unsupported and is rejected rather than partially projected.

Rejection happens at public validation, capability preflight, or output composition. No unsupported provider feature is projected into ordinary text or silently discarded, except for the explicitly documented Responses reasoning omission when the destination is Chat Completions.

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
