# OpenAI compatibility

This document describes the supported OpenAI-compatible boundary. Every route
uses `InputIR`, an OpenAI-compatible SDK endpoint, semantic `OutputEvent` values,
and `OutputIR` before public composition.

## Route matrix

| Request | Model prefix | Execution endpoint | Public output | Status |
| --- | --- | --- | --- | --- |
| OpenAI Chat Completions | `openai/` | Chat Completions | Chat response/chunks | Supported portable subset |
| OpenAI Responses | `openai/` | Responses | Responses response/events | Supported portable subset |
| OpenAI Chat Completions | `openai/` | Chat Completions | Responses response/events | Supported semantic subset |
| OpenAI Responses | `openai/` | Responses | Chat response/chunks | Supported semantic subset |
| OpenAI Chat Completions | `vllm/` | OpenAI-compatible Chat | Chat response/chunks | Explicit `base_url`; subset only |
| OpenAI Responses | `vllm/` | OpenAI-compatible Responses | Responses response/events | Explicit `base_url`; subset only |
| OpenResponses | `openai/` or `vllm/` | OpenAI-compatible endpoint | OpenResponses response/events | Portable subset |
| OpenResponses | `anthropic/` | Anthropic OpenAI-compatible Chat | OpenResponses response/events | Text/function subset |
| Any request | `anthropic/` | Anthropic-native Messages | Any output | Unsupported |

The request model selects the native execution endpoint. The public method
selects the returned output format.

## Supported portable capabilities

- text history and representable image input;
- function definitions, function-call history, function results, and supported
  function choices;
- text, JSON object, and JSON Schema output formats;
- common sampling, token-limit, reasoning-effort, and parallel-call controls;
- generated Chat audio and previous assistant-audio references on the Chat
  Completions route;
- standard OpenAI Responses hosted tools, hosted-tool choices, output items,
  citations, and include selections on the matching Responses route;
- stateless Responses continuation for assistant output, reasoning,
  web-search-call, code-interpreter-call, and image-generation-call history,
  including `store=False` and image-generation ID-only references;
- reasoning items when the selected output contract supports them;
- ordinary completion, incomplete, failed, refusal, and portable usage data.

## Rejected capabilities

Molto rejects a feature when the OpenAI-shaped IR or selected public output
cannot preserve it completely. This includes Responses hosted tools on Chat or
OpenResponses output, MCP where the selected route cannot preserve it,
generated Chat audio when the destination is not Chat Completions,
provider-native events, unsupported annotations/logprobs, unknown `extra_body`
fields, background execution, and provider-affine state on a different
provider.

Stateless replay of MCP, computer-use, file-search, and other hosted output-item
types remains unsupported and is rejected rather than partially projected.

Rejection happens at public validation, capability preflight, or output
composition. No unsupported provider feature is projected into ordinary text or
silently discarded.

## Event contract

The internal event vocabulary contains semantic lifecycle events:

- `response.started`;
- output item added/done;
- content part added/delta/done;
- function-call argument delta/done;
- `response.completed`, `response.incomplete`, and `response.failed`.

It also contains two OpenAI-compatible fidelity events:

- `openai.stream_frame`, containing one canonical SDK-validated Chat chunk or
  Responses stream event;
- `openai.response_snapshot`, containing one complete SDK-validated response.

These are typed IR events, not extension envelopes. Matching-format adapters use
them for exact parsed-SDK parity. Cross-format and OpenResponses adapters ignore
them and compose only semantics they can represent. Provider-specific events
and output items have no public passthrough path.

## Usage mapping

Responses `input_tokens`/`output_tokens` and Chat
`prompt_tokens`/`completion_tokens` map to `Usage.input_tokens` and
`Usage.output_tokens`. Cached input and reasoning output counts are retained when
the OpenAI payload provides them. Supported endpoint detail fields, including
Chat audio token details, are typed in `Usage`; unknown detail fields are
rejected rather than stored as passthrough metadata.

## Verification

Parity tests use official OpenAI SDK parsing and approved cassettes. They compare
the complete supported public response or ordered event sequence after the
request adapter, IR boundary, provider translation, and output adapter have all
run. Local implementation work does not record or replay cassettes; those checks
remain owned by the user or CI.
