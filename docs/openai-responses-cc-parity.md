# OpenAI Responses and Chat Completions Parity Guide

## Overview

OpenAI Responses and Chat Completions overlap substantially, but they are not wire-equivalent formats. A translator can preserve shared request semantics and all returned Chat data that has a defined Responses representation. It cannot recreate values that only an independent Responses request would have generated, such as opaque Responses output-item IDs, assistant-message `phase`, or the exact response completion time.

For this guide, parity means:

- preserve every source value with a defined cross-format mapping;
- retain values from the original request when the target response echoes them;
- apply documented protocol defaults and deterministic translation rules;
- reject, omit, or explicitly extend fields that have no target representation;
- never claim that a generated value reproduces independent server state.

This distinction matters most when a Responses request is sent to an upstream Chat Completions endpoint and its result is translated back into Responses.

`endpoint`, `provider`, and `base_url` are separate route settings. The endpoint selects Responses or Chat Completions wire format. The provider identifies the implementation serving that protocol and selects its capability profile. The base URL only selects the HTTP destination. Molto never infers provider behavior from a URL hostname.

## Document Scope and Current Support

The mapping tables in this guide describe target parity requirements unless a section explicitly says it documents current behavior. They are not a claim that every SDK field is implemented merely because both request wrappers accept it.

The current cross-protocol implementation supports:

- streaming requests only;
- string input and text messages with system, developer, user, and assistant roles;
- raw assistant reasoning and reasoning summaries through Responses reasoning items and the ambiguous Chat `reasoning_content` extension;
- client-defined function tools, tool selection, tool calls, successful results, and streamed argument deltas;
- request translation for `instructions`, model, streaming mode, output-token limits, parallel-tool-call configuration, reasoning effort, temperature, `top_p`, and `include_obfuscation` where the counterpart protocol represents them;
- response IDs, creation times, model names, text and reasoning delta boundaries, lifecycle structure, and semantic token usage; and
- same-protocol paths that validate and yield SDK-native events without crossing the normalized representation.

Refusals, hosted tools, failed tool-result semantics, multiple choices, moderation, annotations, log probabilities, and non-streaming collection are not implemented across the normalized path. The current Chat client reads only `choices[0]`. Normalized streams retain source termination reasons; target projections use exact mappings when available and coerce unrepresentable reasons to normal completion. Responses still maps output-token and content-filter limits exactly to `response.incomplete`. Those limitations are implementation gaps, not evidence that the source values are unavailable in principle.

Fields accepted by the SDK-compatible request wrappers but absent from the supported list above may currently be retained only as extra data or omitted by an upstream request constructor. Treat them as unsupported until they have an explicit mapping or rejection and cassette-backed parity coverage.

## Formats at a Glance

| Concern             | Responses                                           | Chat Completions                                              |
| ------------------- | --------------------------------------------------- | ------------------------------------------------------------- |
| Conversation input  | `input` plus optional `instructions`                | ordered `messages`                                            |
| Final output        | typed `output` items                                | indexed `choices` containing messages                         |
| Streaming           | typed lifecycle, item, content, and delta events    | incremental choice chunks followed by `[DONE]`                |
| Output identity     | response ID plus separate output-item IDs           | completion ID plus tool-call IDs                              |
| Completion time     | `completed_at` on the response                      | no completion timestamp                                       |
| Multiple candidates | one response output sequence                        | `n` indexed choices                                           |
| Tools               | functions, custom tools, and Responses hosted tools | functions, custom tools, and Chat-specific web-search options |

Responses is the richer lifecycle protocol. Translating Chat Completions into Responses therefore requires synthesizing an event envelope around the source chunks. Translating Responses into Chat Completions may require rejecting features that Chat cannot represent.

## Parity Classes

Every field should be assigned to one of these classes before implementing or testing a mapping.

| Class | Source | Required behavior |
| --- | --- | --- |
| Source-backed | Upstream request or result | Preserve exactly after the documented schema transformation. |
| Request-backed | Original Responses request retained by the translator | Echo or transform exactly where the Responses result contract requires it. |
| Default-backed | Unambiguous target-protocol default | Emit the documented default consistently. |
| Translator-generated | Required target structure with no source value | Generate according to a documented policy and validate internal consistency. |
| Unavailable | Independent server state or a source field with no target representation | Reject, omit, or expose through a documented extension. Do not compare it as reconstructable. |

A field is not unavailable merely because the current translator has not mapped it yet. Chat Completions exposes cache-write token usage and output token log probabilities, but their reconstructability depends on direction and lifecycle. The current implementation maps cache-write usage and does not map log probabilities.

Service tier is deliberately target-default rather than source-backed. OpenAI's request and response tier labels do not form a stable cross-provider contract, and even Responses and Chat Completions can expose them at different lifecycle points. Every translated Responses response and every translated Chat Completions chunk therefore reports `service_tier="default"`; the source tier may remain in normalized metadata for diagnostics but is not relabeled as an equivalent target tier. Native same-protocol passthrough remains untouched.

## Request Translation: Responses to Chat Completions

### Target directly compatible fields

The current OpenAI SDK request surfaces share these top-level concepts, but most are not yet forwarded by Molto's normalized path:

- `metadata`;
- `model`;
- `moderation`;
- `parallel_tool_calls`;
- `prompt_cache_key`, `prompt_cache_options`, and `prompt_cache_retention`;
- `safety_identifier`;
- `service_tier` (accepted by both wrappers but not forwarded by the translated path);
- `store`;
- `stream`;
- `temperature`, `top_p`, and `top_logprobs`;
- `tool_choice` and compatible `tools`; and
- `user`.

Shared names do not guarantee identical nested schemas. Validate each nested union before copying it. Responses supports hosted tools that have no general Chat Completions equivalent. Chat also requires `logprobs: true` when `top_logprobs` is set; copying `top_logprobs` alone is insufficient.

Client-defined function tools map Responses `name`, `description`, `parameters`, and `strict` to Chat's nested `function` object. `auto`, `none`, and `required` tool choices map directly; a named Responses function choice maps to Chat's named `function` choice. Hosted, custom-text, and other protocol-specific tools remain unsupported.

Conversation history maps Responses `function_call` items to assistant `tool_calls` and `function_call_output` items to `tool` messages. The portable identifier is Responses `call_id` / Chat `tool_calls[].id`; Responses' enclosing output-item `id` has no Chat equivalent and is omitted when history is reconstructed.

### Fields that require transformation

| Responses request                   | Chat Completions request            | Notes                                                   |
| ----------------------------------- | ----------------------------------- | ------------------------------------------------------- |
| `instructions`                      | leading developer or system message | Preserve ordering relative to the input conversation.   |
| string or message `input`           | `messages`                          | Emit user text as explicit `text` content parts.        |
| `max_output_tokens`                 | `max_completion_tokens`             | Direct scalar rename.                                   |
| `reasoning.effort`                  | `reasoning_effort`                  | Preserve only values accepted by both schemas.          |
| `text.format`                       | `response_format`                   | Transform nested text or JSON schema configuration.     |
| `text.verbosity`                    | `verbosity`                         | Direct semantic mapping when supported by the model.    |
| compatible web-search tool settings | `web_search_options`                | Only for the subset represented by both schemas.        |
| Responses stream options            | Chat stream options                 | Map compatible members; Chat also uses `include_usage`. |

Conversation references, previous-response state, hosted tools, background execution, and other Responses-only request features require a separate emulation strategy or an explicit rejection. Silently dropping them changes request meaning.

User text is projected as an explicit content-part list in both protocols (`input_text` for Responses and `text` for Chat Completions), even when the normalized input originated as a single string. Each normalized `UserInput` maps to exactly one native user message and each ordered `TextInput` maps to one native text part; neither boundary is merged or split. Assistant Responses history is reconstructed as a completed typed message with `output_text` content.

Responses `phase` is not reconstructible from Chat output or the normalized text/reasoning sequence. Translated Responses events and reconstructed request history therefore omit it. Native Responses passthrough retains any phase supplied by the provider.

## Result Translation: Chat Completions to Responses

The implemented result path covers response metadata, text, raw reasoning, reasoning summaries, function calls, lifecycle events, and usage. The mappings below are target requirements for the broader result surface. In particular, refusals, hosted tools, log probabilities, moderation, annotations, and multiple choices remain unsupported.

### Reasoning representation

Responses distinguishes raw reasoning (`response.reasoning_text.*`) from summaries (`response.reasoning_summary_*`). Chat Completions `reasoning_content` does not. The upstream Chat client therefore classifies that field with `ProviderCapabilities.thinking`:

| Source                      | Provider capability | Normalized representation            | Responses projection           |
| --------------------------- | ------------------- | ------------------------------------ | ------------------------------ |
| Chat `reasoning_content`    | `raw`               | `Thinking.content` / `content_delta` | `response.reasoning_text.*`    |
| Chat `reasoning_content`    | `summary`           | `Thinking.summary` / `summary_delta` | `response.reasoning_summary_*` |
| Responses reasoning text    | any                 | `Thinking.content` / `content_delta` | Chat `reasoning_content`       |
| Responses reasoning summary | any                 | `Thinking.summary` / `summary_delta` | Chat `reasoning_content`       |

OpenAI uses `summary`; vLLM uses `raw`. Explicit Responses event types remain authoritative and ignore this capability. Projecting to Chat collapses either normalized form into its one reasoning field.

### Direct result mappings

| Responses concept | Chat source | Rule |
| --- | --- | --- |
| Response ID | `id` | Preserve as the translated response ID. It will not equal an independently generated Responses ID unless fixtures deliberately synchronize it. |
| Creation time | `created` | Map to `created_at`. |
| Model | `model` | Preserve exactly. |
| Service tier | `service_tier` | Emit the Responses target default `default`; do not relabel the source tier. |
| Assistant text | `choices[].message.content` or `choices[].delta.content` | Preserve text and stream boundaries when the source chunks are part of the parity contract. |
| Assistant phase | No standard Chat result field | Treat as unavailable. Do not infer `commentary` or `final_answer` from reasoning presence, message order, or text content. |
| Refusal text | message or delta `refusal` | Map to the corresponding Responses refusal content or events. |
| Completion outcome | `finish_reason` | Apply a documented status and incomplete/error mapping. |
| Function name and arguments | function or tool-call fields | Preserve names and argument bytes. |
| Function call ID | tool-call `id` | Map to Responses `call_id`; do not reuse it as the Responses output-item ID. |
| Token usage | `usage` | Apply the usage mapping below. |
| Token log probabilities | choice `logprobs` | Preserve compatible output-token logprob records when requested. |
| Moderation results | `moderation` | Preserve every result with a Responses counterpart. |
| Compatible annotations | final message `annotations` | Preserve citations or annotations only when the target Responses content type represents them. |

Choice indices and multiple-choice structure need an explicit policy. A translator may reject `n > 1`, select one choice, emit a documented extension, or define a stable mapping into multiple Responses output items. It must not silently merge choices.

The current implementation selects `choices[0]` without rejecting additional choices. The Chat client normalizes supported finish reasons and ends the active part. The Responses adapter emits `response.incomplete` with the corresponding `incomplete_details.reason` for `max_output_tokens` and `content_filter`, the two incomplete reasons represented by the pinned Responses schema. It emits `response.completed` for normal completion and for other normalized outcomes that Responses cannot represent.

It captures `service_tier` from the first Chat chunk as upstream metadata. The Responses adapter intentionally ignores that value and emits the target default `default`.

### Usage mapping

The pinned SDK's `CompletionUsage.prompt_tokens_details` includes `cached_tokens`, `cache_write_tokens`, and `audio_tokens`. These fields are typed source data, not Responses-only defaults.

| Responses usage                           | Chat usage                                   |
| ----------------------------------------- | -------------------------------------------- |
| `input_tokens`                            | `prompt_tokens`                              |
| `output_tokens`                           | `completion_tokens`                          |
| `total_tokens`                            | `total_tokens`                               |
| `input_tokens_details.cached_tokens`      | `prompt_tokens_details.cached_tokens`        |
| `input_tokens_details.cache_write_tokens` | `prompt_tokens_details.cache_write_tokens`   |
| `output_tokens_details.reasoning_tokens`  | `completion_tokens_details.reasoning_tokens` |

Preserve these values exactly when present. Define an explicit default only when the Responses contract makes the omitted Chat value unambiguous. The current implementation defaults omitted cached, cache-write, and reasoning token detail counts to zero. Chat audio and prediction-token details need a documented Responses mapping or extension before they can participate in exact parity.

### Log probabilities

Both non-streaming Chat choices and streaming Chat chunks expose `logprobs`. Responses text delta, text done, and output content structures can carry compatible logprob data. When log probabilities are requested, translate and compare their token text, token bytes, probability values, and top alternatives according to the target schema.

The nested schemas are not identical in every case. Chat token bytes may be optional where the corresponding Responses record requires them, and Chat refusal logprobs do not automatically belong in Responses output-text logprobs. Document a conversion, omission, or rejection for those cases.

When Chat returns no logprobs, a translator may emit the Responses protocol default such as an empty list where required. Tests should compare that value; they should not drop the `logprobs` field merely because a particular fixture did not request probabilities.

The current normalized clients and adapters do not carry log probabilities. Existing translated text events therefore contain target defaults such as empty Responses logprob lists or null Chat logprobs, not source logprob parity.

## Streaming Translation

A Chat stream contains choice deltas, finish reasons, optional usage, and a terminal `[DONE]` marker. A Responses stream additionally requires lifecycle and structural events. A simple text translation commonly emits:

1. `response.created`;
2. `response.in_progress`;
3. `response.output_item.added`;
4. `response.content_part.added`;
5. one `response.output_text.delta` per mapped Chat content delta;
6. `response.output_text.done`;
7. `response.content_part.done`;
8. `response.output_item.done`; and
9. `response.completed`, `response.incomplete`, or `response.failed`.

The exact sequence depends on content and tool types, but the translation policy must keep these invariants:

- sequence numbers are monotonic and follow the emitted event order;
- output and content indices are stable;
- generated item IDs are non-empty and referenced consistently;
- concatenated deltas equal the final content;
- item and response status transitions are valid;
- the terminal event agrees with the mapped finish reason;
- final usage retains source values; target service tier follows the documented default policy; and moderation and logprobs retain source values when their mappings are implemented and available at the corresponding lifecycle point.

Chat stream options expose both `include_usage` and `include_obfuscation`. The generated `ChatCompletionChunk` model may not declare every wire extension as a named field, so a translator that promises obfuscation parity must preserve the raw chunk data rather than discard unknown-but-supported SSE fields.

## Result Translation: Responses to Chat Completions

The current reverse projection maps the Responses ID, creation time, model, text deltas, raw reasoning or summary deltas, function calls, semantic termination reason, and token usage into SDK-native Chat chunks. It emits one initial assistant-role chunk and preserves every matched text, reasoning, and function-argument delta boundary.

Every projected Chat chunk reports the target default `service_tier="default"`, regardless of the source Responses tier. Responses has no Chat `system_fingerprint`, so that field does not participate in exact cross-protocol comparison.

Refusals, hosted tools, log probabilities, moderation, annotations, and multiple choices are not implemented in this direction. Terminal reasons without an explicit Chat mapping are coerced to `finish_reason="stop"`.

## Values That Must Be Generated

Chat Completions does not provide the following Responses structures or values:

- Responses lifecycle events and sequence numbers;
- Responses output-item IDs;
- item and content-part lifecycle events and statuses;
- an exact response completion timestamp;
- empty target-only containers required by the Responses schema; and
- target envelope defaults not present in the Chat result.

These values can be valid and deterministic without being historical server values. A translation policy should document how IDs, timestamps, sequence numbers, statuses, and defaults are produced.

Two values deserve special care:

- A Chat tool-call ID maps to Responses `call_id`. The enclosing Responses function-call item still needs its own generated output-item ID.
- Chat `created` maps to Responses `created_at`; it is not the Responses `completed_at` value.

## Values That Cannot Be Reconstructed

An independently recorded Responses result may contain data unavailable from the Chat request and result:

- opaque Responses output-item IDs;
- Responses assistant-message `phase`;
- the exact Responses completion timestamp;
- other Responses-only opaque IDs or server timestamps;
- server-resolved conversation or previous-response state;
- prompt-cache decisions not represented in Chat usage;
- typed Responses reasoning structure or encrypted reasoning state absent from Chat;
- hosted-tool output and per-tool usage absent from Chat;
- Responses annotations or citations absent from the Chat message; and
- Responses-specific error and incomplete details beyond a defined finish-reason mapping.

Do not hardcode these values from an expected Responses cassette. For comparison against an independent recording, remove them from both sides and compare the remaining reconstructable projection. Removing `phase` is a comparison rule only: native Responses paths must continue to preserve provider-supplied phase values.

## Known Source Data Without a Standard Target Field

Some Chat values are known but do not automatically belong in a standard Responses payload:

- Chat envelope `object` values;
- choice indices and multi-choice structure;
- `system_fingerprint`;
- generated audio and audio usage details; and
- accepted/rejected prediction token details.

Choose one behavior for each field: reject the source shape, intentionally omit the field, or define a documented extension. Absence of a target field is different from inability to read the source value.

The reverse direction has larger gaps. Responses hosted tools, conversation state, background execution, reasoning items, and some typed content parts do not have a general Chat Completions representation.

## Wire Representation

Semantic parity does not imply byte-for-byte SSE parity. Parsing and serializing can change:

- JSON key order;
- omitted fields versus explicit `null`;
- integer and floating-point spelling;
- whitespace;
- SSE field order and framing; and
- provider chunk boundaries.

Same-protocol paths preserve the native protocol and parsed SDK event semantics, but Molto does not promise the original SSE bytes: records are parsed into SDK models before being yielded. Chat-to-Responses translation cannot equal an unmodified independent Responses recording because some IDs, timestamps, and server state never crossed the Chat boundary.

Exact translated SSE is possible only under a canonical fixture contract:

1. synchronize every source-backed value and chunk boundary;
2. retain all request-backed values;
3. define deterministic generated fields;
4. prepare the expected fixture using those policies; and
5. serialize with one canonical SSE serializer.

This proves conformance to the translator's canonical wire format, not recovery of the bytes produced by an independent Responses request.

Request wrappers that serialize with `exclude_none=True` also lose the distinction between omitted fields and explicit `null`. If that distinction matters, retain the original field set or use a serialization policy that preserves it.

## Current Testing Strategy

For same-protocol routes, tests assert the exact request path and body, validate every SDK event, assert exact event count and order, and require each actual event to contain the expected recorded fields. This is event-level semantic comparison, not raw SSE byte equality.

For either translated route, tests validate the downstream request and every parsed SDK event, canonicalize only differences the opposite protocol cannot reproduce, and compare the resulting event lists exactly.

For Responses downstream with Chat Completions upstream:

1. Assert the translated Chat request path and body exactly.
2. Require the translated and expected Responses streams to have the same event count.
3. Validate contiguous sequence numbers and unique generated output-item IDs.
4. Use `reconstructable_responses_event` to remove generated item IDs and Responses-only assistant phases, then retain the fixed reconstructable response envelope plus response fields explicitly present in the downstream request.
5. Compare the projected event lists exactly, preserving every matched text and reasoning delta boundary.

For Chat Completions downstream with Responses upstream:

1. Assert the translated Responses request path and body exactly.
2. Use `reconstructable_chat_completions_chunks` to retain `id`, `object`, `created`, `model`, target-default service tier, choices, deltas, finish reasons, and semantic usage.
3. Omit `system_fingerprint`, which Responses cannot reproduce.
4. Normalize cached, cache-write, and reasoning token details while dropping source-only audio, prediction, and cache-alias fields.
5. Treat a combined finish-and-usage chunk as equivalent to separate finish and usage chunks, then flatten the canonical chunks.
6. Compare the canonical chunk lists exactly. All matched text, reasoning, and function-argument delta boundaries remain observable and must match.

Matched source cassettes must carry all source fields under test. Canonicalization is narrow: it handles known protocol differences but must not aggregate deltas or hide missing mappings.

Repository fixtures and helpers relevant to this contract are:

- `tests/cassettes/matched/simple_text/`;
- `tests/cassettes/matched/reasoning_raw/`, whose `vllm-*` fixtures run with `provider="vllm"`;
- `tests/cassettes/matched/tool_call/`, whose three turns cover calls, results, and continuation history;
- `scripts/match_openai_cassette_events.py`;
- `python/molto/utils/test.py`;
- `tests/core/test_interface.py`; and
- `tests/utils/test_test_utils.py`.

Never record cassettes during normal test runs. Replay approved fixtures only.

## SDK and Schema Maintenance

The exact OpenAI SDK pin in `pyproject.toml` is the source of truth for typed field availability. The request wrappers in `python/molto/types/openai_chat_completions.py` and `python/molto/types/openai_responses.py`, the translation code, and the cassettes must target the same SDK and wire schema.

When the SDK changes:

1. inspect added, removed, and renamed request and result fields;
2. update the reconstructability classification;
3. update request and result mappings;
4. regenerate matched fixtures from approved recordings; and
5. run cassette-backed parity tests.

A permissive event fallback may tolerate a newer raw SSE event, but it cannot make an incompatible request schema or missing result mapping correct.

Do not use a base URL hostname to choose raw-versus-summary behavior. Add or update the explicit provider capability profile and its cassette-backed coverage instead.

## Summary Checklist

- Preserve source-backed fields exactly.
- Retain original request values needed by the target response.
- Emit and compare the target-default `service_tier="default"` on every translated Responses or Chat Completions result; preserve native values only on same-protocol passthrough.
- Treat cache-write usage and logprobs as source data, but compare them only where the target lifecycle can reproduce them; the current implementation maps only the supported subset described above.
- Generate Responses lifecycle structure according to a documented policy.
- Never equate Chat tool-call IDs with Responses output-item IDs.
- Never infer a Responses assistant phase from Chat reasoning or text; omit it only from cross-protocol reconstructed-event comparison.
- Never treat `created` as the exact Responses completion timestamp.
- Reject or document fields with no target representation.
- Use validated event-by-event comparison for same-protocol paths, explicit reconstructable projections for independent cross-format recordings, and canonical byte comparison only for canonical translation fixtures.
