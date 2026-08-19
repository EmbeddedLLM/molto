# OpenAI Responses and Anthropic Messages Parity Guide

## Overview

OpenAI Responses and Anthropic Messages overlap on conversation text, reasoning, streaming lifecycle, termination, and token usage, but they are not wire-equivalent. Molto preserves shared semantics through its normalized representation and keeps provider-owned values provider-specific.

`endpoint`, `provider`, and `base_url` are independent. The endpoint selects Responses or Messages wire format. The provider identifies the implementation serving that protocol and selects its capability profile. The base URL only selects the HTTP destination; Molto never infers provider behavior from its hostname.

Messages `thinking_delta` does not identify its text as raw reasoning or a summary. Anthropic and OpenAI profiles classify it as a summary; vLLM classifies it as raw reasoning. Responses labels these representations explicitly, so `response.reasoning_text.*` and `response.reasoning_summary_*` remain authoritative regardless of provider profile.

For this guide, parity means:

- preserve source values that have a faithful target representation;
- apply documented target defaults where vocabularies are incompatible;
- generate target-required lifecycle values according to a stable policy;
- reject unsupported request semantics and explicitly coerce termination reasons that the target cannot represent;
- omit provider-owned response state that cannot be used by the target; and
- preserve native values exactly on same-protocol passthrough.

## Current Scope

The translated path currently supports:

- streaming requests;
- string, text-message, and ordered mixed text/image user input (remote URL and base64 data-URL images);
- system prompts;
- assistant text, raw reasoning history, and reasoning-summary history;
- client-defined function tools, tool selection, tool calls, successful results, and streamed argument deltas;
- model, output-token limit, temperature, and `top_p`;
- Anthropic adaptive and manual extended thinking with summarized display;
- response IDs, model names, text and reasoning delta boundaries;
- shared termination reasons and Anthropic stop sequences; and
- semantic input, cache-read, cache-write, output, and thinking-token usage.

Image output, failed tool-result semantics outside Messages, citations, refusals as content blocks, redacted thinking, server tools, structured output, multiple candidates, and non-streaming collection are not implemented across this normalized path. Fields accepted by a request wrapper are not supported cross-protocol unless an adapter and client map them explicitly.

The Messages adapter accepts text, thinking, direct client `tool_use`, and text-only `tool_result` content blocks. Anthropic error results remain representable within Messages, but translation to OpenAI protocols rejects them because those histories have no equivalent error outcome. Other variants are rejected explicitly rather than silently discarded.

Official Anthropic thinking history requires its original signature. Molto's local Messages request schema also accepts a missing signature for compatible providers such as vLLM, whose raw-reasoning fixtures do not provide one. That permissive schema does not establish support for unsigned history against the official Anthropic API; official continuation support requires cassette coverage with Anthropic-issued signatures.

## Formats at a Glance

| Concern               | OpenAI Responses                                    | Anthropic Messages                         |
| --------------------- | --------------------------------------------------- | ------------------------------------------ |
| Conversation input    | `input` and optional `instructions`                 | ordered `messages` and optional `system`   |
| Output                | typed `output` items                                | ordered `content` blocks                   |
| Streaming             | response, item, content, and delta lifecycle events | message and content-block lifecycle events |
| Creation timestamp    | `created_at`                                        | not provided                               |
| Termination           | response status and incomplete details              | `stop_reason` and optional `stop_sequence` |
| Reasoning state       | `encrypted_content`                                 | thinking `signature`                       |
| Service tier response | OpenAI tier vocabulary                              | `standard`, `priority`, or `batch`         |
| Keepalive             | transport framing                                   | optional `ping` event                      |

## Parity Classes

| Class                | Required behavior                                                        |
| -------------------- | ------------------------------------------------------------------------ |
| Source-backed        | Preserve after a documented semantic transformation.                     |
| Target-default       | Emit the target protocol's documented compatibility default.             |
| Translator-generated | Generate required target structure and keep it internally consistent.    |
| Provider-owned       | Carry it across protocols only under an explicit same-provider contract. |
| Unsupported          | Reject requests or outcomes whose meaning would otherwise be lost.       |

## Service Tier Policy

Service-tier names are not portable provider semantics. Anthropic request values (`auto` and `standard_only`) are also different from Anthropic response values (`standard`, `priority`, and `batch`). OpenAI tier labels do not have a faithful Anthropic response equivalent.

Molto therefore applies a target-default policy:

| Route                                    | Result value                    |
| ---------------------------------------- | ------------------------------- |
| Native Responses passthrough             | provider value unchanged        |
| Native Chat Completions passthrough      | provider value unchanged        |
| Native Anthropic Messages passthrough    | provider value unchanged        |
| Any translated Responses result          | `service_tier="default"`        |
| Any translated Chat Completions result   | `service_tier="default"`        |
| Any translated Anthropic Messages result | `usage.service_tier="standard"` |

Upstream clients may retain the source tier in normalized lifecycle metadata. Downstream adapters intentionally ignore it when constructing a translated result. This prevents a valid source label from being presented as a false target equivalent.

Request `service_tier` is not forwarded across protocols. Supporting it would require an explicit behavioral policy, not a string rename.

## Request Translation

### Responses to Messages

| Responses request | Messages request | Rule |
| --- | --- | --- |
| `instructions` | `system` | Preserve prompt text before conversation messages. |
| string or text-message `input` | `messages` | Preserve supported roles and ordering. |
| `input_image` part | `image` block | Decode a base64 data URL into a `base64` source; map a remote URL to a `url` source. Reject a non-default `detail`; Anthropic has no image-fidelity control. |
| `max_output_tokens` | `max_tokens` | Direct scalar rename. |
| `model` | `model` | Preserve exactly. |
| `stream` | `stream` | Preserve streaming mode. |
| reasoning configuration | `thinking` and `output_config.effort` | Use adaptive summarized thinking plus effort on compatible models; use manual extended thinking plus a preset budget on older Claude models. |
| `temperature` | `temperature` | Preserve when present. |
| `top_p` | `top_p` | Preserve when present. |
| function `tools` | client `tools` | Rename `parameters` to `input_schema`; preserve name, description, and strict validation. |
| `tool_choice` | `tool_choice` | Map `required` to `any` and a named function to a named `tool`; preserve `auto` and `none`. |
| `parallel_tool_calls` | `tool_choice.disable_parallel_tool_use` | Invert the boolean. |
| function-call history | assistant `tool_use` plus user `tool_result` blocks | Preserve call IDs, names, JSON arguments, results, and ordering. |

User text is always emitted in the target protocol's explicit content-part form: Responses `input_text` parts or Messages `text` blocks. Each normalized `UserInput` maps to exactly one native user message and each ordered `TextInput` maps to one native text part; neither boundary is merged or split. Ordered `ImageUrlInput` parts map the same way: one part becomes one Responses `input_image` part or one Messages `image` block, preserving position.

Tool-result text uses the same explicit-list policy. A native scalar result is normalized as one ordered text part, and translated requests always emit Responses `input_text` output parts or Messages `text` result blocks.

The matched fixtures assume OpenAI serves both APIs and defines Responses `encrypted_content` and Messages `signature` as two protocol names for the same OpenAI-issued opaque state. Molto maps the value under that contract. This does not make an OpenAI-issued value valid against Anthropic's API; official Anthropic continuation still requires an Anthropic-issued signature.

### Messages to Responses

| Messages request | Responses request | Rule |
| --- | --- | --- |
| `system` | leading system input | Preserve prompt text and order. |
| text `messages` | `input` items | Preserve user and assistant turns. |
| `image` block | `input_image` part | Encode a `base64` source as a data URL; map a remote `url` source to `image_url`. Messages has no image-fidelity field, so no `detail` is set. |
| `max_tokens` | `max_output_tokens` | Direct scalar rename. |
| `model` | `model` | Preserve exactly. |
| `stream` | `stream` | Preserve streaming mode. |
| summarized `thinking` | detailed reasoning | Preserve `output_config.effort`; default adaptive thinking to high effort and reduce a manual budget to the nearest effort preset. |
| `temperature` | `temperature` | Preserve when present. |
| `top_p` | `top_p` | Preserve when present. |
| client `tools` | function `tools` | Rename `input_schema` to `parameters`; preserve name, description, and strict validation. |
| `tool_choice` | `tool_choice` | Map `any` to `required` and a named `tool` to a named function; preserve `auto` and `none`. |
| `tool_choice.disable_parallel_tool_use` | `parallel_tool_calls` | Invert the boolean. |
| `tool_use` and `tool_result` history | function-call and function-call-output items | Preserve call IDs, names, JSON arguments, results, and ordering. |

Anthropic has no obfuscation setting. A translated Responses request uses `stream_options.include_obfuscation=false` to make the target behavior explicit. Omitting the field could select a provider default that is not equivalent to Anthropic's stream.

The Messages wrapper also accepts `cache_control`, `container`, `inference_geo`, `metadata`, `service_tier`, `stop_sequences`, `top_k`, and `user_profile_id`. These fields are not currently forwarded through the normalized path and must not be assumed to work cross-protocol. Within `output_config`, only `effort` is currently normalized.

### Anthropic thinking-mode policy

Reasoning effort is the provider-neutral control. For known Claude model IDs, Molto uses the installed Anthropic model profile to select adaptive versus manual extended thinking. Non-Claude model IDs on Anthropic-compatible endpoints default to adaptive thinking, matching the cross-protocol fixtures.

This outbound model-support policy is separate from `ProviderCapabilities.thinking`, which classifies the inbound meaning of the protocol's ambiguous thinking text.

| Normalized effort | Adaptive `output_config.effort`         | Manual `budget_tokens`        |
| ----------------- | --------------------------------------- | ----------------------------- |
| `none`            | thinking disabled                       | thinking disabled             |
| `minimal`         | `low`                                   | 1,024                         |
| `low`             | `low`                                   | 2,048                         |
| `medium`          | `medium`                                | 10,000                        |
| `high` or omitted | `high`                                  | 16,384 or 10,000 when omitted |
| `xhigh`           | `xhigh` when supported, otherwise `max` | 32,768                        |
| `max`             | `max`                                   | 32,768                        |

Manual `budget_tokens` must remain below `max_tokens`; Molto rejects an invalid translated request before transport. Adaptive requests always include summarized thinking and an explicit effort. Reasoning history enables thinking when the current request omits a reasoning object, using high effort for adaptive models or the default 10,000-token manual budget.

## Result Translation: Messages to Responses

| Responses concept | Messages source | Rule |
| --- | --- | --- |
| Response ID | message `id` | Preserve exactly. |
| Model | message `model` | Preserve exactly. |
| Creation time | unavailable | Capture local wall-clock time when normalization observes `message_start`. |
| Service tier | Messages usage tier | Emit target default `default`. |
| Reasoning | thinking deltas and block | Emit a Responses reasoning summary for `summary` providers or reasoning text for `raw` providers, preserving delta boundaries. |
| Assistant text | text deltas and block | Preserve text and delta boundaries. |
| Usage | start and terminal usage | Map semantically as described below. |
| Outcome | `stop_reason` | Normalize the reason and project exact Responses completed or incomplete states. |
| Function call | `tool_use` block and `input_json_delta` | Preserve the call ID and name, and preserve every partial-JSON delta. |

The generated `created_at` is a valid downstream timestamp, not the creation time of an independently recorded OpenAI response. Responses output-item IDs and sequence numbers are also translator-generated.

Responses `phase` is not reconstructible from a Messages stream and is therefore omitted from translated Responses output items and reconstructed request history. Native Responses passthrough remains unchanged.

Under the OpenAI Messages fixture contract, Messages signature deltas become Responses encrypted reasoning state. A signature issued by Anthropic must not be relabeled as OpenAI `encrypted_content`.

## Result Translation: Responses to Messages

| Messages concept | Responses source | Rule |
| --- | --- | --- |
| Message ID | response `id` | Preserve exactly. |
| Model | response `model` | Preserve exactly. |
| Start usage | unavailable at Responses start | Emit zero counts so streaming remains live. |
| Service tier | Responses tier | Emit target default `standard`. |
| Thinking | reasoning text or summary | Collapse either representation into Messages thinking while preserving delta boundaries. |
| Assistant text | output text | Preserve text and delta boundaries. |
| Terminal usage | completed response usage | Map semantically. |
| `stop_reason` | normalized termination reason | Apply the explicit mapping below. |
| Function call | function-call item and argument deltas | Emit a `tool_use` block with the same call ID, name, and partial JSON. |

Terminal usage is authoritative for translated Messages streams. Backfilling `message_start` with terminal counts would require buffering the whole upstream response.

### Opaque reasoning lifecycle

An encrypted Responses reasoning item carries three opaque snapshots. Messages has two corresponding lifecycle slots:

| Responses snapshot | Messages projection | Policy |
| --- | --- | --- |
| `response.output_item.added.item.encrypted_content` | `content_block_start.content_block.signature` | Preserve the initial snapshot. |
| `response.output_item.done.item.encrypted_content` | `signature_delta.signature` | Preserve the part-complete snapshot as a full replacement. |
| reasoning item in `response.completed.response.output` | no independent slot | Reuse the `done` snapshot in a translated terminal response; exact terminal ciphertext parity is intentionally sacrificed. |

The Anthropic SDK replaces the thinking block's initial `signature` when it processes a `signature_delta`; it does not concatenate the two strings. Molto follows that behavior in its normalized state and uses the replacement value for part completion and subsequent history.

In the current recordings, all three distinct ciphertexts were observed to cover the same number of reasoning tokens. Their byte differences may reflect opaque envelope data such as a version, timestamp, or nonce. This equivalence is empirical and non-contractual: applications must not inspect or synthesize these values. Matthew Green's [“Let's talk about encrypted reasoning”](https://blog.cryptographyengineering.com/2026/05/29/fooling-around-with-encrypted-reasoning-blobs/) provides cryptographic background on treating the blobs as opaque authenticated ciphertext and on the metadata their lengths can reveal; it does not establish equivalence among these three snapshots.

Representing the separately recorded terminal snapshot would require a third normalized lifecycle value or buffering until `response.completed`. Molto instead preserves the early and part-complete snapshots, which keeps streaming incremental. The matched two-turn fixtures use the prior turn's `output_item.done` value as the next turn's Responses `encrypted_content` and Messages `signature` history.

## Termination Mapping

Normalized termination preserves source distinctions before target projection.

| Normalized reason               | Anthropic `stop_reason`         | Chat `finish_reason` | Responses projection                        |
| ------------------------------- | ------------------------------- | -------------------- | ------------------------------------------- |
| `stop`                          | `end_turn`                      | `stop`               | `response.completed`                        |
| `max_output_tokens`             | `max_tokens`                    | `length`             | `response.incomplete` (`max_output_tokens`) |
| `stop_sequence`                 | `stop_sequence`                 | `stop`               | `response.completed` (coerced)              |
| `tool_call`                     | `tool_use`                      | `tool_calls`         | `response.completed` (coerced)              |
| `model_context_window_exceeded` | `model_context_window_exceeded` | `stop` (coerced)     | `response.completed` (coerced)              |
| `pause`                         | `pause_turn`                    | `stop` (coerced)     | `response.completed` (coerced)              |
| `refusal`                       | `refusal`                       | `stop` (coerced)     | `response.completed` (coerced)              |
| `content_filter`                | `end_turn` (coerced)            | `content_filter`     | `response.incomplete` (`content_filter`)    |

Anthropic's matched `stop_sequence` is retained separately. A downstream adapter uses the exact target reason when one exists. Otherwise it coerces the outcome to the target's normal-completion value: Anthropic `end_turn`, Chat `stop`, or `response.completed`. This keeps streams consumable but deliberately loses the source termination distinction.

## Usage Mapping

Anthropic reports uncached input, cache creation, and cache reads separately. Normalized total input includes all three categories.

These mappings are semantic interpretations backed by matched fixtures, not assertions that the provider schemas are structurally identical.

| Normalized usage                          | Anthropic source or target                                             |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| `input_tokens`                            | `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` |
| `input_tokens_details.cached_tokens`      | `cache_read_input_tokens`                                              |
| `input_tokens_details.cache_write_tokens` | `cache_creation_input_tokens`                                          |
| `output_tokens`                           | `output_tokens`                                                        |
| `output_tokens_details.reasoning_tokens`  | `output_tokens_details.thinking_tokens`                                |
| `total_tokens`                            | normalized input plus output                                           |

When projecting to Messages, uncached `input_tokens` is recovered by subtracting cache-read and cache-write counts from normalized total input.

Anthropic cache-creation TTL breakdowns, `inference_geo`, and `server_tool_use` have no current normalized equivalents. They are not synthesized. Messages `stop_details` and `container` are likewise unavailable across the current path.

## Provider-Owned and Transport-Only Fields

- OpenAI `encrypted_content` and Messages thinking `signature` are opaque provider-issued continuation state. The matched fixtures map them because both endpoints are assumed to be served by OpenAI. Messages `signature_delta` replaces its initial signature rather than appending to it. Cross-provider OpenAI-to-Anthropic or Anthropic-to-OpenAI continuation cannot reuse that mapping safely.
- Anthropic `ping` is a transport keepalive, not model output, and is excluded from semantic parity.
- Responses obfuscation has no Anthropic wire field. It is disabled when constructing a translated Responses request.
- Exact transport headers, SSE field names, JSON key order, and whitespace are not semantic parity fields.
- Messages `tool_use.caller` is response metadata with no function-call-history counterpart. Native passthrough retains the source value; translated Messages tool-use blocks emit the target default `caller: {type: direct}`, while reconstructed request history omits `caller`.

## Streaming Invariants

Cross-protocol tests require:

- one normalized response start and one response end;
- stable content indices;
- one target delta for every matched source text, raw-reasoning, reasoning-summary, or function-argument delta;
- concatenated deltas equal completed content;
- content blocks and output items close before the response ends;
- target termination fields agree with the normalized reason;
- terminal usage preserves the supported semantic counts; and
- translated service tier equals the target default.

Same-protocol passthrough validates SDK-native events but does not cross the normalized representation.

## Matched Cassette Policy

Responses is the canonical behavioral reference for matched cross-protocol fixtures. Matching scripts may align source-backed IDs, models, usage, text boundaries, reasoning effort, and termination semantics. A continuation with reasoning history defaults to high effort when its canonical request omits the current reasoning configuration. Matchers also set translated target defaults: `standard` for Messages and `default` for Chat Completions.

When a matcher changes an SSE payload discriminator, it must update the corresponding `event:` header. Matched cassettes must retain valid native wire structure; JSON key order and whitespace should otherwise remain stable unless the modified value requires reserialization.

The matcher maps Responses `output_item.added.encrypted_content` to the Messages thinking-block start signature and `output_item.done.encrypted_content` to its replacement signature delta, under the explicit assumption that OpenAI owns both endpoints and the opaque token. It does not buffer for the separately recorded terminal response-output ciphertext. These fixtures prove same-provider cross-protocol compatibility; they do not prove OpenAI-to-Anthropic continuation compatibility.

The `reasoning_raw` fixtures use `vllm-*` filenames and `provider="vllm"`, selecting the raw profile across Responses, Chat Completions, and Messages. The encrypted-reasoning fixtures use the summary profile. A base URL never selects either behavior.

The three-turn `tool_call` fixtures cover a function call, a successful result followed by another call, and a successful result followed by final text. The matcher maps compatible function definitions and represents `parallel_tool_calls` through the inverse Messages `disable_parallel_tool_use` setting. Direct caller metadata remains present on native Messages response blocks and is the translated response default, but it is not added to reconstructed request history.

Committed cassettes remain native wire-format fixtures. Tests always replay them and never record during normal verification.

Relevant files include:

- `scripts/match_anthropic_cassette_events.py`;
- `scripts/match_openai_cassette_events.py`;
- `python/molto/utils/test.py`;
- `tests/core/test_interface.py`;
- `tests/test_match_anthropic_cassette_events.py`; and
- `tests/test_match_openai_cassette_events.py`.

## Maintenance Checklist

- Preserve native passthrough values unchanged.
- Emit `standard` for every translated Messages response.
- Emit `default` for every translated Responses or Chat Completions result.
- Keep request and response service-tier vocabularies distinct.
- Map encrypted reasoning and Messages signatures only under the documented same-provider OpenAI contract.
- Preserve supported stop and finish reasons and document normal-completion coercion for unsupported target outcomes.
- Treat terminal Messages usage as authoritative across protocols.
- Add cassette-backed parity before claiming support for another content block, request field, or provider-specific usage detail.
