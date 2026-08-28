# OpenAI Responses

Use `Molto.responses()` when your application speaks the OpenAI Responses protocol. Molto always returns OpenAI SDK Responses events from this method, whether the provider exposes Responses, Chat Completions, or Anthropic Messages.

This guide describes the Responses wire contract and both directions of translation. For routing, shared parity rules, and provider capability policy, start with the [documentation overview](../README.md).

## At a glance

| Setting                   | Value                                   |
| ------------------------- | --------------------------------------- |
| Application method        | `Molto.responses()`                     |
| Native upstream endpoint  | `endpoint="responses"`                  |
| Default upstream provider | `openai`                                |
| Validated request         | `OpenAIResponsesCreateRequest`          |
| Stream items returned     | OpenAI SDK `ResponseStreamEvent` models |
| Implemented mode          | Streaming only; `stream` must be `true` |
| Other upstream endpoints  | `chat/completions`, `messages`          |

`endpoint` selects the upstream wire protocol, `provider` selects the serving implementation and capability profile, and `base_url` selects only the HTTP destination. Molto never infers provider behavior from a URL.

## When to use Responses

Use Responses as the application protocol when you want typed output items, explicit item and content lifecycles, separate raw-reasoning and reasoning-summary events, or function calls represented as first-class items. It is also Molto's canonical behavioral reference for cross-protocol parity.

Use same-protocol passthrough when the provider already exposes Responses and you need native features beyond the translated subset. Use a translated route when your application must retain the Responses interface but the provider offers only Chat Completions or Messages.

## Quickstart

### Native Responses passthrough

```python
import asyncio

from molto import Molto


async def main() -> None:
    async with Molto() as molto:
        async with await molto.responses(
            body={
                "model": "gpt-5",
                "input": "Explain why the sky is blue in one paragraph.",
                "stream": True,
            },
            endpoint="responses",
            provider="openai",
        ) as stream:
            async for event in stream:
                print(event)


asyncio.run(main())
```

The matching method and endpoint select native passthrough. The request is validated, but neither it nor the returned SDK events cross Molto's normalized representation.

### Responses over another upstream protocol

Change only the route settings. This example sends a Responses request to an Anthropic Messages endpoint and still yields OpenAI Responses events:

```python
async with Molto() as molto:
    async with await molto.responses(
        body={
            "model": "claude-sonnet-5",
            "input": "Hello!",
            "max_output_tokens": 1024,
            "stream": True,
        },
        endpoint="messages",
        provider="anthropic",
    ) as stream:
        async for event in stream:
            print(event)
```

Use `endpoint="chat/completions"` for a Chat Completions upstream. Context-manage both `Molto` and the returned stream so early termination releases the HTTP response. If context managers are impractical, call `await stream.aclose()` and then `await molto.aclose()` in `finally` blocks.

## Native wire contract

### Requests and conversation history

Responses accepts a string `input` or an ordered list of heterogeneous input items. A string is shorthand for one user input. The list can contain message items, reasoning items, function calls, function-call outputs, and other types defined by the pinned OpenAI SDK. Message items can contain ordered content parts.

The pinned SDK accepts `user`, `assistant`, `system`, and `developer` message roles. It does not impose role alternation or require either instruction role to appear first. Schema validity does not imply that every model renderer gives those roles identical precedence.

Top-level `instructions` is separate from `input[]`; it is not a message boundary. Native passthrough preserves this distinction. On translated routes, Molto normalizes it as a leading instruction alongside system and developer messages, so their original source and role identity are not recoverable.

Responses request validation follows the pinned OpenAI SDK schema, with narrow compatibility models for provider history that omits reasoning-item or assistant message IDs and for input images whose default `detail` is omitted. Accepting a field at this boundary does not mean that the normalized path can translate it.

### Output and streaming lifecycle

A Responses stream exposes several nested lifecycles:

1. The response starts with `response.created` and may report `response.in_progress`.
2. Each typed output item has `response.output_item.added` and `response.output_item.done` events at an `output_index`.
3. Text and reasoning inside an item have content or summary lifecycle events and deltas at a `content_index` or `summary_index`.
4. A terminal `response.completed`, `response.incomplete`, or `response.failed` closes the response.

These streaming boundaries describe generated output; they are not request message or model-turn boundaries. Responses output items may also be converted into different model-visible turns by the provider's renderer.

Native Responses can distinguish multiple inner content parts within one output item. Molto's normalized stream represents the output-item index, but not the inner content index. A translated route therefore collapses that distinction and reconstructs a single inner part at index `0`. Compatibility coverage also assumes that output-item lifecycles are not interleaved; interleaved items are not a supported translated-route guarantee.

### Assistant phase

An assistant output message may carry `phase="commentary"` or `phase="final_answer"`. This is source-backed Responses metadata. Native passthrough preserves it. Chat Completions and Messages do not expose an equivalent value, so translated Responses results and reconstructed Responses history omit it. Molto never infers phase from reasoning presence, output order, or text.

## Native passthrough

`Molto.responses(endpoint="responses")` performs these operations:

- validates the body as `OpenAIResponsesCreateRequest`;
- rejects a request unless `stream is True`;
- delegates request serialization, authentication, HTTP, retries, SSE parsing, and OpenAI exception construction to `AsyncOpenAI.responses.create()`; and
- yields the SDK's `ResponseStreamEvent` models directly.

The stream does not cross the normalized representation and retains native event variants, fields, item IDs, phases, lifecycle shape, and service tier. Passthrough is validated SDK transport, not an untyped HTTP proxy. A recognized error is re-projected as a fresh instance of its original SDK class with the normalized cause chain described under [Errors](#errors).

There is one narrow compatible-provider normalization: if a completed Responses event includes usage but omits `input_tokens_details.cache_write_tokens` as `null`, Molto sets it to `0` to satisfy the pinned SDK contract. Otherwise it does not revalidate or rewrite each event.

## When the application speaks Responses

When `endpoint` is `chat/completions` or `messages`, the Responses adapter first normalizes the request. The selected upstream client then emits a normalized event stream, and the same adapter projects that stream into SDK-native Responses events.

```text
Responses request
      |
      v
Normalized request -> selected upstream protocol
      |
      v
Normalized event stream
      |
      v
Responses lifecycle and SDK events
```

### Downstream request normalization

The implemented normalized path accepts:

- string input and text messages with `system`, `developer`, `user`, and `assistant` roles;
- ordered mixed text and image user input, using remote URLs or base64 data URLs;
- raw reasoning and reasoning-summary history;
- client-defined function tools, tool choice, parallel-call control, calls, successful results, and streamed JSON argument deltas; and
- model, streaming mode, output-token limit, reasoning effort, temperature, `top_p`, and `include_obfuscation` when the upstream protocol has an equivalent stream option.

One user message remains one user message, and its ordered text and image parts remain ordered. Consecutive user messages are not merged by Molto, although an upstream provider or model renderer may combine them into one model-visible turn.

System and developer messages both become role-less normalized instructions. A multipart instruction message becomes multiple instruction values. Assistant message containers are flattened: each assistant text part becomes a separate text output, and multiple reasoning parts of the same representation are joined within a normalized reasoning value. These transformations preserve supported content but lose the original instruction role, multipart instruction boundary, and enclosing assistant-message boundary.

Unsupported item types, content parts, non-successful tool outcomes, and populated request-wrapper fields outside the implemented normalization surface are rejected on translated routes. Nested request settings are validated independently; accepting their top-level container does not authorize unsupported members. Optional Responses output-item IDs are the sole exception: they remain valid normalized history, Responses preserves them, and Chat Completions or Messages may omit them because they do not affect model behavior. Messages has no obfuscation field, so a Responses obfuscation request is not portable to that endpoint.

### Downstream event projection

For a simple text result, Molto synthesizes this Responses sequence:

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta...
response.output_text.done
response.content_part.done
response.output_item.done
response.completed or response.incomplete
```

One normalized event can produce several Responses events. Molto owns the sequence numbers, item and content lifecycle, output indices, statuses, and any required item IDs. It preserves source-backed response IDs, models, text, reasoning, function-argument delta boundaries, termination semantics, and usage.

The source of metadata depends on the upstream endpoint:

| Responses value       | Chat Completions source                      | Messages source                              |
| --------------------- | -------------------------------------------- | -------------------------------------------- |
| Response ID           | `id`, preserved                              | message `id`, preserved                      |
| `created_at`          | Chat `created`, preserved                    | Local time when `message_start` is observed  |
| Model                 | `model`, preserved                           | message `model`, preserved                   |
| Output-item ID        | Generated unless a normalized item ID exists | Generated unless a normalized item ID exists |
| Sequence number       | Generated monotonically from `0`             | Generated monotonically from `0`             |
| Inner content index   | Reconstructed as `0`                         | Reconstructed as `0`                         |
| `phase`               | Omitted                                      | Omitted                                      |
| Service tier          | Target default `default`                     | Target default `default`                     |
| Exact completion time | Unavailable and omitted                      | Unavailable and omitted                      |

Generated values are valid target structure, not recovered server state. In particular, a Chat tool-call ID maps to Responses `call_id`; it is never reused as the enclosing function-call item's ID. That output-item ID is independent and may need to be generated.

## When the provider exposes Responses

When another Molto method selects `endpoint="responses"`, the Responses client projects a normalized request to the native Responses schema and normalizes the provider's SDK event stream for the downstream adapter.

### Upstream request projection

The Responses client maps normalized input as follows:

| Normalized value    | Responses request                                                    |
| ------------------- | -------------------------------------------------------------------- |
| Instruction         | `developer` message at the same normalized sequence position         |
| User input          | One `user` message with ordered `input_text` and `input_image` parts |
| Raw reasoning       | Completed `reasoning` item with `reasoning_text` content             |
| Reasoning summary   | Completed `reasoning` item with summary and encrypted state          |
| Assistant text      | Completed assistant `message` with one `output_text` part            |
| Function call       | Completed `function_call` item                                       |
| Function result     | `function_call_output` with ordered text and image parts             |
| Output-token limit  | `max_output_tokens`                                                  |
| Function definition | Responses function tool                                              |
| Tool selection      | Responses `auto`, `none`, `required`, or named function choice       |

Every normalized instruction is emitted as `developer`; an original `system` role cannot be reconstructed. Each flat assistant text output becomes a separate assistant message, so one source assistant message with several parts may split into several Responses messages. Assistant `phase` remains omitted.

Responses supports image fidelity controls, including `original`, but the source protocol may not. Images from Messages therefore have no `detail` value. Images from Chat Completions preserve supported detail values; a value that the target counterpart cannot represent is rejected at that counterpart's boundary instead of being silently discarded.

Only successful tool results are portable to OpenAI protocols. Function `call_id`, name, JSON arguments, results, and ordering are preserved. A normalized function call may also retain its Responses output-item `id`, but the portable identity used to pair a result with a call is `call_id`.

If a normalized request came from Messages, Molto sets `stream_options.include_obfuscation=false` because Messages has no obfuscation setting. More generally, only explicitly normalized request fields are emitted. Responses-only features such as conversation references, previous-response state, hosted tools, and background execution require an explicit mapping or rejection before they can be translated from another protocol.

### Upstream event normalization

The Responses client emits one normalized response start, one part lifecycle per supported output item, each source delta as a matching normalized delta, and one response end. Empty role or lifecycle echoes do not become content deltas.

Supported output items are assistant text, reasoning, and client function calls. A completely empty reasoning `output_item.added` snapshot is deferred until text or encrypted state establishes a real part. If it also completes empty, Molto omits that empty part and reindexes surviving normalized parts densely. An item that starts with content or encrypted state is emitted immediately.

The client recognizes completed and incomplete response terminals. A completed response containing a function call normalizes as `tool_call`; otherwise it normalizes as `stop`. An incomplete response must provide an incomplete reason. Unsupported stream event variants are rejected unless they are explicitly classified as transport or redundant lifecycle events.

## Semantic mappings

### Reasoning representation

Responses is explicit where Chat Completions and Messages are ambiguous:

| Responses events or item content | Normalized representation               |
| -------------------------------- | --------------------------------------- |
| `response.reasoning_text.*`      | Raw reasoning in `Thinking.content`     |
| `response.reasoning_summary_*`   | Summary in `Thinking.summary`           |
| `encrypted_content`              | Opaque snapshot in `Thinking.signature` |

Every explicit Responses event must agree with the selected provider capability profile. OpenAI is classified as summarized reasoning and vLLM as raw reasoning. A base URL never chooses this behavior.

A Responses reasoning lifecycle must use exactly one valid representation:

- raw reasoning content with no summary or encrypted state;
- a reasoning summary accompanied by encrypted state and no raw content; or
- encrypted state with neither summary nor raw content.

The item must not combine or switch representations between its added, delta, and done states. A completely empty start may remain deferred until its representation becomes known. When Responses is the downstream target, summary providers require a signature for a completed reasoning summary.

Chat Completions has one `reasoning_content` extension and no signature. Messages has one `thinking_delta` field plus signature lifecycle. Their configured provider capability decides whether their text is classified as Responses raw reasoning or summary. Only unsigned raw Chat reasoning is currently portable into Responses: a Chat summary cannot supply the signature required at Responses part completion. Responses-to-Chat summary translation also fails because Chat rejects the source signature instead of dropping it. Supported delta boundaries remain observable and are preserved by matched parity coverage.

### Opaque continuation state

Responses can expose three encrypted snapshots for one reasoning item:

| Responses lifecycle                                 | Normalized or Messages lifecycle                         |
| --------------------------------------------------- | -------------------------------------------------------- |
| `response.output_item.added.item.encrypted_content` | Initial part signature or Messages block-start signature |
| `response.output_item.done.item.encrypted_content`  | Full replacement signature delta                         |
| Reasoning item in terminal `response.output`        | No third independent normalized slot                     |

`signature_delta` is a complete replacement snapshot, not text to append. Molto preserves the added and done snapshots to keep streaming incremental. A translated terminal Responses item reuses the done snapshot rather than buffering the stream to reproduce separately observed terminal ciphertext. Exact terminal encrypted parity is therefore intentionally unavailable.

These values are opaque, provider-issued continuation state. Mapping Responses `encrypted_content` to a Messages signature is valid only under the documented same-provider contract used by the OpenAI-owned matched fixtures. Molto currently does not track the issuer and will mechanically project the value across a Responses-to-Messages route. Callers must not reuse that translated state across providers: an OpenAI-issued value is not valid against Anthropic, and an Anthropic-issued signature is not valid OpenAI encrypted content.

### Text, images, and message boundaries

One normalized user input maps to exactly one Responses user message. Each ordered text or image value maps to exactly one `input_text` or `input_image` part, preserving order. Tool-result parts follow the same explicit-list policy.

Cross-protocol translation does not preserve the distinction between `system` and `developer`, an assistant-message container, multipart assistant content inside one output item, or the original Responses inner `content_index`. Provider rendering can cause further model-visible changes after Molto has constructed a valid wire request. See [boundaries and ordering](../README.md#boundaries-and-ordering) and [provider and model renderers](../README.md#provider-and-model-renderers).

### Tools and identifiers

Responses client-defined functions map to the provider-neutral function name, description, JSON schema, and strictness. Named choices and `auto`, `none`, and `required` are preserved where the target has an equivalent. Messages maps `required` to `any` and represents `parallel_tool_calls` by inverting `disable_parallel_tool_use`.

Function-call history preserves call IDs, function names, JSON arguments, successful results, and ordering. Keep these two identifiers distinct:

- `call_id` is the portable identity that pairs a function call with its result.
- The Responses function-call item `id` identifies the enclosing output item and has no Chat Completions equivalent.

Hosted tools, custom-text tools, server tools, and failed tool-result semantics outside Messages are not currently portable.

### Usage, service tier, and termination

Molto constructs Responses usage explicitly from normalized semantic fields:

| Responses usage                           | Normalized meaning                  |
| ----------------------------------------- | ----------------------------------- |
| `input_tokens`                            | Total model input tokens            |
| `input_tokens_details.cached_tokens`      | Cache-read input tokens             |
| `input_tokens_details.cache_write_tokens` | Cache-write input tokens            |
| `output_tokens`                           | Generated output tokens             |
| `output_tokens_details.reasoning_tokens`  | Reasoning or thinking output tokens |
| `total_tokens`                            | Total input plus output tokens      |

Chat maps prompt, completion, cache, and reasoning counts directly to these meanings. Messages input totals combine uncached input, cache creation, and cache reads before projection. Source-only audio, prediction, cache TTL, inference geography, and server-tool usage are not synthesized.

All translated Responses results report `service_tier="default"`. The source tier can remain normalized metadata, but tier vocabularies are not treated as portable. Native Responses passthrough preserves the provider value. Request service tier is not forwarded across protocols.

Termination projection is:

| Normalized reason               | Responses result                           |
| ------------------------------- | ------------------------------------------ |
| `stop`                          | `response.completed`                       |
| `max_output_tokens`             | `response.incomplete` with the same reason |
| `content_filter`                | `response.incomplete` with the same reason |
| `stop_sequence`                 | `response.completed`, coerced              |
| `tool_call`                     | `response.completed`, coerced              |
| `model_context_window_exceeded` | `response.completed`, coerced              |
| `pause`                         | `response.completed`, coerced              |
| `refusal`                       | `response.completed`, coerced              |

Molto retains the more specific source reason in the normalized stream. The coercion occurs only because the pinned Responses schema cannot represent that reason as an incomplete status.

## Errors

Molto normalizes only recognized errors with a protocol-independent meaning and then raises them in the downstream OpenAI SDK namespace, on native passthrough and translated routes alike. Unknown errors are re-raised unchanged as the original SDK exception.

Model-not-found is currently recognized as:

- Responses: OpenAI `APIStatusError`, status `400`, code `model_not_found`;
- Chat Completions: OpenAI `APIStatusError`, status `404`, code `model_not_found`; and
- Messages: Anthropic `APIStatusError`, status `404`, with `model:` in the message.

Error translation is semantic, not a complete HTTP error rewrite. The outward exception is an OpenAI `APIStatusError`, but its status, message, response, and body remain those of the upstream endpoint. Its `__cause__` is Molto's `ModelNotFoundError`, whose cause is the original upstream SDK exception. See the [shared error contract](../README.md#errors) for caller guidance.

## Fidelity

### Preserved on translated routes

- supported model and generation controls;
- user-message boundaries and ordered text and image parts;
- source-backed response ID, model, text, reasoning, and function argument deltas;
- raw reasoning versus reasoning summary after provider-capability classification;
- function `call_id`, name, arguments, successful results, and ordering;
- normalized termination distinctions and supported semantic usage; and
- balanced Responses item and content lifecycles with stable generated IDs and monotonic sequence numbers.

### Generated, defaulted, or omitted

- Responses item IDs are generated when the source has no corresponding ID.
- Responses sequence numbers and lifecycle envelope are generated.
- Messages-to-Responses `created_at` is captured locally at `message_start`.
- Translated service tier is the target default `default`.
- Inner content and summary indices are reconstructed as `0`.
- Exact completion timestamps, assistant `phase`, and independent server state are omitted because the source cannot reconstruct them.
- The terminal encrypted snapshot reuses the part-done snapshot when only two normalized lifecycle slots are available.

### Coerced or structurally changed

- `system` and `developer` become one instruction concept and project as `developer` when Responses is upstream.
- Assistant message and multipart content containers can split or flatten.
- Termination reasons without a Responses incomplete equivalent become normal completion.
- A provider renderer may merge or relocate model-visible turns after receiving the translated wire request.

## Current limitations

The following are implementation gaps, not general claims that the source data cannot exist:

- non-streaming collection;
- multiple Chat Completions choices; the current translated client reads only `choices[0]`;
- summarized reasoning through Chat Completions; Chat has no signature slot, so current translation fails in either direction rather than discarding required continuation state;
- refusals as content, hosted and server tools, failed tool-result semantics, redacted thinking, citations, annotations, moderation, and other multimodal output;
- log probability transport across the normalized representation;
- Chat audio and accepted or rejected prediction-token details;
- Responses conversation references, previous-response state, prompt-cache decisions, background execution, and other Responses-only server state;
- preservation of original instruction role and placement when the target protocol cannot represent it;
- preservation of an output item's multiple inner content indices; and
- guaranteed handling of interleaved Responses output-item lifecycles.
- enforcement of the same-provider requirement for opaque reasoning state; callers must keep continuation state with its issuing provider.

Fields accepted by `OpenAIResponsesCreateRequest` but not listed as supported must be treated as unsupported on translated routes until they have an explicit mapping or rejection and cassette-backed parity coverage. Same-protocol passthrough retains the broader native SDK surface.

## Compatibility and references

Responses is the canonical input for Molto's matched cross-protocol fixtures. Same-protocol tests compare parsed SDK events. Translated tests compare every reconstructible field and delta boundary, then separately validate target lifecycle invariants such as sequence order, generated-ID consistency, balanced parts, usage, and terminal state. Generated IDs, phase, unavailable timestamps, and provider-owned ciphertext are never copied from an independent expected recording to manufacture equality.

Cassette tests are replay-only. Read the repository's [cassette policy](../../tests/cassettes/README.md) before reviewing or changing fixtures.

Implementation and maintenance references:

- [Architecture](../../python/ARCHITECTURE.md)
- [Conventions](../../python/CONVENTIONS.md)
- [Responses request types](../../python/molto/types/openai_responses.py)
- [Responses adapter](../../python/molto/adapters/openai.py)
- [Responses upstream client](../../python/molto/upstream/openai.py)
- [Interface parity tests](../../tests/core/test_interface.py)
- [Chat Completions protocol guide](openai-chat-completions.md)
- [Anthropic Messages protocol guide](anthropic-messages.md)
