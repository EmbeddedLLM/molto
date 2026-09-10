# OpenAI Chat Completions

This guide describes OpenAI Chat Completions as a native protocol and as either side of a Molto translation. For routing, parity classes, shared termination and service-tier policy, and the distinction between wire messages, model-visible turns, and streaming parts, start with the [documentation overview](../README.md).

## At a glance

| Concern              | Value                                     |
| -------------------- | ----------------------------------------- |
| Application method   | `Molto.chat_completions()`                |
| Upstream endpoint    | `endpoint="chat/completions"`             |
| Default provider     | `openai`                                  |
| Request model        | `OpenAIChatCompletionsCreateRequest`      |
| Stream item          | OpenAI SDK `ChatCompletionChunk`          |
| Current request mode | Streaming only; `stream=True` is required |

The method always returns Chat Completions chunks to the application. The `endpoint` argument independently chooses whether the provider receives Chat Completions, Responses, or Anthropic Messages wire data.

## When to use this protocol

Use `chat_completions()` when the application already uses the OpenAI Chat Completions message and chunk contract, or when it depends on native Chat features that should remain intact on a same-protocol route. Select another upstream endpoint when the application must keep that contract but the serving provider exposes Responses or Messages instead.

Prefer native passthrough for protocol-specific features outside Molto's shared translation surface. Cross-protocol translation currently focuses on text, portable image input, reasoning, client-defined function tools, usage, and termination semantics. It is not a general emulation of every Chat field.

## Quickstart

The native route sends Chat Completions to Chat Completions and yields SDK-native chunks:

```python
import asyncio

from molto import Molto


async def main() -> None:
    async with Molto() as molto:
        async with await molto.chat_completions(
            body={
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
            endpoint="chat/completions",
            provider="openai",
        ) as stream:
            async for chunk in stream:
                print(chunk)  # openai.types.chat.ChatCompletionChunk


asyncio.run(main())
```

To keep the same application-facing chunks while changing the upstream wire protocol, change only the route settings:

```python
# OpenAI Responses upstream
stream = await molto.chat_completions(
    body=body,
    endpoint="responses",
    provider="openai",
)

# Anthropic Messages upstream
stream = await molto.chat_completions(
    body=body,
    endpoint="messages",
    provider="anthropic",
)
```

Context-manage the stream, or call `await stream.aclose()` in a `finally` block when managing it manually. Close `Molto` separately with its context manager or `await molto.aclose()`.

## Native wire contract

### Requests and conversation history

Chat Completions represents a conversation as an ordered `messages[]` array. Each element is a native wire-message boundary. The schema includes `developer`, `system`, `user`, `assistant`, `tool`, and legacy `function` messages. It does not impose a positional restriction on system or developer messages. Newer OpenAI reasoning models use developer instructions in place of the older system-message mechanism, but the model or renderer ultimately determines the model-visible instruction behavior.

User messages can contain a string or an ordered content-part array, including text and image parts supported by Molto. Assistant messages can contain text, tool calls, and provider extensions such as `reasoning_content`. Chat tool messages identify the call with `tool_call_id` and carry text content; they do not have a portable image-result representation.

An assistant message containing `tool_calls` must be followed immediately by the complete contiguous block of corresponding `tool` messages before any other role. This ordering requirement also applies to parallel tool calls.

### Streaming chunks

A Chat stream represents one generated assistant message per choice. Each `ChatCompletionChunk` carries an indexed `choices[]` list whose deltas can establish the assistant role, append reasoning or text, append partial function arguments, or report a `finish_reason`. Usage can arrive in a late chunk, often with an empty choices list when `stream_options.include_usage` is enabled.

`[DONE]` is an SSE transport terminator, not a `ChatCompletionChunk`. The OpenAI SDK consumes it and Molto yields only parsed SDK chunk models. Chat does not have native output-part start/end events or per-part statuses; those structures must be synthesized when translating into a richer lifecycle protocol.

Transport chunk boundaries are observable but are not conversation-message boundaries. See [Boundaries and ordering](../README.md#boundaries-and-ordering) for the shared distinction.

## Native passthrough

When both sides use Chat Completions, Molto validates the request with `OpenAIChatCompletionsCreateRequest`, calls the OpenAI SDK's `chat.completions.create()`, and yields its `ChatCompletionChunk` objects directly. The request and result do not cross Molto's normalized representation.

This path preserves system and developer instruction arrays, native choices, result fields, event order, usage placement, service tier, and provider extensions accepted by the pinned SDK. Errors follow the [shared policy](../errors.md). Passthrough is typed SDK processing rather than a byte proxy: Molto does not promise the original JSON or SSE byte representation.

## When the application speaks Chat Completions

For `endpoint="responses"` or `endpoint="messages"`, the downstream adapter normalizes the Chat request and later projects normalized provider events back to Chat chunks.

```text
Chat request -> normalized request -> upstream protocol
Chat chunks <- normalized events  <- upstream stream
```

### Downstream request normalization

The translated path currently maps these request semantics:

| Chat input                           | Normalized behavior                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `model`, `stream`                    | Preserved                                                                                |
| `max_completion_tokens`              | Becomes the portable output-token limit                                                  |
| `reasoning_effort`                   | Becomes portable reasoning effort                                                        |
| `temperature`, `top_p`               | Preserved                                                                                |
| `parallel_tool_calls`                | Preserved; Messages later expresses it by inverting `disable_parallel_tool_use`          |
| Function `tools`                     | Preserves name, description, parameters, and strict validation                           |
| `tool_choice`                        | Preserves `auto`, `none`, `required`, or a named function through target-specific shapes |
| `stream_options.include_obfuscation` | Preserved where the upstream protocol represents it                                      |

One user message becomes one normalized user input. Ordered text and image parts remain ordered, and consecutive user messages remain separate. Scalar user text becomes one text part.

System and developer messages with string content both become role-less instructions. Their sequence position is retained while targeting Responses or Chat, but their original role identity is lost. Text-part arrays serialize correctly and remain available on native passthrough, but the translated downstream adapter currently rejects non-string instruction content. A later Chat projection emits every normalized instruction as `developer`. A Messages upstream additionally hoists all instructions into its top-level `system` value, losing mid-conversation placement and instruction boundaries.

Assistant history is flattened into adjacent reasoning, text, and tool-call items. The enclosing assistant-message boundary is not represented. The adapter accepts assistant string content, optional string `reasoning_content`, and function tool calls; assistant content-part arrays are not supported on translated routes. When a Chat request carries `reasoning_effort`, Molto requests a detailed summary (`reasoning.summary="detailed"` unless the effort is `none`, which disables reasoning) so a Responses or Messages target receives a completed summarized reasoning value.

Tool messages normalize scalar text or ordered text parts as successful tool results. Native Chat tool messages containing image parts are rejected. The downstream adapter does not reinterpret such input as an image tool result.

Fields accepted by the request wrapper but not listed above are not guaranteed to reach a cross-protocol upstream. In particular, accepting a field as extra request data is not evidence that Molto translates it. Native passthrough is the appropriate route when such a field is required.

### Projection back to Chat chunks

Normalized output is projected as choice index `0` in one Chat assistant stream. Molto emits the assistant role once, then preserves supported reasoning, text, and function-argument delta boundaries. A normalized part start can contribute its non-empty initial snapshot; later part starts do not create new assistant messages, and part-end events have no Chat equivalent.

Reasoning and text therefore share one projected assistant message even when the source protocol models them as separate items or blocks. Function calls use stable per-stream Chat tool indexes and preserve the normalized call ID, name, and partial JSON arguments.

The terminal chunk contains the projected `finish_reason` and semantic usage. Molto always emits `service_tier="default"` and `logprobs=None` on translated Chat chunks. It does not invent a `system_fingerprint`.

## When the provider exposes Chat Completions

An upstream Chat client converts another downstream protocol's normalized request into Chat wire messages, consumes native chunks, and emits normalized events for the downstream adapter.

### Upstream request construction

| Normalized input              | Chat request                                                     |
| ----------------------------- | ---------------------------------------------------------------- |
| Instruction                   | `developer` message at the same sequence position                |
| User input                    | One `user` message with an explicit ordered content-part array   |
| Portable image                | `image_url` part, preserving URL or data URL and optional detail |
| Reasoning plus following text | One assistant message with `reasoning_content` and `content`     |
| Adjacent assistant text items | Separate assistant messages                                      |
| Function call                 | Assistant `tool_calls[]` entry                                   |
| Successful text tool result   | `tool` message with the original `tool_call_id`                  |
| Output-token limit            | `max_completion_tokens`                                          |
| Streaming usage               | `stream_options.include_usage=true`                              |

The client groups pending assistant reasoning, text, and tool calls where the Chat shape permits it. A reasoning item is held for a following text item so both can inhabit one assistant message. Because the normalized representation does not retain the original assistant container, this reconstruction can merge or split source assistant messages.

OpenAI Chat tool messages cannot carry images. For a normalized image tool result, Molto leaves a text notice in the `tool` message and emits the image in a later `user` message labeled with the call ID. This relocation follows the approach in [github/copilot-sdk#1922](https://github.com/github/copilot-sdk/issues/1922). It first emits every tool response belonging to the preceding assistant call block, then emits all relocated image prompts. For two parallel calls, the valid order is:

```text
assistant, tool A, tool B, user image A, user image B
```

Interleaving `user image A` before `tool B` would violate Chat's tool-response ordering contract. Non-successful tool outcomes are rejected because OpenAI protocols cannot preserve that semantic distinction.

This relocation matches how vLLM itself handles a `tool_result` image through its Anthropic Messages endpoint, which is likewise a conversion to Chat: text stays in `tool` and the image is remapped to a later `user` prompt. SGLang's Messages-to-Chat conversion instead embeds the `image_url` parts inside the `tool` message's content array. See [Provider and model renderers](../README.md#provider-and-model-renderers).

### Upstream stream normalization

The client reads only `choices[0]`. Non-empty `reasoning_content` and text deltas are preserved for that choice, while an empty role-establishment content chunk does not become a text delta. A transition between reasoning and text closes one normalized part and opens another; later reasoning can open a new thinking part. Tool-call deltas must be contiguous by tool index, and their first chunk must provide the call ID and function name.

`reasoning_content` does not say whether it contains raw reasoning or a reasoning summary. Molto classifies it with the explicit provider capability:

| Provider profile | Interpretation    |
| ---------------- | ----------------- |
| `openai`         | Reasoning summary |
| `vllm`           | Raw reasoning     |

`endpoint` and `base_url` do not select this interpretation. A custom OpenAI-compatible service must use the provider profile matching its actual behavior.

The first native chunk supplies response ID, creation time, model, and source service-tier metadata. A supported `finish_reason` closes active parts, while a late usage chunk can update semantic usage before the normalized response ends. The stream must contain at least one chunk and a supported finish reason.

## Semantic mappings

### Roles, content, and assistant structure

| Chat concept                         | Cross-protocol behavior                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| User message boundary                | Preserved by Molto                                                                    |
| Ordered user text and image parts    | Preserved                                                                             |
| `system` versus `developer` identity | Collapsed into one instruction type                                                   |
| Instruction position                 | Preserved for Responses and Chat targets; hoisted for Messages                        |
| Assistant message container          | Lost in normalization                                                                 |
| `reasoning_content` followed by text | Retained as adjacent reasoning and text; commonly reconstructed into one Chat message |
| Multiple assistant text items        | May become multiple Chat assistant messages                                           |
| Chat choice index                    | Only index `0` is currently normalized                                                |

The provider or model renderer can still merge, move, or reinterpret accepted wire messages before tokenization. Those model-visible transformations are outside the endpoint schema and are covered in the [provider and renderer discussion](../README.md#provider-and-model-renderers).

### Reasoning and continuation state

Chat's single `reasoning_content` extension can be classified as either the normalized raw reasoning field or summary field according to the provider profile. Chat has no reasoning signature or encrypted continuation slot. In the current implementation, only unsigned raw reasoning is portable through Chat on a translated route. A Chat summary has no signature and therefore cannot satisfy the completed-summary contract required by Responses or Messages; in the reverse direction, a Responses or Messages summary carries a signature that Chat projection rejects rather than silently discarding. Opaque continuation state and summarized reasoning are therefore not currently portable through Chat.

### Tools and tool results

Function definitions preserve their portable schema fields. Named selection maps to a named target function or tool, `required` maps to Messages `any`, and parallel-call control maps to each target's native setting. Tool-call IDs are portable call identities; they must not be confused with Responses output-item IDs, which are a separate structure. When Responses history is projected to Chat, those output-item IDs are accepted and omitted as the sole exception to the rule that every accepted input field must affect the target request.

Successful text tool results preserve call ID, content order, and conversation order. Image results require the upstream-Chat relocation described above. Failed-result semantics, hosted tools, server tools, and general custom tool types do not have a complete cross-protocol mapping.

### Termination and usage

| Chat `finish_reason`                   | Normalized reason   | Projection back to Chat |
| -------------------------------------- | ------------------- | ----------------------- |
| `stop`                                 | `stop`              | `stop`                  |
| `length`                               | `max_output_tokens` | `length`                |
| `tool_calls` or legacy `function_call` | `tool_call`         | `tool_calls`            |
| `content_filter`                       | `content_filter`    | `content_filter`        |

Other normalized reasons use `stop` when Chat has no exact finish reason. This keeps the stream consumable but loses the source distinction. Native Chat passthrough does not apply this coercion.

Usage maps semantically rather than by dumping provider objects:

| Normalized usage | Chat usage                                   |
| ---------------- | -------------------------------------------- |
| Input tokens     | `prompt_tokens`                              |
| Cached input     | `prompt_tokens_details.cached_tokens`        |
| Cache writes     | `prompt_tokens_details.cache_write_tokens`   |
| Output tokens    | `completion_tokens`                          |
| Reasoning output | `completion_tokens_details.reasoning_tokens` |
| Total tokens     | `total_tokens`                               |

Missing cached, cache-write, and reasoning detail counts default to zero in the current normalized path. Audio usage and accepted or rejected prediction-token details have no current normalized mapping. When Messages is the source, Molto's normalized input total includes uncached input, cache creation, and cache reads before it is projected into `prompt_tokens`.

Translated Chat Completions always includes terminal usage, matching the other translated protocols. A cross-protocol request may omit `stream_options.include_usage` or set it to `true`; setting it to `false` is rejected.

Translated Chat chunks always use target-default `service_tier="default"`; an upstream tier label is not relabeled as a cross-provider equivalent. Request `service_tier` is not forwarded across protocols. See [Service tiers](../README.md#service-tiers) for the shared rationale.

Chat stream options expose `include_obfuscation`, but the generated `ChatCompletionChunk` and `ChoiceDelta` models declare no `obfuscation` field. Preserving obfuscation parity would therefore require retaining raw chunk data rather than relying on the SDK models, which Molto does not currently do.

## Errors

Recognized errors use the OpenAI SDK namespace and the [shared error contract](../errors.md). A Responses upstream's [mid-stream context-length failure](../errors.md#responses-failures-during-streaming) remains a plain `openai.APIError`; only [missing SDK credentials](../errors.md#missing-sdk-credentials) receive a synthetic `401`.

## Fidelity

### Preserved on translated routes

- One boundary per supported user message and the order of its text and image parts.
- Choice-zero text, classified reasoning, and function-argument delta boundaries.
- Response ID, creation time when supplied by the source, model, supported finish semantics, and semantic token usage.
- Function names, JSON argument bytes, call IDs, successful text results, selection, and parallel-call intent.

### Generated, defaulted, or unavailable

- Projected Chat uses choice index `0`, `object="chat.completion.chunk"`, `service_tier="default"`, and `logprobs=None`.
- A Messages upstream has no creation timestamp, so its normalized client captures local wall-clock time and that generated value becomes Chat `created`.
- Responses has no Chat `system_fingerprint`; translated Chat output omits it.
- Chat has no output-item IDs, content indexes, item statuses, assistant `phase`, exact response-completion timestamp, or encrypted reasoning state. A richer downstream protocol must generate, omit, or reject those values according to its own guide.
- Parsing and serialization can change key order, omitted versus explicit null, whitespace, SSE framing, and provider chunk placement. Cross-protocol parity is semantic, not byte-for-byte wire equality.

## Current implementation limitations

These are implementation gaps rather than inherent claims about the Chat protocol:

- Only streaming requests are supported.
- An upstream Chat stream silently uses `choices[0]`; additional choices are not rejected and are lost. Requests with `n > 1` should not use translation.
- Translated reasoning through Chat currently supports only unsigned raw reasoning, such as the `vllm` provider profile. Summary profiles fail at the target protocol's signature boundary in either direction.
- System and developer text-part arrays are preserved on native Chat routes, but translated Chat input currently supports only string instruction content.
- Translated assistant input supports string content, reasoning text, and function tool calls, but not assistant content-part arrays.
- Refusals, hosted tools, moderation, annotations, audio, prediction output, and log probabilities are not carried through the normalized path.
- Chat text chunks therefore use `logprobs=None`, even when another protocol could theoretically represent compatible token log probabilities.
- Chat has no normalized per-part status. Target lifecycle and status fields are synthesized where required, and normalized terminal reasons without an exact Chat mapping become `finish_reason="stop"`.
- The downstream adapter rejects image content in native Chat tool messages. Image relocation exists only when Chat is the upstream target.
- Cross-protocol request forwarding is limited to the fields listed in this guide. Accepted fields such as deprecated `max_tokens`, `metadata`, prompt-cache settings, `service_tier`, `store`, `top_logprobs`, `verbosity`, and web-search options must not be assumed to work on translated routes.
- There is no focused regression coverage for role-preserving instruction round trips, adjacent assistant-message boundaries, multiple choices, or model-template-specific role rewriting. On the Responses/Chat translated paths, mid-sequence and multipart instruction messages also lack focused coverage; those cases are covered only for top-level system blocks and for the Messages path.

## Compatibility and references

The pinned OpenAI SDK in `pyproject.toml` is the schema source of truth. Molto's request wrapper mirrors the SDK method parameters, while streamed results are aliases of the SDK's `ChatCompletionChunk`. Local compatibility message types replace the SDK's iterable annotations with concrete lists for system, developer, user, and tool content arrays, keeping nested validation eager and serialization repeatable. Changing the SDK requires reviewing request parity, extension fields, result mappings, and cassette-backed behavior.

Same-protocol tests compare validated SDK chunks. Translated tests assert the upstream request body and compare reconstructible ordered chunks, including delta boundaries, finish state, semantic usage, and target-default tier. A finish-and-usage chunk is considered semantically equivalent to separate finish and usage chunks because upstream protocols expose usage at different lifecycle points. Tests always replay committed cassettes and never record during normal verification.

Repository references:

- [Architecture and route composition](../../python/ARCHITECTURE.md)
- [Implementation conventions](../../python/CONVENTIONS.md)
- [Chat request types](../../python/molto/types/openai_chat_completions.py)
- [Chat downstream adapter](../../python/molto/adapters/openai.py)
- [Chat upstream client](../../python/molto/upstream/openai.py)
- [Protocol coordination](../../python/molto/core.py)
- [Interface parity tests](../../tests/core/test_interface.py)
- [OpenAI cassette matcher](../../scripts/match_openai_cassette_events.py) and its [reconstructable Chat Completions canonicalization helpers](../../python/molto/utils/test.py#reconstructable_chat_completions_chunks)
- [Cassette policy](../../tests/cassettes/README.md)
- [Cross-protocol comparison](comparison.md)

External protocol references:

- [OpenAI Chat Completions create reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
