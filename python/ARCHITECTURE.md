# Molto Architecture

## Current scope

Molto is an LLM protocol translation library and the facade for its public coordinators. Its implemented `Molto.responses()` and `Molto.chat_completions()` methods each accept one OpenAI protocol and stream that same protocol back to the caller.

Two upstream OpenAI-compatible endpoints are implemented:

- `responses`: native Responses passthrough or Chat Completions translated through Responses;
- `chat/completions`: native Chat Completions passthrough or Responses translated through Chat Completions.

Only streaming requests are currently supported. Cross-protocol translation currently covers text, raw reasoning content, conversation history, lifecycle metadata, and token usage.

## Architectural goal

The long-term goal from `PLAN.md` remains valid: Molto should be a drop-in protocol and SDK compatibility layer without implementing a translator for every provider pair.

```text
Downstream X protocol
        |
        v
     XAdapter
        |
        v
 NormalizedRequest
        |
        v
     YClient
        |
        v
 Upstream Y protocol

 Upstream Y events
        |
        v
     YClient
        |
        v
 NormalizedEvent stream
        |
        v
     XAdapter
        |
        v
Downstream X events
```

Each downstream protocol should require one adapter and each upstream protocol should require one client. Composition then grows approximately as N + M implementations instead of N x M pairwise translators.

The native protocol passthrough is an identity optimization to this model. When the two protocols are identical, crossing the IR would only reduce fidelity. Cross-protocol routes still compose adapters and clients through the normalized representation.

The following design rules continue to guide the implementation:

1. Normalize semantic concepts such as text, reasoning, tools, usage, and lifecycle state rather than mirroring a provider schema.
2. Treat streaming as canonical. Derive complete responses by collecting the normalized stream instead of maintaining a second execution implementation.
3. Keep adapters free of transport and clients free of downstream formatting.
4. Keep `Molto` thin: it selects and composes components but does not translate provider fields itself.
5. Preserve unsupported provider information through an explicit escape hatch rather than expanding the IR into a union of every provider schema.
6. Optimize the public boundary for SDK compatibility while keeping the internal semantic model provider-neutral.
7. Add adapters and clients, not classes such as `OpenAIToAnthropicTranslator`.

## Dependency direction

```text
                         normalized types
                         /              \
                        /                \
        downstream adapters          upstream clients
                        \                /
                         \              /
                              Molto
```

The layers have distinct ownership:

| Layer | Owns | Must not own |
| --- | --- | --- |
| `core.py` (`Molto`) | Public orchestration facade, protocol coordinators, route selection, composition, credentials | Provider-specific field transformations |
| `adapters/` | Downstream native request to normalized request; normalized events to downstream native events | HTTP transport |
| `upstream/` | Upstream request construction, provider SDK stream integration, compatibility validation where required, upstream response normalization | Downstream response formatting |
| `types/normalized.py` | Provider-neutral request content, response parts, deltas, lifecycle metadata, and usage | Transport or SDK lifecycle objects |
| `types/openai_*.py` | OpenAI request schemas, SDK-native result aliases, and narrow compatibility validators | Cross-protocol orchestration |

Adapters and upstream clients do not depend on each other. Protocol-specific coordinator methods on `Molto` compose them.

## Orchestration facade and coordinators

`Molto` is expected to expose one coordinator method per downstream protocol:

```text
Molto
  |-- responses()         implemented
  |-- chat_completions()  implemented
  |-- messages()          future Anthropic Messages boundary
  `-- chat()              future dispatcher
```

Callers can use a protocol coordinator directly when they need that protocol's native request and SDK-compatible response types. Each coordinator owns its own downstream contract: it validates the native request, selects and composes an upstream client, and projects results back into the same downstream protocol.

A future `Molto.chat()` will be a higher-level dispatcher. It will select one of the underlying protocol coordinators and delegate to it; it should not duplicate their normalization, transport composition, or projection logic. The selection policy may use explicit caller configuration, request type, or capabilities, but that policy is not defined yet.

Each implemented coordinator performs the following work:

1. Validate the public body as its protocol request model.
2. Reject non-streaming requests until collection is implemented.
3. Resolve the API key and injected or default HTTP client.
4. Select the upstream path from `endpoint`.
5. Return an async iterator of the same protocol's OpenAI SDK event models.

Each coordinator deliberately chooses between a native and translated pipeline. This is a downstream protocol decision, not logic that belongs in an adapter or transport client.

`Molto` owns the injected or default `httpx.AsyncClient`. Short-lived `AsyncOpenAI` wrappers borrow that client and are not closed independently. Each returned SDK stream is context-managed so its response is released on completion or early termination; `Molto.close()` remains the only operation that closes the shared HTTP client.

## Native protocol passthrough

When the downstream and upstream protocols match, Molto skips request and response normalization. The Responses path is:

```text
Responses request
      |
      v
OpenAIResponsesCreateRequest validation
      |
      v
Molto.responses(endpoint="responses")
      |
      v
OpenAIResponsesClient.native_events()
      |
      +-- AsyncOpenAI.responses.create()
      +-- SDK-owned HTTP and SSE processing
      +-- direct SDK event yielding
      +-- narrow completed-usage compatibility normalization
      |
      v
Responses SDK event stream
```

The bypass is intentional. Normalizing a request and response that already use the desired protocol would add no semantic value and could lose provider fields, event variants, JSON representation, or native event fidelity.

Molto validates the request and relies on the SDK's native event models. Passthrough means no IR translation or duplicate model validation, not untyped proxying.

The Chat Completions path follows the same shape through `Molto.chat_completions(endpoint="chat/completions")` and `OpenAIChatCompletionsClient.native_events()`. The SDK consumes `[DONE]` as a transport terminator, and Molto exposes the yielded `ChatCompletionChunk` models directly without duplicating SDK validation work.

## Responses through Chat Completions

When `endpoint="chat/completions"`, Molto uses the normalized representation:

```text
Responses request
      |
      v
OpenAIResponsesAdapter.normalize_request()
      |
      v
NormalizedRequest
      |
      v
OpenAIChatCompletionsClient.events()
      |
      +-- construct OpenAIChatCompletionsCreateRequest
      +-- delegate to native_events()
      +-- stream through AsyncOpenAI.chat.completions.create()
      +-- consume ChatCompletionChunk SDK models directly
      +-- emit normalized lifecycle and content events
      |
      v
ModelResponseStreamEvent
      |
      v
OpenAIResponsesAdapter.project_events()
      |
      +-- synthesize Responses lifecycle events
      +-- preserve source-backed metadata and usage
      +-- generate required Responses-only item IDs
      |
      v
Responses SDK event stream
```

The adapter and client remain independently reusable: the adapter knows the downstream Responses contract, while the client knows the upstream Chat Completions contract.

## Chat Completions through Responses

The reverse route uses the same IR rather than a pair-specific reverse translator:

```text
Chat Completions request
      |
      v
OpenAIChatCompletionsAdapter.normalize_request()
      |
      v
NormalizedRequest
      |
      v
OpenAIResponsesClient.events()
      |
      v
ModelResponseStreamEvent
      |
      v
OpenAIChatCompletionsAdapter.project_events()
      |
      v
ChatCompletionChunk SDK stream
```

Raw `reasoning_content` is a provider extension to the OpenAI Chat schema. Molto models that extension only on assistant messages, maps it to `Thinking`, and reconstructs Responses reasoning history. A Chat reasoning request maps to detailed Responses reasoning so the upstream returns the raw reasoning needed by the downstream contract.

## Normalized request model

`NormalizedRequest` holds portable request semantics. The implemented adapter normalizes:

- string input;
- text messages with `system`, `developer`, `user`, and `assistant` roles;
- raw reasoning content in requests and assistant history;
- instructions as a leading `SystemPrompt`;
- model and streaming configuration;
- output-token limits, parallel-tool-call setting, reasoning effort, temperature, and `top_p`.

Assistant conversation history uses `TextOutput`, not user-message metadata. This keeps role semantics explicit when the Chat client reconstructs messages.

Normalized models inherit from `ExtraModel`. Unknown provider fields are retained in `extra_fields` as an escape hatch. Vendor request construction must still map supported fields explicitly; preserving a field does not imply that another protocol supports it.

The IR must not become a lowest-common-denominator provider API. Concepts with clear cross-provider meaning belong in normalized fields; genuinely provider-specific capabilities remain in `extra_fields` until a portable semantic model exists.

## Normalized event protocol

The upstream client exposes streaming as its primitive:

```python
async def events(
    request: NormalizedRequest,
) -> AsyncIterator[ModelResponseStreamEvent]: ...
```

The current normalized grammar is:

```text
ResponseStartEvent
PartStartEvent
PartDeltaEvent*
PartEndEvent
ResponseEndEvent
```

`FinalResultEvent` is also part of the union for future result handling.

For the text and raw-reasoning paths:

- `ResponseStartEvent` carries the upstream response ID, creation time, model, and service tier;
- `PartStartEvent` establishes an empty `Thinking` or `TextOutput` at a stable output index;
- each source reasoning or text chunk becomes one matching `PartDeltaEvent`;
- `PartEndEvent` carries the complete reasoning or text accumulated from the deltas;
- `ResponseEndEvent` carries terminal metadata and normalized usage.

Chat's role chunk starts the first part but its empty content is not emitted as a delta. Transitioning from reasoning to text ends the reasoning part and starts the text part. A finish reason ends the active part. Usage can arrive in a later chunk, so the response ends only after the upstream SDK stream has been consumed.

## Response collection

Streaming remains the canonical execution interface even though non-streaming responses are not implemented yet. The intended non-streaming path is:

```text
YClient.events()
      |
      v
NormalizedEvent stream
      |
      v
generic collect()
      |
      v
NormalizedResponse
      |
      v
XAdapter.project_response()
```

Clients should not gain a separate completion implementation. A generic collector keeps streaming and non-streaming normalization consistent. Until that collector and normalized response model exist, both implemented coordinators reject non-streaming requests explicitly.

## Responses event projection

A normalized event may project to more than one native event. The simple text projection emits:

1. `response.created`;
2. `response.in_progress`;
3. `response.output_item.added`;
4. `response.content_part.added`;
5. one `response.output_text.delta` per normalized text delta;
6. `response.output_text.done`;
7. `response.content_part.done`;
8. `response.output_item.done`;
9. `response.completed`.

The adapter owns Responses sequence numbers, output/content indices, item state, and generated message item IDs. It preserves the Chat response ID, creation time, model, service tier, delta boundaries, and mapped usage.

Chat Completions cannot provide a Responses output-item ID or exact Responses completion timestamp. These are generated or omitted according to the target contract and are not treated as independently recorded server state.

## Usage mapping

The Chat client normalizes token usage and the Responses adapter projects it as:

| Normalized usage                          | Chat Completions source                      | Responses output    |
| ----------------------------------------- | -------------------------------------------- | ------------------- |
| `input_tokens`                            | `prompt_tokens`                              | `input_tokens`      |
| `input_tokens_details.cached_tokens`      | `prompt_tokens_details.cached_tokens`        | same semantic field |
| `input_tokens_details.cache_write_tokens` | `prompt_tokens_details.cache_write_tokens`   | same semantic field |
| `output_tokens`                           | `completion_tokens`                          | `output_tokens`     |
| `output_tokens_details.reasoning_tokens`  | `completion_tokens_details.reasoning_tokens` | same semantic field |
| `total_tokens`                            | `total_tokens`                               | `total_tokens`      |

Projection builds the vendor usage object explicitly. Dumping an `ExtraModel` directly would expose its internal `extra_fields` container at the public boundary.

## OpenAI SDK types

Request bodies are local Pydantic wrappers because Molto needs one validated model whose fields track the corresponding SDK `create()` signature. Nested parameter types come from the pinned OpenAI SDK.

Compatible Responses providers can return reasoning items and assistant output messages without IDs and accept those items as subsequent-turn input. The pinned SDK marks both IDs as required, so Molto uses two narrow Pydantic history models with optional IDs and otherwise SDK-compatible, eagerly validated fields. It does not use an untyped dictionary fallback for Responses input.

Response and stream result types are intentional SDK aliases:

- `OpenAIResponsesResponse` aliases `openai.types.responses.Response`;
- `OpenAIResponsesStreamEvent` annotates the SDK `ResponseStreamEvent` union;
- `OpenAIChatCompletionsResponse` aliases the SDK `ChatCompletion`;
- `OpenAIChatCompletionsStreamChunk` aliases the SDK `ChatCompletionChunk`.

Molto does not subclass or duplicate these result models. The SDK already owns the wire schema, and aliases avoid schema drift while giving callers familiar types.

Molto does not recreate or expose the OpenAI SDK's transport-bearing `AsyncStream` runtime object. It context-manages the SDK stream internally and exposes a Molto-owned async iterator that yields SDK-native Responses events and Chat Completions chunks directly.

Generated protocol event dictionaries are validated directly with SDK model adapters.

## OpenAI SDK transport

Each OpenAI upstream client constructs a short-lived `AsyncOpenAI` wrapper with:

- Molto's shared `httpx.AsyncClient`;
- the resolved API key;
- an explicit supplied or default base URL, so SDK environment settings cannot redirect the request.

An absent vLLM key resolves to the fixed non-secret `EMPTY` credential. OpenAI organization and project environment settings are suppressed so compatible-provider requests cannot leak OpenAI account metadata.

The SDK owns request serialization, authentication, HTTP, SSE decoding, terminal markers, retries, error types, and native model construction for both protocols. Molto accepts the SDK retry policy and OpenAI exceptions rather than wrapping them as the previous `httpx` behavior. Before directly yielding a Responses completion event, Molto narrowly normalizes the compatible-provider omission where `cache_write_tokens` is `None` to `0`.

SDK streams are context-managed, including when iteration ends early. The short-lived SDK wrapper is not closed because doing so would close Molto's injected HTTP client; `Molto.close()` owns that client's lifetime. Adapters never access `httpx`.

## Replay and parity testing

Tests are playback-only. Normal runs must never record or rewrite cassettes.

Matched cross-protocol fixtures use a scenario-first layout:

```text
tests/cassettes/matched/
  simple_text/
    openai_chat_completions-stream-00.yaml
    openai_responses-stream-00.yaml
  reasoning_raw/
    openai_chat_completions-stream-00.yaml
    openai_responses-stream-00.yaml
```

The naming pattern is:

```text
tests/cassettes/matched/{scenario}/openai_{endpoint}-{mode}-{turn}.yaml
```

Each fixture is canonical for its own downstream protocol. Matched pairs preserve source-backed values and aligned delta boundaries so translation can be compared deterministically.

Parity differs by path:

- same-protocol routes assert exact request bodies and parsed SDK stream events;
- Chat Completions to Responses compares reconstructible Responses events and validates event ordering, sequence numbers, and generated IDs;
- Responses to Chat Completions compares ordered text and reasoning deltas, finish state, and semantic usage without requiring unavailable fingerprints or identical usage-chunk placement.

## Current limitations and extension path

The current public paths support streaming text and raw reasoning. Non-streaming collection, encrypted reasoning translation, reasoning summaries, tool calls, multimodal content, and additional providers are not complete cross-protocol implementations yet.

The normalized IR should ultimately be SDK-independent. The current `NormalizedRequest.tools` and `tool_choice` fields still reuse OpenAI SDK parameter types; this is transitional coupling and should be replaced with semantic tool types before additional downstream protocols depend on them.

To add another upstream protocol:

1. add a client that converts `NormalizedRequest` to the upstream request;
2. normalize its native stream into `ModelResponseStreamEvent`;
3. compose it from `Molto` without importing a downstream adapter into the client;
4. add cassette-backed parity tests against the downstream SDK contract.

To add another downstream protocol, add an adapter that normalizes its request and projects normalized events without performing transport.
