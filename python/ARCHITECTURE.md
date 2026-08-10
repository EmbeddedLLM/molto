# Molto Architecture

## Current scope

Molto is an LLM protocol translation library and the facade for its public coordinators. Its first implemented protocol coordinator, `Molto.responses()`, accepts an OpenAI Responses request and streams OpenAI Responses events back to the caller.

Two upstream OpenAI-compatible endpoints are implemented:

- `responses`: native Responses request and response passthrough;
- `chat/completions`: Responses request normalization, Chat Completions execution, and projection back to Responses events.

Only streaming requests are currently supported. The normalized types include additional content and event shapes for future integrations, but the implemented cross-protocol path is text-oriented.

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
| `upstream/` | Upstream request construction, HTTP/SSE transport, upstream response normalization | Downstream response formatting |
| `types/normalized.py` | Provider-neutral request content, response parts, deltas, lifecycle metadata, and usage | Transport or SDK lifecycle objects |
| `types/openai_*.py` | OpenAI request schemas, SDK-native result aliases, and wire parsing helpers | Cross-protocol orchestration |

Adapters and upstream clients do not depend on each other. Protocol-specific coordinator methods on `Molto` compose them.

## Orchestration facade and coordinators

`Molto` is expected to expose one coordinator method per downstream protocol:

```text
Molto
  |-- responses()         implemented
  |-- chat_completions()  future OpenAI Chat Completions boundary
  |-- messages()          future Anthropic Messages boundary
  `-- chat()              future dispatcher
```

Callers can use a protocol coordinator directly when they need that protocol's native request and SDK-compatible response types. Each coordinator owns its own downstream contract: it validates the native request, selects and composes an upstream client, and projects results back into the same downstream protocol.

A future `Molto.chat()` will be a higher-level dispatcher. It will select one of the underlying protocol coordinators and delegate to it; it should not duplicate their normalization, transport composition, or projection logic. The selection policy may use explicit caller configuration, request type, or capabilities, but that policy is not defined yet.

`Molto.responses()` is the first implemented coordinator. It performs the following work:

1. Validate the public body as `OpenAIResponsesCreateRequest`.
2. Reject non-streaming requests until collection is implemented.
3. Resolve the API key and injected or default HTTP client.
4. Select the upstream path from `endpoint`.
5. Return an async iterator of OpenAI Responses SDK event models.

The Responses coordinator deliberately chooses between two different pipelines. This is a downstream protocol decision, not logic that belongs in an adapter or transport client.

## Native Responses passthrough

When the downstream and upstream protocols are both OpenAI Responses, Molto skips request and response normalization:

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
      +-- POST /responses
      +-- split SSE records
      +-- validate each record as an SDK ResponseStreamEvent
      +-- retain the original SSE record as event.as_sse()
      |
      v
Responses SDK event stream
```

The bypass is intentional. Normalizing a request and response that already use the desired protocol would add no semantic value and could lose provider fields, event variants, JSON representation, or SSE framing. Direct passthrough therefore preserves the exact upstream record returned by `event.as_sse()`.

Molto still validates the request and each response event. Passthrough means no IR translation, not untyped proxying.

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
      +-- POST /chat/completions
      +-- parse ChatCompletionChunk SDK models
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

## Normalized request model

`NormalizedRequest` holds portable request semantics. The implemented adapter normalizes:

- string input;
- text messages with `system`, `developer`, `user`, and `assistant` roles;
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

For the text path:

- `ResponseStartEvent` carries the upstream response ID, creation time, model, and service tier;
- `PartStartEvent` establishes an empty `TextOutput` at a stable output index;
- each Chat content chunk becomes one `PartDeltaEvent(TextDelta)`;
- `PartEndEvent` carries the complete text accumulated from the deltas;
- `ResponseEndEvent` carries terminal metadata and normalized usage.

Chat's role chunk starts the text part but its empty content is not emitted as a text delta. A finish reason ends the part. Usage can arrive in a later chunk, so the response ends only after the upstream SSE stream has been consumed.

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

Clients should not gain a separate completion implementation. A generic collector keeps streaming and non-streaming normalization consistent. Until that collector and normalized response model exist, `Molto.responses()` rejects non-streaming requests explicitly.

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

Response and stream result types are intentional SDK aliases:

- `OpenAIResponsesResponse` aliases `openai.types.responses.Response`;
- `OpenAIResponsesStreamEvent` annotates the SDK `ResponseStreamEvent` union;
- `OpenAIChatCompletionsResponse` aliases the SDK `ChatCompletion`;
- `OpenAIChatCompletionsStreamChunk` aliases the SDK `ChatCompletionChunk`.

Molto does not subclass or duplicate these result models. The SDK already owns the wire schema, and aliases avoid schema drift while giving callers familiar types.

Molto also does not recreate the OpenAI SDK's transport-bearing `AsyncStream` runtime object. The library owns the async iterator and yields SDK-native data models from it. This preserves useful SDK compatibility without coupling Molto to the SDK's connection and parsing internals.

Generated Responses event dictionaries are serialized to SSE and parsed through the same SDK event adapter as native records. Both native and translated events therefore expose SDK models and Molto's `as_sse()` helper.

## Transport and SSE framing

`BaseClient` centralizes OpenAI-compatible transport behavior:

- endpoint URL construction from a client-specific `DEFAULT_BASE_URL` or supplied `base_url`;
- optional bearer authentication;
- SSE record splitting;
- extraction of multi-line `data:` payloads.

Concrete clients declare their default base URL and own endpoint-specific request and response handling. Adapters never access `httpx`.

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

Responses fixtures are the canonical downstream output. Matched Chat fixtures preserve source-backed values and aligned delta boundaries so translation can be compared deterministically.

Parity differs by path:

- Responses to Responses asserts exact request bodies and exact `as_sse()` output;
- Chat Completions to Responses compares reconstructible semantics, validates event ordering and sequence numbers, and separately checks generated IDs.

## Current limitations and extension path

The current public path supports streaming text responses. Non-streaming collection, reasoning projection, tool calls, multimodal content, and additional providers are not complete cross-protocol implementations yet.

The normalized IR should ultimately be SDK-independent. The current `NormalizedRequest.tools` and `tool_choice` fields still reuse OpenAI SDK parameter types; this is transitional coupling and should be replaced with semantic tool types before additional downstream protocols depend on them.

To add another upstream protocol:

1. add a client that converts `NormalizedRequest` to the upstream request;
2. normalize its native stream into `ModelResponseStreamEvent`;
3. compose it from `Molto` without importing a downstream adapter into the client;
4. add cassette-backed parity tests against the downstream SDK contract.

To add another downstream protocol, add an adapter that normalizes its request and projects normalized events without performing transport.
