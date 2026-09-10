# Molto Architecture

## Current scope

Molto is an LLM protocol translation library and the facade for its public coordinators. Its implemented `Molto.responses()`, `Molto.chat_completions()`, and `Molto.messages()` methods each accept one native protocol and stream that same protocol back to the caller.

Three upstream endpoints are implemented:

- `responses`: native Responses passthrough or cross-protocol translation through Responses;
- `chat/completions`: native Chat Completions passthrough or cross-protocol translation through Chat Completions;
- `messages`: native Anthropic Messages passthrough or translated through the normalized representation.

Only streaming requests are currently supported. Cross-protocol translation currently covers text, portable image input, raw reasoning, reasoning summaries between Responses and Messages, function tools, tool calls and results, conversation history, lifecycle metadata, and token usage. Because Chat Completions has no signature field, translated routes through Chat currently support only unsigned raw reasoning.

## Route configuration

`endpoint`, `provider`, and `base_url` are independent:

| Concept    | Purpose                                                                                                                       |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `endpoint` | Selects the upstream wire protocol and client: `responses`, `chat/completions`, or `messages`.                                |
| `provider` | Identifies the implementation serving that protocol and selects its `ProviderCapabilities`: `openai`, `anthropic`, or `vllm`. |
| `base_url` | Selects only the HTTP destination.                                                                                            |

Molto never derives provider behavior from a URL hostname. When `provider` is omitted, the coordinator uses the canonical implementation for the selected endpoint as a convenience default: OpenAI for Responses and Chat Completions, and Anthropic for Messages. An explicit provider overrides that default without changing the endpoint, while an explicit base URL changes neither the protocol nor the capability profile.

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

Cross-protocol translation follows one governing principle: Molto is strict during input validation but lenient during output handling. Every accepted input field must be effected, and any output stemming from an accepted input set must be tolerated. This yields three architectural rules:

- Normalized requests are closed and lossless-or-rejected.
- Normalized outputs are open to provider extensions.
- Retaining unknown output data does not authorize projecting it into another protocol.

There are three explicit behavior-neutral exceptions to lossless-or-rejected input handling on translated Responses routes. Chat Completions and Messages may omit an optional Responses output-item ID because it has no equivalent and does not affect model behavior. `include=["reasoning.encrypted_content"]` may be ignored because Molto already preserves any encrypted reasoning present in the response, and `store=false` may be ignored because it requests no provider-side state; `store=true` remains unsupported. Native Responses passthrough forwards these fields unchanged. Portable call identities such as `ToolCall.tool_call_id` remain lossless-or-rejected.

Every semantic rejection while normalizing or projecting a cross-protocol request raises `UnsupportedInput`, a `ValueError` subclass. Public protocol schema validation remains owned by the corresponding Pydantic request model and retains its native validation errors. Molto's local public request models inherit from `ExcludeNoneModel` and reject unknown fields. Output and event validation errors are not `UnsupportedInput`.

The following design rules continue to guide the implementation:

1. Normalize semantic concepts such as text, reasoning, tools, usage, and lifecycle state rather than mirroring a provider schema.
2. Treat streaming as canonical. Derive complete responses by collecting the normalized stream instead of maintaining a second execution implementation.
3. Keep adapters free of transport and clients free of downstream formatting.
4. Keep `Molto` thin: it selects and composes components but does not translate provider fields itself.
5. Preserve unsupported provider output through an explicit escape hatch rather than expanding the IR into a union of every provider schema.
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
| `providers.py` | Provider names, endpoint defaults, and immutable capability profiles | Wire-protocol schemas or HTTP destinations |
| `adapters/` | Downstream native request to normalized request; normalized events to downstream native events | HTTP transport |
| `upstream/` | Upstream request construction, provider SDK stream integration, compatibility validation where required, upstream response normalization | Downstream response formatting |
| `types/normalized.py` | Provider-neutral request content, response parts, deltas, lifecycle metadata, and usage | Transport or SDK lifecycle objects |
| `types/openai_*.py` | OpenAI request schemas, SDK-native result aliases, and narrow compatibility validators | Cross-protocol orchestration |

Adapters and upstream clients do not depend on each other. Protocol-specific coordinator methods on `Molto` compose them.

`BaseAdapter[NativeRequestT, NativeEventT]` owns the lifecycle of translated downstream requests and preserves the concrete downstream request and event types in its interface. A coordinator creates a fresh adapter for each request and calls `normalize_request()` before `project_events()`. Normalization retains the exact validated downstream model as `native_request`, allowing event projection to consult native request fields without adding them to the normalized representation. Event projection fails if normalization has not established that state. The public projection path catches failures raised while consuming or projecting normalized events and delegates them to `_handle_error()` so known IR errors can be expressed as downstream-native exceptions. Same-protocol passthrough skips request normalization and event projection, but still composes the adapter's `project_native_events()` around the native stream so failures reach the same `_handle_error()` contract.

`BaseClient[NativeRequestT, NativeEventT]` preserves the concrete upstream-native types across both streaming paths. `native_events()` accepts a validated request in the upstream protocol, yields that protocol's native SDK events, and delegates failures to `_handle_error()`. The public `events()` path accepts a `NormalizedRequest`, yields `ModelResponseStreamEvent` values from `_events()`, and delegates failures to the same `_handle_error()` so native and translated routes share IR error normalization before downstream projection. Concrete clients own both transformations and the provider transport required by those methods.

## Error translation

Upstream clients normalize recognized SDK failures as `APIException`; downstream adapters project them into their SDK namespace. Both native and translated routes use this pipeline. See [Errors](../docs/errors.md) for recognition rules, status mappings, and special cases.

## Orchestration facade and coordinators

`Molto` is expected to expose one coordinator method per downstream protocol:

```text
Molto
  |-- responses()         implemented
  |-- chat_completions()  implemented
  |-- messages()          implemented
  `-- chat()              future dispatcher
```

Callers can use a protocol coordinator directly when they need that protocol's native request and SDK-compatible response types. Each coordinator owns its own downstream contract: it validates the native request, selects and composes an upstream client, and projects results back into the same downstream protocol.

A future `Molto.chat()` will be a higher-level dispatcher. It will select one of the underlying protocol coordinators and delegate to it; it should not duplicate their normalization, transport composition, or projection logic. The selection policy may use explicit caller configuration, request type, or capabilities, but that policy is not defined yet.

Each implemented coordinator performs the following work:

1. Validate the public body as its protocol request model.
2. Reject non-streaming requests until collection is implemented.
3. Resolve `provider` and its capability profile independently of `base_url`.
4. Resolve the API key and injected or default HTTP client.
5. Select the upstream path from `endpoint`.
6. Return an async iterator of the same downstream protocol's SDK event models.

Every coordinator also accepts request-scoped `extra_body` and `extra_headers` transport options. They bypass downstream request normalization and are applied by the selected provider SDK after upstream request projection. Before forwarding extra headers, the facade removes client credentials, account selectors, connection-specific fields, request framing, and proxy topology headers; the provider SDK then constructs its own authentication and transport headers. This preserves Molto's credential boundary while allowing application-specific headers and provider body extensions.

The returned `MoltoStream` exposes a copy of the successful upstream response headers as `response_headers`. Coordinator prefetching opens the SDK stream and captures those headers before returning, including on translated routes. Molto retains all response headers; an HTTP server using Molto remains responsible for filtering connection-specific and framing headers before constructing its downstream response.

Each coordinator deliberately chooses between a native and translated pipeline. This is a downstream protocol decision, not logic that belongs in an adapter or transport client.

`Molto` owns the injected or default `httpx.AsyncClient`. Short-lived `AsyncOpenAI` wrappers borrow that client and are not closed independently. Each returned SDK stream is context-managed so its response is released on completion or early termination; `Molto.aclose()` remains the only operation that closes the shared HTTP client.

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

Molto validates the request and relies on the SDK's native event models. Passthrough means no IR translation or duplicate model validation, not untyped proxying. Only streaming failures still cross the IR: the upstream client normalizes a recognized error and the downstream adapter re-projects it, giving native passthrough the same error cause chain as a translated route.

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
      +-- project exact completed and incomplete terminal states
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

`reasoning_content` is a provider extension to the OpenAI Chat schema, but the field itself does not distinguish raw reasoning from a reasoning summary. Molto models the extension only on assistant messages and uses `ProviderCapabilities.thinking` to classify it. OpenAI classifies it as a summary and vLLM as raw reasoning. Responses events already distinguish `response.reasoning_text.*` from `response.reasoning_summary_*`; their explicit event types remain authoritative. Classification does not make summaries portable through Chat: Responses and Messages require a signature for completed summaries, while Chat has no signature field.

## Normalized request model

`NormalizedRequest` holds portable request semantics. The implemented adapter normalizes:

- string input;
- text messages with `system`, `developer`, `user`, and `assistant` roles;
- raw reasoning and reasoning summaries in requests and assistant history;
- function tool definitions, tool selection, calls, and successful tool results;
- ordered mixed text/image user messages, including remote URL and data-URL images;
- role-less instructions; each instruction content part produces one `SystemPrompt`, with sequence position preserved (only the Messages projection hoists them into one top-level `system` string);
- model and streaming configuration;
- output-token limits, parallel-tool-call setting, reasoning effort, temperature, and `top_p`.

Assistant conversation history uses `TextOutput`, not a flag on `TextInput`. This keeps role semantics explicit when the Chat client reconstructs messages.

Native request projection always expresses user text as a protocol content-part list: Responses uses `input_text`, Chat Completions uses `text`, and Anthropic Messages uses a `text` block. In the normalized representation, one `UserInput` corresponds to exactly one user message and each child `TextInput` corresponds to one ordered text part. Adapters and clients must not merge or split either boundary. Provider-specific part discriminators remain at the wire boundary.

Tool results follow the same canonical-part rule. `ToolReturn.content` is an ordered list of `TextInput` and `ImageUrlInput` parts. Adapters wrap a native scalar result in one text part. Responses and Messages project both part types directly. Chat Completions tool messages accept only text content, so its upstream client leaves text in the tool message and relocates images into subsequent user messages. It emits every tool response for an assistant's tool-call block contiguously before those user messages.

User images follow the same ordered-part rule: a normalized `ImageUrlInput` holds a remote URL or data URL and projects to a Responses `input_image` part, a Chat Completions `image_url` part, or an Anthropic Messages `image` block. Chat Completions cannot represent the `original` detail value and Anthropic has no fidelity control at all, so those projections reject an unrepresentable `detail` rather than dropping it.

Responses `phase` is intentionally not normalized or synthesized. Native passthrough preserves a provider-supplied phase, but a translated stream or reconstructed assistant-history item omits it because the normalized text/reasoning sequence does not establish whether an assistant message was commentary or a final answer.

Normalized request models are strict. `NormalizedRequest` and every request-only model nested beneath it reject unknown fields. The output models inherit from `ExtraModel` and retain unknown provider fields in `extra_fields` as an escape hatch. The dual-use `Thinking`, `TextOutput`, and `ToolCall` models remain lenient when handling output but reject extras when validated as conversation input.

The IR must not become a lowest-common-denominator provider API. Concepts with clear cross-provider meaning belong in normalized fields. A downstream adapter must map every populated native input field into that representation, apply it while projecting downstream output, or reject it before it is lost. An upstream client must likewise map every populated normalized field into its target protocol or reject it explicitly, with the sole item-ID exception described below. Genuinely provider-specific output remains in `extra_fields` until a portable semantic model exists.

Known normalized fields are subject to the same rule except for Responses output-item IDs. These IDs are distinct from portable tool-call IDs and do not affect model behavior: Responses preserves them when reconstructing Responses history, while Chat Completions and Messages omit them because neither protocol has an equivalent input field. Portable identifiers such as `ToolCall.tool_call_id` remain required and mapped. This is the only input field Molto may accept without effecting on every target protocol.

Reasoning effort is the normalized control for Anthropic thinking. The Messages adapter records `output_config.effort` in `NormalizedRequest.reasoning`; the Anthropic client then selects adaptive thinking with explicit effort or manual extended thinking with a preset token budget according to the target model's capabilities. Thinking mode and `budget_tokens` remain provider projections rather than new IR fields.

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

For the text, reasoning, and function-tool paths:

- `ResponseStartEvent` carries the upstream response ID, creation time, model, and service tier;
- `PartStartEvent` establishes a `Thinking`, `TextOutput`, or `ToolCall` snapshot at a stable output index; native protocols normally start empty, but an initial function-argument fragment is preserved when present;
- each source reasoning, text, or function-argument chunk becomes one matching `PartDeltaEvent`;
- `PartEndEvent` carries the complete reasoning, text, or function call accumulated from the deltas;
- `ResponseEndEvent` carries terminal metadata, a provider-neutral termination reason, an optional matched stop sequence, and normalized usage.

A reasoning part with no content, summary, signature, or deltas carries nothing to translate; upstream clients omit its part lifecycle from the stream rather than emitting empty `Thinking` bookends, and surviving parts are reindexed so normalized indices stay dense. Recorded providers never produce such parts.

Chat's role chunk starts the first part but its empty content is not emitted as a delta. Reasoning and text may interleave, so each transition between them — in either direction — ends the active part and starts a new one. A finish reason ends the active part. Usage can arrive in a later chunk, so the response ends only after the upstream SDK stream has been consumed.

Chat does not expose an end marker for each indexed tool call. Molto treats calls as contiguous: a new call index closes the preceding call, and a later delta cannot resume a closed index. This keeps normalized, Responses and Messages content-block lifecycles balanced without buffering calls.

The normalized representation preserves the distinction that ambiguous wire protocols omit: `Thinking.content` and `ThinkingDelta.content_delta` contain raw reasoning, while `Thinking.summary` and `ThinkingDelta.summary_delta` contain a summary. Chat Completions `reasoning_content` and Messages `thinking_delta` are classified by the resolved provider capability profile. Every downstream adapter accepts the normalized representation selected by that profile; Chat Completions and Messages then emit it through their protocol's single reasoning field, while Responses retains the distinction with separate reasoning-text and reasoning-summary lifecycles.

Opaque reasoning state is represented as snapshots, not text. `Thinking.signature` carries the state observed at part start or part end, and `ThinkingDelta.signature_delta` replaces the prior snapshot instead of appending to it. Responses exposes three encrypted snapshots for a reasoning item (`output_item.added`, `output_item.done`, and terminal response output), while Messages exposes only a thinking-block start signature and a later signature delta. Molto preserves the first two by mapping `added` to part start and `done` to the replacement delta; a translated terminal Responses item reuses the `done` snapshot rather than buffering the stream to reproduce a third ciphertext.

Service-tier labels are retained as upstream metadata, but target protocols do not share one vocabulary. Translated Anthropic Messages responses therefore use the target default `standard`, while translated Responses and Chat Completions responses use `default`. Native passthrough remains untouched. The normalized termination reason maps shared semantics explicitly: normal completion, output limits, stop sequences, tool calls, content filtering, pauses, refusals, and context-window exhaustion. When the target protocol cannot represent a normalized reason, its adapter deliberately coerces it to normal completion (`end_turn`, `stop`, or `response.completed`) so the translated stream remains consumable.

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

Clients should not gain a separate completion implementation. A generic collector keeps streaming and non-streaming normalization consistent. Until that collector and normalized response model exist, all implemented coordinators reject non-streaming requests explicitly.

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

The adapter owns Responses sequence numbers, output/content indices, item state, and generated message item IDs. It preserves the Chat response ID, creation time, model, delta boundaries, and mapped usage. It emits the Responses target-default service tier rather than relabeling the Chat source tier.

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

Two members of the Responses event union are intentionally replaced rather than aliased: `OpenAIResponseCompletedEvent` normalizes a `null` `cache_write_tokens` to `0`, and `OpenAIResponseFunctionCallArgumentsDoneEvent` relaxes the SDK's required `name` on the function-call-arguments-done event. Assert these unions by comparing wrapped members with the SDK union rather than asserting identity.

Molto does not recreate or expose the OpenAI SDK's transport-bearing `AsyncStream` runtime object. It context-manages the SDK stream internally and exposes a Molto-owned async iterator that yields SDK-native Responses events and Chat Completions chunks directly.

Generated protocol event dictionaries are validated directly with SDK model adapters.

## OpenAI SDK transport

Each OpenAI upstream client constructs a short-lived `AsyncOpenAI` wrapper with:

- Molto's shared `httpx.AsyncClient`;
- the resolved API key;
- an explicit supplied or default base URL, so SDK environment settings cannot redirect the request.

An absent vLLM key resolves to the fixed non-secret `EMPTY` credential. OpenAI organization and project environment settings are suppressed so compatible-provider requests cannot leak OpenAI account metadata.

The SDK owns request serialization, authentication, HTTP, SSE decoding, terminal markers, retries, error types, and native model construction for both protocols. Recognized errors on every route follow the [error translation](../docs/errors.md) contract; unknown SDK errors pass through unchanged. Before directly yielding a Responses completion event, Molto narrowly normalizes the compatible-provider omission where `cache_write_tokens` is `None` to `0`.

SDK streams are context-managed, including when iteration ends early. The short-lived SDK wrapper is not closed because doing so would close Molto's injected HTTP client; `Molto.aclose()` owns that client's lifetime. Adapters do not perform HTTP requests; they use `httpx` only to construct the compatibility response required to project local credential failures as SDK authentication errors.

## Replay and parity testing

Tests are playback-only. Normal runs must never record or rewrite cassettes.

Matched cross-protocol fixtures use a scenario-first layout under `tests/cassettes/matched/{scenario}/`. The [full scenario listing](../tests/cassettes/README.md#matched-cassettes) lives in the cassette README; this section only illustrates the layout:

```text
tests/cassettes/matched/
  simple_text/
    anthropic-messages-stream-00.yaml
    openai-chat_completions-stream-00.yaml
    openai-responses-stream-00.yaml
  reasoning_raw/
    vllm-messages-stream-00.yaml
    vllm-chat_completions-stream-00.yaml
    vllm-responses-stream-00.yaml
  tool_call/
    anthropic-messages-stream-00.yaml
    openai-chat_completions-stream-00.yaml
    openai-responses-stream-00.yaml
```

The naming pattern is:

```text
tests/cassettes/matched/{scenario}/{provider}-{endpoint}-{mode}-{turn}.yaml
```

Each fixture is canonical for its own downstream protocol. Matched pairs preserve source-backed values and aligned delta boundaries so translation can be compared deterministically. Responses-only output metadata remains valid native fixture data even when Chat Completions cannot reproduce it. In particular, Chat has no assistant-message `phase` equivalent, so translated event comparison omits that field instead of inferring it from reasoning or text position.

Shared translation policies live in the [documentation overview](../docs/README.md). Protocol-specific contracts live in the [Responses](../docs/protocols/openai-responses.md), [Chat Completions](../docs/protocols/openai-chat-completions.md), and [Anthropic Messages](../docs/protocols/anthropic-messages.md) guides.

Parity differs by path:

- same-protocol routes assert exact request bodies and parsed SDK stream events;
- Chat Completions to Responses compares reconstructible Responses events and validates event ordering, sequence numbers, generated IDs, and function-call identity;
- Responses to Chat Completions compares ordered text, reasoning, and function-argument deltas, finish state, and semantic usage without requiring unavailable fingerprints or identical usage-chunk placement.

## Current limitations and extension path

The current public paths support streaming text, portable image input, raw reasoning, reasoning summaries between Responses and Messages, and client-defined function tools. Routes through Chat Completions currently support only unsigned raw reasoning. Function-tool support includes definitions, selection, parallel-call control, successful results, conversation history, and streamed JSON arguments. Hosted and server tools, failed-result semantics outside Messages, non-streaming collection, other multimodal content, and additional provider capabilities are not complete cross-protocol implementations yet. The matched Responses/Messages fixtures also map the reconstructible opaque reasoning snapshots under the assumption that OpenAI serves both endpoints; opaque state is not portable across providers, the implementation does not enforce its issuer, and the separately recorded terminal Responses ciphertext is not byte-reconstructible through Messages.

Function definitions and tool selection use provider-neutral normalized types. Protocol-specific function schemas and tool-choice shapes remain at their adapters and clients.

Full boundary fidelity is a future-work target, not a current guarantee. Reaching it would require: (1) an instruction item that retains its original role, content-part grouping, and sequence position; (2) a native assistant-message container holding ordered thinking, text, and tool parts; (3) an explicit policy for native features that a target protocol cannot represent; and (4) a model- or renderer-level capability profile distinct from the endpoint schema. The current translated representation approximates these, so cross-protocol boundaries are documented in the [comparison guide](../docs/protocols/comparison.md).

To add another upstream protocol:

1. add a client that converts `NormalizedRequest` to the upstream request;
2. normalize its native stream into `ModelResponseStreamEvent`;
3. compose it from `Molto` without importing a downstream adapter into the client;
4. add cassette-backed parity tests against the downstream SDK contract.

To add another downstream protocol, add an adapter that normalizes its request and projects normalized events without performing transport.
