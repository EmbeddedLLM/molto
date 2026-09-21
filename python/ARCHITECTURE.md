# Molto Architecture

## Current scope

Molto is an LLM protocol translation library and the facade for its public coordinators. Its implemented `Molto.responses()`, `Molto.chat_completions()`, and `Molto.messages()` methods each accept one native protocol and return that same protocol to the caller.

Three upstream endpoints are implemented:

- `responses`: native Responses passthrough or cross-protocol translation through Responses;
- `chat/completions`: native Chat Completions passthrough or cross-protocol translation through Chat Completions;
- `messages`: native Anthropic Messages passthrough or translated through the normalized representation.

Native same-protocol and cross-protocol routes support streaming and non-streaming requests. Cross-protocol routes cover text, portable image input, raw reasoning, reasoning summaries between Responses and Messages, function tools, tool calls and results, conversation history, prompt-cache intent, lifecycle metadata, and token usage. Because Chat Completions has no signature field, translated routes through Chat currently support only unsigned raw reasoning.

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

- Normalized requests are closed and lossless-or-rejected, with documented [cache-lifetime approximations](../docs/README.md#prompt-caching).
- Normalized outputs are open to provider extensions.
- Retaining unknown output data does not authorize projecting it into another protocol.

Documented exceptions allow omission at protocol boundaries. Chat Completions and Messages may omit optional Responses output-item IDs because they have no equivalent and do not affect model behavior. Messages silently drops caller-defined prompt-cache keys because Anthropic computes cache identity implicitly; dropping a key does not enable caching, and remaining cache options are processed independently. Translated Responses routes may ignore `include=["reasoning.encrypted_content"]` because Molto already preserves any encrypted reasoning present in the response, and `store=false` because it requests no provider-side state; `store=true` remains unsupported. Native passthrough forwards these fields unchanged. Responses preserves output-item IDs, including on marked assistant history. Both OpenAI protocols preserve cache keys. Portable call identities such as `ToolCall.tool_call_id` remain lossless-or-rejected.

Every semantic rejection while normalizing or projecting a cross-protocol request raises `UnsupportedInput`, a `ValueError` subclass. Public protocol schema validation remains owned by the corresponding Pydantic request model and retains its native validation errors. Molto's local public request models inherit from `ExcludeNoneModel` and reject unknown fields. Output and event validation errors are not `UnsupportedInput`.

The following design rules continue to guide the implementation:

1. Normalize semantic concepts such as text, reasoning, tools, usage, and lifecycle state rather than mirroring a provider schema.
2. Treat streaming as canonical. Derive complete translated responses by projecting the normalized stream into downstream-native events and collecting those events instead of maintaining a second execution implementation.
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
| `adapters/` | Downstream native request to normalized request; normalized events to downstream-native events; projected native events to a complete downstream-native response | HTTP transport |
| `upstream/` | Upstream request construction, provider SDK stream integration, compatibility validation where required, upstream response normalization | Downstream response formatting |
| `types/normalized.py` | Provider-neutral request content, response parts, deltas, lifecycle metadata, and usage | Transport or SDK lifecycle objects |
| `types/openai_*.py` | OpenAI request schemas, SDK-native result aliases, and narrow compatibility validators | Cross-protocol orchestration |

Adapters and upstream clients do not depend on each other. Protocol-specific coordinator methods on `Molto` compose them.

`BaseAdapter[NativeRequestT, NativeEventT, NativeResponseT]` owns the lifecycle of translated downstream requests and preserves the concrete downstream request, event, and response types in its interface. A coordinator creates a fresh adapter for each translated request and calls `normalize_request()` before `project_events()` or `project_response()`. Normalization retains the exact validated downstream model as `native_request`, allowing projection to consult native request fields without adding them to the normalized representation. Projection fails if normalization has not established that state. `project_events()` translates normalized output into downstream-native events; `project_response()` consumes that projected native event stream to build the complete downstream-native response. The public projection paths catch failures raised while consuming or projecting output and delegate them to `_handle_error()` so known IR errors can be expressed as downstream-native exceptions. Same-protocol passthrough skips request and output translation, but composes `project_native_events()` or `project_native_response()` around native output so failures reach the same `_handle_error()` contract.

`BaseClient[NativeRequestT, NativeEventT, NativeResponseT]` preserves the concrete upstream-native types. `native_events()` accepts a validated streaming request in the upstream protocol and yields that protocol's native SDK events, while `native_response()` accepts a validated non-streaming request and returns its native SDK response. Both native paths delegate failures to `_handle_error()`. The public `events()` path accepts a `NormalizedRequest`, yields `ModelResponseStreamEvent` values from `_events()`, and delegates failures to the same `_handle_error()` so native and translated routes share IR error normalization before downstream projection. Concrete clients own these transformations and the provider transport required by those methods.

## Error translation

Every route uses two stages for errors with a known protocol-independent meaning:

```text
upstream SDK exception
        |
        v
normalized APIException
        |
        v
downstream SDK exception
```

A translated route applies the first stage in the upstream client and the second in the downstream adapter around the normalized event stream. A same-protocol route composes the same two stages around the native event stream or response call, so callers classify errors through the same cause chain regardless of the route taken.

Normalized protocol-independent API errors inherit from `APIException`. The currently recognized subclasses are `AuthenticationError`, `ModelNotFoundError`, and `ResourceNotFoundError`. Upstream clients recognize them as follows:

| Normalized exception    | Upstream protocol       | Upstream SDK exception     | Recognition rule                                      |
| ----------------------- | ----------------------- | -------------------------- | ----------------------------------------------------- |
| `AuthenticationError`   | all                     | protocol `APIStatusError`  | status `401`                                          |
| `ModelNotFoundError`    | OpenAI Responses        | `openai.APIStatusError`    | status `400` and `code == "model_not_found"`          |
| `ModelNotFoundError`    | OpenAI Chat Completions | `openai.APIStatusError`    | status `404` and `code == "model_not_found"`          |
| `ModelNotFoundError`    | Anthropic Messages      | `anthropic.APIStatusError` | status `404` and an error message containing `model:` |
| `ResourceNotFoundError` | all                     | protocol `APIStatusError`  | status `404`                                          |

On translated routes, downstream adapters select the outward SDK class from the original provider's HTTP status. They do not select it from the normalized semantic error and do not translate the HTTP error contract. The status, message, `response`, and body retain their original provider values and shape; only the SDK namespace used to expose that preserved status changes.

| Downstream SDK | Status `400`      | Status `401`          | Status `404`    | Other statuses   |
| -------------- | ----------------- | --------------------- | --------------- | ---------------- |
| OpenAI         | `BadRequestError` | `AuthenticationError` | `NotFoundError` | `APIStatusError` |
| Anthropic      | `APIStatusError`  | `AuthenticationError` | `NotFoundError` | `APIStatusError` |

For example, a missing model reported by Anthropic becomes an OpenAI `NotFoundError` because the original response has status `404`, while retaining Anthropic's body. The reverse route through Messages from Responses preserves the OpenAI status `400` and body and therefore uses Anthropic's base `APIStatusError`. Callers must inspect the normalized exception in the cause chain for the portable semantic category; the outward subclass classifies only the preserved status.

Callers that need the semantic category should inspect the exception chain instead of relying only on status codes or provider-specific bodies:

```text
downstream SDK exception
        caused by normalized Molto exception
        caused by upstream SDK exception
```

This preserves the native provider evidence while exposing a stable semantic classification on every route. A same-protocol route raises a fresh instance of the upstream exception's own class, so native passthrough preserves the exact SDK error type. The original SDK exception remains the innermost cause on every route. Unknown errors are re-raised unchanged because Molto has no justified normalized equivalent.

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
2. For a non-streaming request, return the upstream SDK response directly on a same-protocol route or collect downstream-native projected events on a translated route.
3. Resolve `provider` and its capability profile independently of `base_url`.
4. Resolve the API key and injected or default HTTP client.
5. Select the upstream path from `endpoint`.
6. Return either an async iterator of downstream SDK event models or the downstream SDK response model.

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

Molto validates the request and relies on the SDK's native output models. Passthrough means no IR translation or duplicate model validation, not untyped proxying. Recognized failures still cross the IR on both streaming and non-streaming native paths: the upstream client normalizes the error and the downstream adapter re-projects it, giving native passthrough the same error cause chain as a translated route.

The Chat Completions path follows the same shape through `Molto.chat_completions(endpoint="chat/completions")` and `OpenAIChatCompletionsClient.native_events()`. The SDK consumes `[DONE]` as a transport terminator, and Molto exposes the yielded `ChatCompletionChunk` models directly without duplicating SDK validation work.

For a non-streaming native request, the coordinator instead calls the matching client's `native_response()` method and returns the SDK's `Response`, `ChatCompletion`, or `Message` model directly. This path retains native fidelity without introducing normalized response collection.

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
- role-less instructions; each instruction content part produces one `SystemPrompt`, with sequence position preserved (only the Messages projection hoists them into top-level `system`, preserving separate blocks when any system block is marked);
- model and streaming configuration;
- output-token limits, parallel-tool-call setting, reasoning effort, temperature, and `top_p`;
- prompt-cache intent: an affinity key, breakpoint mode, cache lifetime, retention policy, and positional markers on supported content parts.

Assistant conversation history uses one strict `AssistantInput` per source message, containing ordered `Thinking`, `TextOutput`, and `ToolCall` parts and an optional Responses message ID. Standalone Responses reasoning and function-call items remain top-level `Thinking` and `ToolCall` values. `TextOutput` is an assistant part, not a top-level request item. This retains both the enclosing message boundary and its ordered content. `AssistantInput.id` supplies the history message ID; IDs on nested `TextOutput` parts are retained output metadata and are not used when projecting history.

Native request projection always expresses user text as a protocol content-part list: Responses uses `input_text`, Chat Completions uses `text`, and Anthropic Messages uses a `text` block. In the normalized representation, one `UserInput` corresponds to exactly one user message and each child `TextInput` corresponds to one ordered text part. Adapters and clients must not merge or split either boundary. Provider-specific part discriminators remain at the wire boundary. Assistant text parts follow the same rule: multiple text parts remain in one message, and adjacent assistant messages remain separate. Messages can retain any supported assistant part order. Within one explicit assistant message, Chat accepts runs of reasoning, text, and tool-call parts in that order, with any run optional. Adjacent reasoning parts concatenate in order with a newline separator; text parts retain their array boundaries and markers, and tool calls retain distinct array entries and IDs. Chat rejects reasoning after text or calls and text after calls rather than splitting or reordering the message. This grouping never merges separate explicit assistant messages. Responses has standalone reasoning and call items, so projecting a mixed assistant message emits contiguous text runs as messages interleaved with those items.

Responses and Chat Completions also project every normalized instruction as a `developer` message with one text part, whether or not it has a cache breakpoint. Markers remain attached only to their marked parts.

Tool results follow the same canonical-part rule. `ToolReturn.content` is an ordered list of `TextInput` and `ImageUrlInput` parts. Adapters wrap a native scalar result in one text part. Responses and Messages project both part types directly. Chat Completions tool messages accept only text content, so its upstream client leaves text in the tool message and relocates images into subsequent user messages. It emits every tool response for an assistant's tool-call block contiguously before those user messages, and rejects cache markers whose prefix that relocation would change.

User images follow the same ordered-part rule: a normalized `ImageUrlInput` holds a remote URL or data URL and projects to a Responses `input_image` part, a Chat Completions `image_url` part, or an Anthropic Messages `image` block. Chat Completions cannot represent the `original` detail value and Anthropic has no fidelity control at all, so those projections reject an unrepresentable `detail` rather than dropping it.

Responses `phase` is intentionally not normalized or synthesized. Native passthrough preserves a provider-supplied phase, but a translated stream or reconstructed assistant-history item omits it because the normalized text/reasoning sequence does not establish whether an assistant message was commentary or a final answer.

Normalized request models are strict. `NormalizedRequest` and every request-only model nested beneath it reject unknown fields. The output models inherit from `ExtraModel` and retain unknown provider fields in `extra_fields` as an escape hatch. The dual-use `Thinking`, `TextOutput`, and `ToolCall` models remain lenient when handling output but reject extras when validated as conversation input.

The IR must not become a lowest-common-denominator provider API. Concepts with clear cross-provider meaning belong in normalized fields. A downstream adapter must map every populated native input field into that representation, apply it while projecting downstream output, or reject it before it is lost. An upstream client must likewise map every populated normalized field into its target protocol or reject it explicitly, with the documented omission exceptions and [cache-lifetime approximations](../docs/README.md#prompt-caching). Genuinely provider-specific output remains in `extra_fields` until a portable semantic model exists.

Responses output-item IDs are distinct from portable tool-call IDs and do not affect model behavior: Responses preserves them when reconstructing history, including marked assistant text. Chat Completions and Messages omit them because neither protocol has an equivalent input field. Native passthrough preserves IDs unchanged. Portable identifiers such as `ToolCall.tool_call_id` remain required and mapped. The other omission exception is the caller-defined cache key on Messages, described below.

Reasoning effort is the normalized control for Anthropic thinking. The Messages adapter records `output_config.effort` in `NormalizedRequest.reasoning`; the Anthropic client then selects adaptive thinking with explicit effort or manual extended thinking with a preset token budget according to the target model's capabilities. Thinking mode and `budget_tokens` remain provider projections rather than new IR fields.

`NormalizedRequest.prompt_cache` holds request-wide `PromptCache` intent. A strict `CacheBreakpoint` with optional `ttl` lives on `SystemPrompt`, `TextInput`, `ImageUrlInput`, and `TextOutput` as `cache_breakpoint`. `NormalizedRequest.iter_cache_breakpoints()` traverses those markers in input order, including nested user, assistant, and tool-result content. Adapters normalize supported native markers; clients project them as OpenAI content-part `prompt_cache_breakpoint` or Anthropic block `cache_control`.

OpenAI marker presence resolves omitted mode to `implicit` and omitted lifetime to `30m`. Messages maps that minimum to `1h` on both marked blocks and top-level automatic caching; explicit mode omits the top-level control while retaining block markers. Anthropic block-only caching normalizes to explicit mode, while top-level caching selects implicit mode. OpenAI maps every explicit request or block lifetime to request-wide `30m`, retaining modes and marker positions. This rounds `5m` up but weakens a requested `1h` minimum; mixed block lifetimes collapse to one value. Omitted lifetimes remain unspecified unless another explicit lifetime sets that value. Retention is not synthesized to approximate a minimum lifetime. The [shared cache policy](../docs/README.md#prompt-caching) documents these default and pricing differences. Messages silently drops keys without enabling caching and rejects retention policies.

Projection validates prefix constraints at the target boundary. Messages forwards all supported markers and leaves breakpoint-count limits to the upstream. It rejects conflicting final explicit/automatic lifetimes and one-hour markers after five-minute markers. It also rejects any mid-conversation instruction when markers are present; marked system blocks remain separate. Chat rejects marked tool returns affected by image relocation, including deferred images from earlier parallel tool returns. Responses attaches assistant markers to `output_text` parts and preserves optional output-item IDs. Anthropic outer tool-result, tool-use, thinking, and tool-definition markers remain unsupported because OpenAI has no corresponding content-part marker.

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

Streaming remains the canonical translated execution interface. The cross-protocol non-streaming path is:

```text
YClient.events()
      |
      v
NormalizedEvent stream
      |
      v
XAdapter.project_events()
      |
      v
Downstream-native event stream
      |
      v
XAdapter.project_response()
      |
      v
Downstream-native response
```

The coordinator sets the upstream normalized request to streaming even when the downstream request is non-streaming. This keeps upstream normalization identical for both translated modes. It then passes the resulting normalized events through the same downstream `project_events()` implementation used by a streaming request. Only after translation does the protocol-specific `project_response()` aggregate the projected native events.

Responses returns the inner `Response` from the terminal `response.completed` or `response.incomplete` event. Chat Completions folds projected choice deltas into one assistant message, reconstructs indexed tool calls, and applies the terminal finish reason and usage. Messages accumulates the projected message and content-block lifecycle, including text, thinking, signatures, tool input, terminal stop information, and usage, into one SDK `Message`.

Translated upstream clients do not gain a separate completion implementation, and adapters do not bypass `project_events()` to construct a response directly from normalized events. This keeps streaming and non-streaming translation aligned. Same-protocol non-streaming requests bypass both the normalized and projected-event paths: the matching client's `native_response()` returns the upstream SDK response directly.

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

Compatible Responses providers can return reasoning items and assistant output messages without IDs and accept those items as subsequent-turn input. The pinned SDK marks both IDs as required, so Molto uses two narrow Pydantic history models with optional IDs and eagerly validated fields. Assistant `output_text` also accepts `prompt_cache_breakpoint`, which the live OpenAI API supports but the pinned SDK schema omits. It does not use an untyped dictionary fallback for Responses input.

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

The SDK owns request serialization, authentication, HTTP, SSE decoding, terminal markers, retries, error types, and native model construction for both protocols. Native passthrough and unknown translated errors preserve the OpenAI exception rather than wrapping it as the previous `httpx` behavior; translated errors with a known semantic equivalent follow the [error translation](#error-translation) contract. Before directly yielding a Responses completion event, Molto narrowly normalizes the compatible-provider omission where `cache_write_tokens` is `None` to `0`.

SDK streams are context-managed, including when iteration ends early. The short-lived SDK wrapper is not closed because doing so would close Molto's injected HTTP client; `Molto.aclose()` owns that client's lifetime. Adapters never access `httpx`.

Closure propagates through the complete translated pipeline: adapter wrappers close the projector and its normalized input, client wrappers close the normalizer, and normalizers context-manage their native iterators. This releases the SDK response deterministically on collection failure, cancellation, or explicit early closure without closing the shared HTTP client.

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

- same-protocol routes assert exact request bodies and parsed SDK stream events or non-streaming responses;
- Chat Completions to Responses compares reconstructible Responses events and validates event ordering, sequence numbers, generated IDs, and function-call identity;
- Responses to Chat Completions compares ordered text, reasoning, and function-argument deltas, finish state, and semantic usage without requiring unavailable fingerprints or identical usage-chunk placement.

## Current limitations and extension path

The current public paths support native and translated streaming and non-streaming responses for text, portable image input, raw reasoning, reasoning summaries between Responses and Messages, and client-defined function tools. Routes through Chat Completions currently support only unsigned raw reasoning. Function-tool support includes definitions, selection, parallel-call control, successful results, conversation history, and streamed JSON arguments. Hosted and server tools, failed-result semantics outside Messages, other multimodal content, and additional provider capabilities are not complete yet. The matched Responses/Messages fixtures also map the reconstructible opaque reasoning snapshots under the assumption that OpenAI serves both endpoints; opaque state is not portable across providers, the implementation does not enforce its issuer, and the separately recorded terminal Responses ciphertext is not byte-reconstructible through Messages.

Function definitions and tool selection use provider-neutral normalized types. Protocol-specific function schemas and tool-choice shapes remain at their adapters and clients.

Full boundary fidelity is a future-work target, not a current guarantee. Beyond the implemented assistant-history container, reaching it would require: (1) an instruction item that retains its original role, content-part grouping, and sequence position; (2) generated-output containers that retain native message and content indexes; (3) an explicit policy for native features that a target protocol cannot represent; and (4) a model- or renderer-level capability profile distinct from the endpoint schema. The current translated representation approximates these, so cross-protocol boundaries are documented in the [comparison guide](../docs/protocols/comparison.md).

To add another upstream protocol:

1. add a client that converts `NormalizedRequest` to the upstream request;
2. normalize its native stream into `ModelResponseStreamEvent`;
3. compose it from `Molto` without importing a downstream adapter into the client;
4. add cassette-backed parity tests against the downstream SDK contract.

To add another downstream protocol, add an adapter that normalizes its request and projects normalized events without performing transport.
