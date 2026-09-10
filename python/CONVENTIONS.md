# Conventions

## General style

- Keep implementations simple and flat.
- Follow the existing style in the file being changed.
- Prefer standard-library or existing project functionality over new abstractions.
- Use built-in collection types such as `list`, `dict`, `set`, and `tuple` in annotations. Use `typing` for constructs such as `Annotated`, `Literal`, `Protocol`, and `TypeVar`.
- Use single backticks for inline code references in Python docstrings, not double backticks.

## Layer ownership

`Molto` is the public orchestration facade. Protocol-specific methods such as `responses()`, `chat_completions()`, and `messages()` are coordinators: each validates its downstream request, selects the upstream path, and composes adapters with clients.

- Adapters translate the downstream protocol to and from normalized types. They must not perform HTTP requests.
- Upstream clients construct provider requests, invoke provider SDK transport, apply compatibility validation where required, and normalize provider output. They must not format downstream responses.
- Type modules define schemas and wire helpers. They must not select routes or instantiate clients.

A future `chat()` method is a dispatcher over those coordinators. It should select and delegate to a protocol method rather than duplicate that method's validation, normalization, composition, or projection logic.

Keep same-protocol optimizations visible in the relevant protocol coordinator. Do not hide pair-specific routing inside an adapter or client.

## Endpoint, provider, and transport

Keep these three concepts separate:

- `endpoint` selects the wire protocol and upstream client;
- `provider` identifies the serving implementation and selects its capability profile; and
- `base_url` selects only the HTTP destination.

Do not treat an endpoint or URL hostname as evidence about provider behavior. A coordinator may supply a documented canonical provider when the caller omits one, but an explicit provider must override that convenience default independently of the endpoint and base URL. Keep provider capabilities in the single `PROVIDER_CAPABILITIES` registry and pass the resolved profile explicitly to adapters and clients.

## Adapter and client base classes

Create a fresh `BaseAdapter[NativeRequestT, NativeEventT]` subclass instance for each translated request, specifying its concrete native request and event types. Call its public `normalize_request()` before `project_events()` so it can retain the validated downstream model as `native_request`. Implement protocol-specific translation in `_normalize_request()` and `_project_events()`, and translate known IR failures into downstream-native exceptions in `_handle_error()`. Re-raise errors without a native equivalent. Do not override the public lifecycle methods or bypass their state and error handling. Same-protocol passthrough composes the public `project_native_events()` so native failures receive the same downstream-native error projection.

Every upstream client must subclass `BaseClient[NativeRequestT, NativeEventT]`, specify its concrete native types, and implement its native path, normalized path, and error translation:

- `native_events()` accepts the validated upstream-native request, yields native SDK events, and delegates failures to `_handle_error()` so the native path shares the IR error normalization;
- `_events()` accepts a `NormalizedRequest` and yields normalized events, delegating transport to `native_events()`; and
- `_handle_error()` translates known upstream-native exceptions into IR exceptions and re-raises errors without an IR equivalent.

The public `events()` wrapper owns error interception; do not override it. Clients do not retain adapter request state. Pass normalized requests explicitly and keep downstream-native request handling in the adapter.

## Error translation

Normalize only upstream errors with a known protocol-independent meaning. An upstream client's `_handle_error()` must raise the normalized Molto exception from the original SDK exception. On translated routes, a downstream adapter must select the downstream SDK exception class from the original provider's preserved HTTP status, falling back to the downstream SDK's base `APIStatusError` when that status has no explicit mapping. Do not select the outward class from the normalized error type and do not translate the original status, message, `response`, or body. The normalized cause owns portable semantic classification; the outward SDK subclass classifies only the preserved HTTP status. On same-protocol passthrough, the adapter instead raises a fresh instance of the original SDK exception's class to preserve the native error type; never re-raise the cause instance itself, because chaining it `from` the normalized error would make the two exceptions each other's cause. Re-raise unknown errors unchanged.

Error translation is currently partial. Changing the exception into the downstream SDK namespace does not authorize inventing a target-native status, message, or body. Preserve the upstream status code, message, `response`, and `body` until the implementation has an explicit, documented, cassette-backed mapping for the complete downstream wire error. Callers can use the resulting cause chain for semantic classification even when providers assign different statuses to the same condition.

Document each concrete recognition rule in the architecture's [error translation table](ARCHITECTURE.md#error-translation). Cover every applicable native and translated route with provider-native error cassettes. Tests must assert the outward SDK namespace, upstream status/message/body preservation, normalized cause, and original upstream SDK cause.

## OpenAI SDK schemas

Use local Pydantic request wrappers when Molto needs a validated request model. Their field set must match the pinned SDK method signature, excluding transport arguments such as `timeout`, `extra_headers`, `extra_query`, and `extra_body`.

Use OpenAI SDK response models as aliases when the SDK already owns the wire schema. Do not copy or subclass them only to give the type a Molto module name. Tests should assert alias identity and schema parity:

```python
assert OpenAIResponsesResponse is Response
assert OpenAIChatCompletionsStreamChunk is ChatCompletionChunk
```

For the Responses event union, compare the wrapped union members with the SDK union rather than asserting that the event classes are Molto-owned.

Do not recreate or expose SDK infrastructure objects such as `AsyncStream`. Context-manage returned SDK streams for cleanup and yield SDK-native models through Molto's iterator.

The exact OpenAI SDK pin in `pyproject.toml` is the schema source of truth. When it changes, recheck request-signature parity, response aliases, event union members, field mappings, and cassette-backed parity.

Treat OpenAI Responses as the canonical behavioral reference when matching other protocols. Match their equivalent semantics and lifecycle behavior to Responses wherever the target protocol can represent them, while preserving genuine protocol-specific differences.

When an OpenAI-compatible provider has a documented or cassette-backed extension that the SDK omits, model the narrowest extension possible. For example, extend only assistant messages with `reasoning_content`; do not weaken every Chat message to an untyped dictionary.

Prefer concrete lists over SDK `Iterable` annotations in local compatibility models. Concrete lists validate nested content at request construction time and avoid lazy validation or iterator consumption while Pydantic evaluates union branches.

## Pydantic field metadata

Fields on `ExtraModel` classes must use `Annotated` with a Pydantic `Field`:

```python
content: Annotated[
    str,
    Field(description="The content of the prompt."),
]
```

Use `Field` descriptions for field documentation and keep defaults outside the `Annotated` type:

```python
kind: Annotated[
    Literal["text-content"],
    Field(description="Content type identifier as a discriminator."),
] = "text-content"
```

Keep all field default assignments outside `Annotated`; do not pass defaults such as `default=None` to `Field`.

Models used as API request bodies inherit from `ExcludeNoneModel`, which rejects unknown fields. Call `model_dump()` instead of `BaseModel.model_dump()` so unset and `None` values do not become unintended provider parameters.

The cross-protocol contract is: Molto is strict during input validation but lenient during output handling. Every accepted input field must be effected, and any output stemming from an accepted input set must be tolerated. Apply it through these rules:

- Normalized requests are closed and lossless-or-rejected.
- Normalized outputs are open to provider extensions.
- Retaining unknown output data does not authorize projecting it into another protocol.

Normalized request-only models inherit from `StrictModel` and reject unknown fields. The dual-use `Thinking`, `TextOutput`, and `ToolCall` models reject extras when they are validated through `InputItem`. Normalized output models inherit from `ExtraModel`; `extra_fields` is an internal escape hatch for provider-specific output, not a vendor wire field. Never blindly dump a normalized model into a provider response; project supported fields explicitly.

Raise `UnsupportedInput` for semantic failures encountered while normalizing a downstream request or constructing an upstream request. It subclasses `ValueError` for compatibility. Do not use it for malformed provider output, invalid event order, or output projection failures; those remain output-side validation errors.

## Normalized content and events

Use explicit semantic content types and discriminated unions. In particular, one user message is a `UserInput` containing ordered `TextInput` and `ImageUrlInput` parts, and assistant conversation history is `TextOutput`; do not encode roles as content-part metadata.

Streaming is the primitive. Upstream clients emit normalized lifecycle, part, and delta events; downstream adapters project those events. A single normalized event may produce multiple native lifecycle events.

When non-streaming support is added, derive it with a generic collector over the normalized event stream. Do not add a separate provider completion path that can drift from streaming behavior.

Preserve chunk boundaries when they are observable and covered by parity tests. Do not turn an empty role-establishment chunk into a text delta.

Store raw reasoning in `Thinking.content` and `ThinkingDelta.content_delta`. Store reasoning summaries in `Thinking.summary` and `ThinkingDelta.summary_delta`. Responses wire events label these representations explicitly; preserve that label and require it to match `ProviderCapabilities.thinking`. Chat Completions `reasoning_content` and Messages `thinking_delta` are ambiguous, so classify them with the same capability. Project only the matching normalized representation into each protocol's reasoning field.

A Responses reasoning lifecycle must use exactly one valid representation: raw reasoning content with no summary or signature, a reasoning summary accompanied by a signature with no raw content, or a signature with no reasoning summary or raw content. Preserve that representation across its start, delta, and end state; do not combine or switch representations during the part lifecycle. A completely empty start may remain deferred until its representation is known.

Chat Completions reasoning has no signature and may represent either raw reasoning or a summary according to the provider capability profile. Messages follows the same pairing rule as Responses: raw reasoning has no signature, a reasoning summary must have one, and a signature may appear without reasoning text. Enforce protocol-specific pairing at the relevant wire boundary rather than in the provider-neutral normalized models. Current cross-protocol paths through Chat therefore support only unsigned raw reasoning; reject summaries rather than discarding required opaque state.

Treat `ThinkingDelta.signature_delta` as a complete replacement snapshot of provider-specific opaque state. Never concatenate it with `Thinking.signature`.

Map prompt-cache intent according to the [shared cache policy](../docs/README.md#prompt-caching). Messages silently drops caller-defined cache keys without enabling caching, processes the remaining options independently, and rejects retention policies. Implicit mode maps OpenAI's `30m` minimum lifetime to Anthropic `1h`; document the longer retention and higher cache-write pricing. Preserve `5m` and `1h` on Messages and reject them on OpenAI. Explicit mode adds no automatic caching, so its lifetime has no effect without breakpoints. Never synthesize a positional breakpoint the normalized request cannot carry or reuse a cache key as an unrelated identifier.

Map usage semantically and construct the public usage payload explicitly. This prevents internal `extra_fields` containers and source-only usage details from leaking into the target protocol.

## Same-protocol passthrough

When downstream and upstream use the same protocol, preserve the native protocol instead of round-tripping through the normalized IR.

- Validate the request with that protocol's request model.
- Yield the SDK's native event or chunk models directly.
- Do not route a native event through the normalized IR.

The bypass belongs in the relevant coordinator; transport remains in the matching upstream client.

## OpenAI SDK transport

Construct each short-lived `AsyncOpenAI` wrapper with Molto's injected `httpx.AsyncClient`, the resolved API key, and an explicit supplied or default base URL. Resolve an absent vLLM API key to the fixed non-secret `EMPTY` credential, and suppress OpenAI organization and project environment headers for compatible providers.

Context-manage every returned SDK stream so completion and early termination release its response. Do not close the short-lived SDK wrapper because it borrows Molto's shared HTTP client; only `Molto.aclose()` closes that client.

Let the SDK own request serialization, authentication, HTTP, SSE decoding, terminal markers, retries, non-success responses, and OpenAI exception construction. Preserve the original SDK exception for unknown errors. Map only the known semantic errors covered by the documented error translation contract, on native and translated routes alike; do not add compatibility wrappers for the previous `httpx` errors.

Yield SDK Responses events and `ChatCompletionChunk` models directly to avoid duplicate serialization and validation on every event or delta. Before yielding a Responses completion event, normalize the documented compatible-provider omission where `cache_write_tokens` is `None` to `0`. Do not rewrite an accurate cassette merely to satisfy a newer SDK schema.

## Unsupported semantics

Reject unsupported roles, content parts, request modes, and event types explicitly. Never silently drop a field that changes request meaning. This rule is aspirational for the Messages client's stream loop: it inspects only the six handled `RawMessageStreamEvent` types and silently ignores any other variant (documented in the [Messages guide](../docs/protocols/anthropic-messages.md)), diverging from the Responses client, which raises for unknown stream events. Tightening the Messages path is tracked future work. Retaining a value in `extra_fields` is not evidence that an upstream protocol supports it.

For cross-protocol requests, a downstream adapter must map each populated native field into `NormalizedRequest`, apply it while projecting the downstream response, or reject it. An upstream client must map each populated normalized field into its native request or reject it when the target protocol cannot preserve the semantics. The documented omission exceptions are optional Responses output-item `id` values on Chat Completions and Messages, and caller-defined prompt-cache keys on Messages. Neither omission changes model-visible content; Messages computes cache identity implicitly and must not enable caching from a key alone. These exceptions do not apply to portable identifiers such as `ToolCall.tool_call_id`. Keep target-specific compatibility checks beside that client's request construction rather than in the provider-neutral normalized models.

Prefer semantic normalized types over SDK parameter types. Any SDK type retained inside the normalized IR is transitional coupling and should not spread to new adapters or clients.

## Matched cassettes

Read [`tests/cassettes/README.md`](../tests/cassettes/README.md) before adding or updating cassette fixtures. It distinguishes faithful provider recordings from deliberately edited matched cassettes and documents the encrypted-reasoning exception.

Matched cross-protocol cassettes use:

```text
tests/cassettes/matched/{scenario}/{provider}-{endpoint}-{mode}-{turn}.yaml
```

Examples:

```text
tests/cassettes/matched/simple_text/openai-responses-stream-00.yaml
tests/cassettes/matched/simple_text/openai-chat_completions-stream-00.yaml
tests/cassettes/matched/reasoning_raw/vllm-messages-stream-00.yaml
```

Matched cassettes represent semantically equivalent requests expressed in each protocol's native schema, not identical request bodies. When a behavior-changing setting has protocol-specific field names, include its equivalent in every matched request, even if some protocols make it optional. Retain fields required by a protocol; do not remove them merely because a counterpart protocol makes the equivalent optional. For example, match Anthropic `max_tokens` with Responses `max_output_tokens` and Chat Completions `max_completion_tokens`.

Treat each cassette as authoritative native wire data for its own downstream protocol. For cross-protocol matching, the Responses cassette is the canonical behavioral reference. Paired fixtures may align source-backed IDs, timestamps, model, usage, and delta boundaries, but each must remain valid native wire data.

When Responses exposes `added`, `done`, and terminal encrypted snapshots but Messages exposes only a start signature and a later delta, preserve `added` and `done`. Do not buffer a stream solely to retrofit the separately observed terminal ciphertext. A deliberately curated continuation may use the prior turn's `done` snapshot in both native request fixtures when testing this cross-protocol policy; document and test that exception explicitly.

For native passthrough, compare parsed SDK events. For translated streams, compare reconstructible semantics and separately validate target-protocol lifecycle invariants. Do not require metadata or chunk placement that the source protocol cannot represent without buffering.

Responses assistant-message `phase` is source-backed native metadata, not a default that can be recovered from a Chat Completions result. Preserve it on native Responses paths. For Chat-to-Responses event parity, remove it from both sides of the reconstructible comparison; never infer `commentary` or `final_answer` from reasoning presence, message order, or text content.

Never record cassettes during a normal test run. New cassettes must come from an approved recorded source. Moving or renaming a cassette must not alter its bytes.

## Implementing protocol features

Use the following workflow when adding a feature shared by Responses and one or more other protocols:

1. Record semantically equivalent Responses and counterpart protocol cassettes directly under `tests/cassettes/matched/` from approved official sources. Inspect their requests and events, verify that they contain no secrets, validate them against their native SDK types, and commit the unchanged recordings before modifying them.
2. Add or extend deterministic matching tooling when the current tooling cannot align the new scenario. Use Responses as the canonical input and modify only the non-Responses fixture fields needed to express equivalent native requests, align source-backed response values, and align observable chunk boundaries. Never copy a Responses request body into another protocol or rewrite transport metadata, and keep every transformed fixture valid native wire data.
3. Run the matching tooling on each non-Responses cassette paired with the canonical Responses cassette.
4. Review every resulting cassette diff before implementation. The diff must contain only intentional, representable alignment changes and must not remove meaningful native fields. If it does not, correct the matching logic and repeat the matching and review steps. Commit the reviewed matching-tool and cassette changes before implementation.
5. Add a failing interface test covering all applicable routes: same-protocol passthrough for every participating protocol and both translation directions between Responses and each counterpart protocol. Assert translated upstream request bodies exactly; response parity alone cannot prove that request features reached the upstream protocol.
6. Implement request schemas and native transport support until all same-protocol routes pass without crossing the normalized IR.
7. Add the smallest provider-neutral representation and the required downstream-adapter and upstream-client mappings until all cross-protocol routes pass. Explicitly reject semantics that a target protocol cannot represent, except for the documented omission policies above.

Schema-union note: a shared field name does not guarantee an identical nested shape. Validate each nested union before copying a shared-name field into another protocol. For example, Chat only emits non-empty `logprobs` when a request also sets `logprobs: true`, so copying `top_logprobs` alone is insufficient once structured-output or log-probability support is added.

Run all implementation tests in cassette playback mode. Actual translated output may be written to a temporary location for diagnosis, but it must never be promoted automatically to an expected fixture.

## Verification

Public protocol behavior requires cassette-backed parity coverage against parsed OpenAI SDK events.

Log probabilities are not yet transported through the normalized path. Until support lands, `python/molto/utils/test.py` drops empty `logprobs` from reconstructible Chat chunks behind a TODO, so tests must not assert a `logprobs` value; update the helper when log-probability support is implemented. IR and request-mapping unit tests are supplementary, not a replacement.

After changes, run focused replay tests, full `pytest`, and the formatting and lint commands from `.github/workflows/lint.yml`. Report pre-existing failures without changing unrelated files.
