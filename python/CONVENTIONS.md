# Conventions

## Layer ownership

`Molto` is the public orchestration facade. Protocol-specific methods such as `responses()`, `chat_completions()`, and future `messages()` are coordinators: each validates its downstream request, selects the upstream path, and composes adapters with clients.

- Adapters translate the downstream protocol to and from normalized types. They must not perform HTTP requests.
- Upstream clients construct provider requests, own transport and SSE parsing, and normalize provider output. They must not format downstream responses.
- Type modules define schemas and wire helpers. They must not select routes or instantiate clients.

A future `chat()` method is a dispatcher over those coordinators. It should select and delegate to a protocol method rather than duplicate that method's validation, normalization, composition, or projection logic.

Keep same-protocol optimizations visible in the relevant protocol coordinator. Do not hide pair-specific routing inside an adapter or client.

## OpenAI SDK schemas

Use local Pydantic request wrappers when Molto needs a validated request model. Their field set must match the pinned SDK method signature, excluding transport arguments such as `timeout`, `extra_headers`, `extra_query`, and `extra_body`.

Use OpenAI SDK response models as aliases when the SDK already owns the wire schema. Do not copy or subclass them only to give the type a Molto module name. Tests should assert alias identity and schema parity:

```python
assert OpenAIResponsesResponse is Response
assert OpenAIChatCompletionsStreamChunk is ChatCompletionChunk
```

For the Responses event union, compare the wrapped union members with the SDK union rather than asserting that the event classes are Molto-owned.

Do not recreate SDK infrastructure objects such as `AsyncStream`. Molto owns its async iterator and yields SDK-native data models.

The exact OpenAI SDK pin in `pyproject.toml` is the schema source of truth. When it changes, recheck request-signature parity, response aliases, event union members, field mappings, and cassette-backed parity.

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

Models used as API request bodies inherit from `ExcludeNoneModel`. Call `model_dump()` instead of `BaseModel.model_dump()` so unset and `None` values do not become unintended provider parameters.

Normalized models inherit from `ExtraModel`. `extra_fields` is an internal escape hatch for provider-specific values, not a vendor wire field. Never blindly dump a normalized model into a provider request or response; project supported fields explicitly.

## Normalized content and events

Use explicit semantic content types and discriminated unions. In particular, assistant conversation history is `TextOutput`; do not encode its role in `TextInput.metadata`.

Streaming is the primitive. Upstream clients emit normalized lifecycle, part, and delta events; downstream adapters project those events. A single normalized event may produce multiple native lifecycle events.

When non-streaming support is added, derive it with a generic collector over the normalized event stream. Do not add a separate provider completion path that can drift from streaming behavior.

Preserve chunk boundaries when they are observable and covered by parity tests. Do not turn an empty role-establishment chunk into a text delta.

Map usage semantically and construct the public usage payload explicitly. This prevents internal `extra_fields` containers and source-only usage details from leaking into the target protocol.

## Same-protocol passthrough

When downstream and upstream use the same protocol, preserve the native protocol instead of round-tripping through the normalized IR.

- Validate the request with that protocol's request model.
- Parse every SSE record into that protocol's SDK event model.
- Retain the original record for `event.as_sse()`.
- Do not parse and reserialize a native event when exact passthrough is promised.

The bypass belongs in the relevant coordinator; transport remains in the matching upstream client.

## SSE clients

Reuse `BaseClient` for endpoint construction, authorization headers, SSE record splitting, and `data:` extraction. Each concrete client must define `DEFAULT_BASE_URL` unless it requires a caller-supplied URL.

An upstream client must raise on an unsuccessful HTTP response before yielding events. It must ignore protocol terminal markers such as Chat Completions `data: [DONE]` rather than trying to validate them as event models.

If a compatible provider omits a newly required SDK field, default only that field for typed access and retain the untouched source record in `as_sse()`. Do not rewrite an accurate cassette merely to satisfy a newer SDK schema.

## Unsupported semantics

Reject unsupported roles, content parts, request modes, and event types explicitly. Never silently drop a field that changes request meaning. Retaining a value in `extra_fields` is not evidence that an upstream protocol supports it.

Prefer semantic normalized types over SDK parameter types. Any SDK type retained inside the normalized IR is transitional coupling and should not spread to new adapters or clients.

## Matched cassettes

Matched cross-protocol cassettes use:

```text
tests/cassettes/matched/{scenario}/openai_{endpoint}-{mode}-{turn}.yaml
```

Examples:

```text
tests/cassettes/matched/simple_text/openai_responses-stream-00.yaml
tests/cassettes/matched/simple_text/openai_chat_completions-stream-00.yaml
```

Treat each cassette as canonical for its own downstream protocol. Paired fixtures may align source-backed IDs, timestamps, model, usage, and delta boundaries, but each must remain valid native wire data.

For native passthrough, compare exact parsed events and preserve exact SSE records. For translated streams, compare reconstructible semantics and separately validate target-protocol lifecycle invariants. Do not require metadata or chunk placement that the source protocol cannot represent without buffering.

Never record cassettes during a normal test run. New cassettes must come from an approved recorded source. Moving or renaming a cassette must not alter its bytes.

## Verification

Public protocol behavior requires cassette-backed parity coverage against parsed OpenAI SDK events. IR and request-mapping unit tests are supplementary, not a replacement.

After changes, run focused replay tests, full `pytest`, and the formatting and lint commands from `.github/workflows/lint.yml`. Report pre-existing failures without changing unrelated files.
