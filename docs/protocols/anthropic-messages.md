# Anthropic Messages

Anthropic Messages is both a public protocol that applications can use with Molto and an upstream protocol that Molto can target. This guide describes the Messages-specific wire contract and both translation directions. For routing, the normalized representation, shared parity classes, termination policy, and provider-renderer behavior, start with the [documentation overview](../README.md).

## At a glance

| Concern             | Anthropic Messages in Molto                   |
| ------------------- | --------------------------------------------- |
| Public method       | `Molto.messages()`                            |
| Upstream endpoint   | `endpoint="messages"`                         |
| Default provider    | `anthropic`                                   |
| Request model       | `AnthropicMessagesCreateRequest`              |
| Stream result       | Anthropic SDK `RawMessageStreamEvent` objects |
| Non-stream response | Anthropic SDK `Message`                       |
| Native transport    | `AsyncAnthropic.messages.create()`            |
| Supported modes     | Streaming and non-streaming                   |

The method always accepts an Anthropic Messages request. It returns Anthropic Messages events for a streaming request and an Anthropic SDK `Message` for a non-streaming request. `endpoint` selects the upstream wire protocol independently of `provider` and `base_url`.

## When to use this protocol

Use `Molto.messages()` when the application already speaks Anthropic Messages or depends on Anthropic SDK event and response types. The upstream can still be an Anthropic Messages, OpenAI Responses, or OpenAI Chat Completions endpoint.

Use `endpoint="messages"` from any Molto protocol method when the provider exposes the Messages wire format. When the method is also `messages()`, Molto uses native passthrough. Otherwise it translates through Molto's normalized representation.

## Quickstart

### Native Messages passthrough

With `ANTHROPIC_API_KEY` set, the following request is validated and sent through the Anthropic SDK without protocol translation:

```python
import asyncio

from molto import Molto


async def main() -> None:
    async with Molto() as molto:
        async with await molto.messages(
            body={
                "model": "claude-sonnet-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
            endpoint="messages",
            provider="anthropic",
        ) as stream:
            async for event in stream:
                print(event)  # Native Anthropic SDK event


asyncio.run(main())
```

The returned stream is an async context manager. If it is not used with `async with`, close the stream with `await stream.aclose()` and close Molto with `await molto.aclose()`.

For a non-streaming result, set `stream=False` or omit it and await the same method normally. A native route returns the provider's SDK `Message` directly. A translated route runs the selected upstream protocol as a stream internally, projects Anthropic message events, and accumulates them into the returned `Message`.

### Application speaks Messages, provider speaks Responses

Keep the Messages request and result contract, and change only the upstream route:

```python
stream = await molto.messages(
    body={
        "model": "gpt-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True,
    },
    endpoint="responses",
    provider="openai",
)
```

The provider receives a Responses request, while the application still iterates Anthropic SDK `RawMessageStreamEvent` objects.

### Provider exposes Messages

Select Messages as the endpoint from another public protocol method:

```python
stream = await molto.responses(
    body={
        "model": "claude-sonnet-5",
        "input": "Hello!",
        "max_output_tokens": 1024,
        "stream": True,
    },
    endpoint="messages",
    provider="anthropic",
)
```

Here the provider receives a Messages request and the application receives OpenAI Responses events. The downstream method, not the upstream endpoint, determines the returned SDK types.

## Native wire contract

### Requests and conversation turns

A Messages request contains an ordered `messages` array and an optional top-level `system` prompt. Each message contains a string or an ordered list of content blocks. A string and a single text block are shorthand forms with the same meaning.

Messages does not have a `developer` role. Molto's request model accepts `user`, `assistant`, and `system` message roles. The distinction between wire messages and model-visible turns matters:

- Anthropic combines consecutive `user` messages into one conversational turn and does the same for consecutive `assistant` messages.
- A final `assistant` message is a response prefill, so the final role changes generation behavior.
- Ordered blocks remain part of one wire message even when their types differ.

Molto can preserve separate user wire objects during translation, but it cannot make Anthropic treat consecutive same-role objects as separate model turns.

### Top-level and mid-conversation system prompts

The top-level `system` field is not a conversation message. Selected Anthropic models also support `system` messages within `messages`, subject to these placement rules:

- A mid-conversation system message cannot be first.
- It must immediately follow a user message, or an assistant message that ends in a server-tool result.
- It must be last or immediately followed by an assistant message.
- It cannot appear between a tool use and its tool result.
- Consecutive system messages form one section.
- A later system section overrides earlier instructions for following turns.

Molto's local request model validates the `system` role but does not enforce these placement rules or check that the selected model supports the feature. Native providers remain responsible for those checks.

### Streaming lifecycle

A native stream describes one generated assistant message:

```text
message_start
content_block_start(index)
content_block_delta(index)*
content_block_stop(index)
...
message_delta
message_stop
```

Text, thinking, and tool-use blocks have stable indexes. `message_delta` carries terminal usage and stop information. A `ping` event is a transport keepalive, not model output.

The native stream may start a block with content already present in the `content_block_start` snapshot. Deltas represent only later additions; code must not assume that all blocks start empty.

## Native passthrough

`Molto.messages(endpoint="messages")` validates the body with `AnthropicMessagesCreateRequest` and calls `AsyncAnthropic.messages.create()`. It yields the Anthropic SDK's native events directly for a streaming request and returns the SDK's native `Message` directly for a non-streaming request. Neither path crosses Molto's normalized representation.

This path retains native content variants, event variants, provider service tiers, cache metadata, `tool_use.caller`, and other fields the Anthropic SDK supports. Wire-level `ping` keepalives are filtered by the SDK before events are yielded, so callers never observe them on this path either. Errors follow the [shared policy](../errors.md). The request is still validated rather than passed through as an untyped HTTP proxy.

## When the application speaks Messages

When `Molto.messages()` targets `responses` or `chat/completions`, the `AnthropicMessagesAdapter` converts the Messages request to normalized content and projects the upstream normalized event stream back into Anthropic SDK events. For a translated non-streaming request, those projected events are consumed by `AnthropicMessagesAdapter.project_response()` instead of being returned to the application. The upstream call still runs in streaming mode, so both translated result modes share the same normalized and projected events.

### Request normalization

The adapter currently applies these rules:

- A string message becomes one text block.
- One user message becomes one normalized user input. Ordered text and image blocks remain ordered, and remote URL and base64 image sources remain distinguishable.
- Top-level system text is placed before conversation input. Each system text block and each mid-conversation system text block becomes a role-less system prompt. The distinction between top-level and inline system instructions is lost.
- An assistant message becomes one `AssistantInput` with ordered thinking, text, and direct client tool-call parts. Its enclosing boundary and text-part markers are retained.
- A `tool_result` becomes a tool return with ordered text or image content. `is_error=true` becomes a failed outcome.
- `tool_use` history preserves its ID, name, and JSON input. Only direct caller metadata is translatable; server or other caller types are rejected. Reconstructed request-history `tool_use` blocks carry only `id`, `name`, and `input`; `caller` is omitted.
- Custom tool definitions preserve name, description, input schema, and strict validation. Unsupported custom-tool fields and non-custom tools are rejected.
- `tool_choice` preserves `auto` and `none`, maps `any` to required use, and maps a named `tool` to a named function. `disable_parallel_tool_use` becomes the inverse `parallel_tool_calls` value.
- `max_tokens`, model, stream mode, temperature, `top_p`, and normalized reasoning effort are retained.
- Top-level `cache_control` becomes the normalized prompt-cache intent: its presence selects provider-chosen (implicit) breakpoints, and its `ttl` carries the requested lifetime.

Within a user message, tool results must precede ordinary user content for the current normalized path. A failed Messages tool result is representable inside Molto, but OpenAI request histories have no equivalent failed outcome, so a Messages-to-OpenAI route rejects it rather than silently changing its meaning.

`cache_control` on system text, user text/images, assistant text, and nested tool-result text/images becomes a positional `CacheBreakpoint` with optional lifetime. Block-only caching selects normalized explicit mode; top-level `cache_control` selects implicit mode while preserving block markers. Markers on outer `tool_result`, `tool_use`, thinking blocks, and tool definitions are rejected on translated routes because OpenAI has no corresponding content-part marker. Native passthrough preserves these provider-native fields.

The request wrapper also accepts `container`, `inference_geo`, `metadata`, `service_tier`, `stop_sequences`, `top_k`, and `user_profile_id`. These fields are available in native passthrough but are not forwarded across the normalized path. Within `output_config`, only `effort` is normalized.

### Messages event projection

The adapter projects normalized events as one Anthropic message with indexed content blocks:

- A response start produces `message_start`. Because terminal usage is not yet available, its counts are zero and its translated service tier is `standard`.
- A part start produces `content_block_start` for thinking, text, or a client `tool_use`. Non-empty thinking and text snapshots are preserved in that start event. Initial tool arguments are emitted immediately as an `input_json_delta` because a tool-use start requires an object.
- Each reasoning, text, or partial-JSON delta becomes a matching indexed `content_block_delta`.
- A part end produces `content_block_stop`.
- A response end produces `message_delta` with authoritative terminal usage and stop information, followed by `message_stop`.

Translated output never synthesizes `ping`. Translated `tool_use` blocks use the target default `caller: {type: direct}` because normalized function calls do not carry Messages caller metadata.

### Messages non-streaming response projection

`AnthropicMessagesAdapter.project_response()` consumes the already projected Anthropic SDK event stream. It initializes the response from `message_start`, appends indexed blocks from `content_block_start`, folds text, thinking, and citation deltas into their blocks, replaces a thinking signature from `signature_delta`, and assembles partial JSON into each client or server tool invocation input. It validates content-block lifecycle and index order, and requires tool input to be a JSON object. When generation stops within tool arguments, collection preserves the complete JSON values available so far, matching the Anthropic SDK's partial JSON parsing. A final `message_delta` supplies stop information and authoritative usage before `message_stop` confirms that the accumulated SDK `Message` is complete.

The response projector does not consume normalized events directly. Because it uses the output of `project_events()`, the non-streaming message retains the same projected block order, target-default service tier, content, reasoning, tool calls, stop mapping, and semantic usage as the streaming route.

## When the provider exposes Messages

When another Molto method targets `endpoint="messages"`, the `AnthropicMessagesClient` constructs a Messages request from normalized input, uses the Anthropic SDK for transport, and converts native Messages events into the shared streaming lifecycle.

### Request construction

The client applies these Messages-specific rules:

- Each normalized user input becomes one user message. Ordered text and image parts become ordered `text` and `image` blocks.
- Base64 data URLs become Anthropic base64 image sources and remote URLs become URL sources. Messages has no image-fidelity control, so an explicit non-default image `detail` is rejected.
- Each explicit assistant input becomes one assistant message with ordered thinking, text, and tool-use blocks. Adjacent explicit assistant messages stay separate. Standalone Responses reasoning may accompany the following text message, and following standalone function calls may join that message; those items have no explicit shared container.
- Adjacent tool returns are grouped into one user message containing `tool_result` blocks. Successful and failed outcomes map directly; `is_error=true` represents failure. Denied and interrupted outcomes are rejected because Messages cannot preserve them.
- Tool parameters become `input_schema`; required use becomes `any`; and parallel-call control becomes the inverse `disable_parallel_tool_use` value.
- Tool choice has edge rules: a required or named choice with no tools raises; when only parallelism is set, `disable_parallel_tool_use` rides on an explicit `auto` choice; the flag is omitted when tool use is disabled; and `tool_choice` is dropped entirely when no tools are present.
- Implicit normalized cache intent becomes top-level automatic `cache_control: {type: ephemeral}`. Lifetimes `5m` and `1h` pass through; OpenAI's `30m` minimum maps to `1h`. OpenAI marker presence resolves an omitted lifetime to `30m` before projection; without markers, omission uses Anthropic's five-minute default. A caller-defined cache key is silently dropped without enabling caching or discarding the remaining options. Explicit mode omits top-level `cache_control` but preserves marked blocks, which inherit the request lifetime unless they specify their own. Retention policies are rejected.

Messages requires an integer `max_tokens`. Responses and Chat Completions make their corresponding output-token limit optional, but a translated route targeting Messages must provide it so Molto can construct a valid Messages request. If the downstream request omits the limit, validation fails before transport.

Normalized system prompts are projected into top-level `system`. If any system block is marked, the client keeps all system blocks separate and projects their markers. Otherwise it joins instruction text with blank lines. Instruction role and enclosing message boundaries remain unavailable in the normalized representation.

Without markers, mid-conversation instructions are hoisted, losing their original placement and later-instruction priority. If any marker is present, the client rejects mid-conversation instructions because hoisting would change the cached prefix. A native Messages route does not hoist or merge system content through the normalized representation.

For example, translating either Responses or Chat to Messages without cache markers collects both instruction roles into the top-level `system` field:

```text
Source sequence: system A → user → assistant → developer B → user
Messages request:
  system = "A\n\nB"
  messages = [user, assistant, user]
```

The instructions retain their order relative to each other, but `B` loses its position after the assistant message. The `system` field is separate from the `messages` array. With any cache marker present in the request, this translation is rejected.

### Stream normalization

The client preserves the response ID, model, and supported block order. Each emitted block keeps a stable index across its lifecycle. Text delta boundaries, thinking delta boundaries, tool call identity, and every partial-JSON delta are also preserved. Anthropic does not provide a creation timestamp, so Molto captures local wall-clock time when it observes `message_start` for downstream protocols that require one.

`thinking_delta` does not say whether its text is raw reasoning or a summary. Molto classifies it using the explicit provider capability profile: Anthropic and OpenAI profiles use summaries, while vLLM uses raw reasoning. The endpoint or base URL does not select this behavior.

A thinking block that starts with text or a signature begins immediately so the observed snapshot cannot be lost. A completely empty thinking start is deferred until a non-empty thinking or signature delta establishes content. If the block stays empty, Molto omits it and assigns dense normalized indexes to the remaining blocks. Native `ping` events are ignored on this semantic path.

Only the six handled raw events are inspected here; any other `RawMessageStreamEvent` variant is silently ignored rather than rejected. This matches the fact that `ping` is transport-only, but it diverges from `python/CONVENTIONS.md`'s "reject unsupported event types explicitly" rule and from the Responses client, which raises for unknown stream events. The silent-ignore is a documented current behavior, not a deliberate contract to rely on.

Only directly invoked `tool_use` output is normalized. A server-owned caller is rejected because normalized function-call history has no equivalent caller metadata.

## Semantic mappings

### Request fields

The normalized representation is the composition point, so these mappings are directional rather than pair-specific:

| Messages concept | Normalized meaning | Messages projection |
| --- | --- | --- |
| `system` or system message | System prompt | Top-level `system`; separate blocks when any system block is marked |
| User `text` block | Ordered text input | User `text` block |
| User `image` block | Ordered image input | Base64 or URL `image` block |
| Assistant `text` block | Assistant text history | Assistant `text` block |
| Assistant `thinking` block | Raw reasoning or summary plus opaque state | Assistant `thinking` block |
| `max_tokens` | Output-token limit | `max_tokens` |
| Custom tool `input_schema` | Function parameters | Custom tool `input_schema` |
| `tool_use` | Function call | `tool_use` |
| `tool_result` | Tool return and outcome | `tool_result`, with `is_error` for failure |
| `tool_choice.type=any` | Required tool use | `tool_choice.type=any` |
| `disable_parallel_tool_use` | Inverse parallel-call control | Inverted on projection |
| Top-level `cache_control` | Prompt cache with implicit breakpoints | Top-level `cache_control` |
| Supported text/image block `cache_control` | Positional `CacheBreakpoint` with optional lifetime | Block `cache_control` |

User text and tool-result text are always projected in explicit content-block lists. One normalized user input remains one Messages user message, and its ordered parts are not merged or split by Molto.

### Thinking modes and effort

Reasoning effort is the portable control. For known Claude model IDs, Molto uses the installed Anthropic model profile to select adaptive thinking or manual extended thinking. Non-Claude model IDs on Messages-compatible endpoints default to adaptive thinking, as covered by the cross-protocol fixtures. In addition, current and future adaptive Claude families are force-treated as adaptive and `xhigh`-capable even when the pinned model table has no entry yet: `claude-mythos-preview` and any `claude-{fable,haiku,mythos,opus,sonnet}-<major>=5+` model ID.

When the application speaks Messages, an inbound thinking block is normalized as summarized reasoning: Molto sets `reasoning.summary="detailed"` on the normalized request for both adaptive and `enabled` manual modes.

| Normalized effort | Adaptive `output_config.effort`         | Manual `budget_tokens` |
| ----------------- | --------------------------------------- | ---------------------- |
| `none`            | Thinking disabled                       | Thinking disabled      |
| `minimal`         | `low`                                   | 1,024                  |
| `low`             | `low`                                   | 2,048                  |
| `medium`          | `medium`                                | 10,000                 |
| `high`            | `high`                                  | 16,384                 |
| Omitted           | `high`                                  | 10,000                 |
| `xhigh`           | `xhigh` when supported, otherwise `max` | 32,768                 |
| `max`             | `max`                                   | 32,768                 |

Adaptive requests always use summarized display and an explicit effort. A manual `budget_tokens` value must be less than `max_tokens`; Molto rejects the request before transport when the budget is invalid. Reasoning history enables thinking when the current request omits reasoning configuration, using high effort for adaptive models or the default 10,000-token manual budget.

Inbound `thinking_delta` classification is a separate concern from the outbound adaptive-versus-manual model policy. A Messages thinking output lifecycle contains one of:

- raw reasoning without a signature;
- a reasoning summary with an optional signature; or
- a signature without reasoning text.

It must not mix or switch representations within one block. Output processing accepts a missing summary signature and projects it as an empty Messages signature without inventing continuation state. Request normalization preserves unsigned summary history for upstream validation. Chat upstreams accept the summary text, while Responses and Messages upstream builders reject it before transport. Official Anthropic continuation requires the original Anthropic-issued signature. Molto's local request model also permits an absent signature for compatible providers such as vLLM; this does not make unsigned thinking history valid against Anthropic's official API.

### Opaque thinking state

A Messages thinking block has an initial `signature` and may later receive a `signature_delta`. The delta is a complete replacement snapshot, not text to append to the initial value: the Anthropic SDK replaces the block's initial signature when it processes a `signature_delta`. Molto stores and reuses the replacement value for block completion and subsequent history.

OpenAI Responses exposes three encrypted snapshots where Messages has only two slots:

| Responses snapshot       | Messages slot                                      |
| ------------------------ | -------------------------------------------------- |
| Output item added        | Thinking-block start `signature`                   |
| Output item done         | Replacement `signature_delta`                      |
| Terminal response output | No independent Messages slot; reuse the done value |

Molto does not buffer the whole stream to reproduce the third ciphertext. This mapping is valid only under an explicit same-provider contract in which the provider defines both fields as the same opaque continuation state. The matched fixtures make that assumption because OpenAI serves both endpoints. Molto currently does not track or enforce the issuer and will mechanically project the value. Callers must not reuse translated continuation state across providers: an Anthropic-issued signature is not valid OpenAI `encrypted_content`, and an OpenAI-issued value is not valid for continuation against Anthropic merely because the wire fields look similar.

### Termination and service tier

The [shared termination table](../README.md#termination) defines every route. For Messages, normal completion is `end_turn`, output exhaustion is `max_tokens`, a matched stop sequence is `stop_sequence`, and a function call ends with `tool_use`. The matched sequence is retained separately. A source reason with no Messages equivalent is deliberately coerced to `end_turn` so the stream remains consumable; the source distinction is then lost.

Messages request tier values (`auto` and `standard_only`) are distinct from response values (`standard`, `priority`, and `batch`) and have no portable OpenAI equivalent. Following the [shared service-tier policy](../README.md#service-tiers), native passthrough preserves the provider value, every translated Messages result uses `standard`, and request `service_tier` is not forwarded.

### Prompt caching

Anthropic derives cache identity from the matching tools, system, and message prefix. Molto silently drops OpenAI `prompt_cache_key` without enabling caching and independently processes options and content markers; `prompt_cache_retention` is rejected. Top-level `cache_control` normalizes to implicit mode, while block-only caching normalizes to explicit mode. Supported text/image block markers retain their position and optional lifetime.

OpenAI markers resolve an omitted mode to `implicit` and omitted lifetime to `30m`. Messages maps that minimum to `1h` for both marked blocks and automatic caching; explicit mode omits only the automatic control. This selects longer retention and cache-write pricing of 2x the base input-token price, compared with 1.25x for five minutes. Without markers, implicit mode with no lifetime uses Anthropic's five-minute default. Explicit `5m` and `1h` lifetimes pass through to Messages. On OpenAI targets, all explicit request and block lifetimes map to one request-wide `30m` TTL, preserving mode and marker positions. This rounds `5m` up but weakens `1h` to a thirty-minute minimum; mixed block lifetimes lose their distinctions. Omitted lifetimes remain unspecified unless another explicit lifetime supplies the shared TTL. This approximation follows the [shared cache policy](../README.md#prompt-caching).

Messages forwards all supported markers and leaves breakpoint-count limits to the upstream. Conflicting automatic and final explicit lifetimes are rejected, and one-hour markers must precede five-minute markers. Mid-conversation instructions are rejected whenever any marker is present. Marked system blocks stay separate. Markers on outer tool-result, tool-use, thinking, or tool-definition objects remain provider-native. See [Prompt caching](../README.md#prompt-caching) for the complete mapping and provider references.

### Usage and cache accounting

Anthropic reports uncached input, cache creation, and cache reads separately. As defined by the [shared usage policy](../README.md#usage), normalized input is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. Cache reads map to cached input, cache creation maps to cache writes, and `thinking_tokens` maps to reasoning output.

When projecting to Messages, uncached `input_tokens` is recovered by subtracting cache-read and cache-write counts from normalized total input. Terminal usage is authoritative. Filling `message_start` with those final counts would require buffering the complete upstream response, so translated Messages starts with zero counts and reports the mapped totals in `message_delta`.

Cache-creation TTL breakdowns, `inference_geo`, and `server_tool_use` have no current normalized equivalents and are not synthesized. `stop_details` and `container` are likewise unavailable on translated results.

## Errors

Recognized errors use the Anthropic SDK namespace and the [shared error contract](../errors.md). A Responses upstream's [mid-stream context-length failure](../errors.md#responses-failures-during-streaming) becomes a plain `anthropic.APIError`; only [missing SDK credentials](../errors.md#missing-sdk-credentials) receive a synthetic `401`.

## Fidelity

### Preserved on translated routes

- One user wire message per normalized user input, with ordered supported text and image parts.
- Supported thinking, text, and direct tool-use block ordering, with a stable index for every emitted block lifecycle.
- Response ID, model, text and reasoning delta boundaries, call IDs, tool names, and partial JSON argument deltas.
- Successful tool results, failed tool results when the target can represent them, stop sequences, and semantic cache usage.
- Automatic caching and supported text/image block markers, with `30m` minimum lifetime mapped to `1h` under the shared policy.

### Generated, defaulted, or coerced

- Translated Messages results use service tier `standard`.
- Translated `message_start` usage begins at zero; terminal usage is authoritative.
- Translated tool-use output receives `caller: {type: direct}`.
- Downstream protocols that require a creation time receive a local timestamp captured at `message_start`.
- Stop reasons without a Messages equivalent become `end_turn`.

### Inherently unavailable or lost

- Messages has no `developer` instruction role or image detail control.
- Responses output-item IDs are accepted in normalized history but omitted when Messages is the target; portable tool-call IDs remain mapped.
- Anthropic can merge consecutive same-role wire messages and interprets a final assistant message as prefill.
- Responses has no shared assistant container for reasoning, text messages, and calls, so reconstructing those standalone items cannot recover an original Messages container. Generated-output normalization also does not retain arbitrary assistant-message boundaries.
- Unmarked translation may hoist instructions and lose placement, priority, and block boundaries. Marked system blocks stay separate; requests with any markers and mid-conversation instructions are rejected.
- Caller-defined prompt-cache keys are silently dropped because Anthropic computes cache identity implicitly; dropping a key does not enable caching.
- Markers on outer tool-result, tool-use, thinking, and tool-definition objects have no supported OpenAI equivalent.
- `ping` is transport-only and is not projected across protocols.
- Provider-owned signatures are not portable across providers.

## Hosted web search

Declare `{"type":"web_search_20250305","name":"web_search"}` in `tools`. Translation to Responses supports allowed domains, approximate location, and search budgets; explicit blocked domains are rejected.

With Responses upstream, Messages receives server-tool blocks. `Molto.messages(..., citation_policy="buffer")` also preserves URL citations with reduced source metadata by withholding every text part until completion, including uncited parts. `"auto"` buffers only when the normalized request declares hosted search or carries search or citation history; otherwise it streams text without citations. Opaque `extra_body` extensions do not select automatic buffering. The default `"drop"` always streams text without citations. All policies preserve request-history attribution; native Messages passthrough is unaffected. With Chat upstream, the policy is a no-op: text streams immediately with unchanged chunk boundaries. See [Hosted web search](../web-search.md) for mappings, placeholders, and continuation limits.

When Messages is the translated upstream and the provider is Anthropic, missing, empty, or `"N/A"` encrypted search-source or citation state raises `UnsupportedInput` before transport. Preserve the original state through the Responses metadata extension or explicitly choose text-only history. Native Messages continues to forward history for provider validation; other provider profiles retain reduced-history support.

## Current implementation limitations

Cross-protocol Messages support includes streaming and non-streaming requests for the explicitly mapped text, portable image input, thinking, custom function tools, direct tool calls, successful tool results, limited failed-result paths, conversation history, lifecycle metadata, termination, and usage semantics described above.

A route targeting Messages must supply a downstream output-token limit because Messages requires `max_tokens`. Unsigned reasoning summary output can pass through Chat Completions, but present signatures cannot because Chat has no signature slot. Unsigned summary history can be translated to a Chat upstream; Responses and Messages upstream builders still require signatures. Molto also does not enforce the same-provider requirement for opaque continuation state; callers must keep signatures with their issuing provider.

The normalized path does not currently implement image output, non-web citations, refusals as content blocks, redacted thinking, hosted tools other than basic web search, structured output, or multiple candidates. Request fields accepted by the local Messages wrapper are not cross-protocol features unless the adapter and client map them explicitly.

Model support for mid-conversation system instructions is not locally validated. Without cache markers, translated requests targeting Messages hoist those instructions even when the source protocol could retain their original positions. With any markers present, Molto rejects mid-conversation instructions to preserve cache prefixes. These are current Molto limitations, distinct from Anthropic's inherent same-role merging and prefill semantics.

## Compatibility and references

Responses is the canonical behavioral reference for matched cross-protocol fixtures. Same-protocol tests compare parsed Anthropic SDK events or non-streaming responses. Translated streaming tests compare reconstructible semantics and separately assert balanced lifecycles, stable indexes, matching delta accumulation, terminal usage, termination, and target-default service tier; non-streaming tests verify accumulation of those projected native events into an SDK `Message`. Cassettes are replayed only; see the [cassette policy](../../tests/cassettes/README.md).

Relevant implementation and test entry points:

- [`Molto.messages()`](../../python/molto/core.py)
- [`AnthropicMessagesAdapter`](../../python/molto/adapters/anthropic.py)
- [`AnthropicMessagesClient`](../../python/molto/upstream/anthropic.py)
- [Messages request and SDK aliases](../../python/molto/types/anthropic_messages.py)
- [Normalized content and events](../../python/molto/types/normalized.py)
- [Interface parity tests](../../tests/core/test_interface.py)
- [Anthropic cassette matcher](../../scripts/match_anthropic_cassette_events.py)
- [Cross-protocol comparison](comparison.md)

External protocol references:

- [Anthropic Messages create reference](https://platform.claude.com/docs/en/api/messages/create)
- [Anthropic streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Anthropic mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- [vLLM Anthropic serving conversion](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/anthropic/serving/)
- [SGLang Anthropic-compatible API](https://docs.sglang.io/docs/basic_usage/anthropic_api)
