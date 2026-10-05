# Molto documentation

Molto is a Python library for translating between LLM API protocols without giving up native SDK compatibility. An application can speak OpenAI Responses, OpenAI Chat Completions, or Anthropic Messages while the provider exposes any of those protocols.

Molto has two design goals:

- Support any implemented downstream protocol through any implemented upstream protocol with one provider-neutral translation model.
- Bypass translation when both sides use the same protocol, preserving native requests, events, and responses with the [shared error policy](errors.md).

Streaming and non-streaming requests are supported on native and translated routes. Streaming is the translated execution primitive; translated non-streaming responses are collected from the downstream-native event stream produced by the same event projector.

## Cross-protocol contract

> For cross protocol, Molto is strict during input validation but lenient during output handling. Every accepted input field must be effected, and any output stemming from an accepted input set must be tolerated.

Three rules make this principle concrete:

- Normalized requests are closed and lossless-or-rejected, with documented [cache-lifetime approximations](#prompt-caching).
- Normalized outputs are open to provider extensions.
- Retaining unknown output data does not authorize projecting it into another protocol.

“Closed” applies recursively to nested request models. “Lossless-or-rejected” means each populated input field must affect the normalized request or target request, or Molto must reject it. Documented omission exceptions are optional Responses output-item IDs on Chat Completions and Messages, caller-defined prompt-cache keys on Messages, and the behavior-neutral Responses settings `include=["reasoning.encrypted_content"]` and `store=false` on translated routes. Native passthrough forwards these fields unchanged, while `store=true` remains unsupported for translation. Portable call identities such as `tool_call_id` are not part of these exceptions.

Cross-protocol semantic input rejections raise `molto.types.exceptions.UnsupportedInput`, a `ValueError` subclass. Schema validation at the public protocol boundary continues to use the validation errors provided by its Pydantic request model. Molto's local public request models always reject unknown top-level fields; this behavior is not configurable through environment settings.

## Choose a protocol

The public method selects the protocol returned to the application. The `endpoint` argument selects the protocol sent to the provider.

| Application protocol | Molto method | Native endpoint | Stream values | Non-stream value | Guide |
| --- | --- | --- | --- | --- | --- |
| OpenAI Responses | `Molto.responses()` | `responses` | OpenAI Responses SDK events | OpenAI SDK `Response` | [OpenAI Responses](protocols/openai-responses.md) |
| OpenAI Chat Completions | `Molto.chat_completions()` | `chat/completions` | OpenAI `ChatCompletionChunk` values | OpenAI SDK `ChatCompletion` | [OpenAI Chat Completions](protocols/openai-chat-completions.md) |
| Anthropic Messages | `Molto.messages()` | `messages` | Anthropic Messages SDK events | Anthropic SDK `Message` | [Anthropic Messages](protocols/anthropic-messages.md) |

Use Responses when an application benefits from typed output items and a rich streaming lifecycle. Use Chat Completions for applications built around the widely supported `messages` and choice-delta interface. Use Messages for applications that need the Anthropic SDK contract and its content-block model. The selected upstream endpoint does not have to match that choice.

## Quickstart

This example accepts an OpenAI Responses request, sends an Anthropic Messages request upstream, and yields native OpenAI Responses events:

```python
import asyncio

from molto import Molto


async def main() -> None:
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


asyncio.run(main())
```

Use the same protocol for the method and `endpoint` to select native passthrough. Each protocol guide includes native and translated examples.

`Molto` owns its shared `httpx.AsyncClient`. A returned stream releases its response when its context exits, including after early termination. Closing `Molto`, either with `async with` or `await molto.aclose()`, closes the shared HTTP client.

## Planned server mode

A future `python -m molto.server` entry point will expose Molto as an OpenAI Responses-compatible FastAPI upstream for Codex CLI. The implementation, model-catalog contract, compatibility settings, and remaining tests are specified in the [Codex server handoff](../codex-server.md). The server module is not implemented yet.

## Routing

Four independent choices define a route:

| Concept       | Responsibility                                                                                |
| ------------- | --------------------------------------------------------------------------------------------- |
| Public method | Selects the downstream request and SDK event or response contract exposed to the application. |
| `endpoint`    | Selects the upstream wire protocol and upstream client.                                       |
| `provider`    | Selects the serving implementation and its immutable capability profile.                      |
| `base_url`    | Selects only the HTTP destination.                                                            |

Molto never infers provider behavior from a hostname. When `provider` is omitted, the endpoint supplies a convenience default: OpenAI for Responses and Chat Completions, and Anthropic for Messages. An explicit provider overrides that default without changing the endpoint or base URL.

Provider capabilities matter even when two implementations expose the same schema. For example, Chat Completions `reasoning_content` and Messages `thinking_delta` do not label their text as raw reasoning or a summary. The provider profile supplies that interpretation: OpenAI and Anthropic profiles classify it as a summary, while vLLM classifies it as raw reasoning.

## How translation works

A translated request crosses a provider-neutral intermediate representation:

```text
application protocol request
        |
        v
downstream protocol adapter
        |
        v
normalized request
        |
        v
upstream protocol client
        |
        v
provider protocol request

provider protocol events
        |
        v
upstream protocol client
        |
        v
normalized event stream
        |
        v
downstream protocol adapter
        |
        v
application protocol events
```

The public `Molto` facade validates the downstream body, resolves routing and transport configuration, and composes the adapter and client. Adapters own downstream request normalization, downstream event projection, and collection of projected native events into native responses. Upstream clients own provider request construction, SDK transport, and upstream event normalization. Protocol-specific field transformations do not belong in the facade.

When the application and provider protocols match, Molto skips the normalized representation. This native passthrough validates requests and returns the SDK-native event stream or non-streaming response without losing protocol-only structure. Errors follow the [shared policy](errors.md).

Streaming is the execution primitive. The normalized event grammar is:

```text
ResponseStartEvent
PartStartEvent
PartDeltaEvent*
PartEndEvent
ResponseEndEvent
```

For a translated non-streaming request, the coordinator runs the selected upstream client in streaming mode and reuses the complete downstream event projection before collecting a response:

```text
normalized upstream event stream
        |
        v
downstream adapter.project_events()
        |
        v
downstream-native event stream
        |
        v
downstream adapter.project_response()
        |
        v
downstream-native response
```

Collection follows the downstream protocol's native event grammar. Responses returns the inner `Response` carried by the terminal `response.completed` or `response.incomplete` event. Chat Completions accumulates projected choice deltas, including content, reasoning, indexed tool calls, finish reason, and usage, into one `ChatCompletion`. Messages accumulates the projected message and content-block lifecycle through `message_stop` into one `Message`.

This order makes projected native events the source of truth for both translated modes. It does not introduce a second normalized-to-response translation path. Native same-protocol non-streaming requests take a separate fidelity-preserving fast path and return the matching client's SDK response directly without streaming or crossing the normalized representation.

## Translation contract

The [cross-protocol contract](#cross-protocol-contract) governs every translated route. Input validation establishes what Molto promises to effect; output handling tolerates provider additions without turning them into portable semantics.

These guides describe **target parity requirements**: their tables state what a correct translation must produce, not what Molto necessarily produces today. A section documents current behavior only when it says so explicitly. Likewise, a field accepted through an SDK-backed request wrapper is not an implementation promise; it must be mapped or rejected before it is treated as supported.

Cross-protocol parity means preserving semantics that the target can represent; it does not mean reproducing an independently generated response or identical SSE bytes. Every mapped value belongs to one of these classes:

| Class                | Required behavior                                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Source-backed        | Preserve the upstream request or result value after the documented schema transformation.                        |
| Request-backed       | Retain a downstream request value when the target result contract must echo it.                                  |
| Target-default       | Emit a documented target value when source and target vocabularies are not equivalent.                           |
| Translator-generated | Generate required target structure using a stable, internally consistent policy.                                 |
| Provider-owned       | Translate only under an explicit contract that the same provider owns both representations.                      |
| Unsupported input    | Reject before translation, except for documented omissions and [cache-lifetime approximations](#prompt-caching). |
| Unavailable output   | Preserve, omit, coerce, or expose through a documented extension; never claim exact parity.                      |

Normalized requests reject unknown fields, including unknown fields nested in request models. Every accepted input field must be mapped, handled under a documented exception, or explicitly rejected by the target client. The [input contract](#cross-protocol-contract) describes permitted omissions, and the [cache policy](#prompt-caching) describes lifetime approximations. Responses preserves output-item IDs, including on marked assistant history. Both OpenAI protocols preserve cache keys. Portable call identities such as `tool_call_id` remain lossless-or-rejected.

Normalized output models retain unknown values in `extra_fields` so output from any accepted input set remains tolerable. Request constructors and event projectors still map supported fields explicitly; retained output data is not evidence that another protocol can represent it.

Parsed event parity is also different from raw transport parity. Parsing and serialization may change JSON key order, omitted fields versus explicit null, number spelling, whitespace, SSE framing, and chunk placement. Request models that serialize with `exclude_none=True` cannot preserve omitted-versus-null distinctions unless they retain the original field set separately.

## Boundaries and ordering

Conversation behavior has three distinct kinds of boundary:

1. A wire-message boundary is an element of Responses `input[]`, Chat Completions `messages[]`, or Messages `messages[]` and its content blocks.
2. A model-visible turn is produced after a provider or model renderer applies role, merging, and chat-template rules.
3. A streaming boundary describes generated output items, parts, blocks, choices, and deltas. It is not an arbitrary conversation-message boundary.

Molto preserves each normalized user message and its ordered text or image parts. Assistant history likewise retains each source message and its ordered reasoning, text, and tool-call parts, including text-part cache markers. Adjacent explicit assistant messages are not merged. Messages preserves these ordered parts directly; Chat rejects ordering that its reasoning, content, and tool-call fields cannot represent within one message. Responses projects mixed assistant content as contiguous text messages interleaved with standalone reasoning and function-call items because its protocol has no shared container for them. Same-protocol passthrough retains more native structure. On translated routes, the current representation does not fully preserve:

- `system` versus `developer` identity;
- multipart instruction-message boundaries;
- valid mid-sequence instruction placement when targeting Messages;
- generated Responses assistant containers and inner content-part indexes; or
- Chat Completions choices beyond index zero.

Wire preservation does not guarantee a distinct model-visible turn. Anthropic may combine consecutive messages with the same role, and open-model renderers may merge, move, or rewrite instructions.

## Current feature support

The translated path currently supports:

- streaming and non-streaming requests;
- string input and ordered text messages;
- ordered mixed text and image user input using remote URLs or base64 data URLs;
- system prompts and supported assistant history;
- raw reasoning across all three protocols, and reasoning summaries on supported translations between Responses and Messages;
- client-defined function tools, selection, parallel-call control, calls, successful results, history, and streamed argument deltas;
- model, output-token limit, reasoning effort, temperature, and `top_p`;
- prompt-cache settings and supported text/image breakpoints under the [shared mapping policy](#prompt-caching), including silent omission of caller-defined keys on Messages;
- response identity, model, lifecycle metadata, termination, and semantic token usage; and
- native protocol passthrough that remains outside the normalized path.

The shared translated path does not currently implement image output, hosted or server tools, citations, refusals as content blocks, redacted thinking, structured output, multiple candidates, moderation, or log probabilities. Anthropic error tool results remain representable inside Messages but cannot be translated to OpenAI protocols without losing the error outcome. Fields accepted by a local request wrapper are not supported cross-protocol until an adapter and client implement their mapping, documented omission, or rejection.

Protocol guides distinguish these current implementation gaps from information that the target protocol can never represent faithfully.

## Shared field policies

### Reasoning

The normalized representation distinguishes raw reasoning from reasoning summaries:

- `Thinking.content` and `ThinkingDelta.content_delta` contain raw reasoning.
- `Thinking.summary` and `ThinkingDelta.summary_delta` contain a summary.

Responses events label those representations explicitly and must agree with the configured provider capability. Chat Completions `reasoning_content` and Messages `thinking_delta` are ambiguous, so the capability profile classifies them before normalization.

A Responses reasoning lifecycle must contain raw reasoning without a summary or signature, a summary accompanied by a signature without raw reasoning, or a signature without reasoning text. A Messages thinking lifecycle follows the same pairing rule. A completely empty start may be deferred until its representation is known, and a lifecycle must not switch representations.

Although a provider profile can classify Chat `reasoning_content` as a summary, Chat has no signature field. Current translated routes through Chat therefore support only unsigned raw reasoning. Summary translation through Chat fails at the Responses or Messages signature boundary rather than silently discarding required continuation state.

### Opaque continuation state

Responses `encrypted_content` and a Messages thinking `signature` are provider-issued continuation state. They are opaque; applications must not inspect or synthesize these values. They are not generally portable across providers. Molto currently carries the value through the normalized representation without tracking or validating its issuer. Callers must not reuse translated continuation state across providers. Matched fixtures map it only under the explicit assumption that OpenAI owns both endpoints.

Opaque state is represented as replacement snapshots, not append-only text. A Messages `signature_delta` replaces the block-start signature. Responses exposes three encrypted snapshots where Messages exposes two corresponding slots; translated streams preserve the Responses `added` and `done` snapshots and reuse `done` at terminal projection instead of buffering for a third value.

### Prompt caching

Prompt caching changes cost and latency without changing model-visible content or conversation history. Messages silently drops caller-defined cache keys because Anthropic computes cache identity implicitly. Dropping a key does not enable caching or discard the remaining options.

`NormalizedRequest.prompt_cache` carries request-wide intent; `CacheBreakpoint` on supported content parts carries the position and optional lifetime of an explicit marker:

| Normalized field | Responses and Chat Completions | Anthropic Messages |
| --- | --- | --- |
| `PromptCache.key` | `prompt_cache_key` | Silently dropped |
| `PromptCache.mode` | `prompt_cache_options.mode` | `implicit` enables top-level automatic `cache_control`; `explicit` caches only marked blocks |
| `PromptCache.ttl` | `5m`, `30m`, and `1h` map to request-wide `prompt_cache_options.ttl: "30m"` | `30m` maps to `1h`; `5m` and `1h` pass through |
| `PromptCache.retention` | `prompt_cache_retention` | Rejected |
| Content `cache_breakpoint` | `prompt_cache_breakpoint: {"mode": "explicit"}` | Block `cache_control: {"type": "ephemeral", "ttl": ...}` |

A key-only request or empty options object without markers does not enable Messages caching. Implicit mode enables automatic caching; a lifetime without a mode also enables it. Without markers, implicit mode with no lifetime uses Anthropic's five-minute default. Explicit mode omits top-level automatic caching but preserves marked blocks. When projecting normalized input to Messages, a block lifetime overrides the normalized request lifetime; an omitted block lifetime inherits it.

When an OpenAI request has any content marker, normalization resolves omitted mode to `implicit` and omitted lifetime to OpenAI's `30m` default. Messages therefore uses `1h` for both marked blocks and automatic caching, unless the caller selected explicit mode to disable the latter. An Anthropic request with only block markers normalizes to explicit mode; adding top-level `cache_control` selects implicit mode while retaining the blocks.

OpenAI defines `30m` as a minimum lifetime, so Molto maps it to Anthropic `1h`. This selects longer retention and cache-write pricing of 2x the base input-token price, compared with 1.25x for Anthropic's five-minute default. It is a documented mapping policy, not exact lifetime or pricing equivalence. See [OpenAI cache lifetime](https://developers.openai.com/api/docs/guides/prompt-caching#cache-lifetime) and [Anthropic cache pricing](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

In the reverse direction, OpenAI supports only `30m` in `prompt_cache_options.ttl`. Molto maps every explicit normalized request or block lifetime (`5m`, `30m`, or `1h`) to that request-wide value. This rounds `5m` up but approximates `1h` with a shorter minimum: an OpenAI target does not guarantee the requested hour. Mixed block lifetimes collapse to one `30m` value for automatic caching and all explicit breakpoints, while mode and marker positions remain intact. This is a documented exception to the lossless-or-rejected input contract. Molto does not synthesize `prompt_cache_retention: "24h"`: retention is a separate maximum policy and does not establish a one-hour minimum.

An omitted Anthropic lifetime remains unspecified unless another request or block lifetime supplies the shared OpenAI TTL; otherwise OpenAI's `30m` default applies. Molto does not invent an explicit five-minute requirement from Anthropic's default. Native passthrough retains the original provider lifetimes.

Supported marker positions are system/developer text, user text/images, assistant text, and nested tool-result text/images. Chat tool messages support only text. Translation rejects markers that cannot retain their prefix:

- Anthropic outer `tool_result`, `tool_use`, thinking blocks, and tool definitions have no corresponding OpenAI content-part marker.
- Messages rejects a different automatic and final explicit lifetime. One-hour breakpoints must precede five-minute breakpoints.
- Messages rejects mid-conversation instructions whenever any marker is present, because hoisting would change the cached prefix.
- Chat rejects marked tool returns when image relocation would move content past a marker, including a marked later parallel tool return with earlier images still waiting to be emitted.

Messages forwards all supported markers and leaves breakpoint-count limits to the upstream. It keeps system blocks separate when any system block is marked. Responses projects assistant history as `output_text` parts, preserving both markers and optional output-item IDs. Native passthrough forwards provider-native cache fields unchanged.

### Service tiers

Service-tier labels are not portable provider semantics. Native passthrough preserves the provider value. Translated results use target defaults:

| Downstream result       | Target value                        |
| ----------------------- | ----------------------------------- |
| OpenAI Responses        | `default`                           |
| OpenAI Chat Completions | `default`                           |
| Anthropic Messages      | `standard` in `message_start.usage` |

Request `service_tier` is not forwarded across protocols. Anthropic request values and response values also use different vocabularies and must not be treated as direct aliases.

### Termination

Molto first preserves a provider-neutral termination reason, then projects it into the target protocol:

| Normalized reason               | Messages        | Chat Completions | Responses                       |
| ------------------------------- | --------------- | ---------------- | ------------------------------- |
| `stop`                          | `end_turn`      | `stop`           | completed                       |
| `max_output_tokens`             | `max_tokens`    | `length`         | incomplete: `max_output_tokens` |
| `max_messages`                  | `end_turn`      | `stop`           | incomplete: `max_messages`      |
| `steered`                       | `end_turn`      | `stop`           | incomplete: `steered`           |
| `stop_sequence`                 | `stop_sequence` | `stop`           | completed                       |
| `tool_call`                     | `tool_use`      | `tool_calls`     | completed                       |
| `model_context_window_exceeded` | same value      | `stop`           | completed                       |
| `pause`                         | `pause_turn`    | `stop`           | completed                       |
| `refusal`                       | `refusal`       | `stop`           | completed                       |
| `content_filter`                | `end_turn`      | `content_filter` | incomplete: `content_filter`    |

Responses preserves `max_messages` and `steered` as incomplete states in streaming and collected output. `max_messages` is not interpreted as token exhaustion; accepting `steered` does not implement WebSocket steering or successor responses. Values shown as normal completion for a target that lacks the source distinction are deliberate coercions. A matched Anthropic stop sequence is retained separately.

### Usage

Usage is mapped by meaning rather than by field spelling:

| Normalized usage | Responses | Chat Completions | Messages |
| --- | --- | --- | --- |
| Input tokens | `input_tokens` | `prompt_tokens` | uncached + cache creation + cache read |
| Cached input | `input_tokens_details.cached_tokens` | `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` |
| Cache writes | `input_tokens_details.cache_write_tokens` | `prompt_tokens_details.cache_write_tokens` | `cache_creation_input_tokens` |
| Output tokens | `output_tokens` | `completion_tokens` | `output_tokens` |
| Reasoning tokens | `output_tokens_details.reasoning_tokens` | `completion_tokens_details.reasoning_tokens` | `output_tokens_details.thinking_tokens` |

When projecting Messages, Molto recovers uncached input by subtracting cache reads and writes from normalized total input. Terminal Messages usage is authoritative; start usage is zero when a source protocol cannot provide final counts without buffering. Unsupported provider-specific usage details are not synthesized.

### Errors

Recognized failures expose a downstream SDK exception with a normalized Molto cause; unknown errors pass through unchanged. See [Errors](errors.md) for the cause chain, recognition rules, status mappings, and the streaming and missing-credentials cases.

## Provider and model renderers

An endpoint schema describes accepted wire data, not necessarily the turns shown to the model. This is especially important for vLLM, whose Responses, Chat Completions, and Messages endpoints all construct a conversation for the selected model renderer.

For Responses, vLLM converts input and output items into chat messages before rendering. Reasoning and tool-call items can belong to one reconstructed chat message, so Responses item boundaries are not necessarily model-turn boundaries. For Chat Completions, `openai` content mode retains structured parts inside one message, while `string` mode joins text parts with newlines and loses their individual identities.

The Hugging Face renderer has a developer compatibility fallback. If a template does not explicitly support `developer`, vLLM converts developer messages to system messages and may consolidate all system messages at the beginning. A Messages conversion can similarly extract and merge inline system messages when a template requires system-first ordering. These transformations sacrifice instruction role, placement, and message boundaries for template compatibility.

Tool-result images are also constrained by the server's Chat Completions substrate. Both vLLM and SGLang implement the Anthropic Messages `/v1/messages` endpoint as a conversion to their own Chat Completions protocol, and OpenAI Chat `tool` messages carry only text content, so a `tool_result` that returns an image has no native Chat representation. The two servers resolve this differently:

- vLLM keeps the text in the `tool` message and remaps the image parts into a later `user` content array, converting base64 sources to data URIs. A returned image therefore reaches the model as a follow-up user prompt. This is the same relocation that Molto's Chat-carrying routes perform. Older vLLM silently discarded such images (it stringified the content list); current vLLM preserves them, but still relocates because native Chat `tool` messages with `image_url` parts remain unsupported by at least some renderers.
- SGLang instead embeds the `image_url` parts inside the `tool` message's own content array next to the text, because its Chat backend accepts multimodal content arrays in tool messages. A tool-result image therefore stays tied to its tool call instead of being moved to a user turn.

Molto's Chat-carrying route always uses the vLLM-style form (a text notice in the `tool` message and the image relocated to a later user message), which is compatible with both servers: vLLM expects relocated images, and SGLang also accepts images in user messages.

Model-specific behavior includes:

- Kimi K3 uses vLLM's dedicated XTML renderer. It preserves ordinary conversation order but may reorder consecutive tool results to match the preceding assistant tool-call order. A distinct developer instruction level is not documented.
- Qwen uses ChatML turn markers and documents system, user, and assistant roles. A template accepting the literal word `developer` does not prove that the model was trained to treat it as a separate instruction level. vLLM generally applies its developer-to-system fallback when the template lacks explicit support. Qwen3 does not insert a default system message, and Qwen tool results are rendered as special user messages.
- Muse Glimmer's ATEM template has explicit system, user, tool, and assistant branches but no developer branch. Direct template application would omit an unsupported developer message, while vLLM preprocessing is expected to convert it to system and may move or merge it. If no system message exists, the template inserts a default leading system message. Supplied system messages also receive reasoning, tool-definition, and recipient metadata.

For open-weight models, the selected renderer or chat template is the most authoritative description of model-visible roles and boundaries. Do not infer these capabilities from `endpoint`, `provider`, or `base_url` alone.

## Compatibility and maintenance

OpenAI request wrappers track the exact SDK version pinned in `pyproject.toml`. OpenAI result and event types are SDK aliases rather than duplicated Molto models, except two narrow Responses event replacements: `OpenAIResponseCompletedEvent` (sets a `null` `cache_write_tokens` to `0`) and `OpenAIResponseFunctionCallArgumentsDoneEvent` (relaxes the required `name`). Compatible-provider extensions are modeled narrowly; accepting an extension must not weaken every message or event to an untyped dictionary.

The OpenAI schema monitor derives its baseline from the exact SDK test pin in `pyproject.toml` and compares that release's `api_reference/openapi.transformed.yml` against the same file in the latest published stable SDK release. Updating the verified SDK test pin advances the baseline automatically; OpenAPI `main` is no longer monitored directly. Native Responses accepts `ultrafast`, `access_programs`, cache prewarming/comparison IDs, asynchronous function tools/history, web-search `external_web_access`, and image-generation `xhigh`/`max` quality through the pinned SDK. Native events include compaction progress and shell command/output-content variants. These features do not widen translation: populated `prewarm`, `comparison_response_id`, and function `async` values (including `false`) are explicitly rejected, as are service tiers, access programs, hosted tools/history, and context management.

Native SDK responses retain diagnostics, access-program metadata, image-generation failures, Chat metadata, obfuscation, and detailed usage. Translation maps aggregate, cache, and reasoning token counts and omits source-only diagnostics, access-program metadata, text/image usage breakdowns, and obfuscation. Shell/compaction translation and WebSocket steering remain outside the supported scope; content-bearing shell events are never silently ignored. Arbitrary model IDs, Python integer Chat seeds, optional compatible-provider history IDs, and omitted function-call argument-done names remain supported on native routes. Chat image `detail="original"` is retained by native requests and translation to Responses; Messages rejects that detail.

Existing behavior is verified through offline cassette replay. Cassette-backed coverage of the new schema fields, events, metadata, and incomplete terminal reasons is deferred to a follow-up PR using real provider recordings; live-provider acceptance remains unverified. The monitor reports subsequent released SDK schema drift. Comparing release schemas does not verify live-provider behavior or cross-protocol feature support.

Responses is the canonical behavioral reference for matched cross-protocol cassettes. Same-protocol tests compare native request bodies and parsed SDK events. Translated tests compare reconstructable semantics and separately validate lifecycle ordering, generated identifiers, target defaults, and exact upstream request construction. Normal test runs replay committed cassettes and never record or rewrite them.

Implementation and maintenance details remain canonical in:

- [Python architecture](../python/ARCHITECTURE.md)
- [Python conventions](../python/CONVENTIONS.md)
- [Cassette policy](../tests/cassettes/README.md)

The protocol-specific contracts are:

- [OpenAI Responses](protocols/openai-responses.md)
- [OpenAI Chat Completions](protocols/openai-chat-completions.md)
- [Anthropic Messages](protocols/anthropic-messages.md)
- [Cross-protocol comparison](protocols/comparison.md)

## Future work

The [Codex server roadmap](roadmap/codex-server.md) describes a proposed stateless HTTP Responses server, its library prerequisites, and the milestones required for Codex CLI compatibility. The server is not yet implemented.

## Primary references

- [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses)
- [OpenAI Chat Completions create reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Anthropic Messages create reference](https://platform.claude.com/docs/en/api/messages/create)
- [Anthropic mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- [Anthropic streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM Responses conversion utilities](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/responses/utils/)
- [vLLM Anthropic serving conversion](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/anthropic/serving/)
- [SGLang Anthropic-compatible API](https://docs.sglang.io/docs/basic_usage/anthropic_api)
- [SGLang OpenAI-compatible Chat Completions](https://docs.sglang.io/docs/basic_usage/openai_api_completions)
- [vLLM Hugging Face renderer](https://docs.vllm.ai/en/latest/api/vllm/renderers/hf/)
- [vLLM Kimi K3 renderer](https://docs.vllm.ai/en/latest/api/vllm/renderers/kimi_k3/)
- [Qwen chat-template concepts](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md#chat-template)
- [Muse Glimmer chat template](https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/chat_template.jinja)
