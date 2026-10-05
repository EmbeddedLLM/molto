# Cross-protocol comparison

This guide collects the side-by-side views that were split across the per-protocol guides: boundary fidelity, request/output formats, role placement, and provider/model-renderer behavior. It is a comparison lens over the per-protocol contracts, not an independent source of truth. For routing, the normalized representation, parity classes, termination, usage, and service-tier policy, see the [documentation overview](../README.md). For protocol-specific wire contracts see the [Responses](../protocols/openai-responses.md), [Chat Completions](../protocols/openai-chat-completions.md), and [Anthropic Messages](../protocols/anthropic-messages.md) guides.

## Boundary fidelity at a glance

Every row below holds for a translated route (one protocol to another through the normalized representation) unless it says "native passthrough".

| Fidelity aspect | Across translated routes | Source of value | Protocol exception |
| --- | --- | --- | --- |
| One normalized user input → one wire message | Preserved | User message model | Provider renderer may merge consecutive same-role messages into one model-visible turn |
| Ordered user text parts | Preserved | Ordered text parts | Chat `string` content mode can join text parts with newlines and lose individual identity |
| Ordered image parts | Preserved | Remote URL or base64 data URL | Chat tool messages cannot carry images; image tool results are relocated |
| Consecutive user messages | Not merged by Molto | One `UserInput` per message | Upstream provider or renderer may combine them |
| `system` versus `developer` identity | Lost; one instruction concept | Role-less `SystemPrompt` | Native passthrough preserves the distinction |
| Instruction position | Preserved for Responses and Chat targets; hoisted for Messages | `SystemPrompt` sequence position | Messages uses top-level `system`, keeps marked system blocks separate, and rejects mid-conversation instructions when markers exist |
| Multipart instruction boundaries | Lost | One `SystemPrompt` per instruction part | Text-part content and markers survive; the enclosing instruction-message boundary does not |
| Assistant-history message container | Preserved where the target can represent it | One `AssistantInput` per source message | Responses separates reasoning and calls from text messages; Chat rejects incompatible part ordering |
| Ordered assistant-history parts | Preserved, including multipart text and markers | Ordered `AssistantInput.content` | Responses emits each contiguous text run as one message interleaved with standalone reasoning and calls; generated-output containers remain a separate limitation |
| Raw versus summary reasoning | Classified by provider capability profile | Capability profile | OpenAI/Anthropic profiles = summary; vLLM = raw; explicit Responses events are authoritative |
| Function `call_id`, name, args, results, ordering | Preserved | Portable call identity | Responses output-item `id` is a separate structure with no Chat equivalent |
| Termination reason | Projected per the shared table | Provider-neutral termination | Reasons without a target equivalent are coerced to that target's normal completion |
| Semantic token usage | Preserved | Mapped by meaning, not field spelling | Source-only audio, prediction, cache TTL, geographic, server-tool details are not synthesized |
| Native cache prewarming/comparison and async function settings | Rejected before transport | Native Responses settings | No portable semantic mapping |
| Source-only diagnostics, usage details, and obfuscation | Omitted | SDK-owned native metadata | Aggregate/cache/reasoning usage remains mapped |
| Prompt-cache intent | Projected under the [shared policy](../README.md#prompt-caching) | `PromptCache` plus positional `CacheBreakpoint` | Messages silently drops keys and maps `30m` minimum lifetime to `1h` on blocks and automatic caching; key-only requests do not enable caching; supported text/image markers preserve position; retention and unsupported marker positions are rejected; OpenAI collapses explicit request/block lifetimes to `30m`, rounding `5m` up and weakening the `1h` minimum |
| Response ID and model | Preserved | Source response | A Messages upstream has no creation timestamp; `created_at` is captured locally |
| Exact response-completion timestamp | Unavailable and omitted | None | Responses `created` is not `completed_at` |
| Assistant `phase` | Not normalizable | None | Native Responses passthrough retains it; Chat and Messages omit it |
| Opaque reasoning state | Not portable across providers | Provider-owned | Mapped only under an explicit same-provider contract (OpenAI owns both endpoints) |

Full boundary fidelity would require more than the current normalized representation can hold; the design reasoning for that is in the [architecture's boundary-fidelity commentary](../../python/ARCHITECTURE.md#current-limitations-and-extension-path).

## Formats at a glance

| Concern | Responses (downstream target) | Chat Completions (downstream target) | Messages (downstream target) |
| --- | --- | --- | --- |
| Application method | `Molto.responses()` | `Molto.chat_completions()` | `Molto.messages()` |
| Native endpoint | `responses` | `chat/completions` | `messages` |
| Request input | String `input` or heterogeneous `input[]` | Ordered `messages[]` | Ordered `messages[]` + top-level `system` |
| Output identity | Typed output items with IDs and lifecycles | Chunks in `choices[]`, index `0` only | One message with indexed content blocks |
| Streaming events | `ResponseStreamEvent` lifecycles | `ChatCompletionChunk` deltas | `RawMessageStreamEvent` blocks |
| `created_at` | Source-backed or, from Messages, local `message_start` time | Source-backed; local time from Messages upstream | Captured locally when required by a downstream protocol |
| Termination | `completed` / `incomplete` with reason, including `max_messages` and `steered` | `finish_reason`; new Responses reasons fall back to `stop` | `message_delta.stop_reason`; new Responses reasons fall back to `end_turn` |
| Reasoning representation | Explicit `reasoning_text.*` vs `reasoning_summary_*` | Ambiguous `reasoning_content`, capability-classified | Ambiguous `thinking_delta`, capability-classified |
| Service tier (translated) | Target default `default` | Target default `default` | `standard` in `message_start.usage` |
| Keepalive | None specific on translated output | None specific on translated output | `ping` is transport-only, filtered by the SDK, never synthesized |
| Multiple candidates | Not supported | Only `choices[0]` normalized; `n > 1` should not use translation | Not supported |
| Tools | `function_call` items | `tool_calls[]` in assistant messages | `tool_use` / `tool_result` blocks |
| Request prompt caching | `prompt_cache_key`, `prompt_cache_options`, per-part `prompt_cache_breakpoint` | `prompt_cache_key`, `prompt_cache_options`, per-part `prompt_cache_breakpoint` | top-level or per-block `cache_control` |

Responses is the richer lifecycle protocol. Translating into Responses requires synthesizing output-item IDs, sequence numbers, item/content lifecycles, and empty target-only containers; translating into Chat or Messages flattens that lifecycle into deltas or blocks. The conversion is not symmetric: a native result must be mapped, defaulted, generated, or rejected rather than copied.

## Role and placement

The normalized representation has one instruction type (role-less `SystemPrompt`), so `system` and `developer` identity is not recoverable across protocols. What happens to a mid-conversation instruction depends on the translation target:

| Translation target | Mid-sequence `system` | Mid-sequence `developer` | Qualification |
| --- | --- | --- | --- |
| Native passthrough | Preserved with its role and position | Preserved with its role and position | Same-protocol identity optimization retains native structure |
| Responses (translated) | Emitted as role-less instruction; projects as `developer` at the same position | Instruction; projects as `developer` | Original `system` identity is not reconstructible |
| Chat Completions (translated) | Emitted as role-less instruction with position retained | Instruction with position retained | Projection re-emits every instruction as `developer`; text-part arrays and markers are supported |
| Messages (translated) | Top-level `system`; separate blocks when system markers exist | Hoisted only when no markers exist; otherwise rejected | Unmarked hoisting loses placement, priority, and block boundaries |

## Provider and model renderers

An endpoint schema is only the first stage of a conversation: the selected chat template or renderer decides the model-visible roles, merging, and boundaries. Molto never infers these from `endpoint`, `provider`, or `base_url` alone. The renderer below is only relevant for vLLM-style open-weight serving.

- Kimi K3 uses vLLM's dedicated Python XTML renderer rather than a generic Jinja chat template. Structural markers encode the conversation, and user or tool text is kept as ordinary text rather than re-tokenized as structure. A distinct `developer` instruction level is not documented.
- Qwen uses ChatML turn markers and documents `system`, `user`, and `assistant` roles; no `developer` level is documented. Tokenizing the literal word `developer` is not proof of an instruction role. Some Qwen templates explicitly require the system message to be first. Tool results render as special user messages, and Qwen3 does not insert a default system message.
- Muse Glimmer's official ATEM template branches for `system`, `user`, `tool`, and `assistant` and has no `developer` branch. It emits four structural marker kinds: start, message, end-of-message, and end-of-turn. A supplied system message is rendered at its position, and the template inserts a default leading system message when none is supplied. Supplied system messages carry reasoning, tool-definition, and valid-recipient metadata. Because the template lacks a `developer` branch, vLLM's Hugging Face fallback converts a developer message to system, and direct template application would render no developer message at all.

These renderer details are condensed into the [provider and model renderers](../README.md#provider-and-model-renderers) overview; this guide preserves the properties that the overview compresses to one bullet per model.
