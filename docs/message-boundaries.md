# Message Boundaries Across Responses, Chat Completions, and Messages

_Status: investigation snapshot as of 2026-08-18._

## Executive summary

Message boundaries exist at three distinct layers:

1. **Wire messages and content parts** are the objects accepted by an endpoint.
2. **Model-visible turns** are produced after a provider applies its conversation rules or chat template.
3. **Streaming parts and deltas** describe generated output and are not necessarily request-message boundaries.

Molto currently preserves user-message boundaries and the ordered text parts inside each user message on translated routes. It does not fully preserve:

- `system` versus `developer` role identity;
- the placement of mid-sequence instructions when targeting Messages;
- enclosing assistant-message boundaries;
- Responses content-part boundaries inside one output item; or
- multiple Chat Completions choices.

Same-protocol routes bypass the normalized representation and therefore retain substantially more native structure. The losses described below primarily affect cross-protocol translation.

For vLLM, the endpoint schema is only the first stage. Responses, Chat Completions, and Messages requests are ultimately converted into a conversation and processed by the selected model renderer. The renderer or chat template determines the model-visible roles and message boundaries. Consequently, these semantics are model- or renderer-specific and cannot be inferred from `endpoint`, `provider`, or `base_url` alone.

## The three kinds of boundary

### Wire-message boundaries

These are elements of a protocol request array, such as:

- Responses `input[]` items;
- Chat Completions `messages[]` entries; and
- Messages `messages[]` entries and their content blocks.

Wire boundaries determine serialization, validation, caching opportunities, and which protocol fields can be represented. They do not guarantee that the serving implementation presents each element to the model as an independent turn.

### Model-visible turn boundaries

The provider or model renderer can merge or rewrite the wire sequence before tokenization. Examples include:

- Anthropic combining consecutive Messages entries with the same role into one conversational turn;
- vLLM converting unsupported `developer` messages to `system` messages; and
- vLLM consolidating system messages at the beginning for templates that require system-first ordering.

For open-weight models, the chat template's control tokens are the most authoritative description of the model-visible boundaries.

### Streaming boundaries

Streaming boundaries describe one generated response, not an arbitrary list of conversation messages:

- Responses exposes output-item and inner content-part lifecycles.
- Chat Completions exposes delta chunks for one assistant message per choice.
- Messages exposes one assistant message containing indexed content blocks.

Transport chunk boundaries are observable and can matter for parity, but they should not be treated as conversation-turn boundaries.

## Protocol comparison

| Protocol | Native request structure | Instruction roles | Native output structure |
| --- | --- | --- | --- |
| Responses | Ordered heterogeneous `input[]`; message items contain ordered parts | `system` and `developer` are schema-valid message roles | Indexed output items, each with indexed content parts |
| Chat Completions | Ordered `messages[]`; messages may contain content parts | `system` and `developer` are message variants | One streamed assistant message per choice |
| Messages | Ordered `messages[]`; messages contain ordered content blocks | Top-level `system` plus model-gated mid-sequence `system`; no `developer` | One assistant message containing indexed blocks |

## Responses

### Request and history boundaries

Responses accepts either a string input or an ordered list of heterogeneous input items. A string is shorthand for one user input. Message items can contain multiple ordered text parts, while reasoning, tool calls, tool results, and other items have their own item types.

The pinned OpenAI SDK schema accepts `user`, `assistant`, `system`, and `developer` easy-input messages and does not impose role alternation or a first-message restriction. The local Molto request schema likewise accepts system and developer messages mid-sequence. This establishes schema legality; model-specific instruction precedence remains a separate concern.

Responses also has a top-level `instructions` field. It is not itself an `input[]` message boundary, even though Molto currently normalizes it into the same instruction representation used for system and developer messages.

### Molto translation behavior

`OpenAIResponsesAdapter` processes input items in order:

- One user message becomes one `UserInput` with ordered `TextInput` parts.
- Consecutive user messages remain separate `UserInput` values.
- System and developer messages both become role-less `SystemPrompt` values.
- A multipart system or developer message becomes multiple `SystemPrompt` values.
- Each assistant text part becomes a flat `TextOutput`, losing its parent assistant-message identity.
- Multiple reasoning-text or reasoning-summary parts are joined into one string per representation.
- `instructions` is prepended as another `SystemPrompt` and is no longer distinguishable from an input message.

When targeting Responses again through the normalized path, each `SystemPrompt` is emitted as a `developer` message. Every `TextOutput` becomes a separate assistant output message containing one text part. A source assistant message containing several parts can therefore be split into several output messages.

### Streaming boundaries

Responses natively distinguishes:

- an output item, identified by `output_index`;
- an inner content part, identified by `content_index` or `summary_index`; and
- deltas within that part.

Molto preserves one output item as one normalized part lifecycle. It does not represent the inner content index, so multiple content parts inside one output item collapse. Projection reconstructs one inner part at index `0`.

The current Responses projection also keeps one active output-item state rather than state keyed by `output_index`. It therefore assumes output-item lifecycles are not interleaved.

## Chat Completions

### Request and history boundaries

Each `messages[]` entry is a native message boundary. The official schema includes developer, system, user, assistant, tool, and legacy function messages. It does not state a positional restriction for system or developer messages. For newer OpenAI reasoning models, developer instructions replace the older system-message mechanism.

Molto's translated Chat path currently supports text-bearing system, developer, user, and assistant messages:

- One user message becomes one `UserInput`.
- Ordered user text parts remain ordered `TextInput` parts.
- Consecutive user messages remain separate.
- System and developer both become `SystemPrompt`, retaining sequence position but losing role identity.
- Assistant reasoning and text are represented as adjacent `Thinking` and `TextOutput` values rather than an explicit assistant-message container.

Projection to Chat emits every `SystemPrompt` as `developer`. A `Thinking` value is held until the following `TextOutput` so both can be emitted in one assistant message. Adjacent `TextOutput` values become separate assistant messages. An assistant message with multipart content, a tool-call-only assistant message, and non-text user parts are not supported by the translated normalizer.

Chat Completions `tool` messages accept only text content. When an upstream Chat request receives a normalized image tool result, Molto keeps a text notice in the `tool` message and injects the image as a later `user` prompt, following [github/copilot-sdk#1922](https://github.com/github/copilot-sdk/issues/1922). Every `tool` message responding to an assistant's `tool_calls` must appear contiguously and immediately after that assistant message, before any other role. Molto therefore emits the complete tool-response block before any relocated image prompts, including for parallel tool calls.

There is also a local schema issue with the pinned SDK: valid system or developer content-part arrays can validate but serialize as empty arrays because only the user-message wrapper replaces the SDK's iterable annotation with a concrete list. This behavior is not covered by a regression test.

### Streaming boundaries

A Chat Completions stream represents one assistant message per choice. Role establishment and content or reasoning deltas are chunk-level events, not independent messages.

Molto currently:

- normalizes only `choices[0]`, silently losing additional choices;
- preserves each non-empty content or reasoning delta for choice zero;
- represents the transition from reasoning to text as separate normalized parts; and
- projects all normalized output back into one Chat assistant stream.

Chat has no native representation for the normalized part lifecycle. Projection emits the initial role-bearing chunk once, skips later part starts, and ignores part ends. A source chunk containing both reasoning and text can become two projected chunks.

## Anthropic Messages

### Request and provider turn boundaries

Each `messages[]` entry contains a string or an ordered list of content blocks. A string and a single text block are documented shorthand forms and are semantically equivalent.

Anthropic combines consecutive user entries or consecutive assistant entries into a single conversational turn. Molto may preserve the separate wire objects, but it cannot claim that Anthropic preserves them as separate model-visible turns. The final role also matters: a final assistant message acts as a response prefill.

Messages has no `developer` role. Current Anthropic schemas include `user`, `assistant`, and `system`.

### Mid-conversation system messages

Anthropic supports mid-conversation system messages on selected models. They are not interchangeable with a top-level system prompt:

- A mid-conversation system message cannot be first.
- It must immediately follow a user message, or an assistant message ending in a server-tool result.
- It must be last or immediately followed by an assistant message.
- It cannot appear between a tool use and its tool result.
- Consecutive system messages form one section.
- Later system messages override earlier instructions for following turns.

The local Pydantic request model validates the `system` role but does not enforce placement or model-support constraints.

### Molto translation behavior

The Messages adapter preserves one user message as one `UserInput` and retains its ordered text blocks. It flattens every assistant message into top-level `Thinking` and `TextOutput` values, discarding the enclosing assistant-message boundary.

When targeting Messages, the upstream client:

- reconstructs each `UserInput` as one user message;
- groups all adjacent assistant parts into one assistant message; and
- gathers every `SystemPrompt` from the entire sequence, joins the strings with blank lines, and emits one top-level `system` value.

That last transformation destroys valid mid-conversation placement, instruction priority, cache-prefix semantics, and the original system message and block boundaries.

### Streaming boundaries

Anthropic emits one generated assistant message with:

- `message_start`;
- indexed `content_block_start`, delta, and stop events;
- message-level deltas; and
- `message_stop`.

Molto preserves supported text and thinking block indexes and their order. A normalized `ThinkingDelta` can produce separate Anthropic thinking and signature delta events. Raw reasoning and reasoning summaries both collapse into Messages' single thinking field.

Projection currently emits empty initial text or thinking at block start and expects subsequent deltas to carry the content. A generic non-empty normalized part start would therefore lose that initial content.

## Role and placement matrix

| Target | Mid-sequence `system` | Mid-sequence `developer` | Important qualification |
| --- | --- | --- | --- |
| OpenAI Responses | Schema-valid | Schema-valid | Molto preserves position but converts both roles to `developer` on translated Responses output |
| OpenAI Chat Completions | Schema-valid | Schema-valid | Modern OpenAI models prefer developer instructions; Molto loses the original role |
| Anthropic Messages | Supported only with model and placement constraints | Not supported | Molto currently hoists all translated instructions to top-level `system` |
| vLLM OpenAI-compatible endpoints | Accepted at the API layer according to the endpoint schema | Accepted by OpenAI-shaped schemas | The renderer may rewrite, merge, reject, or ignore a role |

## vLLM: endpoint schema versus model renderer

vLLM exposes OpenAI-compatible Chat Completions and Responses endpoints and an Anthropic-compatible Messages endpoint. All three eventually construct a conversation for a model renderer.

For Responses, vLLM converts input and output items into chat messages before rendering. Some Responses items, such as reasoning plus tool calls, can belong to one reconstructed chat message. Responses item boundaries are therefore not necessarily model-turn boundaries.

For Chat Completions, vLLM parses messages in order. In `openai` content mode, structured parts remain in one message. In `string` mode, text parts are joined with newlines, so part identity is reduced to separators.

For Messages, vLLM converts the Anthropic request into OpenAI-shaped conversation messages. It tests whether the selected template requires system-first ordering. If so, inline system messages are extracted and merged into the leading system block.

The Hugging Face renderer also has a developer compatibility fallback. If a conversation contains a developer message and the template does not explicitly support `developer`, vLLM converts developer messages to system messages and consolidates all system messages into one message at position zero. This intentionally sacrifices instruction placement and system-message boundaries so templates requiring a leading system prompt can run.

## Model-specific findings under vLLM

### Kimi K3

Kimi K3 uses vLLM's dedicated Python XTML renderer rather than a generic Jinja chat template. Structural markers encode the conversation, and user- or tool-provided text is kept as ordinary text rather than retokenized as structure.

The renderer preserves the conversation order except for one documented tool behavior: consecutive tool-result messages can be reordered to match the preceding assistant tool-call order. System, user, and assistant structures are documented. A distinct developer role is not documented and should not be assumed to have OpenAI developer-role semantics.

### Qwen

Qwen uses ChatML control tokens around each model-visible turn:

```text
<|im_start|>{{role}}
{{content}}<|im_end|>
```

Qwen documents `system`, `user`, and `assistant` semantics. Starting with Qwen3, no default system message is inserted. A distinct developer role is not documented as a trained instruction level.

Although a simple template may interpolate an arbitrary role string, tokenizing the word `developer` is not evidence that the model treats it as an instruction role. With the current vLLM fallback, templates that do not explicitly mention developer generally receive a developer message as system, with system messages potentially merged and moved to the beginning. Some Qwen templates explicitly require the system message to be first.

Qwen tool results are rendered as special user messages. This is another example of a wire role not mapping one-to-one to the model-visible role.

### Muse Glimmer

Muse Glimmer's official ATEM template has explicit branches for `system`, `user`, `tool`, and `assistant`, but no developer branch. The template emits structural start, message, end-of-message, and end-of-turn markers.

System messages are rendered at their supplied sequence location by the template itself. If no system message exists anywhere in the input, the template inserts a default leading system message. Each supplied system message also receives reasoning, tool-definition, and valid-recipient metadata.

When served through vLLM's Hugging Face renderer, a developer message is expected to trigger the developer-to-system fallback because the template does not explicitly support that role. That can move and merge system instructions. Applying the template directly without that preprocessing would not render a developer message at all because no branch handles it.

## Current Molto fidelity summary

| Semantic feature                            | Current translated fidelity                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| User-message boundary                       | Preserved                                                                   |
| Ordered text parts inside one user message  | Preserved                                                                   |
| Consecutive user wire messages              | Preserved by Molto; Anthropic may combine them semantically                 |
| `system` versus `developer` identity        | Lost                                                                        |
| Instruction sequence position               | Preserved for Responses and Chat targets; lost for Messages targets         |
| Multipart system/developer message boundary | Lost                                                                        |
| Assistant-message container                 | Lost                                                                        |
| Ordered assistant text/thinking parts       | Usually retained by adjacency, but may be merged or split during projection |
| Responses output-item boundary              | Preserved for non-interleaved items                                         |
| Responses inner content-part boundary       | Lost                                                                        |
| Chat choices beyond index zero              | Lost                                                                        |
| Messages output content-block boundary      | Preserved for supported block types                                         |
| Same-protocol native structure              | Preserved through passthrough, subject to request-wrapper validation        |

## Architecture implications

The current `ProviderCapabilities.thinking` profile remains suitable for classifying ambiguous raw reasoning versus reasoning summaries. Message-boundary behavior is different: it depends on the serving implementation and the selected model renderer or chat template.

Full cross-protocol boundary fidelity would minimally require:

1. An instruction item that retains `system` versus `developer`, content-part grouping, and sequence position.
2. An assistant-message container holding ordered thinking, text, and tool parts.
3. Explicit policy for protocol-native features that cannot be represented by the target, such as multiple Chat choices or Responses inner content indexes.
4. Model- or renderer-level capabilities for supported instruction roles, mid-sequence instructions, required role ordering, developer fallback, and same-role merging.

These model capabilities should not be inferred from `base_url`. `endpoint` still selects the wire protocol, `provider` identifies the serving implementation, and the model renderer determines how the accepted wire conversation becomes tokens.

## Test coverage and gaps

Existing tests establish that Molto preserves user messages and ordered user text parts in both directions. Matched cassette tests cover ordinary single-choice streams, Responses output-item transitions, and Messages thinking followed by text blocks.

There is no focused regression coverage for:

- system versus developer round trips;
- either instruction role in the middle of a translated conversation;
- multipart system or developer messages;
- two source assistant messages with adjacent parts;
- Responses `content_index > 0`;
- interleaved Responses output items;
- multiple Chat choices;
- Chat system/developer content-part serialization; or
- model-template-specific vLLM role rewriting.

## Primary references

- [OpenAI Chat Completions create reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Anthropic Messages create reference](https://platform.claude.com/docs/en/api/messages/create)
- [Anthropic mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- [Anthropic streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM Responses conversion utilities](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/responses/utils/)
- [vLLM Anthropic serving conversion](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/anthropic/serving/)
- [vLLM Hugging Face renderer system-message consolidation](https://docs.vllm.ai/en/latest/api/vllm/renderers/hf/)
- [vLLM Kimi K3 renderer](https://docs.vllm.ai/en/latest/api/vllm/renderers/kimi_k3/)
- [Qwen chat-template concepts](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md#chat-template)
- [Muse Glimmer chat template](https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/chat_template.jinja)

## Relevant Molto implementation

- [`NormalizedRequest` content types](../python/molto/types/normalized.py)
- [`OpenAIResponsesAdapter` and `OpenAIChatCompletionsAdapter`](../python/molto/adapters/openai.py)
- [`AnthropicMessagesAdapter`](../python/molto/adapters/anthropic.py)
- [`OpenAIResponsesClient` and `OpenAIChatCompletionsClient`](../python/molto/upstream/openai.py)
- [`AnthropicMessagesClient`](../python/molto/upstream/anthropic.py)
- [Protocol coordination and same-protocol passthrough](../python/molto/core.py)
