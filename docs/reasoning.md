# Reasoning and signatures

This guide describes current implemented behavior for reasoning text, provider profiles, opaque signatures, and continuation across Responses, Chat Completions, and Messages.

The public Molto method selects the application's protocol; `endpoint` selects the provider's protocol. On translated routes, upstream clients produce IR from provider output and consume IR to build provider requests. Downstream adapters produce IR from application requests and consume IR to return application output. These directions have different signature rules.

When both protocols match, native passthrough bypasses reasoning normalization and projection. Provider-specific fields remain in the SDK request/response path, subject to the documented transport adjustments. See [routing](README.md#routing) and [provider compatibility](provider-compatibility.md).

## Text, summaries, and opaque state

The intermediate representation uses `Thinking` parts and `ThinkingDelta` updates:

| IR field                                        | Meaning                                                      |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `Thinking.content`                              | Raw reasoning text                                           |
| `Thinking.summary`                              | A reasoning summary                                          |
| `Thinking.signature`                            | Provider-specific opaque continuation state                  |
| `ThinkingDelta.content_delta` / `summary_delta` | Text to append to the corresponding field                    |
| `ThinkingDelta.signature_delta`                 | A replacement for the previous signature, not text to append |

A signature is not reasoning text. Preserve it without parsing, concatenating, or inventing a value, and return it to its issuing provider with the associated history. Carrying a token across protocol schemas does not make it portable between providers.

The current IR permits raw text, a summary with an optional signature, or signature-only state. It rejects raw text combined with a summary or signature. These are current model constraints, not restrictions imposed by every provider's wire schema.

## Provider profiles

Chat's `reasoning_content` and Messages' `thinking` do not label their text as raw reasoning versus a summary. `ProviderCapabilities.thinking` supplies that classification when these protocols produce IR, including continuation-history normalization.

| Provider | Current `thinking` profile | Qualification |
| --- | --- | --- |
| OpenAI | `summary` | Used for ambiguous reasoning text |
| Anthropic | `summary` | Signed Messages thinking is retained |
| Bedrock | `summary` | Chat reasoning classification remains unverified; signed Messages continuation is verified |
| Gemini | `summary` | Signature-only continuation is verified; thought-marker text mapping is not implemented |
| Fireworks | `raw` | Also has the Responses override described below |
| OpenRouter | `raw` | Registered for Responses only; also classifies ambiguous downstream Chat/Messages history |
| Together AI (`togetherai`) | `raw` | Used for ambiguous reasoning text |
| vLLM | `raw` | Used for ambiguous reasoning text |

Classification places the same text in an IR field; it does not summarize, expand, or regenerate it. Reasoning effort and token-budget settings are separate request controls. `base_url` does not select a profile, and Molto does not infer one from the model name.

The keyword-only coordinator option `reasoning_representation` applies only to clients declaring `REQUIRES_REASONING_REPRESENTATION`. Such clients require `raw` or `summary` on translated routes before normalization or HTTP. No production client currently enables that flag; fixed clients retain their profiles, so this option is not a general override. Native passthrough is exempt.

Profiles currently also constrain parts of output projection and request construction. They are not yet restricted to classification alone. In particular, a summary produced by a Responses client can still fail a downstream projector's raw-profile check.

## Responses labels and provider exceptions

Responses represents raw text as `content[].type="reasoning_text"`, summaries as `summary[].type="summary_text"`, and opaque state as `encrypted_content`. Its shared parser maps those fields to `Thinking.content`, `Thinking.summary`, and `Thinking.signature`. The shared upstream client then validates the result against the effective profile.

Two clients have special behavior:

- **OpenRouter** accepts explicit raw or summary wire labels, using its `responses_reasoning="wire"` capability and validation override. It accepts `format="unknown"` only for raw-only reasoning; named formats and formatted summary/signature state reject. The current stream normalizer still rejects mixed representations within one reasoning item.
- **Fireworks** converts observed `summary_text` snapshots and summary deltas into raw IR text. When building Responses continuation requests, it converts that raw text back into the recorded `summary_text` shape. Native Responses passthrough preserves the original labels. This is a Molto interpretation policy, not evidence that Fireworks performs or omits summarization internally.

**Current OpenRouter limitation:** its Responses client can produce summary IR, but its effective raw profile makes downstream Messages projection reject that summary. Unsigned summaries can project to Chat, whose output projector accepts either text representation. A Responses caller using OpenRouter Responses follows native passthrough and preserves the original labels. Successful raw-reasoning continuation recordings do not establish summary continuation through translated Messages.

## Chat signatures by direction

### Provider Chat response into IR

| Upstream client | Signature handling during translation |
| --- | --- |
| Generic shared `ChatCompletionsClient` | Ignores unmodeled output extensions, including all signature values; no signature enters IR |
| OpenAI | Rejects populated message-level or tool-call `extra_content` |
| vLLM | Inherits Generic behavior; ignores signature extensions |
| Bedrock | Inherits Generic behavior; ignores signature extensions |
| Together AI | Inherits Generic behavior; ignores signature extensions |
| Fireworks | Inherits Generic behavior; ignores signature extensions |
| Gemini | Extracts `extra_content.google.thought_signature` from messages and tool calls and preserves it as signature-only IR |

"Ignores" means neither the first signature nor later values are mapped. It does not mean that the first is retained and later updates are discarded. The shared client still maps supported reasoning text independently of these extensions. OpenRouter and Anthropic have no registered Chat clients.

### IR into a provider Chat request

The shared builder rejects a `Thinking` part whose `signature` is not `None`. Generic, OpenAI, vLLM, Bedrock, Together AI, and Fireworks clients use this behavior. Unsigned raw or summary text becomes `reasoning_content`.

Gemini overrides signature handling: a signature-only `Thinking` prefix attaches to the following text message or tool call through `extra_content.google.thought_signature`. A remaining trailing token becomes message-level state. A signed `Thinking` part containing reasoning text and multiple signatures targeting one attachment reject. This is a Gemini-specific mapping, not a generic namespace mechanism.

### Application Chat requests and responses

The shared downstream Chat adapter applies the same rules regardless of the selected provider:

- **Chat request into IR:** populated `extra_content` on a message, tool call, or content block rejects. The adapter does not extract signatures from application-supplied Chat history.
- **IR into Chat output:** nonempty signatures reject at part start, signature delta, or part end, including signature-only parts. Both streaming and collected non-streaming responses use this projection.

Consequently, a Responses or Messages application can preserve signatures while using Gemini's Chat upstream, but signed IR cannot be returned as translated Chat output. Native Chat-to-Chat passthrough bypasses these checks and preserves SDK extension fields.

## Gemini attachment and buffering

Gemini records message-level signatures separately from signatures attached to individual tool calls. Its client keeps the latest observed token for each attachment and emits signature-only `Thinking` prefixes:

- A tool-call signature prefixes that call.
- A message signature prefixes the first text part when text exists.
- Without text, a message signature prefixes the first call if that call is unsigned. If the call already has its own signature, the separate message token remains standalone.

History restoration attaches prefixes forward. Do not mix history produced by this convention with the former trailing-signature convention. The [provider guide](provider-compatibility.md#provider-differences) describes the evidence and limitations of the no-text attachment approximation.

**Translated Gemini output currently buffers until the upstream stream completes.** Only `ResponseStartEvent` is yielded immediately; the remaining normalized events, including text and tool-call deltas, are retained. After completion the client inserts signature prefixes, reindexes the parts, and emits the buffered events. Native passthrough does not use this buffering.

Late message signatures are recorded behavior. In [Gemini text turn 00](../tests/cassettes/upstream_ir/text_chat/gemini-chat_completions-stream-00.yaml), JSON chunks 1-22 contain text and chunk 23 supplies the message signature with `finish_reason="stop"`. Both committed tool-bearing streams instead carry arguments and their signature together. These recordings justify handling late message signatures; they do not establish late tool signatures or signature revisions. The implementation nevertheless buffers tool-only translated responses too.

## Continuation and unsigned summaries

Supporting reasoning output does not guarantee that the same state can be sent to every upstream endpoint:

| Upstream request builder | Current continuation constraint                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| Responses                | Rejects a nonempty summary without a signature; raw history does not require one           |
| Messages                 | Rejects an unsigned summary and requires the reasoning representation to match its profile |
| Shared Chat              | Accepts unsigned raw or summary text; rejects signatures                                   |
| Gemini Chat              | Additionally accepts its signature-only prefix convention                                  |

Downstream request normalization can preserve an unsigned summary even when a later upstream builder rejects it. Shared output parsing can also accept unsigned summaries, subject to profile checks during projection. Accepting the text does not recover missing opaque state.

Chat and Messages collapse reasoning into one unlabeled text field. Reusing that output as history classifies it again using `thinking`, so the original raw/summary label can be lost. Incoming Messages history under a raw profile rejects nonempty signatures; the upstream Messages raw-profile normalizer does not retain signature state. These are current restrictions, not a recommendation to strip signed provider history.

## Streaming and accumulation

Text deltas append; IR signature deltas replace. Molto's Messages adapter forwards signature updates, and its collector retains the latest value per thinking block. The Anthropic SDK follows the same replacement behavior.

For translated non-streaming Chat output, Molto uses OpenAI's `ChatCompletionStreamState` to collect projected chunks. That SDK accumulator recursively concatenates strings at the same field path, including unknown extensions. For example, two hypothetical deltas for the same choice:

```jsonl
{"extra_content": {"anthropic": {"thought_signature": "A"}}}
{"extra_content": {"anthropic": {"thought_signature": "B"}}}
```

accumulate to signature `AB`, not `B`. Intervening text does not reset the field. This was verified with OpenAI SDK 2.53.0; raw `create(stream=True)` iteration yields the separate values without collecting them.

Current translated Chat output rejects signatures, so it does not emit this hypothetical sequence. Adding generic signature output would require an explicit accumulation and attachment contract. The existing implementation does not implement signature selection per contiguous reasoning run.

Chat has one output message per choice. Multiple chunks or IR parts do not create new output messages, and different choices represent alternative completions. Request history can contain multiple assistant messages, but that does not give a single Chat response multiple sequential message slots.

## Current field-mapping limits

- Shared Chat normalization reads `reasoning_content`, not the alternate `reasoning` field. There is no caller-selectable field-name override yet.
- Gemini recognizes `extra_content.google.thought_signature`, but not the `extra_content.google.thought` marker as a reasoning classifier. Marked `content` currently follows ordinary text handling.
- `<reasoning>` tags inside normal text remain normal text; Molto does not extract them.
- There is no generic `extra_content.<provider>.thought_signature` translation contract.

See the [provider evidence matrix](provider-compatibility.md#replay-evidence) for tested model/settings combinations. A profile or accepted schema does not establish provider-wide reasoning support.
