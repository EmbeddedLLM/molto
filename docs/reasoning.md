# Reasoning and signatures

On translated routes, Molto stores raw reasoning in `Thinking.content`, a summary in `Thinking.summary`, and opaque continuation state in `Thinking.signature`. All three fields are independent. Text deltas append to their corresponding field. A signature delta replaces the previous value for its attachment. Return opaque state to the provider that issued it.

## Protocol mapping

| Protocol | Protocol to IR | IR to protocol |
| --- | --- | --- |
| Responses | `reasoning_text` supplies raw content, `summary_text` supplies a summary, and `encrypted_content` supplies the signature. | Preserve both text fields and any signature independently, including an item whose only populated field is its ID. |
| Chat Completions | Classify unlabeled reasoning text using the provider profile. Read signatures from the configured extension namespace. | Emit one text representation in the selected Chat field. Carry signatures at their message or tool-call attachment when a namespace is configured. |
| Messages | Classify `thinking` text using the provider profile. Preserve the block signature independently. | Emit one text representation in a thinking block and preserve any signature. |

A complete history part sent to Chat or Messages uses nonempty raw content first, then the summary. When projecting a stream, the first nonempty representation wins for that part. If one snapshot supplies both, raw content wins. For example, summary A, raw B, summary C produces A and C. A signature or an empty start leaves the choice open. Each new part selects independently. Translated non-streaming responses collect the same projected stream.

Responses and Messages request builders accept unsigned summaries and omit absent signature state. Endpoint and model requirements still apply. A provider that requires its original signature may reject unsigned history.

## Provider profiles

The `thinking` profile only classifies text that Chat or Messages leaves unlabeled. It applies to provider output and to application-supplied continuation. OpenAI, Anthropic, Gemini, and Bedrock Responses and Messages use `summary`. vLLM, Together, OpenRouter, Fireworks, and Bedrock Chat use `raw`. The provider and selected client determine this profile. A base URL does not select it.

A Responses summary projected into Chat or Messages loses its wire label. If returned as history through a raw profile, the next Responses request carries that text as `reasoning_text`. This applies to OpenRouter and Fireworks. Molto preserves the text without inferring how the provider will use it.

The registered clients use fixed profiles. Native same-protocol requests bypass reasoning translation.

## Chat fields and signatures

Chat accepts both `reasoning_content` and `reasoning`. The configured nonempty field takes precedence when both occur. vLLM and Bedrock Chat default to `reasoning`. Other Chat clients default to `reasoning_content`. Set `chat_reasoning_field="reasoning_content"` on the Molto call for an older vLLM server. The selected spelling controls translated output and outgoing Chat history.

When `extra_content.google.thought` is `true`, its associated content is reasoning. False or absent markers leave the text visible. Literal `<reasoning>` tags in normal content remain visible text.

Use `chat_signature_namespace` to select the issuer's Chat extension. For example, `chat_signature_namespace="google"` maps `extra_content.google.thought_signature`. Gemini selects `google` by default. The namespace identifies the wire field. It does not establish that another provider accepts the token. Signed history or output projected into Chat requires a namespace.

Within a contiguous reasoning run, Molto holds the latest message signature and keeps streaming reasoning text. It emits the signature at the next text or tool boundary, or at normal completion. Empty metadata does not close a run. A signature-only IR part directly before a tool call supplies that call’s signature. If two signature-only parts precede the call, the first carries message state. A message-only prefix immediately before an unsigned call uses the same position as a call signature and cannot distinguish those two attachments. Recorded Gemini tool calls supply their signature with the arguments. Gemini also sends a message signature after the last visible text chunk. Molto emits this trailing state at completion and restores it on the assistant message when it builds continuation. Text and tool output stream without waiting for a late message signature.

Chat history can carry independent reasoning parts in separate assistant messages. Chat output has one message per choice. Streaming delivers the final signature for each run separately. OpenAI SDK 2.53.0 concatenates strings at the same extension path when it collects a choice, even across intervening text. Two runs that emit message signatures `B` and `D` therefore appear as `BD` in an SDK-collected message, including translated non-streaming Chat. Raw chunk iteration retains each value. Separate tool indices retain separate attachment paths.

See [provider compatibility](provider-compatibility.md#replay-evidence) for the models, settings, and continuation evidence.
