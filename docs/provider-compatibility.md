# Provider compatibility

Each added provider has one explicit upstream client, choosing Responses when available, then Messages, then Chat Completions. Thin subclasses share the protocol implementation. `molto.upstream.registry.CLIENTS` selects the client; unsupported provider/endpoint pairs fail locally instead of silently switching protocols.

Existing OpenAI Responses and Chat Completions, Anthropic Messages, and vLLM routes remain available. Cross-vendor configurations such as OpenAI Messages and Anthropic Responses are unsupported.

## Selected protocols and configuration

| Provider    | Client                        | Default SDK base URL                                      | Credential                 |
| ----------- | ----------------------------- | --------------------------------------------------------- | -------------------------- |
| Together AI | `TogetherAIResponsesClient`   | `https://api.together.ai/v1`                              | `TOGETHER_AI_API_KEY`      |
| Fireworks   | `FireworksResponsesClient`    | `https://api.fireworks.ai/inference/v1`                   | `FIREWORKS_API_KEY`        |
| OpenRouter  | `OpenRouterResponsesClient`   | `https://openrouter.ai/api/v1`                            | `OPENROUTER_API_KEY`       |
| Bedrock     | `BedrockResponsesClient`      | Explicit regional URL required                            | `AWS_BEARER_TOKEN_BEDROCK` |
| Gemini      | `GeminiChatCompletionsClient` | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY`           |

An explicit `api_key` overrides the configured provider key; an empty key fails locally. Missing provider keys cannot fall back to OpenAI credentials. An explicit `base_url` overrides the default without changing the provider or protocol. Authentication and account headers supplied through `extra_headers` cannot override the selected upstream identity.

Select the upstream protocol explicitly when it differs from the public method. For example, `Molto.chat_completions(..., provider="openrouter", endpoint="responses")` translates Chat input through the Responses client. The existing method defaults remain unchanged.

Protocol availability is documented by [Together](https://docs.together.ai/docs/inference/responses-api), [Fireworks](https://docs.fireworks.ai/api-reference/post-responses), [OpenRouter](https://openrouter.ai/docs/api_reference/responses/overview), [Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-responses-api.html), and [Gemini](https://ai.google.dev/gemini-api/docs/openai). Availability does not establish support for every model or feature. Together's Responses documentation lists GLM-5.2 and GLM-5.3 families; its earlier Chat model recordings do not validate this protocol.

## Provider differences

OpenRouter Responses identifies raw text and summaries explicitly. Its client accepts unsigned raw reasoning, including the recorded `format: "unknown"`, or a summary with its required signature. Named formats, formatted summaries, mixed representations, and mid-item representation changes reject. Native passthrough preserves the original fields. Raw reasoning projects to Chat and Messages; signed summaries can project to Messages. Incoming unsigned Messages thinking still follows the summary-input contract and rejects, so this output support does not establish a complete Messages reasoning round trip.

Fireworks Responses uses the summary profile. Its recorded low-reasoning output contains unsigned summaries, which remain valid native evidence but reject during normalization. Together uses the raw reasoning profile.

Gemini Chat preserves opaque `extra_content.google` attachments on messages, content parts, and function calls, including nested nulls. Translation rejects populated opaque state. Missing streamed tool indices are repaired only when the call identity is unambiguous; ambiguous output rejects rather than losing calls.

## Replay evidence

Provider conversations share `tests/upstream/test_multi_turn.py`. Cassettes use the existing scenario-first layout, for example `tests/cassettes/upstream_ir/tool_use/togetherai-responses-stream-00.yaml`. Each provider uses its own recordings. Canonical OpenAI/Anthropic fixtures are not reused as evidence of hosted-provider behavior.

The suite checks native SDK parity, actual-output history, normalized lifecycle and semantics, exact expected subsequent requests, and focused public translation or rejection cases. Independent SDK responses supply the comparison values. Expected normalized request copies account for documented wire representations and omitted optional metadata; recorded fixtures are not rewritten to match Molto output.

| Provider/model | Retained conversation evidence | Remaining limits |
| --- | --- | --- |
| Together `zai-org/GLM-5.2` | Two-turn text and three-turn sequential tools, streaming and non-streaming, with minimal reasoning | Other models, images, parallel calls, and reasoning-enabled continuation are unverified |
| Fireworks `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` | Text, sequential tools, and parallel tools in both native modes with reasoning disabled; streamed unsigned-summary tool history | Unsigned summaries cannot translate; other models and images are unverified |
| OpenRouter `nvidia/nemotron-3.5-lightning:free` | Text in both native modes; non-streaming sequential tools; streamed raw-reasoning sequential tools | Parallel Responses calls and public translated Messages reasoning round trips are unverified |
| Gemini | Configuration, Google attachment, and missing-index regressions | Live recording was blocked by the provider's location restriction |
| Bedrock | Configuration and credential isolation | No live recording: credentials and model/profile were unavailable |

The retained OpenRouter captures use an explicit free model, disabled provider fallbacks, zero price caps, and no plugins. Reported costs were zero. Responses history is manual with `store=false`. Captures establish only the recorded model, settings, and date; replay does not prove that translated request bodies were executed live.

The duplicate PR #76 corpus, per-run manifests, source snapshots, and unsupported-protocol captures are omitted. Full issue #78 coverage still requires the remaining provider/model cells and the separate Anthropic continuation work in #74/#92.
