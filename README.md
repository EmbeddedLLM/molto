# Molto

Molto is a Python gateway for OpenAI-compatible Chat Completions and Responses
requests. Every request follows the same boundary:

```text
public request -> InputIR -> OpenAI-compatible SDK endpoint
              -> OutputEvent stream -> OutputIR -> public response
```

The IR emits OpenAI-compatible semantic response events plus typed, validated
OpenAI stream-frame and response-snapshot events. Provider-native event models
and generic passthrough payloads are not part of the contract; data that cannot
be represented completely is rejected.

`Molto.responses()` accepts `OpenAIResponsesRequest` and returns OpenAI Responses
output. `Molto.chat_completions()` accepts `OpenAIChatCompletionRequest` and
returns Chat Completions output. Their keyword-only `upstream` parameter selects
the execution endpoint and defaults to the matching protocol. Cross-endpoint
conversion is limited to data that both protocols can represent, except for the
documented omission of Responses reasoning items from Chat output.

Models use these prefixes:

- `openai/`: official OpenAI endpoints.
- `vllm/`: OpenAI-compatible endpoints with an explicit `base_url`.
- `anthropic/`: Anthropic's OpenAI-compatible Chat endpoint; the Anthropic-native
  Messages API is not supported by Molto.

Generated Chat audio is supported through the Chat Completions destination.
Standard OpenAI Responses hosted tools are supported on the matching Responses
route. Hosted tools, MCP, reasoning, or audio fail closed when the selected
cross-format destination cannot represent the complete result.

Stateless Responses continuation is supported for assistant messages,
reasoning, web-search calls, code-interpreter calls, and image-generation calls
by appending the prior `response.output` items to the next request's `input` and
setting `store=False`. Image generation also supports the OpenAI-documented
`image_generation_call` ID-only reference.

See [Architecture](python/ARCHITECTURE.md), the
[OpenAI compatibility matrix](python/OPENAI_COMPATIBILITY.md), and
[Onboarding](ONBOARDING.md).
