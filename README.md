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

`Molto.responses()` accepts an `OpenAIResponsesBody` or a validated mapping and
returns OpenAI Responses output. `Molto.chat_completions()` likewise accepts an
`OpenAIChatCompletionBody` or mapping and returns Chat Completions output.
Pass the provider and its unprefixed model separately, while keyword-only
`endpoint` selects the execution endpoint:

```python
await Molto().responses(
    {"input": "Summarize this document."},
    provider="openai",
    model="gpt-5.6",
    endpoint="responses",
)
```

The endpoint is consumed before the request becomes `InputIR`; the public
method continues to own the output format. Legacy complete request models with
provider-qualified `model` values remain accepted for direct adapter callers.
Cross-endpoint
conversion is limited to data that both protocols can represent, except for the
documented omission of Responses reasoning items from Chat output.

Transport configuration is per call: both methods accept `api_key`, `base_url`,
`max_retries`, and `http_client`, so a single `Molto` instance can serve
multiple providers and credentials. `Molto` is constructed with only the
`strict` flag, which controls unsupported request parameters: by default they
are dropped and ignored, while `Molto(strict=True)` raises
`ModelCapabilityError` before provider construction.

Models use these prefixes:

- `openai/`: official OpenAI endpoints.
- `vllm/`: OpenAI-compatible endpoints with an explicit `base_url`.
- `anthropic/`: Anthropic's OpenAI-compatible Chat endpoint; the Anthropic-native
  Messages API is not supported by Molto. Both public methods default to this
  Chat upstream for Anthropic models.

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
