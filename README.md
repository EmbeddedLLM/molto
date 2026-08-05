# Molto

Molto is a Python gateway for OpenAI-compatible Chat Completions and Responses.
Every request follows the same path:

```text
public request -> InputIR -> OpenAI-compatible SDK endpoint
              -> OutputEvent stream -> OutputIR -> public response
```

The IR produces OpenAI-compatible response events. It also produces typed,
validated OpenAI stream frames and response snapshots. Molto does not expose
provider-native events or generic passthrough data. It rejects data that it
cannot represent completely.

`Molto.responses()` accepts a complete OpenAI Responses request and returns a
Responses result. `Molto.chat_completions()` does the same for Chat Completions.
The `model` field stays in the OpenAI-style request body. A bare model name uses
the OpenAI-compatible adapter. A provider-qualified model name selects that
provider's adapter. The keyword-only `endpoint` argument chooses where to run
the request:

```python
await Molto().responses(
    {"model": "gpt-5.6", "input": "Summarize this document."},
    endpoint="responses",
)
```

Molto removes `endpoint` before converting the request to `InputIR`. The public
method still controls the output format. Use `OpenAIResponsesBody` or
`OpenAIChatCompletionBody` when you want to pass the provider and model
separately.

Molto converts data between endpoints only when both formats can represent it.
The one documented exception is that Responses reasoning items are omitted
from Chat output (Will be supported in the following updates).

Both methods accept `api_key`, `base_url`, `max_retries`, and `http_client` for
each call. This lets one `Molto` instance use different providers and
credentials. The constructor only accepts `strict`. By default, Molto ignores
unsupported request parameters. With `Molto(strict=True)`, it raises
`ModelCapabilityError` before creating the provider.

Models use these prefixes:

- `openai/`: official OpenAI endpoints.
- `vllm/`: OpenAI-compatible endpoints that use an explicit `base_url`.
- `anthropic/`: Anthropic's OpenAI-compatible Chat endpoint. Molto does not
  support Anthropic's native Messages API. Both public methods use the Chat
  endpoint by default for Anthropic models.

Molto supports generated Chat audio through the Chat Completions destination.
It supports standard OpenAI hosted tools through the matching Responses route.
Molto rejects hosted tools, MCP, reasoning, or audio when the chosen destination
cannot represent the complete result.

To continue a Responses request without stored state, append the previous
`response.output` items to the next request's `input` and set `store=False`.
This works with assistant messages, reasoning, web search, code interpreter,
and image generation. Image generation also supports the documented
ID-only `image_generation_call` reference.

See [Architecture](python/ARCHITECTURE.md), the
[OpenAI compatibility matrix](python/OPENAI_COMPATIBILITY.md).
