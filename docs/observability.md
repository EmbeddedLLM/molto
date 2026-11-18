# Observability

Molto emits one OpenTelemetry span for each coordinator call: `Molto.responses()`, `Molto.chat_completions()`, or `Molto.messages()`. The span shows what Molto did with the call: the route it chose, the downstream and upstream protocols it translated between, the provider, token usage, finish reasons, and why a call failed.

Molto depends only on `opentelemetry-api`. It obtains its tracer with `trace.get_tracer("molto", __version__)` and never configures a tracer provider, exporter, sampler, or resource. With no SDK configured, the API returns no-op spans and calls behave identically.

Spans for the application's own work, such as request handling, authentication, guardrails, or billing, belong to the application, not to Molto. Molto does not emit metrics or log events, and does not wrap the OpenAI or Anthropic SDKs with third-party instrumentors.

## Enabling export

The application owns the OpenTelemetry SDK and its configuration:

1. Install `opentelemetry-sdk` and an exporter, such as `opentelemetry-exporter-otlp-proto-http`.
2. Configure a tracer provider once, at application startup.
3. Configure the exporter and resource with the standard `OTEL_*` environment variables, such as `OTEL_SERVICE_NAME` and `OTEL_EXPORTER_OTLP_ENDPOINT`.

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
```

### HTTP child spans

Molto calls providers through the OpenAI and Anthropic SDKs over `httpx`, and does not create HTTP spans itself. To trace the upstream HTTP requests, install `opentelemetry-instrumentation-httpx` and instrument `httpx`:

```python
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

HTTPXClientInstrumentor().instrument()
```

Global instrumentation wraps `httpx`'s network transport, so it covers Molto's default client and any injected `httpx.AsyncClient` that uses that transport, including clients created before `instrument()` is called. It does not see a client built on a custom transport, such as `httpx.MockTransport`.

To instrument only the client Molto uses, or a client with a custom transport, pass an instrumented `httpx.AsyncClient` instead:

```python
from httpx import AsyncClient
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from molto import Molto

http_client = AsyncClient()
HTTPXClientInstrumentor.instrument_client(http_client)
molto = Molto(http_client=http_client)
```

The SDK call runs while the Molto span is current, so the HTTP spans nest under it, on native and translated routes, streaming and non-streaming.

An `httpx` client span ends when the response headers arrive, not when a stream finishes. For a streaming call it measures only the time to first byte. The Molto span covers the whole generation, and its `molto.first_token` event records the time to first token.

## Span contract

The span follows the [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) for inference spans. Those conventions have Development status, so attribute keys may still change upstream.

| Property | Value                                                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Kind     | `CLIENT`                                                                                                                            |
| Name     | `{gen_ai.operation.name} {gen_ai.request.model}`, for example `chat gpt-5`; `{gen_ai.operation.name}` when the request has no model |
| Status   | `UNSET` on success and on cancellation; `ERROR` only on a genuine failure                                                           |
| Parent   | The span active when the coordinator is awaited                                                                                     |

Molto creates no separate spans for adapters, normalizers, upstream clients, or translation. Translation runs interleaved with streaming, event by event, rather than as a separate phase, so a translation span would not isolate its cost.

### Operation name

`gen_ai.operation.name` is `chat` for all three coordinators, so a Messages call is also named `chat {model}`. The attribute holds the cross-vendor operation type, multi-turn generation, not Molto's protocol name. Observability backends recognize LLM spans by its well-known values, so a custom value such as `responses` or `messages` would leave Molto spans unrecognized. The protocols are carried by `molto.downstream.protocol` and `molto.upstream.endpoint`.

### Streaming lifetime

A streaming coordinator returns a `MoltoStream` after prefetching the first event. The span outlives that `await`: it ends exactly once, when the stream is exhausted, fails, is closed with `aclose()`, or is garbage-collected. A failure during prefetch ends the span in the coordinator.

A stream that is neither consumed nor closed leaks its upstream HTTP response, because closing it requires an `await` that garbage collection cannot perform. Use `async with` or `aclose()`. To make the leak visible, a dropped stream ends its span when it is garbage-collected, with `molto.stream.outcome=abandoned`, at the time of the last event the caller received rather than the collection time. CPython collects an object as soon as its last reference goes, later when it is part of a reference cycle, and possibly never at interpreter shutdown, so an abandoned span can arrive late or not at all.

Every streaming span records how the stream ended in `molto.stream.outcome`:

| Value | When |
| --- | --- |
| `completed` | The caller read the stream to its end. Closing it afterwards keeps `completed`. |
| `closed` | The caller called `aclose()` before the end, for example by leaving `async with` early. |
| `cancelled` | The caller's task was cancelled, while reading the stream or before the first event arrived. This is how a client hanging up appears. The status stays `UNSET` and the cancellation is re-raised. Any other exit that is not an `Exception`, such as `KeyboardInterrupt`, also reports `cancelled`, because it too is the caller stopping the call rather than the call failing. |
| `error` | The call failed, while the caller read the stream or before the first event. `molto.error.phase` says which, and the [error attributes](#errors) describe the failure. |
| `abandoned` | The caller dropped the stream without consuming or closing it. The span ends at the last event received; its upstream response was leaked. |

Non-streaming spans do not carry the attribute.

The span starts in the caller's context, so iterating the stream from another task does not reparent it. The Molto span is current only while the upstream request is sent: for the whole call when non-streaming and for the prefetch when streaming. It is not current while the caller iterates the stream.

### Attributes known at start

Molto resolves these values before transport.

| Key | Value |
| --- | --- |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | Resolved `provider`: `openai`, `anthropic`, or `vllm` |
| `gen_ai.request.model` | Request `model` |
| `gen_ai.request.max_tokens`, `gen_ai.request.temperature`, `gen_ai.request.top_p` | Request fields, only when set with the expected type |
| `gen_ai.request.stream` | Set only when streaming |
| `server.address`, `server.port` | Host and port of the effective `base_url`; the scheme's default port when the URL has none |
| `molto.downstream.protocol` | `responses`, `chat/completions`, or `messages`, from the coordinator |
| `molto.upstream.endpoint` | Effective `endpoint` |
| `molto.route` | `native` or `translated` |

Native request bodies reach the SDK unvalidated, so a request field becomes an attribute only when it has the convention's type: a non-empty string model, an integer output-token limit, and numeric sampling values, reported as doubles. Booleans do not count as numbers. Chat Completions reports `max_completion_tokens`, falling back to the deprecated `max_tokens`. A `base_url` without a host, or one `httpx` cannot parse, omits the server attributes and leaves the error to the SDK.

`vllm` is not a well-known `gen_ai.provider.name` value; the conventions allow a custom value for other providers. Molto has no separate `molto.provider` attribute, because it would duplicate this value and `server.address` already identifies the host.

### Attributes known at end

| Key | Translated route | Native route |
| --- | --- | --- |
| `gen_ai.response.id` | `ResponseStartEvent` | SDK response or terminal event |
| `gen_ai.response.model` | `ResponseStartEvent` | SDK response or terminal event |
| `gen_ai.response.finish_reasons` | `ResponseEndEvent.termination_reason` | SDK finish or stop reason; for Responses, the incomplete reason or terminal status |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | `ResponseEndEvent` normalized usage | SDK usage |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | Normalized usage details, when non-zero | SDK usage details, when non-zero |
| `error.type`, `http.response.status_code`, `molto.error.phase` | See [Errors](#errors) | See [Errors](#errors) |
| `molto.stream.outcome` | Streaming only; see [Streaming lifetime](#streaming-lifetime) | Same |

Translated spans report Molto's normalized termination reason, such as `stop`, `max_output_tokens`, or `tool_call`. Native spans report the provider's own value, such as Chat Completions `stop` or `length` and Messages `end_turn`. The same call can therefore report different finish reasons on its two routes: an Anthropic call that ends normally reports `["end_turn"]` natively and `["stop"]` through a translated route. Group by `molto.route` to compare them. Responses has no finish-reason field, so a native Responses span reports `incomplete_details.reason` for an incomplete response, such as `max_output_tokens`, and the terminal `status`, such as `completed`, otherwise.

Normalized usage stores a missing cache count as zero, so both cache attributes appear only when non-zero, on every route, as the conventions recommend. Native cache counts come from Responses `input_tokens_details.cached_tokens` and `cache_write_tokens`, Chat Completions `prompt_tokens_details.cached_tokens` and `cache_write_tokens`, and Messages `cache_read_input_tokens` and `cache_creation_input_tokens`.

Molto reports the provider's token counts. The one adjustment is native Messages input: the Anthropic SDK `input_tokens` excludes cached tokens, so a native Messages span reports `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. This matches the normalized usage a translated route reports for the same call.

`gen_ai.usage.cache_write.input_tokens` is the current spec key. Molto defines attribute keys as its own string constants rather than importing `opentelemetry-semantic-conventions`, which as of 0.66b2 defines only the older `gen_ai.usage.cache_creation.input_tokens`. A backend built on that package may not recognize the new key yet.

A native Chat Completions stream reports usage only when the caller sets `stream_options.include_usage`. Molto never injects that option, because native passthrough forwards the request unchanged, so these spans omit the usage attributes when the stream reports none.

### Errors

On failure, the span status is `ERROR`, the exception is recorded with `record_exception`, and these attributes are set:

| Key | Value |
| --- | --- |
| `error.type` | The normalized `APIException` subclass name when the exception or its `__cause__` is one, for example `ContextLengthExceededError`; otherwise the raw exception class name, for example `RateLimitError` or `UnsupportedInput` |
| `http.response.status_code` | Set only from a real upstream error response with status 400 or higher: the normalized cause's `response` when a Molto cause exists, otherwise the status of an SDK `APIStatusError`. Omitted when no such response exists |
| `molto.error.phase` | `setup` when the failure is raised before the coordinator returns, such as a local rejection, credential error, or HTTP error; `stream` when it is raised while the caller iterates |

For a recognized upstream failure, the caller receives a downstream SDK exception whose `__cause__` is the normalized Molto exception, so `error.type` checks `__cause__` rather than reporting the caller-specific SDK class. Native and translated routes therefore report the same `error.type` for the same upstream failure. Unknown errors pass through unchanged and have no Molto cause, and `UnsupportedInput` is raised directly, so their own class is the `error.type`. A raw class name such as `RateLimitError` exists in both SDKs; `molto.upstream.endpoint` identifies which SDK raised it. See [Errors](errors.md) for the cause chain and recognition rules.

`http.response.status_code` is never synthesized. A missing-credentials error carries a synthetic `401` on the outward exception, but no HTTP call occurs and its normalized cause has no response, so the attribute is omitted. A Responses context-length failure during streaming arrives on an HTTP `200` response; the attribute is omitted there too, and `error.type` and `molto.error.phase` describe the failure.

The exception is recorded as one `exception` event without its message, because exception messages can contain request content: a rejected request's validation error quotes its input, and an upstream error can echo a request field. The event's `exception.message` is the exception's class name, and its `exception.stacktrace` lists the class and stack frames of the exception and each `__cause__` in its chain, without any messages. Frame lines show code locations, never runtime values. The type, frames, and error attributes identify the failure without the message, which never appears on an attribute, so `error.type` stays low-cardinality.

### Events

| Event | When |
| --- | --- |
| `molto.first_token` | Streaming only, at the first delta with non-empty content: text, a refusal, reasoning, or tool-call arguments. Translated routes use the first such `PartDeltaEvent`; native routes use the first such delta event of their protocol. |

The event marks the first content delta rather than the first protocol event. Responses `response.created` arrives before any generation, so the first-event time would only repeat the `httpx` span's time to first byte.

Content deltas are, per protocol:

| Route | Content deltas |
| --- | --- |
| Translated | `TextDelta`; `ThinkingDelta` with reasoning or a summary; `ToolCallDelta` with arguments |
| Native Responses | `response.output_text.delta`, `response.refusal.delta`, `response.reasoning_text.delta`, `response.reasoning_summary_text.delta`, `response.function_call_arguments.delta`, `response.custom_tool_call_input.delta`, `response.mcp_call_arguments.delta` |
| Native Chat Completions | A choice delta with `content`, `refusal`, the compatible-provider `reasoning_content` extension, or tool-call `function.arguments` |
| Native Messages | `content_block_delta` with `text_delta`, `thinking_delta`, or `input_json_delta` |

Empty deltas do not count, and neither do deltas that carry no generated content: a reasoning signature, which replaces opaque state, a tool call's name or ID alone, or a role.

## Content and privacy

Spans never carry prompts, completions, tool arguments, headers, or credentials. Content capture is not available; an opt-in would be a separate, explicitly approved change.

Telemetry is data, not behavior. Span attributes are derived from values the coordinator already resolved or from events the pipeline already emitted. Instrumentation never changes routing, validation, request construction, projection, or errors, and it never reads the `extra_body` or `extra_headers` request extensions. Native passthrough does not validate upstream output, so a field a provider omits is left off the span rather than raising.
