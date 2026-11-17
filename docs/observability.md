# Observability

Status: planned, not implemented yet. This page defines the tracing contract Molto will implement; Molto does not emit spans today.

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

To instrument only the client Molto uses, pass an instrumented `httpx.AsyncClient` instead:

```python
from httpx import AsyncClient
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from molto import Molto

http_client = AsyncClient()
HTTPXClientInstrumentor.instrument_client(http_client)
molto = Molto(http_client=http_client)
```

The SDK call runs while the Molto span is current, so the HTTP spans nest under it.

An `httpx` client span ends when the response headers arrive, not when a stream finishes. For a streaming call it measures only the time to first byte. The Molto span covers the whole generation, and its `molto.first_token` event records the time to first token.

## Span contract

The span follows the [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) for inference spans. Those conventions have Development status, so attribute keys may still change upstream.

| Property | Value                                                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Kind     | `CLIENT`                                                                                                                            |
| Name     | `{gen_ai.operation.name} {gen_ai.request.model}`, for example `chat gpt-5`; `{gen_ai.operation.name}` when the request has no model |
| Status   | `UNSET` on success; `ERROR` only on a genuine failure                                                                               |
| Parent   | The span active when the coordinator is awaited                                                                                     |

Molto creates no separate spans for adapters, normalizers, upstream clients, or translation. Translation runs interleaved with streaming, event by event, rather than as a separate phase, so a translation span would not isolate its cost.

### Operation name

`gen_ai.operation.name` is `chat` for all three coordinators, so a Messages call is also named `chat {model}`. The attribute holds the cross-vendor operation type, multi-turn generation, not Molto's protocol name. Observability backends recognize LLM spans by its well-known values, so a custom value such as `responses` or `messages` would leave Molto spans unrecognized. The protocols are carried by `molto.downstream.protocol` and `molto.upstream.endpoint`.

### Streaming lifetime

A streaming coordinator returns a `MoltoStream` after prefetching the first event. The span outlives that `await`: it ends exactly once, when the stream is exhausted, fails, or is closed with `aclose()`. A failure during prefetch ends the span in the coordinator.

A stream that is neither consumed nor closed leaves its span unended, and the span is not exported, just as its upstream response is not released. Use `async with` or `aclose()`.

The span starts in the caller's context, so iterating the stream from another task does not reparent it. The Molto span is current only while the upstream request is sent: for the whole call when non-streaming and for the prefetch when streaming. It is not current while the caller iterates the stream.

### Attributes known at start

Molto resolves these values before transport.

| Key | Value |
| --- | --- |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | Resolved `provider`: `openai`, `anthropic`, or `vllm` |
| `gen_ai.request.model` | Request `model` |
| `gen_ai.request.max_tokens`, `gen_ai.request.temperature`, `gen_ai.request.top_p` | Request fields, only when set |
| `gen_ai.request.stream` | Set only when streaming |
| `server.address`, `server.port` | Effective `base_url` |
| `molto.downstream.protocol` | `responses`, `chat/completions`, or `messages`, from the coordinator |
| `molto.upstream.endpoint` | Effective `endpoint` |
| `molto.route` | `native` or `translated` |

`vllm` is not a well-known `gen_ai.provider.name` value; the conventions allow a custom value for other providers. Molto has no separate `molto.provider` attribute, because it would duplicate this value and `server.address` already identifies the host.

### Attributes known at end

| Key | Translated route | Native route |
| --- | --- | --- |
| `gen_ai.response.id` | `ResponseStartEvent` | SDK response or terminal event |
| `gen_ai.response.model` | `ResponseStartEvent` | SDK response or terminal event |
| `gen_ai.response.finish_reasons` | `ResponseEndEvent.termination_reason` | SDK finish or stop reason |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | `ResponseEndEvent` normalized usage | SDK usage |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | Normalized usage details, when present | SDK usage details, when present |
| `error.type`, `http.response.status_code`, `molto.error.phase` | See [Errors](#errors) | See [Errors](#errors) |

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

The exception message appears only on the `record_exception` event, never on an attribute, so `error.type` stays low-cardinality. The recorded message must not contain request content.

### Events

| Event | When |
| --- | --- |
| `molto.first_token` | Streaming only, at the first content delta: text, reasoning, or tool-call arguments. Translated routes use the first `PartDeltaEvent`; native routes use the first delta event of their protocol. |

The event marks the first content delta rather than the first protocol event. Responses `response.created` arrives before any generation, so the first-event time would only repeat the `httpx` span's time to first byte.

## Content and privacy

Spans never carry prompts, completions, tool arguments, headers, or credentials. Content capture is not available; an opt-in would be a separate, explicitly approved change.

Telemetry is data, not behavior. Span attributes are derived from values the coordinator already resolved or from events the pipeline already emitted. Instrumentation never changes routing, validation, request construction, projection, or errors, and it never reads the `extra_body` or `extra_headers` request extensions.
