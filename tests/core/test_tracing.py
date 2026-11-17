import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import openai
import pytest
from httpx import (
    AsyncByteStream,
    AsyncClient,
    AsyncHTTPTransport,
    Headers,
    MockTransport,
    Request,
    Response,
)
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import SecretStr, ValidationError

from molto.configs import ENV_CONFIG
from molto.core import Molto, MoltoStream, _prefetch_stream, _traced
from molto.types.exceptions import ContextLengthExceededError
from molto.utils.io import json_dumps, json_loads, read_yaml
from molto.utils.telemetry import ATTRIBUTE_KEYS
from molto.utils.test import replay_transport, replayed_outcomes
from molto.version import __version__

BODY = {"model": "gpt-5", "input": "Hello!"}
# Placed in request content, extensions, headers, and credentials; it must never
# reach a span.
CONTENT_MARKER = "secret-content-marker"
# Event attribute keys Molto may record; anything else could carry content.
EVENT_ATTRIBUTE_KEYS = {
    "molto.first_token": set(),
    "exception": {
        "exception.type",
        "exception.message",
        "exception.stacktrace",
        "exception.escaped",
    },
}
CASSETTES = Path(__file__).parents[1] / "cassettes"


@pytest.fixture
def tracer_provider(monkeypatch: pytest.MonkeyPatch) -> TracerProvider:
    """Route Molto's tracer to a local SDK provider, leaving the global provider untouched."""
    provider = TracerProvider()
    monkeypatch.setattr("molto.core.get_tracer", lambda: provider.get_tracer("molto"))
    return provider


@pytest.fixture(autouse=True)
def exporter(
    tracer_provider: TracerProvider, caplog: pytest.LogCaptureFixture
) -> Iterator[InMemorySpanExporter]:
    """Collect every span the local provider ends, and fail if one ends twice.

    The SDK ignores a second `end()` and only logs a warning, so the warning is the
    evidence that a span was ended more than once.
    """
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    # `caplog.records` holds only the current (teardown) phase; check the test body.
    records = caplog.get_records("call")
    assert not [record for record in records if "ended span" in record.getMessage()]


def _molto_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    (span,) = exporter.get_finished_spans()
    return span


def _assert_cancelled(span: ReadableSpan) -> None:
    """Cancellation ends the span as UNSET, recording no error (decision 9)."""
    assert span.status.status_code is StatusCode.UNSET
    assert "error.type" not in span.attributes
    assert "molto.error.phase" not in span.attributes
    assert span.events == ()


async def _events(*events: str, error: BaseException | None = None) -> AsyncIterator[str]:
    for event in events:
        yield event
    if error is not None:
        raise error


async def _trace[ResultT](call: Callable[[], Awaitable[ResultT]]) -> ResultT:
    return await _traced("responses", "responses", "openai", None, BODY, lambda _: call())


async def _traced_stream(events: AsyncIterator[str]) -> MoltoStream[str]:
    return await _trace(lambda: _prefetch_stream(events, Headers()))


def _cassette(name: str) -> dict[str, Any]:
    """Read the cassette at `name`, a path under `tests/cassettes` without `.yaml`."""
    return read_yaml(str(CASSETTES / f"{name}.yaml"))


def _replay(cassette_name: str) -> AsyncClient:
    """Serve one recorded response for any request, without checking the request."""
    return AsyncClient(transport=replay_transport(_cassette(cassette_name)))


def _recorded_response_id(cassette_name: str) -> str:
    """Return the response ID a recording carries, from its body or first identified event."""
    response = _cassette(cassette_name)["response"]
    if "body" in response:
        return response["body"]["id"]
    for line in "".join(response["sse"]).splitlines():
        if line.startswith("data:") and line != "data: [DONE]":
            data = json_loads(line.removeprefix("data:"))
            identified = data.get("response") or data.get("message") or data
            if "id" in identified:
                return identified["id"]
    raise AssertionError(f"{cassette_name} has no response ID")


def _request_body(cassette_name: str) -> dict[str, Any]:
    return _cassette(cassette_name)["request"]["body"]


async def test_non_streaming_call_ends_span_on_return(exporter: InMemorySpanExporter):
    """Ensures a non-streaming call ends one CLIENT span with UNSET status when it returns."""

    async def call() -> str:
        return "response"

    assert await _trace(call) == "response"

    span = _molto_span(exporter)
    assert span.name == "chat gpt-5"
    assert span.kind is SpanKind.CLIENT
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes["molto.route"] == "native"


async def test_non_streaming_failure_marks_span_error(exporter: InMemorySpanExporter):
    """Asserts that a failed call ends its span with ERROR and re-raises the original error."""

    async def call() -> str:
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        await _trace(call)

    span = _molto_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    # The message must not reach the status description (docs/observability.md).
    assert not span.status.description


async def test_cancelled_call_ends_span_without_error(exporter: InMemorySpanExporter):
    """Ensures cancellation ends the span with UNSET status, since the call did not fail."""

    async def call() -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _trace(call)

    _assert_cancelled(_molto_span(exporter))


async def test_span_parents_to_caller_and_is_current_only_during_the_call(
    exporter: InMemorySpanExporter,
):
    """Ensures the span nests under the caller's span and is current only while the call runs.

    HTTP client spans parent to the current span, so the Molto span must be current
    while the upstream request is sent and must not leak into the caller's context.
    """
    current_during_call = []

    async def call() -> str:
        current_during_call.append(trace.get_current_span())
        return "response"

    with TracerProvider().get_tracer("caller").start_as_current_span("caller") as caller:
        await _trace(call)
        assert trace.get_current_span() is caller

    span = _molto_span(exporter)
    assert span.parent.span_id == caller.get_span_context().span_id
    assert current_during_call[0].get_span_context().span_id == span.context.span_id


async def test_stream_span_outlives_coordinator_and_ends_on_exhaustion(
    exporter: InMemorySpanExporter,
):
    """Ensures a streaming span stays open after the coordinator returns and ends at exhaustion."""
    stream = await _traced_stream(_events("a", "b"))

    assert exporter.get_finished_spans() == ()
    assert [event async for event in stream] == ["a", "b"]

    assert _molto_span(exporter).status.status_code is StatusCode.UNSET
    await stream.aclose()


async def test_empty_stream_ends_span_on_first_iteration(exporter: InMemorySpanExporter):
    """Ensures a stream with no events, already exhausted by the prefetch, still ends its span."""
    stream = await _traced_stream(_events())

    assert [event async for event in stream] == []

    assert _molto_span(exporter).status.status_code is StatusCode.UNSET


async def test_stream_span_ends_once_on_early_aclose(exporter: InMemorySpanExporter):
    """Ensures closing a partly consumed stream ends its span once, without ERROR."""
    stream = await _traced_stream(_events("a", "b", "c"))

    assert await anext(stream) == "a"
    await stream.aclose()
    await stream.aclose()

    assert _molto_span(exporter).status.status_code is StatusCode.UNSET


async def test_stream_failure_marks_span_error(exporter: InMemorySpanExporter):
    """Asserts that an upstream error while the caller iterates ends the span with ERROR once."""
    stream = await _traced_stream(_events("a", error=RuntimeError("stream failed")))

    with pytest.raises(RuntimeError, match="stream failed"):
        async with stream:
            async for _ in stream:
                pass

    assert _molto_span(exporter).status.status_code is StatusCode.ERROR


async def test_failing_close_marks_span_error(exporter: InMemorySpanExporter):
    """Asserts that an error while closing the source stream ends the span with ERROR."""

    class FailingClose:
        def __aiter__(self) -> "FailingClose":
            return self

        async def __anext__(self) -> str:
            return "a"

        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    stream = await _traced_stream(FailingClose())

    with pytest.raises(RuntimeError, match="close failed"):
        await stream.aclose()

    assert _molto_span(exporter).status.status_code is StatusCode.ERROR


async def test_stream_cancellation_ends_span_without_error(exporter: InMemorySpanExporter):
    """Ensures a stream cancelled mid-iteration ends its span with UNSET status."""
    stream = await _traced_stream(_events("a", error=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        async for _ in stream:
            pass

    _assert_cancelled(_molto_span(exporter))


async def test_prefetch_failure_ends_span_in_coordinator(exporter: InMemorySpanExporter):
    """Asserts that a failure before the first event ends the span with ERROR, with no stream."""
    with pytest.raises(RuntimeError, match="setup failed"):
        await _traced_stream(_events(error=RuntimeError("setup failed")))

    assert _molto_span(exporter).status.status_code is StatusCode.ERROR


async def test_prefetch_cancellation_ends_span_without_error(exporter: InMemorySpanExporter):
    """Ensures a call cancelled before its first event ends the span with UNSET status."""
    with pytest.raises(asyncio.CancelledError):
        await _traced_stream(_events(error=asyncio.CancelledError()))

    _assert_cancelled(_molto_span(exporter))


async def test_stream_span_is_not_current_while_the_caller_iterates(
    exporter: InMemorySpanExporter,
):
    """Ensures iterating a stream runs in the caller's context, not under the Molto span.

    The span is current for the prefetch only; later iteration, possibly from another
    task, must not make it current or reparent spans the caller creates.
    """
    current = []

    async def events() -> AsyncIterator[str]:
        current.append(trace.get_current_span())
        yield "a"
        current.append(trace.get_current_span())
        yield "b"

    stream = await _traced_stream(events())
    async with stream:
        assert [event async for event in stream] == ["a", "b"]

    span = _molto_span(exporter)
    prefetch, iteration = current
    assert prefetch.get_span_context().span_id == span.context.span_id
    assert iteration.get_span_context().span_id != span.context.span_id


COORDINATOR_CASES = [
    # (method, protocol, endpoint, request cassette, response cassette, finish reason)
    (
        "responses",
        "responses",
        "responses",
        "matched/simple_text/openai-responses-stream-00",
        None,
        "completed",
    ),
    (
        "responses",
        "responses",
        "responses",
        "matched/simple_text/openai-responses-non_stream-00",
        None,
        "completed",
    ),
    (
        "chat_completions",
        "chat/completions",
        "chat/completions",
        "matched/simple_text/openai-chat_completions-stream-00",
        None,
        "stop",
    ),
    (
        "chat_completions",
        "chat/completions",
        "chat/completions",
        "matched/simple_text/openai-chat_completions-non_stream-00",
        None,
        "stop",
    ),
    (
        "messages",
        "messages",
        "messages",
        "matched/simple_text/anthropic-messages-stream-00",
        None,
        "end_turn",
    ),
    (
        "messages",
        "messages",
        "messages",
        "matched/simple_text/anthropic-messages-non_stream-00",
        None,
        "end_turn",
    ),
    # Translated spans report Molto's normalized termination reason.
    (
        "responses",
        "responses",
        "chat/completions",
        "matched/simple_text/openai-responses-stream-00",
        "matched/simple_text/openai-chat_completions-stream-00",
        "stop",
    ),
    (
        "messages",
        "messages",
        "responses",
        "matched/simple_text/anthropic-messages-non_stream-00",
        # Translated calls always stream from the upstream, even when non-streaming.
        "matched/simple_text/openai-responses-stream-00",
        "stop",
    ),
    (
        "responses",
        "responses",
        "messages",
        "matched/simple_text/openai-responses-non_stream-00",
        "matched/simple_text/anthropic-messages-stream-00",
        "stop",
    ),
    (
        "chat_completions",
        "chat/completions",
        "messages",
        "matched/simple_text/openai-chat_completions-stream-00",
        "matched/simple_text/anthropic-messages-stream-00",
        "stop",
    ),
    (
        "chat_completions",
        "chat/completions",
        "responses",
        "matched/simple_text/openai-chat_completions-non_stream-00",
        "matched/simple_text/openai-responses-stream-00",
        "stop",
    ),
    (
        "messages",
        "messages",
        "chat/completions",
        "matched/simple_text/anthropic-messages-stream-00",
        "matched/simple_text/openai-chat_completions-stream-00",
        "stop",
    ),
]


async def _run_coordinator(
    method: str,
    endpoint: str,
    request_cassette: str,
    response_cassette: str | None,
    **body_fields: Any,
) -> bool:
    """Call one coordinator against a replayed response, consuming any stream."""
    body = {**_request_body(request_cassette), **body_fields}
    client = _replay(response_cassette or request_cassette)
    async with Molto(http_client=client) as molto:
        result = await getattr(molto, method)(body=body, endpoint=endpoint, api_key="test-key")
        if not isinstance(result, MoltoStream):
            return False
        async with result:
            [event async for event in result]
    return True


@pytest.mark.parametrize(
    ("method", "protocol", "endpoint", "request_cassette", "response_cassette", "finish_reason"),
    COORDINATOR_CASES,
)
async def test_coordinator_emits_one_span_per_call(
    exporter: InMemorySpanExporter,
    method: str,
    protocol: str,
    endpoint: str,
    request_cassette: str,
    response_cassette: str | None,
    finish_reason: str,
):
    """Ensures each coordinator emits one complete span per call on native and translated routes.

    The span is a CLIENT child of the caller's span and carries the start attributes,
    the response identity, finish reason, and usage, and, for a streaming caller only,
    one `molto.first_token` event.
    """
    with TracerProvider().get_tracer("caller").start_as_current_span("caller") as caller:
        # The replay ignores the request, so sampling values only reach the span.
        streamed = await _run_coordinator(
            method, endpoint, request_cassette, response_cassette, temperature=0.5, top_p=0.9
        )

    span = _molto_span(exporter)
    attributes = span.attributes
    assert span.name == "chat gpt-4.1-nano"
    assert span.kind is SpanKind.CLIENT
    assert span.parent.span_id == caller.get_span_context().span_id
    assert span.status.status_code is StatusCode.UNSET
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.provider.name"] == (
        "anthropic" if endpoint == "messages" else "openai"
    )
    assert attributes["server.address"] == (
        "api.anthropic.com" if endpoint == "messages" else "api.openai.com"
    )
    assert attributes["server.port"] == 443
    assert attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert attributes["gen_ai.request.max_tokens"] == 1024
    assert attributes["gen_ai.request.temperature"] == 0.5
    assert attributes["gen_ai.request.top_p"] == 0.9
    assert attributes["molto.downstream.protocol"] == protocol
    assert attributes["molto.upstream.endpoint"] == endpoint
    assert attributes["molto.route"] == ("native" if protocol == endpoint else "translated")
    assert attributes.get("gen_ai.request.stream") is (True if streamed else None)
    assert attributes["gen_ai.response.id"] == _recorded_response_id(
        response_cassette or request_cassette
    )
    assert attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert attributes["gen_ai.response.finish_reasons"] == (finish_reason,)
    assert attributes["gen_ai.usage.input_tokens"] == 15
    assert isinstance(attributes["gen_ai.usage.output_tokens"], int)
    # The recorded calls used no prompt cache, so no zero-valued cache counts appear.
    assert "gen_ai.usage.cache_read.input_tokens" not in attributes
    assert "gen_ai.usage.cache_write.input_tokens" not in attributes
    assert [event.name for event in span.events] == (["molto.first_token"] if streamed else [])


async def test_native_and_translated_messages_spans_report_the_same_usage(
    exporter: InMemorySpanExporter,
):
    """Asserts that the same Messages upstream call reports identical usage on both routes.

    Native Messages sums Anthropic's uncached and cached input tokens, matching the
    normalized usage that the translated route reports.
    """
    await _run_coordinator(
        "messages", "messages", "matched/simple_text/anthropic-messages-stream-00", None
    )
    await _run_coordinator(
        "responses",
        "messages",
        "matched/simple_text/openai-responses-stream-00",
        "matched/simple_text/anthropic-messages-stream-00",
    )

    native, translated = exporter.get_finished_spans()
    usage_keys = ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")
    assert [native.attributes[key] for key in usage_keys] == [
        translated.attributes[key] for key in usage_keys
    ]
    assert native.attributes["gen_ai.response.finish_reasons"] == ("end_turn",)
    assert translated.attributes["gen_ai.response.finish_reasons"] == ("stop",)


def _not_found(request: Request) -> Response:
    return Response(
        404,
        json={"error": {"message": "secret-request-marker not found", "type": "not_found"}},
    )


@pytest.mark.parametrize(
    ("method", "body", "endpoint"),
    [
        ("responses", {"model": "gpt-5", "input": "Hello!"}, "responses"),
        (
            "chat_completions",
            {"model": "gpt-5", "messages": [{"role": "user", "content": "Hello!"}]},
            "responses",
        ),
    ],
)
async def test_upstream_error_reports_the_same_error_type_on_both_routes(
    exporter: InMemorySpanExporter, method: str, body: dict, endpoint: str
):
    """Asserts that an upstream HTTP error records the normalized error on native and translated routes.

    The caller receives an SDK-specific exception, but `error.type` reports the
    normalized class so both routes agree, and the recorded exception event carries
    no message text from the upstream error.
    """
    async with Molto(http_client=AsyncClient(transport=MockTransport(_not_found))) as molto:
        with pytest.raises(Exception, match="secret-request-marker"):
            await getattr(molto, method)(body=body, endpoint=endpoint, api_key="test-key")

    span = _molto_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "ResourceNotFoundError"
    assert span.attributes["http.response.status_code"] == 404
    assert span.attributes["molto.error.phase"] == "setup"
    # `use_span` must not record its own exception event with the full message.
    (event,) = span.events
    assert event.name == "exception"
    assert event.attributes["exception.message"] == "NotFoundError"
    assert "secret-request-marker" not in event.attributes["exception.stacktrace"]
    assert "ResourceNotFoundError" in event.attributes["exception.stacktrace"]


async def test_validation_error_records_no_request_content(exporter: InMemorySpanExporter):
    """Asserts that a rejected request's values never reach the span, though the error message holds them."""
    async with Molto() as molto:
        with pytest.raises(ValidationError, match="secret-request-marker"):
            await molto.responses(
                body={"model": "gpt-5", "input": "Hello!", "extra": "secret-request-marker"},
                endpoint="messages",
                api_key="test-key",
            )

    span = _molto_span(exporter)
    assert span.attributes["error.type"] == "ValidationError"
    assert "http.response.status_code" not in span.attributes
    (event,) = span.events
    assert event.attributes["exception.message"] == "ValidationError"
    recorded = [*span.attributes.values(), *event.attributes.values()]
    assert not any("secret-request-marker" in str(value) for value in recorded)


async def test_mid_stream_error_reports_stream_phase(exporter: InMemorySpanExporter):
    """Ensures an error raised while the caller iterates is attributed to the stream phase."""
    stream = await _traced_stream(_events("a", error=RuntimeError("secret-request-marker")))

    with pytest.raises(RuntimeError):
        async for _ in stream:
            pass

    span = _molto_span(exporter)
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.attributes["molto.error.phase"] == "stream"
    (event,) = span.events
    assert "secret-request-marker" not in str(dict(event.attributes))


async def test_coordinator_reports_caller_base_url(exporter: InMemorySpanExporter):
    """Asserts that a caller-supplied `base_url`, not the client default, sets the server attributes."""
    body = _request_body("matched/simple_text/openai-responses-non_stream-00")
    async with Molto(
        http_client=_replay("matched/simple_text/openai-responses-non_stream-00")
    ) as molto:
        await molto.responses(body=body, base_url="http://localhost:8000/v1", api_key="test-key")

    span = _molto_span(exporter)
    assert span.attributes["server.address"] == "localhost"
    assert span.attributes["server.port"] == 8000


async def test_coordinator_rejection_is_recorded_on_its_span(exporter: InMemorySpanExporter):
    """Asserts that a coordinator's local rejection ends its span with ERROR.

    The span starts before request validation, so failures such as an invalid
    translated body are visible in traces. The effective base URL is the upstream
    client's default when the caller passes none.
    """
    async with Molto() as molto:
        with pytest.raises(ValidationError):
            await molto.responses(
                body={"model": "claude-sonnet-5", "input": "Hello!", "unknown_field": 1},
                endpoint="messages",
                api_key="test-key",
            )

    span = _molto_span(exporter)
    assert span.name == "chat claude-sonnet-5"
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["molto.route"] == "translated"
    assert span.attributes["gen_ai.provider.name"] == "anthropic"
    assert span.attributes["server.address"] == "api.anthropic.com"


async def test_cached_input_is_reported_alike_on_every_route(exporter: InMemorySpanExporter):
    """Asserts that one cached call reports the same input and cache-read counts on all routes.

    The recorded Anthropic call reports 34 uncached and 7440 cache-read input tokens;
    the matched Responses call reports 7474 input tokens with 7440 cached. Native
    Messages must add the cached tokens to match Responses and the translated route.
    """
    scenario = "matched/prompt_cache_explicit"
    await _run_coordinator(
        "responses", "responses", f"{scenario}/openai-responses-stream-00", None
    )
    await _run_coordinator(
        "messages", "messages", f"{scenario}/anthropic-messages-stream-00", None
    )
    await _run_coordinator(
        "responses",
        "messages",
        f"{scenario}/openai-responses-stream-00",
        f"{scenario}/anthropic-messages-stream-00",
    )

    for span in exporter.get_finished_spans():
        assert span.attributes["gen_ai.usage.input_tokens"] == 7474
        assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 7440
        assert "gen_ai.usage.cache_write.input_tokens" not in span.attributes


async def test_incomplete_native_responses_stream_reports_its_reason(
    exporter: InMemorySpanExporter,
):
    """Ensures a native Responses stream ending in `response.incomplete` reports the incomplete reason."""
    await _run_coordinator(
        "responses", "responses", "errors/incomplete_response/openai-responses-stream-00", None
    )

    span = _molto_span(exporter)
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes["gen_ai.response.finish_reasons"] == ("max_output_tokens",)


async def test_missing_credentials_report_no_http_status(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    """Asserts that a call failing for missing credentials reports no HTTP status.

    No request is sent; the outward SDK error carries a synthetic 401 that must not
    be reported as an upstream response.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ENV_CONFIG, "openai_api_key", SecretStr(""))
    async with Molto(http_client=AsyncClient(transport=MockTransport(_not_found))) as molto:
        with pytest.raises(Exception) as raised:
            await molto.responses(body={"model": "gpt-5", "input": "Hello!"})

    assert raised.value.response.status_code == 401
    span = _molto_span(exporter)
    assert span.attributes["error.type"] == "AuthenticationError"
    assert span.attributes["molto.error.phase"] == "setup"
    assert "http.response.status_code" not in span.attributes


async def test_failure_after_a_successful_stream_response_reports_the_stream_phase(
    exporter: InMemorySpanExporter,
):
    """Asserts that a Responses error arriving on an HTTP 200 stream is a stream-phase failure.

    The first event arrives before the error, so the coordinator returns a stream
    and the failure surfaces while the caller iterates. Its HTTP response succeeded,
    so no status code is reported.
    """
    body = _request_body("errors/context_length_exceeded/openai-responses-stream-00")
    client = _replay("errors/context_length_exceeded/openai-responses-stream-00")
    async with Molto(http_client=client) as molto:
        stream = await molto.responses(body=body, api_key="test-key")
        with pytest.raises(Exception) as raised:
            async with stream:
                [event async for event in stream]

    assert isinstance(raised.value.__cause__, ContextLengthExceededError)
    span = _molto_span(exporter)
    assert span.attributes["error.type"] == "ContextLengthExceededError"
    assert span.attributes["molto.error.phase"] == "stream"
    assert "http.response.status_code" not in span.attributes


async def test_coordinator_stream_closed_early_ends_its_span_once(exporter: InMemorySpanExporter):
    """Ensures a coordinator stream closed after its first event ends its span once, without ERROR."""
    name = "matched/simple_text/openai-responses-stream-00"
    async with Molto(http_client=_replay(name)) as molto:
        stream = await molto.responses(body=_request_body(name), api_key="test-key")
        await anext(stream)
        await stream.aclose()
        await stream.aclose()

    span = _molto_span(exporter)
    assert span.status.status_code is StatusCode.UNSET
    assert "error.type" not in span.attributes


async def test_streaming_coordinator_failure_before_the_first_event_ends_its_span(
    exporter: InMemorySpanExporter,
):
    """Asserts that a streaming call failing before its first event ends the span in the coordinator."""
    async with Molto(http_client=AsyncClient(transport=MockTransport(_not_found))) as molto:
        with pytest.raises(Exception, match="secret-request-marker"):
            await molto.responses(
                body={"model": "gpt-5", "input": "Hello!", "stream": True}, api_key="test-key"
            )

    span = _molto_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "ResourceNotFoundError"
    assert span.attributes["molto.error.phase"] == "setup"


_CONTENT_FIELDS = {
    "responses": {"input": CONTENT_MARKER},
    "chat/completions": {"messages": [{"role": "user", "content": CONTENT_MARKER}]},
    "messages": {"messages": [{"role": "user", "content": CONTENT_MARKER}]},
}
_MARKED_EXTENSIONS = {
    "api_key": f"{CONTENT_MARKER}-key",
    "extra_body": {"metadata": {"note": CONTENT_MARKER}},
    "extra_headers": {
        "x-trace-note": CONTENT_MARKER,
        "authorization": f"Bearer {CONTENT_MARKER}",
    },
}


def _assert_carries_no_content(span: ReadableSpan) -> None:
    """Spans hold only contract keys, and no recorded value contains the marker."""
    assert set(span.attributes) <= ATTRIBUTE_KEYS
    recorded = [*span.attributes.values()]
    for event in span.events:
        assert set(event.attributes) <= EVENT_ATTRIBUTE_KEYS[event.name]
        recorded += [event.name, *event.attributes.values()]
    assert not any(CONTENT_MARKER in str(value) for value in recorded)


async def _consume(result: Any) -> None:
    if isinstance(result, MoltoStream):
        async with result:
            [event async for event in result]


@pytest.mark.parametrize(
    ("method", "protocol", "endpoint", "request_cassette", "response_cassette", "finish_reason"),
    COORDINATOR_CASES,
)
async def test_spans_carry_no_content_headers_or_credentials(
    exporter: InMemorySpanExporter,
    method: str,
    protocol: str,
    endpoint: str,
    request_cassette: str,
    response_cassette: str | None,
    finish_reason: str,
):
    """Asserts that request content, extensions, headers, and credentials never reach a span.

    Covers every coordinator route. Spans hold only the contract's attribute keys, so
    completion text cannot appear either.
    """
    body = {**_request_body(request_cassette), **_CONTENT_FIELDS[protocol]}
    async with Molto(http_client=_replay(response_cassette or request_cassette)) as molto:
        await _consume(
            await getattr(molto, method)(body=body, endpoint=endpoint, **_MARKED_EXTENSIONS)
        )

    _assert_carries_no_content(_molto_span(exporter))


@pytest.mark.parametrize(
    ("method", "protocol", "endpoint", "request_cassette", "response_cassette"),
    [
        (
            "responses",
            "responses",
            "responses",
            "matched/simple_text/openai-responses-non_stream-00",
            "errors/not_found/openai-responses-non_stream-00",
        ),
        (
            "responses",
            "responses",
            "responses",
            "matched/simple_text/openai-responses-stream-00",
            "errors/not_found/openai-responses-stream-00",
        ),
        (
            "chat_completions",
            "chat/completions",
            "responses",
            "matched/simple_text/openai-chat_completions-stream-00",
            "errors/not_found/openai-responses-stream-00",
        ),
    ],
)
async def test_failed_spans_carry_no_content_headers_or_credentials(
    exporter: InMemorySpanExporter,
    method: str,
    protocol: str,
    endpoint: str,
    request_cassette: str,
    response_cassette: str,
):
    """Asserts that a failed call's error attributes and exception event carry no request content."""
    body = {**_request_body(request_cassette), **_CONTENT_FIELDS[protocol]}
    async with Molto(http_client=_replay(response_cassette)) as molto:
        with pytest.raises(openai.NotFoundError):
            await _consume(
                await getattr(molto, method)(body=body, endpoint=endpoint, **_MARKED_EXTENSIONS)
            )

    span = _molto_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    _assert_carries_no_content(span)


@pytest.mark.parametrize(
    ("endpoint", "response_cassette"),
    [
        ("responses", None),
        ("messages", "matched/simple_text/anthropic-messages-stream-00"),
    ],
)
async def test_request_extensions_never_change_span_attributes(
    exporter: InMemorySpanExporter, endpoint: str, response_cassette: str | None
):
    """Asserts that `extra_body`, `extra_headers`, and the API key never influence the span.

    The same call with and without request extensions produces identical attributes,
    so telemetry cannot be reading them.
    """
    request_cassette = "matched/simple_text/openai-responses-stream-00"
    body = _request_body(request_cassette)
    for extensions in ({"api_key": "test-key"}, _MARKED_EXTENSIONS):
        async with Molto(http_client=_replay(response_cassette or request_cassette)) as molto:
            await _consume(await molto.responses(body=body, endpoint=endpoint, **extensions))

    plain, extended = exporter.get_finished_spans()
    assert dict(plain.attributes) == dict(extended.attributes)


class _StalledStream(AsyncByteStream):
    """Send `head`, then wait forever, as an upstream that stops responding."""

    def __init__(self, head: bytes, sent: asyncio.Event) -> None:
        self._head = head
        self._sent = sent

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._head:
            yield self._head
        self._sent.set()
        await asyncio.Event().wait()


def _stalled_client(head: bytes, sent: asyncio.Event) -> AsyncClient:
    def handler(request: Request) -> Response:
        return Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_StalledStream(head, sent),
        )

    return AsyncClient(transport=MockTransport(handler))


async def test_cancelling_a_coordinator_call_ends_its_span_without_error(
    exporter: InMemorySpanExporter,
):
    """Ensures cancelling a real coordinator call before its first event ends its span as UNSET."""
    sent = asyncio.Event()
    async with Molto(http_client=_stalled_client(b"", sent)) as molto:
        call = asyncio.create_task(
            molto.responses(body={**BODY, "stream": True}, api_key="test-key")
        )
        await sent.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    _assert_cancelled(_molto_span(exporter))


async def test_cancelling_a_coordinator_stream_ends_its_span_without_error(
    exporter: InMemorySpanExporter,
):
    """Ensures cancelling the task iterating a real coordinator stream ends its span once, as UNSET."""
    name = "matched/simple_text/openai-responses-stream-00"
    # The first recorded SSE event, after which the upstream stops responding.
    first_event = "".join(_cassette(name)["response"]["sse"]).split("\n\n")[0] + "\n\n"
    sent = asyncio.Event()
    async with Molto(http_client=_stalled_client(first_event.encode(), sent)) as molto:
        stream = await molto.responses(body=_request_body(name), api_key="test-key")
        consumer = asyncio.create_task(_consume(stream))
        await sent.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    _assert_cancelled(_molto_span(exporter))


# Recorded calls compared with and without an SDK: native and translated, streaming and
# non-streaming, and an upstream failure in both modes.
FRESH_INTERPRETER_CASES = [
    {
        "method": method,
        "endpoint": endpoint,
        "request": str(CASSETTES / f"{request}.yaml"),
        "response": str(CASSETTES / f"{response or request}.yaml"),
    }
    for method, endpoint, request, response in [
        ("responses", "responses", "matched/simple_text/openai-responses-stream-00", None),
        ("responses", "responses", "matched/simple_text/openai-responses-non_stream-00", None),
        (
            "responses",
            "chat/completions",
            "matched/simple_text/openai-responses-stream-00",
            "matched/simple_text/openai-chat_completions-stream-00",
        ),
        (
            "messages",
            "responses",
            "matched/simple_text/anthropic-messages-non_stream-00",
            "matched/simple_text/openai-responses-stream-00",
        ),
        (
            "responses",
            "responses",
            "matched/simple_text/openai-responses-non_stream-00",
            "errors/not_found/openai-responses-non_stream-00",
        ),
        (
            "responses",
            "responses",
            "matched/simple_text/openai-responses-stream-00",
            "errors/not_found/openai-responses-stream-00",
        ),
    ]
]
_FRESH_INTERPRETER_PREAMBLE = (
    "import asyncio, json, sys\n"
    "from molto.utils.test import replayed_outcomes\n"
    "cases = json.loads(sys.argv[1])\n"
)


def _run_in_fresh_interpreter(code: str, cases: list[dict[str, Any]]) -> Any:
    """Run `code` in a new interpreter with `cases` bound, and return its JSON stdout.

    A new interpreter has no tracer provider unless `code` installs one; the API
    lets a process set its global provider only once. The conftest network block does
    not apply there, but `replayed_outcomes` serves every response through
    `MockTransport`, so no request leaves the process.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH_INTERPRETER_PREAMBLE + code, json_dumps(cases)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return json_loads(completed.stdout)


def _without_item_ids(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_item_ids(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _without_item_ids(item)
        for key, item in value.items()
        if key != "item_id" and not (key == "id" and "type" in value)
    }


def _comparable(case: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """Drop the output-item IDs a translated Responses route generates afresh on every call.

    They differ between any two runs, traced or not (`python/ARCHITECTURE.md`,
    "Responses through Chat Completions"); every other value is compared exactly.
    """
    translated_responses = case["method"] == "responses" and case["endpoint"] != "responses"
    return _without_item_ids(outcome) if translated_responses else outcome


async def test_calls_behave_identically_without_an_sdk(exporter: InMemorySpanExporter):
    """Asserts that with no SDK configured, every call yields exactly what a traced call yields.

    Covers native and translated routes, streaming and non-streaming, and failures,
    comparing events, responses, and error classes and messages. The SDK must never be
    imported, because Molto depends only on the API.
    """
    traced = await replayed_outcomes(FRESH_INTERPRETER_CASES)

    untraced = _run_in_fresh_interpreter(
        "outcomes = asyncio.run(replayed_outcomes(cases))\n"
        "sdk_imported = 'opentelemetry.sdk' in sys.modules\n"
        "print(json.dumps({'outcomes': outcomes, 'sdk_imported': sdk_imported}))\n",
        FRESH_INTERPRETER_CASES,
    )

    assert [
        _comparable(case, outcome)
        for case, outcome in zip(FRESH_INTERPRETER_CASES, untraced["outcomes"], strict=True)
    ] == [
        _comparable(case, outcome)
        for case, outcome in zip(FRESH_INTERPRETER_CASES, traced, strict=True)
    ]
    assert untraced["sdk_imported"] is False
    assert [outcome.keys() & {"events", "response", "error"} for outcome in traced] == [
        {"events"},
        {"response"},
        {"events"},
        {"response"},
        {"error"},
        {"error"},
    ]
    assert len(exporter.get_finished_spans()) == len(FRESH_INTERPRETER_CASES)


def test_spans_use_the_application_tracer_provider():
    """Ensures Molto's tracer comes from the application's global provider, scoped as `molto`.

    The other tests route the tracer to a local provider; this checks the real
    accessor in an interpreter whose application installs the SDK globally.
    """
    result = _run_in_fresh_interpreter(
        "from opentelemetry import trace\n"
        "from opentelemetry.sdk.trace import TracerProvider\n"
        "from opentelemetry.sdk.trace.export import SimpleSpanProcessor\n"
        "from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter\n"
        "exporter = InMemorySpanExporter()\n"
        "provider = TracerProvider()\n"
        "provider.add_span_processor(SimpleSpanProcessor(exporter))\n"
        "trace.set_tracer_provider(provider)\n"
        "asyncio.run(replayed_outcomes(cases))\n"
        "(span,) = exporter.get_finished_spans()\n"
        "scope = span.instrumentation_scope\n"
        "print(json.dumps({'name': span.name, 'scope': scope.name, 'version': scope.version}))\n",
        FRESH_INTERPRETER_CASES[:1],
    )

    assert result == {"name": "chat gpt-4.1-nano", "scope": "molto", "version": __version__}


HTTPX_SCOPE = "opentelemetry.instrumentation.httpx"


class HttpSpanCase(NamedTuple):
    """One coordinator call whose upstream HTTP request should be traced."""

    method: str
    endpoint: str
    request: str
    stream: bool
    response_override: str | None = None

    @property
    def response(self) -> str:
        """The recording that answers the call: the override, or the request's own."""
        return self.response_override or self.request


HTTP_SPAN_CASES = [
    HttpSpanCase("responses", "responses", "matched/simple_text/openai-responses-stream-00", True),
    HttpSpanCase(
        "responses", "responses", "matched/simple_text/openai-responses-non_stream-00", False
    ),
    HttpSpanCase(
        "responses",
        "messages",
        "matched/simple_text/openai-responses-stream-00",
        True,
        "matched/simple_text/anthropic-messages-stream-00",
    ),
    HttpSpanCase(
        "messages",
        "responses",
        "matched/simple_text/anthropic-messages-non_stream-00",
        False,
        "matched/simple_text/openai-responses-stream-00",
    ),
]


def _case_id(case: HttpSpanCase) -> str:
    route = "native" if case.method.replace("_", "/") == case.endpoint else "translated"
    return f"{case.method}-{route}-{'stream' if case.stream else 'non_stream'}"


async def _call(molto: Molto, case: HttpSpanCase) -> None:
    """Make the case's coordinator call and consume any stream."""
    await _consume(
        await getattr(molto, case.method)(
            body=_request_body(case.request), endpoint=case.endpoint, api_key="test-key"
        )
    )


@contextmanager
def _instrumented_httpx(tracer_provider: TracerProvider) -> Iterator[None]:
    """Instrument `httpx` globally, as host applications do, for the duration of the block.

    A no-op meter provider keeps the instrumentor's duration histograms off the
    process-wide meter provider, so no other test sees them.
    """
    instrumentor = HTTPXClientInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=NoOpMeterProvider())
    try:
        yield
    finally:
        instrumentor.uninstrument()


def _replay_network(monkeypatch: pytest.MonkeyPatch, cassette_name: str) -> None:
    """Answer httpx's network transport from a recording, so no request leaves the process.

    Global instrumentation wraps this transport, which every default `AsyncClient` uses.
    """
    replay = replay_transport(_cassette(cassette_name))

    async def handle_async_request(self: AsyncHTTPTransport, request: Request) -> Response:
        return await replay.handle_async_request(request)

    monkeypatch.setattr(AsyncHTTPTransport, "handle_async_request", handle_async_request)


def _http_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope.name == HTTPX_SCOPE
    ]


def _assert_http_span_nests_under_molto_span(
    exporter: InMemorySpanExporter, case: HttpSpanCase
) -> None:
    """Assert one HTTP CLIENT span is a child of the Molto span, ending at the response headers.

    For a stream, the HTTP span ends before the first token: it measures only the time
    to first byte, while the Molto span covers the whole generation.
    """
    (http_span,) = _http_spans(exporter)
    (molto_span,) = [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope.name == "molto"
    ]
    assert http_span.kind is SpanKind.CLIENT
    assert http_span.parent.span_id == molto_span.context.span_id
    assert http_span.context.trace_id == molto_span.context.trace_id
    first_tokens = [event for event in molto_span.events if event.name == "molto.first_token"]
    if case.stream:
        (first_token,) = first_tokens
        assert http_span.end_time <= first_token.timestamp
    else:
        assert first_tokens == []


@pytest.mark.parametrize("injected", [False, True], ids=["default_client", "injected_client"])
@pytest.mark.parametrize("case", HTTP_SPAN_CASES, ids=_case_id)
async def test_global_httpx_instrumentation_nests_http_spans_under_the_molto_span(
    exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    monkeypatch: pytest.MonkeyPatch,
    injected: bool,
    case: HttpSpanCase,
):
    """Ensures globally instrumented `httpx` spans parent to the Molto span on every route.

    Covers Molto's default client and an injected one, both created before the
    instrumentation is installed, for native and translated, streaming and
    non-streaming calls. This guards the rule that the Molto span is current while
    the upstream request is sent.
    """
    _replay_network(monkeypatch, case.response)
    molto = Molto(http_client=AsyncClient() if injected else None)
    with _instrumented_httpx(tracer_provider):
        async with molto:
            await _call(molto, case)

    _assert_http_span_nests_under_molto_span(exporter, case)


@pytest.mark.parametrize("case", HTTP_SPAN_CASES, ids=_case_id)
async def test_instrumented_client_nests_http_spans_under_the_molto_span(
    exporter: InMemorySpanExporter, tracer_provider: TracerProvider, case: HttpSpanCase
):
    """Ensures an injected client instrumented with `instrument_client` nests its spans under Molto's.

    This is the per-client setup `docs/observability.md` shows, which works with any
    `httpx` transport.
    """
    http_client = _replay(case.response)
    HTTPXClientInstrumentor.instrument_client(
        http_client, tracer_provider=tracer_provider, meter_provider=NoOpMeterProvider()
    )
    async with Molto(http_client=http_client) as molto:
        await _call(molto, case)

    _assert_http_span_nests_under_molto_span(exporter, case)


async def test_global_httpx_instrumentation_ignores_custom_transports(
    exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Asserts that global instrumentation does not trace a client built on a custom transport.

    It wraps only `httpx`'s network transport, so a `MockTransport` client produces
    no HTTP span; `docs/observability.md` points such clients to `instrument_client`.
    """
    case = HTTP_SPAN_CASES[0]
    with _instrumented_httpx(tracer_provider):
        async with Molto(http_client=_replay(case.response)) as molto:
            await _call(molto, case)

    assert _http_spans(exporter) == []
    assert len(exporter.get_finished_spans()) == 1
