import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient, Headers, MockTransport, Request, Response
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import ValidationError

from molto.core import Molto, MoltoStream, _prefetch_stream, _traced
from molto.utils.io import read_yaml

BODY = {"model": "gpt-5", "input": "Hello!"}
SIMPLE_TEXT = Path(__file__).parents[1] / "cassettes" / "matched" / "simple_text"


@pytest.fixture(autouse=True)
def exporter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> Iterator[InMemorySpanExporter]:
    """Route Molto's tracer to a local SDK provider, leaving the global provider untouched.

    The SDK ignores a second `end()` and only logs a warning, so the warning is the
    evidence that a span was ended more than once.
    """
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("molto.core.get_tracer", lambda: provider.get_tracer("molto"))
    yield exporter
    # `caplog.records` holds only the current (teardown) phase; check the test body.
    records = caplog.get_records("call")
    assert not [record for record in records if "ended span" in record.getMessage()]


def _molto_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    (span,) = exporter.get_finished_spans()
    return span


async def _events(*events: str, error: BaseException | None = None) -> AsyncIterator[str]:
    for event in events:
        yield event
    if error is not None:
        raise error


async def _trace[ResultT](call: Callable[[], Awaitable[ResultT]]) -> ResultT:
    return await _traced("responses", "responses", "openai", None, BODY, call)


async def _traced_stream(events: AsyncIterator[str]) -> MoltoStream[str]:
    return await _trace(lambda: _prefetch_stream(events, Headers()))


def _replay(cassette_name: str) -> AsyncClient:
    """Serve one recorded response for any request, without checking the request."""
    response = read_yaml(str(SIMPLE_TEXT / cassette_name))["response"]

    def handler(request: Request) -> Response:
        if "sse" in response:
            return Response(
                response["status_code"],
                content="".join(response["sse"]).encode(),
                headers={"content-type": "text/event-stream"},
            )
        return Response(response["status_code"], json=response["body"])

    return AsyncClient(transport=MockTransport(handler))


def _request_body(cassette_name: str) -> dict[str, Any]:
    return read_yaml(str(SIMPLE_TEXT / cassette_name))["request"]["body"]


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

    assert _molto_span(exporter).status.status_code is StatusCode.UNSET


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

    assert _molto_span(exporter).status.status_code is StatusCode.UNSET


async def test_prefetch_failure_ends_span_in_coordinator(exporter: InMemorySpanExporter):
    """Asserts that a failure before the first event ends the span with ERROR, with no stream."""
    with pytest.raises(RuntimeError, match="setup failed"):
        await _traced_stream(_events(error=RuntimeError("setup failed")))

    assert _molto_span(exporter).status.status_code is StatusCode.ERROR


async def test_prefetch_cancellation_ends_span_without_error(exporter: InMemorySpanExporter):
    """Ensures a call cancelled before its first event ends the span with UNSET status."""
    with pytest.raises(asyncio.CancelledError):
        await _traced_stream(_events(error=asyncio.CancelledError()))

    assert _molto_span(exporter).status.status_code is StatusCode.UNSET


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


@pytest.mark.parametrize(
    ("method", "protocol", "endpoint", "request_cassette", "response_cassette"),
    [
        (
            "responses",
            "responses",
            "responses",
            "openai-responses-stream-00.yaml",
            "openai-responses-stream-00.yaml",
        ),
        (
            "responses",
            "responses",
            "responses",
            "openai-responses-non_stream-00.yaml",
            "openai-responses-non_stream-00.yaml",
        ),
        (
            "chat_completions",
            "chat/completions",
            "chat/completions",
            "openai-chat_completions-stream-00.yaml",
            "openai-chat_completions-stream-00.yaml",
        ),
        (
            "chat_completions",
            "chat/completions",
            "chat/completions",
            "openai-chat_completions-non_stream-00.yaml",
            "openai-chat_completions-non_stream-00.yaml",
        ),
        (
            "messages",
            "messages",
            "messages",
            "anthropic-messages-stream-00.yaml",
            "anthropic-messages-stream-00.yaml",
        ),
        (
            "messages",
            "messages",
            "messages",
            "anthropic-messages-non_stream-00.yaml",
            "anthropic-messages-non_stream-00.yaml",
        ),
        (
            "responses",
            "responses",
            "chat/completions",
            "openai-responses-stream-00.yaml",
            "openai-chat_completions-stream-00.yaml",
        ),
        (
            "messages",
            "messages",
            "responses",
            "anthropic-messages-non_stream-00.yaml",
            # Translated calls always stream from the upstream, even when non-streaming.
            "openai-responses-stream-00.yaml",
        ),
    ],
)
async def test_coordinator_emits_one_span_per_call(
    exporter: InMemorySpanExporter,
    method: str,
    protocol: str,
    endpoint: str,
    request_cassette: str,
    response_cassette: str,
):
    """Ensures each coordinator emits one span per call on native and translated routes.

    A streaming coordinator hands its span to the returned stream, so the span stays
    open until the caller exhausts the stream.
    """
    body = _request_body(request_cassette)
    async with Molto(http_client=_replay(response_cassette)) as molto:
        result = await getattr(molto, method)(body=body, endpoint=endpoint, api_key="test-key")
        if isinstance(result, MoltoStream):
            assert exporter.get_finished_spans() == ()
            async with result:
                [event async for event in result]

    span = _molto_span(exporter)
    assert span.name == f"chat {body['model']}"
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes["molto.downstream.protocol"] == protocol
    assert span.attributes["molto.upstream.endpoint"] == endpoint
    assert span.attributes["molto.route"] == ("native" if protocol == endpoint else "translated")
    assert span.attributes.get("gen_ai.request.stream") is (
        True if isinstance(result, MoltoStream) else None
    )


async def test_coordinator_reports_caller_base_url(exporter: InMemorySpanExporter):
    """Asserts that a caller-supplied `base_url`, not the client default, sets the server attributes."""
    body = _request_body("openai-responses-non_stream-00.yaml")
    async with Molto(http_client=_replay("openai-responses-non_stream-00.yaml")) as molto:
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
