import asyncio
from collections.abc import AsyncIterator
from itertools import product
from typing import Any, Literal

import anthropic
import httpx
import openai
import pytest

from molto import Molto
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient

Protocol = Literal["responses", "chat/completions", "messages"]
SDKErrorKind = Literal["connection", "timeout", "response_validation", "api"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions", "messages")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))
SDK_ERROR_KINDS: tuple[SDKErrorKind, ...] = (
    "connection",
    "timeout",
    "response_validation",
    "api",
)


def _request_body(protocol: Protocol) -> dict[str, Any]:
    if protocol == "responses":
        return {
            "model": "test-model",
            "input": "hi",
            "max_output_tokens": 16,
            "stream": True,
        }
    if protocol == "chat/completions":
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 16,
            "stream": True,
        }
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "stream": True,
    }


def _upstream_client_type(protocol: Protocol) -> type:
    return {
        "responses": OpenAIResponsesClient,
        "chat/completions": OpenAIChatCompletionsClient,
        "messages": AnthropicMessagesClient,
    }[protocol]


def _sdk(protocol: Protocol):
    return anthropic if protocol == "messages" else openai


def _sdk_error(protocol: Protocol, kind: SDKErrorKind) -> Exception:
    sdk = _sdk(protocol)
    origin = "https://api.anthropic.com" if protocol == "messages" else "https://api.openai.com"
    request = httpx.Request("POST", f"{origin}/v1/test")
    if kind == "connection":
        return sdk.APIConnectionError(message="Portable connection failure.", request=request)
    if kind == "timeout":
        return sdk.APITimeoutError(request)
    if kind == "response_validation":
        response = httpx.Response(200, request=request)
        return sdk.APIResponseValidationError(
            response,
            {"malformed": True},
            message="Portable response validation failure.",
        )
    return sdk.APIError("Portable API failure.", request, body={"detail": "source detail"})


def _expected_error_type(protocol: Protocol, kind: SDKErrorKind) -> type[Exception]:
    sdk = _sdk(protocol)
    return {
        "connection": sdk.APIConnectionError,
        "timeout": sdk.APITimeoutError,
        "response_validation": sdk.APIResponseValidationError,
        "api": sdk.APIError,
    }[kind]


async def _call_molto(client: Molto, downstream: Protocol, upstream: Protocol):
    coordinator = {
        "responses": client.responses,
        "chat/completions": client.chat_completions,
        "messages": client.messages,
    }[downstream]
    return await coordinator(
        body=_request_body(downstream),
        endpoint=upstream,
        api_key="test-api-key",
    )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("kind", SDK_ERROR_KINDS)
async def test_sdk_errors_are_raised_eagerly_and_translated(
    monkeypatch: pytest.MonkeyPatch,
    downstream: Protocol,
    upstream: Protocol,
    kind: SDKErrorKind,
):
    source_error = _sdk_error(upstream, kind)

    async def raise_source_error(*args, **kwargs):
        raise source_error

    monkeypatch.setattr(_upstream_client_type(upstream), "native_events", raise_source_error)

    async with httpx.AsyncClient() as http_client:
        client = Molto(http_client=http_client)
        with pytest.raises(_expected_error_type(downstream, kind)) as exc_info:
            await _call_molto(client, downstream, upstream)

    error = exc_info.value
    assert type(error) is _expected_error_type(downstream, kind)
    if downstream == upstream:
        assert error is source_error
        assert error.__cause__ is None
        return

    assert error is not source_error
    assert error.__cause__ is source_error
    assert error.request.method == source_error.request.method
    assert error.request.url == source_error.request.url
    assert error.body == source_error.body
    if kind != "timeout":
        assert error.message == source_error.message
    if kind == "response_validation":
        assert error.status_code == source_error.status_code


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_cancelled_error_is_never_translated(
    monkeypatch: pytest.MonkeyPatch,
    downstream: Protocol,
    upstream: Protocol,
):
    cancellation = asyncio.CancelledError()

    async def raise_cancelled_error(*args, **kwargs):
        raise cancellation

    monkeypatch.setattr(_upstream_client_type(upstream), "native_events", raise_cancelled_error)

    async with httpx.AsyncClient() as http_client:
        client = Molto(http_client=http_client)
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await _call_molto(client, downstream, upstream)

    assert exc_info.value is cancellation
    assert cancellation.__cause__ is None


class _CountingIterator(AsyncIterator[Any]):
    def __init__(self) -> None:
        self.next_calls = 0

    async def __anext__(self) -> Any:
        self.next_calls += 1
        raise StopAsyncIteration


class _FailingIterator(_CountingIterator):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def __anext__(self) -> Any:
        self.next_calls += 1
        raise self.error


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_eager_stream_does_not_consume_first_event(
    monkeypatch: pytest.MonkeyPatch,
    downstream: Protocol,
    upstream: Protocol,
):
    source_stream = _CountingIterator()
    start_calls = 0

    async def native_events(*args, **kwargs):
        nonlocal start_calls
        start_calls += 1
        return source_stream

    monkeypatch.setattr(_upstream_client_type(upstream), "native_events", native_events)

    async with httpx.AsyncClient() as http_client:
        client = Molto(http_client=http_client)
        await _call_molto(client, downstream, upstream)
        assert start_calls == 1
        assert source_stream.next_calls == 0

    assert source_stream.next_calls == 0


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_sdk_api_error_during_iteration_is_translated(
    monkeypatch: pytest.MonkeyPatch,
    downstream: Protocol,
    upstream: Protocol,
):
    source_error = _sdk_error(upstream, "api")
    source_stream = _FailingIterator(source_error)

    async def native_events(*args, **kwargs):
        return source_stream

    monkeypatch.setattr(_upstream_client_type(upstream), "native_events", native_events)

    async with httpx.AsyncClient() as http_client:
        client = Molto(http_client=http_client)
        stream = await _call_molto(client, downstream, upstream)
        with pytest.raises(_expected_error_type(downstream, "api")) as exc_info:
            await anext(stream)

    error = exc_info.value
    if downstream == upstream:
        assert error is source_error
        assert error.__cause__ is None
    else:
        assert error is not source_error
        assert error.__cause__ is source_error
    assert source_stream.next_calls == 1
