from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIError, APIStatusError, RateLimitError

from molto.providers.openai import map_openai_error
from molto.types.openai.responses import OpenAIResponsesRequest
from molto.utils.exceptions import (
    ContextOverflowError,
    ExternalAuthError,
    ModelOverloadError,
    RateLimitExceedError,
)
from tests.support import make_molto_client, make_openai_client, prepend_model_prefix

CONTEXT_OVERFLOW_INPUT = "Repeat this exactly: " + ("x " * 300_000).strip()

EXCEPTION_REQUESTS: dict[str, dict[str, Any]] = {
    "context_length_exceeded": {
        "model": "o3-mini",
        "input": [{"role": "user", "content": CONTEXT_OVERFLOW_INPUT}],
        "include": ["reasoning.encrypted_content"],
    },
    "thinking_budget_exceeding_max_tokens": {
        "model": "o4-mini",
        "input": [{"role": "user", "content": "Think very carefully about this."}],
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 1,
        "reasoning": {"effort": "high"},
    },
    "incorrect_api_key": {
        "model": "gpt-4.1-nano",
        "input": [{"role": "user", "content": "hi"}],
    },
    "missing_api_key": {
        "model": "gpt-4.1-nano",
        "input": [{"role": "user", "content": "hi"}],
    },
}

EXCEPTION_STATUS_CODES = {
    "context_length_exceeded": 400,
    "thinking_budget_exceeding_max_tokens": 400,
    "incorrect_api_key": 401,
    "missing_api_key": 401,
}


def _provider_error_request_body(name: str) -> dict[str, Any]:
    """Return the provider request body used for error cassette replay."""
    return {**EXCEPTION_REQUESTS[name], "stream": True}


async def _record_or_replay_provider_error(
    cassette_manager,
    name: str,
    *,
    api_key: str | None = None,
) -> None:
    """Record or replay the OpenAI SDK error cassette used by Molto playback."""
    req_body = _provider_error_request_body(name)
    client = make_openai_client(
        cassette_manager,
        cassette_suffix=name,
        api_key=api_key,
    )
    with pytest.raises(APIError) as exc_info:
        response = await client.responses.create(**req_body)
        if req_body.get("stream"):
            async for _ in response:
                pass

    exc = exc_info.value
    assert not isinstance(exc, APIConnectionError)
    if isinstance(exc, APIStatusError):
        assert exc.status_code == EXCEPTION_STATUS_CODES[name]
    else:
        assert name == "context_length_exceeded"
        assert "context window" in str(exc).lower()


def _cassette_request_params(name: str) -> dict[str, Any]:
    """Return Molto request params for an exception cassette name."""
    body = EXCEPTION_REQUESTS[name]
    params: dict[str, Any] = prepend_model_prefix(
        {
            "model": body["model"],
            "input": body["input"],
            "stream": True,
        }
    )
    for key in ("include", "reasoning", "max_output_tokens"):
        if key in body:
            params[key] = body[key]
    return params


class TestBadInputError:
    async def test_context_window_exceeded(self, cassette_manager):
        """Map OpenAI context-window errors to ContextOverflowError."""
        await _record_or_replay_provider_error(cassette_manager, "context_length_exceeded")

        molto = make_molto_client(
            cassette_manager,
            cassette_suffix="context_length_exceeded",
        )
        req = OpenAIResponsesRequest(**_cassette_request_params("context_length_exceeded"))
        with pytest.raises(ContextOverflowError):
            response = await molto.responses(req)
            async for _ in response:
                pass


class TestModelOverloadError:
    def test_connection_error_preserves_cassette_assertion(self):
        """Map SDK connection failures without a Pydantic AI wrapper."""
        assertion = AssertionError("OpenAI cassette request mismatch.")
        connection_error = APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com")
        )
        connection_error.__cause__ = assertion
        mapped = map_openai_error(connection_error)
        assert type(mapped) is ModelOverloadError
        assert str(mapped) == "Connection error."

    async def test_network_connectivity_503(self):
        """Map HTTP 503 model errors to ModelOverloadError."""
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(503, request=request)
        exc = APIStatusError(
            "Service unavailable",
            response=response,
            body={"error": {"message": "Service unavailable"}},
        )
        mapped = map_openai_error(exc)
        assert type(mapped) is ModelOverloadError
        with pytest.raises(ModelOverloadError):
            raise mapped

    async def test_server_downtime_529(self):
        """Map HTTP 529 model errors to ModelOverloadError."""
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(529, request=request)
        exc = APIStatusError(
            "Site is overloaded",
            response=response,
            body={"error": {"message": "Site is overloaded"}},
        )
        mapped = map_openai_error(exc)
        assert type(mapped) is ModelOverloadError
        with pytest.raises(ModelOverloadError):
            raise mapped


class TestExternalAuthError:
    async def test_invalid_api_key(self, cassette_manager):
        """Map invalid OpenAI credentials to ExternalAuthError."""
        await _record_or_replay_provider_error(
            cassette_manager,
            "incorrect_api_key",
            api_key="sk-invalid12345",
        )

        molto = make_molto_client(
            cassette_manager,
            cassette_suffix="incorrect_api_key",
        )
        req = OpenAIResponsesRequest(**_cassette_request_params("incorrect_api_key"))
        with pytest.raises(ExternalAuthError):
            await molto.responses(req)

    async def test_missing_api_key(self, cassette_manager):
        """Map missing OpenAI credentials to ExternalAuthError."""
        await _record_or_replay_provider_error(
            cassette_manager,
            "missing_api_key",
            api_key="",
        )

        molto = make_molto_client(
            cassette_manager,
            cassette_suffix="missing_api_key",
        )
        req = OpenAIResponsesRequest(**_cassette_request_params("missing_api_key"))
        with pytest.raises(ExternalAuthError):
            await molto.responses(req)


class TestRateLimitExceedError:
    def test_rate_limit_headers_populate_typed_fields(self):
        """Map 429 provider rate-limit headers into the typed error fields."""
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(
            429,
            request=request,
            headers={
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "0",
            },
        )
        exc = RateLimitError("rate limited", response=response, body=None)

        mapped = map_openai_error(exc)

        assert type(mapped) is RateLimitExceedError
        assert mapped.limit == 60
        assert mapped.remaining == 0
