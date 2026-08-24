from collections.abc import AsyncIterator

import anthropic
import httpx
import openai
import pytest

from molto.adapters.anthropic import _anthropic_api_status_error
from molto.adapters.openai import OpenAIResponsesAdapter, _openai_api_status_error
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.normalized import (
    ModelResponseStreamEvent,
    NormalizedAPIStatusError,
    NormalizedAuthenticationError,
    NormalizedBadRequestError,
    NormalizedConflictError,
    NormalizedInternalServerError,
    NormalizedNotFoundError,
    NormalizedOverloadedError,
    NormalizedPermissionDeniedError,
    NormalizedRateLimitError,
    NormalizedRequestTooLargeError,
    NormalizedUnprocessableEntityError,
)
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.anthropic import (
    _normalized_api_status_error as normalized_anthropic_status_error,
)
from molto.upstream.openai import (
    _normalized_api_status_error as normalized_openai_status_error,
)

NormalizedErrorType = type[NormalizedAPIStatusError]


def _normalized_error(
    error_type: NormalizedErrorType,
    status_code: int,
) -> NormalizedAPIStatusError:
    return error_type(
        "request failed",
        status_code=status_code,
        request_id="req_source",
        response_headers={
            "content-length": "999",
            "content-type": "application/json",
            "x-source": "kept",
        },
        request_method="POST",
        request_url="https://upstream.example/v1/responses",
    )


@pytest.mark.parametrize(
    ("normalized_type", "status_code", "expected_type"),
    [
        (NormalizedBadRequestError, 400, openai.BadRequestError),
        (NormalizedAuthenticationError, 401, openai.AuthenticationError),
        (NormalizedPermissionDeniedError, 403, openai.PermissionDeniedError),
        (NormalizedNotFoundError, 404, openai.NotFoundError),
        (NormalizedConflictError, 409, openai.ConflictError),
        (NormalizedUnprocessableEntityError, 422, openai.UnprocessableEntityError),
        (NormalizedRateLimitError, 429, openai.RateLimitError),
        (NormalizedInternalServerError, 500, openai.InternalServerError),
        (NormalizedRequestTooLargeError, 413, openai.APIStatusError),
        (NormalizedOverloadedError, 529, openai.InternalServerError),
        (NormalizedAPIStatusError, 418, openai.APIStatusError),
    ],
)
def test_normalized_status_error_projects_to_openai(
    normalized_type: NormalizedErrorType,
    status_code: int,
    expected_type: type[openai.APIStatusError],
) -> None:
    projected = _openai_api_status_error(_normalized_error(normalized_type, status_code))

    assert type(projected) is expected_type
    assert projected.status_code == status_code
    assert str(projected.request.url) == "https://upstream.example/v1/responses"
    assert projected.request_id == "req_source"
    assert projected.body["message"] == "request failed"
    assert projected.response.headers["x-source"] == "kept"
    assert projected.response.headers["content-length"] != "999"


@pytest.mark.parametrize(
    ("normalized_type", "status_code", "expected_type", "expected_error_type"),
    [
        (NormalizedBadRequestError, 400, anthropic.BadRequestError, "invalid_request_error"),
        (
            NormalizedAuthenticationError,
            401,
            anthropic.AuthenticationError,
            "authentication_error",
        ),
        (
            NormalizedPermissionDeniedError,
            403,
            anthropic.PermissionDeniedError,
            "permission_error",
        ),
        (NormalizedNotFoundError, 404, anthropic.NotFoundError, "not_found_error"),
        (NormalizedConflictError, 409, anthropic.ConflictError, "conflict_error"),
        (
            NormalizedUnprocessableEntityError,
            422,
            anthropic.UnprocessableEntityError,
            "invalid_request_error",
        ),
        (NormalizedRateLimitError, 429, anthropic.RateLimitError, "rate_limit_error"),
        (
            NormalizedInternalServerError,
            500,
            anthropic.InternalServerError,
            "api_error",
        ),
        (
            NormalizedRequestTooLargeError,
            413,
            anthropic.RequestTooLargeError,
            "request_too_large",
        ),
        (NormalizedOverloadedError, 529, anthropic.OverloadedError, "overloaded_error"),
        (NormalizedAPIStatusError, 418, anthropic.APIStatusError, "api_error"),
    ],
)
def test_normalized_status_error_projects_to_anthropic(
    normalized_type: NormalizedErrorType,
    status_code: int,
    expected_type: type[anthropic.APIStatusError],
    expected_error_type: str,
) -> None:
    projected = _anthropic_api_status_error(_normalized_error(normalized_type, status_code))

    assert type(projected) is expected_type
    assert projected.status_code == status_code
    assert str(projected.request.url) == "https://upstream.example/v1/responses"
    assert projected.request_id == "req_source"
    assert projected.body == {
        "type": "error",
        "error": {
            "type": expected_error_type,
            "message": "request failed",
        },
        "request_id": "req_source",
    }
    assert projected.response.headers["x-source"] == "kept"
    assert projected.response.headers["content-length"] != "999"
    assert "x-request-id" not in projected.response.headers


async def _failed_events(
    error: NormalizedAPIStatusError,
    cause: Exception,
) -> AsyncIterator[ModelResponseStreamEvent]:
    raise error from cause
    yield


async def test_base_adapter_projects_normalized_error_to_other_sdk():
    source_request = httpx.Request("POST", "https://upstream.example/v1/messages")
    source_response = httpx.Response(
        404,
        request=source_request,
        json={
            "type": "error",
            "error": {"type": "not_found_error", "message": "model not found"},
            "request_id": "req_anthropic",
        },
    )
    source_error = anthropic.NotFoundError(
        "model not found",
        response=source_response,
        body=source_response.json(),
    )
    normalized_error = NormalizedNotFoundError(
        "model not found",
        status_code=404,
        request_id="req_anthropic",
        response_headers=dict(source_response.headers),
        request_method=source_request.method,
        request_url=str(source_request.url),
    )
    adapter = OpenAIResponsesAdapter(PROVIDER_CAPABILITIES["anthropic"])
    request = OpenAIResponsesCreateRequest(input="hi", model="missing", stream=True)
    normalized_request = adapter.normalize_request(request)

    with pytest.raises(openai.NotFoundError) as exc_info:
        async for _ in adapter.project_events(
            _failed_events(normalized_error, source_error), normalized_request
        ):
            pass

    assert exc_info.value.__cause__ is source_error
    assert exc_info.value.status_code == 404
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.request_id == "req_anthropic"


async def test_base_adapter_reuses_same_sdk_error_and_its_cause():
    request = httpx.Request("POST", "https://upstream.example/v1/chat/completions")
    response = httpx.Response(400, request=request)
    transport_error = RuntimeError("transport failed")
    source_error = openai.BadRequestError(
        "bad request",
        response=response,
        body={"message": "bad request", "type": "invalid_request_error"},
    )
    source_error.__cause__ = transport_error
    normalized_error = NormalizedBadRequestError(
        "bad request",
        status_code=400,
        request_id=None,
        response_headers={},
        request_method=request.method,
        request_url=str(request.url),
    )
    adapter = OpenAIResponsesAdapter(PROVIDER_CAPABILITIES["openai"])
    native_request = OpenAIResponsesCreateRequest(input="hi", model="test", stream=True)
    normalized_request = adapter.normalize_request(native_request)

    with pytest.raises(openai.BadRequestError) as exc_info:
        async for _ in adapter.project_events(
            _failed_events(normalized_error, source_error), normalized_request
        ):
            pass

    assert exc_info.value is source_error
    assert source_error.__cause__ is transport_error


def _openai_status_error(
    error_type: type[openai.APIStatusError],
    status_code: int,
) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://upstream.example/v1/responses")
    response = httpx.Response(
        status_code,
        request=request,
        headers={"x-request-id": "req_openai"},
    )
    return error_type(
        f"Error code: {status_code}",
        response=response,
        body={"message": "request failed", "type": "invalid_request_error"},
    )


@pytest.mark.parametrize(
    ("source_type", "status_code", "expected_type"),
    [
        (openai.BadRequestError, 400, NormalizedBadRequestError),
        (openai.AuthenticationError, 401, NormalizedAuthenticationError),
        (openai.PermissionDeniedError, 403, NormalizedPermissionDeniedError),
        (openai.NotFoundError, 404, NormalizedNotFoundError),
        (openai.ConflictError, 409, NormalizedConflictError),
        (openai.UnprocessableEntityError, 422, NormalizedUnprocessableEntityError),
        (openai.RateLimitError, 429, NormalizedRateLimitError),
        (openai.InternalServerError, 500, NormalizedInternalServerError),
        (openai.APIStatusError, 418, NormalizedAPIStatusError),
    ],
)
def test_openai_status_error_normalizes_to_expected_class(
    source_type: type[openai.APIStatusError],
    status_code: int,
    expected_type: NormalizedErrorType,
) -> None:
    normalized = normalized_openai_status_error(_openai_status_error(source_type, status_code))

    assert type(normalized) is expected_type
    assert normalized.status_code == status_code
    assert normalized.message == "request failed"
    assert normalized.request_id == "req_openai"


def _anthropic_status_error(
    error_type: type[anthropic.APIStatusError],
    status_code: int,
) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://upstream.example/v1/messages")
    response = httpx.Response(status_code, request=request)
    body = {
        "type": "error",
        "error": {"type": "api_error", "message": "request failed"},
        "request_id": "req_anthropic",
    }
    return error_type(
        f"Error code: {status_code}",
        response=response,
        body=body,
    )


@pytest.mark.parametrize(
    ("source_type", "status_code", "expected_type"),
    [
        (anthropic.BadRequestError, 400, NormalizedBadRequestError),
        (anthropic.AuthenticationError, 401, NormalizedAuthenticationError),
        (anthropic.PermissionDeniedError, 403, NormalizedPermissionDeniedError),
        (anthropic.NotFoundError, 404, NormalizedNotFoundError),
        (anthropic.ConflictError, 409, NormalizedConflictError),
        (anthropic.RequestTooLargeError, 413, NormalizedRequestTooLargeError),
        (
            anthropic.UnprocessableEntityError,
            422,
            NormalizedUnprocessableEntityError,
        ),
        (anthropic.RateLimitError, 429, NormalizedRateLimitError),
        (anthropic.OverloadedError, 529, NormalizedOverloadedError),
        (anthropic.InternalServerError, 500, NormalizedInternalServerError),
        (anthropic.APIStatusError, 418, NormalizedAPIStatusError),
    ],
)
def test_anthropic_status_error_normalizes_to_expected_class(
    source_type: type[anthropic.APIStatusError],
    status_code: int,
    expected_type: NormalizedErrorType,
) -> None:
    normalized = normalized_anthropic_status_error(
        _anthropic_status_error(source_type, status_code)
    )

    assert type(normalized) is expected_type
    assert normalized.status_code == status_code
    assert normalized.message == "request failed"
    assert normalized.request_id == "req_anthropic"
