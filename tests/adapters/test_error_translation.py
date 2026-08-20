from collections.abc import AsyncIterator

import anthropic
import httpx
import openai
import pytest

from molto.adapters.anthropic import _anthropic_api_status_error
from molto.adapters.openai import OpenAIResponsesAdapter, _openai_api_status_error
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.normalized import ModelResponseStreamEvent, NormalizedAPIStatusError
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.anthropic import (
    _normalized_api_status_error as normalized_anthropic_status_error,
)
from molto.upstream.openai import (
    _normalized_api_status_error as normalized_openai_status_error,
)


def _normalized_error(status_code: int) -> NormalizedAPIStatusError:
    return NormalizedAPIStatusError(
        "request failed",
        status_code=status_code,
        source_protocol="openai",
        source_type="invalid_request_error",
        code="source_code",
        param="model",
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
    ("status_code", "expected_type"),
    [
        (400, openai.BadRequestError),
        (401, openai.AuthenticationError),
        (402, openai.APIStatusError),
        (403, openai.PermissionDeniedError),
        (404, openai.NotFoundError),
        (409, openai.ConflictError),
        (413, openai.APIStatusError),
        (422, openai.UnprocessableEntityError),
        (429, openai.RateLimitError),
        (500, openai.InternalServerError),
        (529, openai.InternalServerError),
    ],
)
def test_openai_status_projection_matches_sdk_dispatch(status_code, expected_type):
    projected = _openai_api_status_error(_normalized_error(status_code))

    assert type(projected) is expected_type
    assert projected.status_code == status_code
    assert str(projected.request.url) == "https://upstream.example/v1/responses"
    assert projected.request_id == "req_source"
    assert projected.body == {
        "message": "request failed",
        "type": "invalid_request_error",
        "param": "model",
        "code": "source_code",
    }
    assert projected.response.headers["x-source"] == "kept"
    assert projected.response.headers["content-length"] != "999"


@pytest.mark.parametrize(
    ("status_code", "expected_type", "expected_error_type"),
    [
        (400, anthropic.BadRequestError, "invalid_request_error"),
        (401, anthropic.AuthenticationError, "authentication_error"),
        (402, anthropic.APIStatusError, "billing_error"),
        (403, anthropic.PermissionDeniedError, "permission_error"),
        (404, anthropic.NotFoundError, "not_found_error"),
        (409, anthropic.ConflictError, "conflict_error"),
        (413, anthropic.RequestTooLargeError, "request_too_large"),
        (422, anthropic.UnprocessableEntityError, "invalid_request_error"),
        (429, anthropic.RateLimitError, "rate_limit_error"),
        (500, anthropic.InternalServerError, "api_error"),
        (503, anthropic.InternalServerError, "api_error"),
        (504, anthropic.InternalServerError, "timeout_error"),
        (529, anthropic.OverloadedError, "overloaded_error"),
    ],
)
def test_anthropic_status_projection_matches_sdk_dispatch(
    status_code, expected_type, expected_error_type
):
    projected = _anthropic_api_status_error(_normalized_error(status_code))

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
    try:
        raise cause
    except Exception as source:
        raise error from source
    yield


async def test_base_adapter_projects_error_with_original_upstream_cause():
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
    normalized_error = NormalizedAPIStatusError(
        "model not found",
        status_code=404,
        source_protocol="anthropic",
        source_type="not_found_error",
        code=None,
        param=None,
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
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.request_id == "req_anthropic"


def test_openai_status_normalization_preserves_structured_fields():
    request = httpx.Request("POST", "https://upstream.example/v1/responses")
    response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": "req_openai"},
    )
    source_error = openai.BadRequestError(
        "Error code: 400",
        response=response,
        body={
            "message": "max_tokens is too low",
            "type": "invalid_request_error",
            "param": "max_output_tokens",
            "code": "invalid_value",
        },
    )

    normalized = normalized_openai_status_error(source_error)

    assert normalized.message == "max_tokens is too low"
    assert normalized.category == "invalid_request"
    assert normalized.source_type == "invalid_request_error"
    assert normalized.param == "max_output_tokens"
    assert normalized.code == "invalid_value"
    assert normalized.request_id == "req_openai"


def test_anthropic_status_normalization_uses_envelope_request_id():
    request = httpx.Request("POST", "https://upstream.example/v1/messages")
    response = httpx.Response(404, request=request)
    body = {
        "type": "error",
        "error": {"type": "not_found_error", "message": "model not found"},
        "request_id": "req_envelope",
    }
    source_error = anthropic.NotFoundError(
        "Error code: 404",
        response=response,
        body=body,
    )

    normalized = normalized_anthropic_status_error(source_error)

    assert normalized.message == "model not found"
    assert normalized.category == "not_found"
    assert normalized.source_type == "not_found_error"
    assert normalized.request_id == "req_envelope"
