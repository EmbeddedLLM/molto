from typing import Any

import pytest
from openai import APIConnectionError, APIError, APIStatusError
from pydantic_ai import ModelHTTPError

from molto.providers import OpenAIResponsesProvider
from molto.runner import _map_pydantic_ai_error
from molto.types.ir import Input
from molto.types.openai.responses import OpenAIResponsesRequest
from molto.utils.exceptions import (
    BadInputError,
    ContextOverflowError,
    ExternalAuthError,
    ModelOverloadError,
)
from molto.utils.test import make_molto_client, make_openai_client, prepend_model_prefix

CONTEXT_OVERFLOW_INPUT = "Repeat this exactly: " + ("x " * 300_000).strip()

EXCEPTION_REQUESTS: dict[str, dict[str, Any]] = {
    "context_length_exceeded": {
        "model": "o3-mini",
        "input": CONTEXT_OVERFLOW_INPUT,
    },
    "thinking_budget_exceeding_max_tokens": {
        "model": "o4-mini",
        "input": "Think very carefully about this.",
        "max_output_tokens": 1,
        "reasoning": {"effort": "high"},
    },
    "incorrect_api_key": {
        "model": "gpt-4.1-nano",
        "input": "hi",
    },
    "missing_api_key": {
        "model": "gpt-4.1-nano",
        "input": "hi",
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


def _molto_provider_error_request_body(name: str) -> dict[str, Any]:
    """Return the OpenAI request body pydantic-ai sends for Molto replay."""
    body = _provider_error_request_body(name)
    return {
        **body,
        "input": [{"role": "user", "content": body["input"]}],
    }


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
        expected_request_body=req_body,
        assert_attrs=["model", "input"],
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
            "stream": False,
        }
    )
    for key in ("reasoning", "max_output_tokens"):
        if key in body:
            params[key] = body[key]
    return params


def _make_ir(**overrides) -> Input:
    """Return an Input IR with complete defaults for provider validation tests."""
    defaults = dict(
        model="openai/gpt-4.1-nano",
        instructions=None,
        temperature=None,
        top_p=None,
        max_output_tokens=None,
        reasoning_effort=None,
        reasoning_summary=None,
        top_logprobs=None,
        parallel_tool_calls=None,
        max_tool_calls=None,
        text_format=None,
        verbosity=None,
        tools=None,
        tool_choice=None,
        include=None,
        store=None,
        previous_response_id=None,
        metadata=None,
        conversation_id=None,
        prompt=None,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        safety_identifier=None,
        service_tier=None,
        stream_obfuscation=None,
        truncation=None,
        extra_body=None,
        stream=None,
    )
    defaults.update(overrides)
    return Input(**defaults)


class TestBadInputError:
    async def test_context_window_exceeded(self, cassette_manager):
        """Map OpenAI context-window errors to ContextOverflowError."""
        await _record_or_replay_provider_error(cassette_manager, "context_length_exceeded")

        molto = make_molto_client(
            cassette_manager,
            cassette_suffix="context_length_exceeded",
            expected_request_body=_molto_provider_error_request_body("context_length_exceeded"),
            assert_attrs=["model", "input"],
        )
        req = OpenAIResponsesRequest(**_cassette_request_params("context_length_exceeded"))
        with pytest.raises(ContextOverflowError):
            await molto.chat(req)

    async def test_invalid_temperature(self):
        """Reject temperatures below the OpenAI-supported range."""
        ir = _make_ir(temperature=-1.0)
        provider = OpenAIResponsesProvider()
        with pytest.raises(BadInputError) as exc_info:
            await provider.parse_agent_settings(ir)
        assert "temperature" in str(exc_info.value)

    async def test_invalid_temperature_above_range(self):
        """Reject temperatures above the OpenAI-supported range."""
        ir = _make_ir(temperature=5.0)
        provider = OpenAIResponsesProvider()
        with pytest.raises(BadInputError) as exc_info:
            await provider.parse_agent_settings(ir)
        assert "temperature" in str(exc_info.value)

    async def test_thinking_budget_exceeding_max_tokens(self, cassette_manager):
        """Map invalid reasoning/max-token provider errors to BadInputError."""
        await _record_or_replay_provider_error(
            cassette_manager,
            "thinking_budget_exceeding_max_tokens",
        )

        molto = make_molto_client(
            cassette_manager,
            cassette_suffix="thinking_budget_exceeding_max_tokens",
            expected_request_body=_molto_provider_error_request_body(
                "thinking_budget_exceeding_max_tokens"
            ),
            assert_attrs=["model", "input"],
        )
        req = OpenAIResponsesRequest(
            **_cassette_request_params("thinking_budget_exceeding_max_tokens")
        )
        with pytest.raises(BadInputError):
            await molto.chat(req)


class TestModelOverloadError:
    async def test_network_connectivity_503(self):
        """Map HTTP 503 model errors to ModelOverloadError."""
        exc = ModelHTTPError(
            status_code=503,
            model_name="gpt-4.1-nano",
            body="Service unavailable",
        )
        mapped = _map_pydantic_ai_error(exc)
        assert type(mapped) is ModelOverloadError
        with pytest.raises(ModelOverloadError):
            raise mapped

    async def test_server_downtime_529(self):
        """Map HTTP 529 model errors to ModelOverloadError."""
        exc = ModelHTTPError(
            status_code=529,
            model_name="gpt-4.1-nano",
            body="Site is overloaded",
        )
        mapped = _map_pydantic_ai_error(exc)
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
            expected_request_body=_molto_provider_error_request_body("incorrect_api_key"),
            assert_attrs=["model", "input"],
        )
        req = OpenAIResponsesRequest(**_cassette_request_params("incorrect_api_key"))
        with pytest.raises(ExternalAuthError):
            await molto.chat(req)

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
            expected_request_body=_molto_provider_error_request_body("missing_api_key"),
            assert_attrs=["model", "input"],
        )
        req = OpenAIResponsesRequest(**_cassette_request_params("missing_api_key"))
        with pytest.raises(ExternalAuthError):
            await molto.chat(req)
