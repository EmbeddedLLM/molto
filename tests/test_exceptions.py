from pathlib import Path
from typing import Any

import pytest
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
from molto.utils.test import make_molto_client, prepend_model_prefix, read_cassette_request

CASSETTE_DIR = Path("tests/cassettes/openai/exceptions")


def _cassette_request_params(path: Path) -> dict:
    body = read_cassette_request(path)
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
        tools=None,
        tool_choice=None,
        stream=None,
    )
    defaults.update(overrides)
    return Input(**defaults)


def _cassette(name: str) -> Path:
    return CASSETTE_DIR / f"{name}.yaml"

# TEMP
class TestBadInputError:
    async def test_context_window_exceeded(self):
        path = _cassette("context_length_exceeded")
        assert path.exists(), f"Cassette not found: {path}"

        molto = make_molto_client(cassette_path=path)
        req = OpenAIResponsesRequest(**_cassette_request_params(path))
        with pytest.raises(ContextOverflowError):
            await molto.chat(req)

    async def test_invalid_temperature(self):
        ir = _make_ir(temperature=-1.0)
        provider = OpenAIResponsesProvider()
        with pytest.raises(BadInputError) as exc_info:
            await provider.parse_agent_settings(ir)
        assert "temperature" in str(exc_info.value)

    async def test_invalid_temperature_above_range(self):
        ir = _make_ir(temperature=5.0)
        provider = OpenAIResponsesProvider()
        with pytest.raises(BadInputError) as exc_info:
            await provider.parse_agent_settings(ir)
        assert "temperature" in str(exc_info.value)

    async def test_thinking_budget_exceeding_max_tokens(self):
        path = _cassette("thinking_budget_exceeding_max_tokens")
        assert path.exists(), f"Cassette not found: {path}"

        molto = make_molto_client(cassette_path=path)
        req = OpenAIResponsesRequest(**_cassette_request_params(path))
        with pytest.raises(BadInputError):
            await molto.chat(req)


class TestModelOverloadError:
    async def test_network_connectivity_503(self):
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
    async def test_invalid_api_key(self):
        path = _cassette("incorrect_api_key")
        assert path.exists(), f"Cassette not found: {path}"

        molto = make_molto_client(cassette_path=path)
        req = OpenAIResponsesRequest(**_cassette_request_params(path))
        with pytest.raises(ExternalAuthError):
            await molto.chat(req)

    async def test_missing_api_key(self):
        path = _cassette("missing_api_key")
        assert path.exists(), f"Cassette not found: {path}"

        molto = make_molto_client(cassette_path=path)
        req = OpenAIResponsesRequest(**_cassette_request_params(path))
        with pytest.raises(ExternalAuthError):
            await molto.chat(req)
