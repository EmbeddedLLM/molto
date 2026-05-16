from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic_ai import ModelHTTPError

from molto import Molto
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

CASSETTE_DIR = Path("tests/cassettes/openai/exceptions")


def _load_cassette(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _cassette_request_params(path: Path) -> dict:
    cassette = _load_cassette(path)
    body = cassette["request"]["body"]
    params: dict[str, Any] = {
        "model": f"openai/{body['model']}",
        "input": body["input"],
        "stream": False,
    }
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
        verbosity=None,
        tools=None,
        tool_choice=None,
        include=None,
        store=None,
        previous_response_id=None,
        metadata=None,
        background=None,
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


def _make_error_handler(cassette_path: Path):
    cassette = _load_cassette(cassette_path)
    resp = cassette["response"]
    status_code: int = resp["status_code"]
    body: Any = resp["body"]
    raw_headers = resp.get("headers") or {}
    headers = {k: v for k, v in raw_headers.items() if k.lower() != "content-length"}

    def handler(request: httpx.Request, load_stream_cassette: bool) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status_code, content=body.encode(), headers=headers)
        return httpx.Response(status_code, json=body, headers=headers)

    return handler


def _make_molto_from_cassette(cassette_path: Path) -> Molto:
    handler = _make_error_handler(cassette_path)
    return Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(partial(handler, load_stream_cassette=True)),
            base_url="http://testserver",
        ),
    )


def _cassette(name: str) -> Path:
    return CASSETTE_DIR / f"{name}.yaml"


class TestBadInputError:
    async def test_context_window_exceeded(self):
        path = _cassette("context_length_exceeded")
        assert path.exists(), f"Cassette not found: {path}"

        molto = _make_molto_from_cassette(path)
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

        molto = _make_molto_from_cassette(path)
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

        molto = _make_molto_from_cassette(path)
        req = OpenAIResponsesRequest(**_cassette_request_params(path))
        with pytest.raises(ExternalAuthError):
            await molto.chat(req)

    async def test_missing_api_key(self):
        path = _cassette("missing_api_key")
        assert path.exists(), f"Cassette not found: {path}"

        molto = _make_molto_from_cassette(path)
        req = OpenAIResponsesRequest(**_cassette_request_params(path))
        with pytest.raises(ExternalAuthError):
            await molto.chat(req)
