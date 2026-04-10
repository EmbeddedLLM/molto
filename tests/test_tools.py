from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
)

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.openai.responses import (
    OpenAIFunctionToolCall,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamText,
    _OpenAIStreamEvent,
)
from molto.utils.io import json_loads, read_yaml
from molto.utils.test import STREAM_PARAMS

CASSETTE_DIR = Path("tests/cassettes")


def _read_cassette(path: Path, stream: bool) -> dict[str, Any]:
    """Read the response body from a cassette YAML file."""
    if stream:
        path = path.with_stem(f"{path.stem}-stream")
    data = read_yaml(path)["response"]
    return data["sse"] if stream else data["body"]


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_tool_choice_required(stream: bool):

    req_body = dict(
        model="openai/gpt-4.1-nano",
        input=[{"role": "user", "content": "What number am I thinking of?"}],
        tools=[
            {
                "type": "function",
                "name": "guess_number",
                "description": "Guess a number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer", "description": "The guessed number."},
                    },
                    "required": ["number"],
                    "additionalProperties": False,
                },
            },
        ],
        tool_choice="required",
        stream=stream,
    )

    def handler(request: httpx.Request, load_stream_cassette: bool) -> httpx.Response:
        try:
            assert request.method == "POST"
            body = json_loads(request.content)
            for attr in ["model", "input", "tools", "tool_choice"]:
                assert body.get(attr) == req_body[attr], (
                    f"Expected {attr} to be {req_body[attr]!r}, got {body.get(attr)!r}"
                )
            assert len(body.get("input", [])) == 1
            assert body.get("stream", False) is load_stream_cassette
            data = _read_cassette(
                CASSETTE_DIR / "openai" / "responses" / "tool_choice-required.yaml",
                stream=load_stream_cassette,
            )
            if load_stream_cassette:
                return httpx.Response(
                    200,
                    content="".join(data).encode(),
                    headers={"content-type": "text/event-stream; charset=utf-8"},
                )
            return httpx.Response(200, json=data)
        except AssertionError as e:
            return httpx.Response(
                400,
                json={"error": f"Assertions failed: {e!r}"},
            )
        except Exception as e:
            return httpx.Response(
                400,
                json={"error": f"Invalid request: {e!r}"},
            )

    client = AsyncOpenAI(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(partial(handler, load_stream_cassette=stream)),
            base_url="http://testserver",
        ),
        api_key=ENV_CONFIG.openai_api_key_plain,
        # base_url="http://0.0.0.0:8080/v1",
        max_retries=1,  # Disable retries to surface assertion errors immediately
    )
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert ResponseFunctionCallArgumentsDeltaEvent in {type(e) for e in events}
        assert "response.function_call_arguments.delta" in {e.type for e in events}
        assert isinstance(events[-1], ResponseCompletedEvent)
        response = events[-1].response
    assert isinstance(response, Response)
    assert isinstance(response.output, list)
    assert isinstance(response.output[0], ResponseFunctionToolCall)
    assert response.output[0].name == "guess_number"

    client = Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(partial(handler, load_stream_cassette=True)),
            base_url="http://testserver",
        ),
    )
    response = await client.chat(OpenAIResponsesRequest(**req_body))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.function_call_arguments.delta" in {e.type for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)
    assert isinstance(response.output, list)
    assert isinstance(response.output[0], OpenAIFunctionToolCall)
    assert response.output[0].name == "guess_number"
