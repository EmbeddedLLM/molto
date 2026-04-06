from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.responses import Response, ResponseFunctionToolCall

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.openai.responses import (
    OpenAIFunctionToolCall,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
)
from molto.utils.io import json_loads, read_yaml

CASSETTE_DIR = Path("tests/cassettes")


def _read_cassette_response_body(path: Path) -> dict[str, Any]:
    """Read the response body from a cassette YAML file."""
    return read_yaml(path)["response"]["body"]


async def test_tool_choice_required():
    def handler(request: httpx.Request) -> httpx.Response:
        try:
            assert request.method == "POST"
            body = json_loads(request.content)
            assert body.get("tool_choice") == "required"
            assert len(body.get("input", [])) == 1
            assert body.get("stream", False) is False
            return httpx.Response(
                200,
                json=_read_cassette_response_body(
                    CASSETTE_DIR / "openai" / "responses" / "tool_choice-required.yaml"
                ),
            )
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

    body = dict(
        model="gpt-4.1-nano",
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
    )
    client = AsyncOpenAI(
        api_key=ENV_CONFIG.openai_api_key_plain,
        # base_url="http://0.0.0.0:8080/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        ),
        max_retries=1,  # Disable retries to surface assertion errors immediately
    )
    response = await client.responses.create(**body)
    assert isinstance(response, Response)
    assert isinstance(response.output, list)
    assert isinstance(response.output[0], ResponseFunctionToolCall)
    assert response.output[0].name == "guess_number"

    client = Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        )
    )
    response = await client.chat(OpenAIResponsesRequest(**body))
    assert isinstance(response, OpenAIResponsesResponse)
    assert isinstance(response.output, list)
    assert isinstance(response.output[0], OpenAIFunctionToolCall)
    assert response.output[0].name == "guess_number"
