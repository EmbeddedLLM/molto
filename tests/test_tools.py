from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.openai.responses import OpenAIResponsesRequest
from molto.utils.io import json_loads, read_yaml

CASSETTE_DIR = Path("tests/cassettes")


def _read_cassette_response_body(path: Path) -> dict[str, Any]:
    """Read the response body from a cassette YAML file."""
    return read_yaml(path)["response"]["body"]


async def test_tool_choice_required():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = json_loads(request.content)
        try:
            assert body["tool_choice"] == "required"
            assert len(body["messages"]) == 1
            return httpx.Response(
                200,
                json=_read_cassette_response_body(
                    CASSETTE_DIR / "openai" / "tool-choice-required.yaml"
                ),
            )
        except AssertionError:
            return httpx.Response(
                400,
                json={"error": "Invalid request"},
            )

    body = dict(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What number am I thinking of?"}],
        tools=[
            {
                "type": "function",
                "function": {
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
            }
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
    )
    response = await client.chat.completions.create(**body)
    assert isinstance(response, ChatCompletion)
    tool_calls = response.choices[0].message.tool_calls
    assert isinstance(tool_calls, list)
    assert isinstance(tool_calls[0], ChatCompletionMessageFunctionToolCall)
    assert tool_calls[0].function.name == "guess_number"

    client = Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        )
    )
    await client.chat(
        OpenAIResponsesRequest(
            model="gpt-4o",
            input=[{"role": "user", "content": "Hello, world!"}],
            tool_choice="required",
        )
    )
