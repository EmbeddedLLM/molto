from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseIncompleteEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import ResponseOutputText

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
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
    if stream:
        path = path.with_stem(f"{path.stem}-stream")
    data = read_yaml(path)["response"]
    return data["sse"] if stream else data["body"]


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_incomplete_response(stream: bool):

    req_body = dict(
        model="gpt-4.1-nano",
        input=[
            {"role": "system", "content": "You are a financial advisor"},
            {"role": "user", "content": "Who are you?"},
        ],
        instructions="You are a financial advisor for the user.",
        max_output_tokens=20,
        stream=stream,
    )

    def handler(request: httpx.Request, load_stream_cassette: bool) -> httpx.Response:
        try:
            assert request.method == "POST"
            body = json_loads(request.content)
            for attr in ["model", "input", "instructions", "max_output_tokens"]:
                assert body.get(attr) == req_body[attr], (
                    f"Expected {attr} to be {req_body[attr]!r}, got {body.get(attr)!r}"
                )
            assert body.get("stream", False) is load_stream_cassette
            data = _read_cassette(
                CASSETTE_DIR / "openai" / "responses" / "incomplete.yaml",
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
        max_retries=1,
    )
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert ResponseTextDeltaEvent in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert ResponseIncompleteEvent in {type(e) for e in events}
        assert isinstance(events[-1], ResponseIncompleteEvent)
        response = events[-1].response
    assert isinstance(response, Response)

    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    assert response.max_output_tokens == 20
    assert response.instructions == "You are a financial advisor for the user."

    assert len(response.output) == 1
    msg = response.output[0]
    assert isinstance(msg, ResponseOutputMessage)
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.status == "incomplete"
    assert len(msg.content) == 1

    text_content = msg.content[0]
    assert isinstance(text_content, ResponseOutputText)
    assert text_content.type == "output_text"
    assert len(text_content.text) > 0
    assert "financial advisor" in text_content.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens == 20
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )

    client = Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(partial(handler, load_stream_cassette=True)),
            base_url="http://testserver",
        ),
    )
    from copy import deepcopy

    copy_body = deepcopy(req_body)
    copy_body["model"] = "openai/" + copy_body["model"]
    response = await client.chat(OpenAIResponsesRequest(**copy_body))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.incomplete"
        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    assert response.max_output_tokens == 20
    assert response.instructions == "You are a financial advisor for the user."

    assert len(response.output) == 1
    msg_molto = response.output[0]
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "incomplete"
    assert len(msg_molto.content) == 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0
    assert "financial advisor" in text_molto.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens == 20
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )
