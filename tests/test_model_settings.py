from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionWebSearch,
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
    OpenAIWebSearchToolCall,
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
async def test_model_settings(stream: bool):

    req_body = dict(
        model="gpt-5.1",
        input=[
            {"role": "user", "content": "Who are you?"},
        ],
        instructions="The user would like to know about the weather in Taiwan today.",
        max_output_tokens=400,
        max_tool_calls=1,
        parallel_tool_calls=True,
        prompt_cache_key="my_cache_key",
        prompt_cache_retention="24h",
        safety_identifier="safety",
        store=False,
        service_tier="flex",
        temperature=0.45,
        tool_choice="required",
        top_logprobs=1,
        include=[
            "reasoning.encrypted_content",
            "code_interpreter_call.outputs",
            "web_search_call.action.sources",
            "message.output_text.logprobs",
        ],
        tools=[{"type": "web_search_preview"}, {"type": "code_interpreter"}],
        stream=stream,
    )

    def handler(request: httpx.Request, load_stream_cassette: bool) -> httpx.Response:
        try:
            assert request.method == "POST"
            body = json_loads(request.content)
            for attr in [
                "model",
                "input",
                "instructions",
                "max_output_tokens",
                "max_tool_calls",
                "parallel_tool_calls",
                "prompt_cache_key",
                "prompt_cache_retention",
                "safety_identifier",
                "service_tier",
                "temperature",
                "top_logprobs",
            ]:
                assert body.get(attr) == req_body[attr], (
                    f"Expected {attr} to be {req_body[attr]!r}, got {body.get(attr)!r}"
                )
            assert body.get("store") == req_body["store"], (
                f"Expected store to be {req_body['store']!r}, got {body.get('store')!r}"
            )
            assert body.get("stream", False) is load_stream_cassette
            data = _read_cassette(
                CASSETTE_DIR / "openai" / "responses" / "model_settings.yaml",
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

    # ── OpenAI SDK client ──────────────────────────────────────────────
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
        assert isinstance(events[-1], ResponseCompletedEvent)
        response = events[-1].response
    assert isinstance(response, Response)

    # Response-level fields from cassette
    assert response.status == "completed"
    assert response.incomplete_details is None
    assert response.model == "gpt-5.1-2025-11-13"
    assert response.temperature == req_body["temperature"]
    assert response.max_output_tokens == req_body["max_output_tokens"]
    assert response.max_tool_calls == req_body["max_tool_calls"]
    assert response.parallel_tool_calls is True
    assert response.top_logprobs == 1
    assert (
        response.instructions == "The user would like to know about the weather in Taiwan today."
    )
    assert response.tool_choice == "required"
    assert response.service_tier == "flex"
    assert response.store is False
    assert response.safety_identifier == "safety"
    assert response.prompt_cache_key == "my_cache_key"
    assert response.prompt_cache_retention == "24h"
    assert response.frequency_penalty == 0.0
    assert response.presence_penalty == 0.0
    assert response.top_p == 1.0
    assert response.truncation == "disabled"
    assert response.reasoning.effort == "none"

    # Output structure: web_search_call + message
    assert len(response.output) == 2

    # Web search call
    ws = response.output[0]
    assert isinstance(ws, ResponseFunctionWebSearch)
    assert ws.type == "web_search_call"
    assert ws.status == "completed"
    assert ws.action.type == "search"
    assert ws.action.query == "weather: Taiwan"
    assert ws.action.queries == ["weather: Taiwan"]
    assert ws.action.sources is not None
    assert len(ws.action.sources) == 1
    assert ws.action.sources[0].type == "api"
    assert ws.action.sources[0].name == "oai-weather"

    # Message output
    msg = response.output[1]
    assert isinstance(msg, ResponseOutputMessage)
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.status == "completed"
    assert len(msg.content) == 1

    text_content = msg.content[0]
    assert isinstance(text_content, ResponseOutputText)
    assert text_content.type == "output_text"
    assert len(text_content.text) > 0

    # Logprobs (top_logprobs=1 means 1 alternative per token)
    assert text_content.logprobs is not None
    assert len(text_content.logprobs) > 0
    for lp in text_content.logprobs:
        assert len(lp.top_logprobs) == 1
        assert isinstance(lp.token, str)
        assert isinstance(lp.logprob, float)
        assert isinstance(lp.bytes, list)

    # Usage
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cached_tokens >= 0
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.reasoning_tokens >= 0

    # ── Molto client (always streams internally) ───────────────────────
    client = Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(partial(handler, load_stream_cassette=True)),
            base_url="http://testserver",
        ),
    )
    copy_body = deepcopy(req_body)
    copy_body["model"] = "openai/" + copy_body["model"]
    response = await client.chat(OpenAIResponsesRequest(**copy_body))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"
        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    # Response-level fields echoed from the IR (request values)
    assert response.status == "completed"
    assert response.incomplete_details is None
    assert response.model == "openai/" + req_body["model"]
    assert response.temperature == req_body["temperature"]
    assert response.max_output_tokens == req_body["max_output_tokens"]
    assert response.max_tool_calls == req_body["max_tool_calls"]
    assert response.parallel_tool_calls is True
    assert response.top_logprobs == 1
    assert (
        response.instructions == "The user would like to know about the weather in Taiwan today."
    )
    assert response.tool_choice == "required"
    assert response.service_tier == "flex"
    assert response.store is False
    assert response.safety_identifier == "safety"
    assert response.prompt_cache_key == "my_cache_key"
    assert response.prompt_cache_retention == "24h"
    assert response.frequency_penalty == 0.0
    assert response.presence_penalty == 0.0
    assert response.top_p == 1.0
    assert response.truncation == "disabled"
    assert response.reasoning is not None
    assert response.reasoning.effort == "medium"

    # Tools echoed from request
    assert len(response.tools) == 2
    assert response.tools[0].type == "web_search"
    assert response.tools[1].type == "code_interpreter"

    # Completed at from API response
    assert response.completed_at is not None
    assert isinstance(response.completed_at, int)
    assert response.completed_at > 0

    # Output structure: web_search_call + message
    assert len(response.output) == 2

    # Web search call
    ws_molto = response.output[0]
    assert isinstance(ws_molto, OpenAIWebSearchToolCall)
    assert ws_molto.type == "web_search_call"
    assert ws_molto.status == "completed"
    assert ws_molto.action.type == "search"
    assert ws_molto.action.query == "weather: Taiwan"
    assert ws_molto.action.queries == ["weather: Taiwan"]
    assert ws_molto.action.sources is not None
    assert len(ws_molto.action.sources) == 1
    assert ws_molto.action.sources[0].type == "api"
    assert ws_molto.action.sources[0].name == "oai-weather"

    # Message output
    msg_molto = response.output[1]
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0

    # Logprobs (top_logprobs=1 means 1 alternative per token) disabled for now due to a bug in pydantic-ai issue #5091
    # assert text_molto.logprobs is not None
    # assert len(text_molto.logprobs) > 0

    # Usage
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cached_tokens >= 0
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.reasoning_tokens >= 0
