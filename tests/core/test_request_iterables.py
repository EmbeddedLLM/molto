"""Reusable validated histories with nested SDK iterable fields."""

from copy import deepcopy

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from molto import Molto
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from tests.core.test_interface import (
    Protocol,
    _coordinator,
    _get_http_client,
    _matched_cassette_path,
    _read_cassette,
)


@pytest.mark.parametrize("protocol", ["messages", "chat/completions"])
async def test_native_model_history_can_be_sent_twice(protocol: Protocol):
    """Ensures repeated model sends retain tool history and citations with exact SDK parity."""
    path = _matched_cassette_path("tool_call", protocol, "stream-01.yaml")
    body = deepcopy(_read_cassette(path)["request"]["body"])
    if protocol == "messages":
        # Synthetic citations exercise repeatability without changing the recording.
        body["messages"][1]["content"][0]["citations"] = [
            {
                "type": "web_search_result_location",
                "cited_text": "A source",
                "encrypted_index": "opaque-index",
                "title": "Example",
                "url": "https://example.com/",
            }
        ]
        request_model = AnthropicMessagesCreateRequest
    else:
        request_model = OpenAIChatCompletionsCreateRequest
    request = request_model.model_validate(body)

    async with _get_http_client([path], protocol, expected_body=body) as http_client:
        if protocol == "messages":
            create = AsyncAnthropic(
                api_key="test-api-key", http_client=http_client
            ).messages.create
        else:
            create = AsyncOpenAI(
                api_key="test-api-key", http_client=http_client
            ).chat.completions.create
        reference = await create(**body)
        async with reference:
            reference = [event async for event in reference]

    requests = []
    async with _get_http_client(
        [path, path], protocol, expected_body=body, requests=requests
    ) as http_client:
        coordinator = _coordinator(Molto(http_client=http_client), protocol)
        for _ in range(2):
            result = await coordinator(body=request, endpoint=protocol, api_key="test-api-key")
            async with result:
                events = [event async for event in result]
            assert [type(event) for event in events] == [type(event) for event in reference]
            assert [event.model_dump() for event in events] == [
                event.model_dump() for event in reference
            ]
    assert len(requests) == 2
    assert request.model_dump() == body
