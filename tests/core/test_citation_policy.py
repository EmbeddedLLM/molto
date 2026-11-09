"""Explicit Messages citation policies against recorded provider SDK streams."""

import pytest
from anthropic import AsyncAnthropic
from httpx import AsyncClient, MockTransport, Response

from molto import Molto
from molto.utils.io import dumps_sse
from tests.core.test_interface import _collect, _get_http_client
from tests.core.test_web_search import (
    _assert_lifecycle,
    _messages_semantics,
    _path,
    _request_pair,
    _sdk_result,
)


@pytest.mark.parametrize("citation_policy", [None, "drop", "buffer", "auto"])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("tools_in_extra_body", [False, True])
async def test_messages_citation_policy_matches_sdk(citation_policy, stream, tools_in_extra_body):
    """Ensures auto uses declared search while buffer also preserves opaque search citations."""
    body, expected_request = _request_pair("messages", "responses", stream)
    extra_body = None
    if tools_in_extra_body:
        body.pop("tools")
        body.pop("tool_choice")
        extra_body = {"tools": [{"type": "web_search"}], "tool_choice": "required"}
        expected_request.pop("include")
        expected_request.update(extra_body)
    expected = _messages_semantics(_collect("messages", await _sdk_result("messages", True)))
    sdk_events = await _sdk_result("responses", True)
    buffer_text = citation_policy == "buffer" or (
        citation_policy == "auto" and not tools_in_extra_body
    )
    if not buffer_text:
        content = []
        for block in expected["content"]:
            if block["type"] != "text":
                content.append(block)
            elif content and content[-1]["type"] == "text":
                content[-1]["text"] += block["text"]
            else:
                content.append({**block, "citations": None})
        expected["content"] = content
    options = {} if citation_policy is None else {"citation_policy": citation_policy}
    async with _get_http_client(
        [_path("responses")], "responses", expected_body=expected_request
    ) as http_client:
        result = await Molto(http_client=http_client).messages(
            body=body,
            endpoint="responses",
            api_key="test-api-key",
            extra_body=extra_body,
            **options,
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle("messages", events)
            actual = _collect("messages", events)
            if not buffer_text:
                assert [
                    event.delta.text
                    for event in events
                    if event.type == "content_block_delta" and event.delta.type == "text_delta"
                ] == [
                    event.delta
                    for event in sdk_events
                    if event.type == "response.output_text.delta"
                ]
        else:
            actual = result
    assert _messages_semantics(actual) == expected


@pytest.mark.parametrize("citation_policy", ["buffer", "auto"])
async def test_buffered_citations_preserve_sdk_text_stream(citation_policy):
    """Ensures SDK text iteration exposes the recorded answer without losing citation blocks."""
    body, expected_request = _request_pair("messages", "responses", True)
    reference = _collect("responses", await _sdk_result("responses", True))
    async with _get_http_client(
        [_path("responses")], "responses", expected_body=expected_request
    ) as http_client:
        stream = await Molto(http_client=http_client).messages(
            body=body,
            endpoint="responses",
            api_key="test-api-key",
            citation_policy=citation_policy,
        )
        events = [event async for event in stream]

    payload = "".join(dumps_sse(event, event.type) for event in events)
    transport = MockTransport(
        lambda _: Response(200, text=payload, headers={"content-type": "text/event-stream"})
    )
    async with AsyncAnthropic(
        api_key="test-api-key", http_client=AsyncClient(transport=transport)
    ) as client:
        sdk_body = {key: value for key, value in body.items() if key != "stream"}
        async with client.messages.stream(**sdk_body) as sdk_stream:
            text_chunks = [text async for text in sdk_stream.text_stream]
            message = await sdk_stream.get_final_message()

    assert "".join(text_chunks) == reference.output_text
    text_blocks = [block for block in message.content if block.type == "text"]
    assert text_chunks == [block.text for block in text_blocks]
    expected = _collect("messages", await _sdk_result("messages", True))
    assert [block.model_dump() for block in text_blocks] == [
        block.model_dump() for block in expected.content if block.type == "text"
    ]


@pytest.mark.parametrize("citation_policy", ["drop", "buffer", "auto"])
@pytest.mark.parametrize("stream", [True, False])
async def test_native_messages_ignore_citation_policy(citation_policy, stream):
    """Ensures projection policy never changes native Messages requests or SDK responses."""
    body, expected_request = _request_pair("messages", "messages", stream)
    expected = await _sdk_result("messages", stream)
    async with _get_http_client(
        [_path("messages", stream)], "messages", expected_body=expected_request
    ) as http_client:
        result = await Molto(http_client=http_client).messages(
            body=body,
            api_key="test-api-key",
            citation_policy=citation_policy,
        )
        if stream:
            result = [event async for event in result]
            assert [event.model_dump() for event in result] == [
                event.model_dump() for event in expected
            ]
        else:
            assert result.model_dump() == expected.model_dump()
