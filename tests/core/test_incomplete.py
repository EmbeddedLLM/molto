"""SDK parity for synthetic incomplete terminals on the simple-text recording."""

from copy import deepcopy
from typing import Any

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI

from molto import Molto
from molto.utils.io import dumps_sse, json_loads, load_sse
from tests.core.test_interface import (
    PROTOCOLS,
    Protocol,
    _collect,
    _coordinator,
    _matched_cassette_path,
    _read_cassette,
    _sse_records,
)
from tests.core.test_web_search import _assert_lifecycle


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("downstream", PROTOCOLS)
@pytest.mark.parametrize(
    ("incomplete_fields", "reason", "messages_reason", "chat_reason"),
    [
        pytest.param(
            {"incomplete_details": {"reason": "max_output_tokens"}},
            "max_output_tokens",
            "max_tokens",
            "length",
            id="max_output_tokens",
        ),
        pytest.param(
            {"incomplete_details": {"reason": "content_filter"}},
            "content_filter",
            "end_turn",
            "content_filter",
            id="content_filter",
        ),
        pytest.param({"incomplete_details": None}, None, "end_turn", "stop", id="null_details"),
        pytest.param({}, None, "end_turn", "stop", id="absent_details"),
        pytest.param(
            {"incomplete_details": {"reason": None}},
            None,
            "end_turn",
            "stop",
            id="null_reason",
        ),
        pytest.param({"incomplete_details": {}}, None, "end_turn", "stop", id="absent_reason"),
    ],
)
async def test_synthetic_incomplete_responses_preserve_sdk_text_usage_and_reason(
    stream: bool,
    downstream: Protocol,
    incomplete_fields: dict[str, Any],
    reason: str | None,
    messages_reason: str,
    chat_reason: str,
) -> None:
    """Ensures unknown incomplete reasons never invent a pause, while text and SDK usage survive."""
    source = _read_cassette(_matched_cassette_path("simple_text", "responses", "stream-00.yaml"))
    events = [load_sse(record) for record in _sse_records(source, "responses")]
    assert events[-1]["type"] == "response.completed"
    assert not source["request"]["body"].get("tools")
    assert all(not event.get("response", {}).get("tools") for event in events)
    assert not any(item["type"] == "web_search_call" for item in events[-1]["response"]["output"])

    # Only the terminal state is synthetic; the cassette's text, usage, and
    # preceding lifecycle remain the SDK reference for every downstream mode.
    events[-1]["type"] = "response.incomplete"
    terminal = events[-1]["response"]
    terminal.update(status="incomplete", completed_at=None)
    terminal.pop("incomplete_details", None)
    terminal.update(incomplete_fields)
    payload = "".join(dumps_sse(event) for event in events)
    upstream_stream = stream or downstream != "responses"
    expected_body = {**source["request"]["body"], "stream": upstream_stream}

    def handler(request: Request) -> Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        assert json_loads(request.content) == expected_body
        if upstream_stream:
            return Response(200, text=payload, headers={"content-type": "text/event-stream"})
        return Response(200, json=terminal)

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        reference = await sdk.responses.create(**expected_body)
        if upstream_stream:
            async with reference:
                reference_events = [event async for event in reference]
            assert reference_events[-1].type == "response.incomplete"
            reference = reference_events[-1].response
    assert reference.status == "incomplete"
    assert (
        reference.incomplete_details.reason if reference.incomplete_details else None
    ) == reason
    assert reference.output_text and reference.usage.output_tokens > 0

    body = deepcopy(
        _read_cassette(_matched_cassette_path("simple_text", downstream, "stream-00.yaml"))[
            "request"
        ]["body"]
    )
    body["stream"] = stream
    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        actual = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body, endpoint="responses", api_key="test-api-key"
        )
        if stream:
            async with actual:
                actual_events = [event async for event in actual]
            _assert_lifecycle(downstream, actual_events)
            actual = _collect(downstream, actual_events)
            if downstream == "responses":
                assert [event.model_dump() for event in actual_events] == [
                    event.model_dump() for event in reference_events
                ]

    assert actual.id == reference.id
    assert actual.model == reference.model
    if downstream == "responses":
        assert actual.model_dump() == reference.model_dump()
    elif downstream == "chat/completions":
        assert actual.choices[0].finish_reason == chat_reason
        assert actual.choices[0].message.content == reference.output_text
        assert actual.usage.prompt_tokens == reference.usage.input_tokens
        assert actual.usage.completion_tokens == reference.usage.output_tokens
        assert actual.usage.total_tokens == reference.usage.total_tokens
        assert (
            actual.usage.prompt_tokens_details.cached_tokens
            == reference.usage.input_tokens_details.cached_tokens
        )
        assert (
            actual.usage.completion_tokens_details.reasoning_tokens
            == reference.usage.output_tokens_details.reasoning_tokens
        )
    else:
        assert actual.stop_reason == messages_reason
        assert "".join(block.text for block in actual.content if block.type == "text") == (
            reference.output_text
        )
        input_details = reference.usage.input_tokens_details
        assert actual.usage.input_tokens == (
            reference.usage.input_tokens
            - input_details.cached_tokens
            - input_details.cache_write_tokens
        )
        assert actual.usage.cache_read_input_tokens == input_details.cached_tokens
        assert actual.usage.cache_creation_input_tokens == input_details.cache_write_tokens
        assert actual.usage.output_tokens == reference.usage.output_tokens
