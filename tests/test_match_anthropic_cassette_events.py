from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import RawMessageStreamEvent
from pydantic import TypeAdapter

from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml
from scripts.match_anthropic_cassette_events import main, match_messages_events

MESSAGE_EVENT_ADAPTER = TypeAdapter(RawMessageStreamEvent)
MATCHED_CASSETTES = Path(__file__).parent / "cassettes" / "matched" / "reasoning_raw"


def _sse_unit(event: dict[str, Any], *, event_name: str | None = None) -> list[str]:
    return [
        f"event: {event_name or event['type']}\n\n",
        f"data: {json_dumps(event)}\n\n",
        "\n",
    ]


def _messages_cassette(
    *,
    thinking_deltas: list[str] | None,
    text_deltas: list[str],
    request_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events: list[tuple[dict[str, Any], str | None]] = [
        (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_original",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-original",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 3,
                        "cache_creation_input_tokens": 1,
                        "cache_read_input_tokens": 2,
                        "output_tokens": 1,
                        "service_tier": "priority",
                    },
                },
            },
            None,
        )
    ]

    text_index = 0
    if thinking_deltas is not None:
        events.extend(
            [
                (
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        },
                    },
                    None,
                ),
                ({"type": "ping"}, None),
                *[
                    (
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": delta,
                            },
                        },
                        None,
                    )
                    for delta in thinking_deltas
                ],
                (
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "signature_delta",
                            "signature": "signed-thinking",
                        },
                    },
                    None,
                ),
                ({"type": "content_block_stop", "index": 0}, None),
            ]
        )
        text_index = 1

    events.append(
        (
            {
                "type": "content_block_start",
                "index": text_index,
                "content_block": {"type": "text", "text": ""},
            },
            None,
        )
    )
    events.extend(
        (
            {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": delta},
            },
            None,
        )
        for delta in text_deltas
    )
    events.extend(
        [
            (
                {"type": "content_block_stop", "index": text_index},
                "content_block_delta",
            ),
            (
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {
                        "input_tokens": 5,
                        "cache_creation_input_tokens": 1,
                        "cache_read_input_tokens": 2,
                        "output_tokens": 4,
                    },
                },
                None,
            ),
            ({"type": "message_stop"}, None),
        ]
    )

    sse_entries = [
        entry for event, event_name in events for entry in _sse_unit(event, event_name=event_name)
    ]
    return {
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "headers": {"anthropic-version": "2023-06-01"},
            "body": {
                "max_tokens": 20_000,
                "messages": deepcopy(
                    request_messages
                    if request_messages is not None
                    else [{"content": "old prompt", "role": "user"}]
                ),
                "model": "claude-original",
                "stream": True,
                "thinking": {"display": "summarized", "type": "adaptive"},
            },
        },
        "response": {"status_code": 200, "sse": sse_entries},
    }


def _responses_cassette(
    *,
    reasoning_deltas: list[str] | None,
    text_deltas: list[str],
    request_input: Any = "canonical prompt",
) -> dict[str, Any]:
    response = {
        "id": "resp_reference",
        "created_at": 123,
        "model": "model-response-reference",
    }
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": response},
    ]
    if reasoning_deltas is not None:
        events.extend(
            {"type": "response.reasoning_text.delta", "delta": delta} for delta in reasoning_deltas
        )
        events.append(
            {
                "type": "response.reasoning_text.done",
                "text": "".join(reasoning_deltas),
            }
        )
    events.extend({"type": "response.output_text.delta", "delta": delta} for delta in text_deltas)
    events.extend(
        [
            {"type": "response.output_text.done", "text": "".join(text_deltas)},
            {
                "type": "response.completed",
                "response": {
                    **response,
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 90,
                        "input_tokens_details": {
                            "cache_write_tokens": 7,
                            "cached_tokens": 11,
                        },
                        "output_tokens": 17,
                        "output_tokens_details": {
                            "reasoning_tokens": 13 if reasoning_deltas is not None else 0
                        },
                        "total_tokens": 107,
                    },
                },
            },
        ]
    )
    return {
        "request": {
            "method": "POST",
            "path": "/v1/responses",
            "body": {
                "input": deepcopy(request_input),
                "model": "model-request-reference",
                "stream": True,
                "stream_options": {"include_obfuscation": False},
            },
        },
        "response": {
            "status_code": 200,
            "sse": [f"data: {json_dumps(event)}\n\n" for event in events],
        },
    }


def _data_events(cassette: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        json_loads(entry.strip().removeprefix("data:").lstrip())
        for entry in cassette["response"]["sse"]
        if entry.strip().startswith("data:")
    ]


def test_matches_messages_reasoning_and_text_events() -> None:
    template_request_messages = [
        {"content": "old question", "role": "user"},
        {
            "content": [
                {
                    "signature": "signed-history",
                    "thinking": "old reasoning",
                    "type": "thinking",
                },
                {"text": "old answer", "type": "text"},
            ],
            "role": "assistant",
        },
        {"content": "old follow-up", "role": "user"},
    ]
    messages_cassette = _messages_cassette(
        thinking_deltas=["old thinking"],
        text_deltas=["old", " text"],
        request_messages=template_request_messages,
    )
    original_request = deepcopy(messages_cassette["request"])
    canonical_input = [
        {"content": "canonical question", "role": "user"},
        {
            "content": [{"text": "canonical reasoning", "type": "reasoning_text"}],
            "status": "completed",
            "summary": [],
            "type": "reasoning",
        },
        {
            "content": [
                {
                    "annotations": [],
                    "logprobs": [],
                    "text": "canonical answer",
                    "type": "output_text",
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        },
        {"content": "canonical follow-up", "role": "user"},
    ]
    responses_cassette = _responses_cassette(
        reasoning_deltas=["new", " reasoning", "!"],
        text_deltas=["new text"],
        request_input=canonical_input,
    )

    match_messages_events(messages_cassette, responses_cassette)

    events = _data_events(messages_cassette)
    for event in events:
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)

    message = next(event["message"] for event in events if event["type"] == "message_start")
    assert message["id"] == "resp_reference"
    assert message["model"] == "model-response-reference"
    assert message["usage"]["service_tier"] == "standard"
    assert message["usage"]["input_tokens"] == 72
    assert message["usage"]["cache_creation_input_tokens"] == 7
    assert message["usage"]["cache_read_input_tokens"] == 11
    assert message["usage"]["output_tokens"] == 0
    assert [
        event["delta"]["thinking"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "thinking_delta"
    ] == ["new", " reasoning", "!"]
    assert [
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
    ] == ["new text"]
    assert (
        next(
            event["delta"]["signature"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "signature_delta"
        )
        == "signed-thinking"
    )

    usage = next(event["usage"] for event in events if event["type"] == "message_delta")
    assert usage == {
        "input_tokens": 72,
        "cache_creation_input_tokens": 7,
        "cache_read_input_tokens": 11,
        "output_tokens": 17,
        "output_tokens_details": {"thinking_tokens": 13},
    }
    assert sum(event["type"] == "ping" for event in events) == 1
    assert sum(event["type"] == "message_start" for event in events) == 1
    assert sum(event["type"] == "message_delta" for event in events) == 1
    assert sum(event["type"] == "message_stop" for event in events) == 1
    assert sum(event["type"] == "content_block_start" for event in events) == 2
    assert sum(event["type"] == "content_block_stop" for event in events) == 2
    expected_request = {
        **original_request,
        "body": {
            "max_tokens": 20_000,
            "messages": [
                {"content": "canonical question", "role": "user"},
                {
                    "content": [
                        {
                            "signature": "signed-history",
                            "thinking": "canonical reasoning",
                            "type": "thinking",
                        },
                        {"text": "canonical answer", "type": "text"},
                    ],
                    "role": "assistant",
                },
                {"content": "canonical follow-up", "role": "user"},
            ],
            "model": "model-request-reference",
            "stream": True,
            "thinking": {"display": "summarized", "type": "adaptive"},
        },
    }
    assert messages_cassette["request"] == expected_request

    stop_data_index = next(
        index
        for index, entry in enumerate(messages_cassette["response"]["sse"])
        if entry.strip().startswith("data:")
        and json_loads(entry.strip().removeprefix("data:").lstrip()).get("type")
        == "content_block_stop"
        and json_loads(entry.strip().removeprefix("data:").lstrip()).get("index") == 1
    )
    assert (
        messages_cassette["response"]["sse"][stop_data_index - 1].strip()
        == "event: content_block_delta"
    )


def test_matches_messages_without_reasoning_via_cli(tmp_path: Path) -> None:
    messages_path = tmp_path / "messages.yaml"
    responses_path = tmp_path / "responses.yaml"
    output_path = tmp_path / "matched.yaml"
    dump_yaml(
        _messages_cassette(thinking_deltas=None, text_deltas=["old"]),
        str(messages_path),
        sort_keys=False,
    )
    dump_yaml(
        _responses_cassette(reasoning_deltas=None, text_deltas=["new", " text"]),
        str(responses_path),
        sort_keys=False,
    )

    assert (
        main(
            [
                str(messages_path),
                str(responses_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    events = _data_events(read_yaml(str(output_path)))
    for event in events:
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)
    assert [
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
    ] == ["new", " text"]
    assert not any(
        event["type"] == "content_block_start" and event["content_block"]["type"] == "thinking"
        for event in events
    )
    assert read_yaml(str(output_path))["request"]["body"]["messages"] == [
        {"content": "canonical prompt", "role": "user"}
    ]


@pytest.mark.parametrize(
    ("message_reasoning", "response_reasoning"),
    [(None, ["reasoning"]), (["thinking"], None)],
)
def test_rejects_reasoning_shape_mismatch(
    message_reasoning: list[str] | None, response_reasoning: list[str] | None
) -> None:
    messages_cassette = _messages_cassette(thinking_deltas=message_reasoning, text_deltas=["text"])
    responses_cassette = _responses_cassette(
        reasoning_deltas=response_reasoning, text_deltas=["text"]
    )

    with pytest.raises(ValueError, match="disagree on the presence of 'thinking'"):
        match_messages_events(messages_cassette, responses_cassette)


def test_rejects_unrepresentable_service_tier() -> None:
    messages_cassette = _messages_cassette(thinking_deltas=None, text_deltas=["text"])
    responses_cassette = _responses_cassette(reasoning_deltas=None, text_deltas=["text"])
    completed = next(
        event["response"]
        for event in _data_events(responses_cassette)
        if event["type"] == "response.completed"
    )
    completed["service_tier"] = "flex"
    responses_cassette["response"]["sse"][-1] = (
        f"data: {json_dumps({'type': 'response.completed', 'response': completed})}\n\n"
    )

    with pytest.raises(ValueError, match="cannot be represented by Anthropic Messages"):
        match_messages_events(messages_cassette, responses_cassette)


def test_matches_real_two_turn_request_body() -> None:
    messages_cassette = read_yaml(str(MATCHED_CASSETTES / "anthropic_responses-stream-01.yaml"))
    responses_cassette = read_yaml(str(MATCHED_CASSETTES / "openai_responses-stream-01.yaml"))
    turn_zero = read_yaml(str(MATCHED_CASSETTES / "anthropic_responses-stream-00.yaml"))
    turn_zero_events = _data_events(turn_zero)
    signature = next(
        event["delta"]["signature"]
        for event in turn_zero_events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "signature_delta"
    )

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette["request"]["body"] == {
        "max_tokens": 20_000,
        "messages": [
            {"content": "What is 2+2?", "role": "user"},
            {
                "content": [
                    {
                        "signature": signature,
                        "thinking": "We need answer simple. Need comply. 2+2=4.",
                        "type": "thinking",
                    },
                    {"text": "4", "type": "text"},
                ],
                "role": "assistant",
            },
            {"content": "What about 2*2?", "role": "user"},
        ],
        "model": "deepseek-v4-flash",
        "stream": True,
        "thinking": {"display": "summarized", "type": "adaptive"},
    }
    message_start = next(
        event["message"]
        for event in _data_events(messages_cassette)
        if event["type"] == "message_start"
    )
    assert message_start["usage"]["output_tokens"] == 0
