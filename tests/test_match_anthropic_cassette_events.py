from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import RawMessageStreamEvent
from pydantic import TypeAdapter

from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml
from scripts.match_anthropic_cassette_events import main, match_messages_events

MESSAGE_EVENT_ADAPTER = TypeAdapter(RawMessageStreamEvent)
MATCHED_CASSETTES = Path(__file__).parent / "cassettes" / "matched"
REASONING_RAW_CASSETTES = MATCHED_CASSETTES / "reasoning_raw"
TOOL_CALL_CASSETTES = MATCHED_CASSETTES / "tool_call"


def _sse_unit(event: dict[str, Any], *, event_name: str | None = None) -> list[str]:
    return [
        f"event: {event_name or event['type']}\n\n",
        f"data: {json_dumps(event)}\n\n",
        "\n",
    ]


def _messages_cassette(
    *,
    thinking_deltas: list[str] | None,
    text_deltas: list[str] | None,
    tool_calls: list[dict[str, Any]] | None = None,
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

    next_block_index = 0
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
        next_block_index = 1

    if text_deltas is not None:
        events.append(
            (
                {
                    "type": "content_block_start",
                    "index": next_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
                None,
            )
        )
        events.extend(
            (
                {
                    "type": "content_block_delta",
                    "index": next_block_index,
                    "delta": {"type": "text_delta", "text": delta},
                },
                None,
            )
            for delta in text_deltas
        )
        events.append(
            (
                {"type": "content_block_stop", "index": next_block_index},
                "content_block_delta",
            )
        )
        next_block_index += 1

    for tool_call in tool_calls or []:
        events.append(
            (
                {
                    "type": "content_block_start",
                    "index": next_block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": {},
                    },
                },
                None,
            )
        )
        events.extend(
            (
                {
                    "type": "content_block_delta",
                    "index": next_block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": delta,
                    },
                },
                None,
            )
            for delta in tool_call["argument_deltas"]
        )
        events.append(({"type": "content_block_stop", "index": next_block_index}, None))
        next_block_index += 1

    events.extend(
        [
            (
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use" if tool_calls else "end_turn",
                        "stop_sequence": None,
                    },
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
    reasoning_summary_deltas: list[str] | None = None,
    text_deltas: list[str] | None,
    tool_calls: list[dict[str, Any]] | None = None,
    request_input: Any = "canonical prompt",
    reasoning_encrypted_content: str | None = None,
) -> dict[str, Any]:
    response = {
        "id": "resp_reference",
        "created_at": 123,
        "model": "model-response-reference",
    }
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": response},
    ]
    if reasoning_deltas is not None or reasoning_summary_deltas is not None:
        if reasoning_summary_deltas is not None:
            events.extend(
                [
                    {
                        "type": "response.reasoning_summary_part.added",
                        "part": {"type": "summary_text", "text": ""},
                    },
                    *(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "delta": delta,
                        }
                        for delta in reasoning_summary_deltas
                    ),
                    {
                        "type": "response.reasoning_summary_text.done",
                        "text": "".join(reasoning_summary_deltas),
                    },
                    {
                        "type": "response.reasoning_summary_part.done",
                        "part": {
                            "type": "summary_text",
                            "text": "".join(reasoning_summary_deltas),
                        },
                    },
                ]
            )
        reasoning_deltas = reasoning_deltas or []
        events.extend(
            {"type": "response.reasoning_text.delta", "delta": delta} for delta in reasoning_deltas
        )
        if reasoning_deltas:
            events.append(
                {
                    "type": "response.reasoning_text.done",
                    "text": "".join(reasoning_deltas),
                }
            )
        reasoning_item = {
            "type": "reasoning",
            "content": [
                {
                    "type": "reasoning_text",
                    "text": "".join(reasoning_deltas),
                }
            ]
            if reasoning_deltas
            else [],
            "summary": [
                {
                    "type": "summary_text",
                    "text": "".join(reasoning_summary_deltas),
                }
            ]
            if reasoning_summary_deltas is not None
            else [],
        }
        if reasoning_encrypted_content is not None:
            reasoning_item["encrypted_content"] = reasoning_encrypted_content
        events.append({"type": "response.output_item.done", "item": reasoning_item})
    if text_deltas is not None:
        events.extend(
            {"type": "response.output_text.delta", "delta": delta} for delta in text_deltas
        )
        events.append({"type": "response.output_text.done", "text": "".join(text_deltas)})
    for output_index, tool_call in enumerate(
        tool_calls or [], start=1 if text_deltas is not None else 0
    ):
        events.extend(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": tool_call["item_id"],
                "delta": delta,
            }
            for delta in tool_call["argument_deltas"]
        )
        arguments = "".join(tool_call["argument_deltas"])
        events.extend(
            [
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": tool_call["item_id"],
                    "arguments": arguments,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "status": "completed",
                        "id": tool_call["item_id"],
                        "call_id": tool_call["call_id"],
                        "name": tool_call["name"],
                        "arguments": arguments,
                    },
                },
            ]
        )
    events.append(
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
                        "reasoning_tokens": 13
                        if reasoning_deltas is not None or reasoning_summary_deltas is not None
                        else 0
                    },
                    "total_tokens": 107,
                },
            },
        }
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
    assert not any(
        event["type"] == "content_block_delta" and event["delta"]["type"] == "signature_delta"
        for event in events
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
                        {"thinking": "canonical reasoning", "type": "thinking"},
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


def test_matches_responses_reasoning_summary_to_messages_thinking() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        reasoning_summary_deltas=["summary", " text"],
        reasoning_encrypted_content="canonical-signature",
        text_deltas=["answer"],
        request_input=[
            {"content": "question", "role": "user"},
            {
                "content": [],
                "encrypted_content": "history-signature",
                "summary": [{"text": "history summary", "type": "summary_text"}],
                "type": "reasoning",
            },
            {"content": "prior answer", "role": "assistant"},
            {"content": "follow-up", "role": "user"},
        ],
    )
    messages_cassette = _messages_cassette(
        thinking_deltas=["template thinking"],
        text_deltas=["template answer"],
    )

    match_messages_events(messages_cassette, responses_cassette)

    events = _data_events(messages_cassette)
    assert [
        event["delta"]["thinking"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "thinking_delta"
    ] == ["summary", " text"]
    assert [
        event["delta"]["signature"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "signature_delta"
    ] == ["canonical-signature"]
    assert messages_cassette["request"]["body"]["messages"][1]["content"][0] == {
        "signature": "history-signature",
        "thinking": "history summary",
        "type": "thinking",
    }


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


def test_matches_tool_call_history_and_argument_deltas() -> None:
    response_tool_calls = [
        {
            "item_id": "fc_weather",
            "call_id": "call_weather",
            "name": "Weather",
            "argument_deltas": ['{"city"', ':"Paris"}'],
        },
        {
            "item_id": "fc_time",
            "call_id": "call_time",
            "name": "DateTime",
            "argument_deltas": ["{}"],
        },
    ]
    messages_cassette = _messages_cassette(
        thinking_deltas=None,
        text_deltas=["template text"],
        tool_calls=[
            {
                "id": "toolu_template_weather",
                "name": "TemplateWeather",
                "argument_deltas": ["", "template"],
            },
            {
                "id": "toolu_template_time",
                "name": "TemplateTime",
                "argument_deltas": [""],
            },
        ],
    )
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=["I will call both tools."],
        tool_calls=response_tool_calls,
        request_input=[
            {"content": "Check Paris.", "role": "user"},
            {"content": "I will call both tools.", "role": "assistant"},
            {
                "arguments": '{"city":"Paris"}',
                "call_id": "call_weather",
                "id": "fc_weather_history",
                "name": "Weather",
                "type": "function_call",
            },
            {
                "arguments": "{}",
                "call_id": "call_time",
                "id": "fc_time_history",
                "name": "DateTime",
                "type": "function_call",
            },
            {
                "call_id": "call_weather",
                "output": "Sunny",
                "type": "function_call_output",
            },
            {
                "call_id": "call_time",
                "output": "12:00Z",
                "type": "function_call_output",
            },
        ],
    )

    match_messages_events(messages_cassette, responses_cassette)

    events = _data_events(messages_cassette)
    for event in events:
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)
    tool_starts = [
        event
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
    ]
    assert [
        (
            event["content_block"]["id"],
            event["content_block"]["name"],
            event["content_block"]["input"],
        )
        for event in tool_starts
    ] == [
        ("call_weather", "Weather", {}),
        ("call_time", "DateTime", {}),
    ]
    assert [
        [
            event["delta"]["partial_json"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["index"] == tool_start["index"]
            and event["delta"]["type"] == "input_json_delta"
        ]
        for tool_start in tool_starts
    ] == [
        ['{"city"', ':"Paris"}'],
        ["{}"],
    ]
    assert messages_cassette["request"]["body"]["messages"] == [
        {"content": "Check Paris.", "role": "user"},
        {
            "content": [
                {"text": "I will call both tools.", "type": "text"},
                {
                    "id": "call_weather",
                    "input": {"city": "Paris"},
                    "name": "Weather",
                    "type": "tool_use",
                },
                {
                    "id": "call_time",
                    "input": {},
                    "name": "DateTime",
                    "type": "tool_use",
                },
            ],
            "role": "assistant",
        },
        {
            "content": [
                {
                    "content": "Sunny",
                    "tool_use_id": "call_weather",
                    "type": "tool_result",
                },
                {
                    "content": "12:00Z",
                    "tool_use_id": "call_time",
                    "type": "tool_result",
                },
            ],
            "role": "user",
        },
    ]


def test_matches_tool_only_stream_without_text_block() -> None:
    tool_call = {
        "item_id": "fc_tool_only",
        "call_id": "call_tool_only",
        "name": "DateTime",
        "argument_deltas": ["{}"],
    }
    messages_cassette = _messages_cassette(
        thinking_deltas=None,
        text_deltas=None,
        tool_calls=[
            {
                "id": "toolu_template",
                "name": "Template",
                "argument_deltas": [""],
            }
        ],
    )
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=None,
        tool_calls=[tool_call],
    )

    match_messages_events(messages_cassette, responses_cassette)

    events = _data_events(messages_cassette)
    for event in events:
        MESSAGE_EVENT_ADAPTER.validate_python(event)
    assert not any(
        event["type"] == "content_block_start" and event["content_block"]["type"] == "text"
        for event in events
    )
    tool_start = next(
        event
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
    )
    assert tool_start["content_block"]["id"] == "call_tool_only"
    assert [
        event["delta"]["partial_json"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta"
    ] == ["{}"]


def test_rejects_tool_call_shape_mismatch() -> None:
    messages_cassette = _messages_cassette(
        thinking_deltas=None,
        text_deltas=["text"],
    )
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=["text"],
        tool_calls=[
            {
                "item_id": "fc_missing",
                "call_id": "call_missing",
                "name": "Missing",
                "argument_deltas": ["{}"],
            }
        ],
    )

    with pytest.raises(ValueError, match="disagree on the number of tool calls"):
        match_messages_events(messages_cassette, responses_cassette)


def test_maps_responses_image_urls_to_messages_sources() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=["answer"],
        request_input=[
            {
                "content": [
                    {"text": "What is this?", "type": "input_text"},
                    {
                        "image_url": "data:image/jpeg;base64,aW1hZ2UtZGF0YQ==",
                        "type": "input_image",
                    },
                    {
                        "image_url": "https://example.com/image.png",
                        "type": "input_image",
                    },
                ],
                "role": "user",
            }
        ],
    )
    messages_cassette = _messages_cassette(
        thinking_deltas=None,
        text_deltas=["template"],
    )

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette["request"]["body"]["messages"] == [
        {
            "content": [
                {"text": "What is this?", "type": "text"},
                {
                    "source": {
                        "data": "aW1hZ2UtZGF0YQ==",
                        "media_type": "image/jpeg",
                        "type": "base64",
                    },
                    "type": "image",
                },
                {
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.png",
                    },
                    "type": "image",
                },
            ],
            "role": "user",
        }
    ]


@pytest.mark.parametrize("image_url", [None, "data:image/png,not-base64"])
def test_rejects_invalid_responses_image_urls(image_url: str | None) -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=["answer"],
        request_input=[
            {
                "content": [{"image_url": image_url, "type": "input_image"}],
                "role": "user",
            }
        ],
    )
    messages_cassette = _messages_cassette(
        thinking_deltas=None,
        text_deltas=["template"],
    )

    with pytest.raises(ValueError, match="Responses image"):
        match_messages_events(messages_cassette, responses_cassette)


def test_maps_responses_encrypted_content_to_messages_signatures() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=["canonical reasoning"],
        reasoning_encrypted_content="canonical-signature",
        text_deltas=["canonical answer"],
        request_input=[
            {"content": "question", "role": "user"},
            {
                "content": [{"text": "prior reasoning", "type": "reasoning_text"}],
                "encrypted_content": "canonical-history-signature",
                "type": "reasoning",
            },
            {"content": "prior answer", "role": "assistant"},
            {"content": "follow-up", "role": "user"},
        ],
    )
    messages_cassette = _messages_cassette(
        thinking_deltas=["template reasoning"],
        text_deltas=["template answer"],
        request_messages=[
            {"content": "template question", "role": "user"},
            {
                "content": [
                    {
                        "signature": "template-history-signature",
                        "thinking": "template reasoning",
                        "type": "thinking",
                    },
                    {"text": "template answer", "type": "text"},
                ],
                "role": "assistant",
            },
        ],
    )

    match_messages_events(messages_cassette, responses_cassette)

    events = _data_events(messages_cassette)
    assert [
        event["delta"]["signature"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "signature_delta"
    ] == ["canonical-signature"]
    assert messages_cassette["request"]["body"]["messages"][1]["content"][0] == {
        "signature": "canonical-history-signature",
        "thinking": "prior reasoning",
        "type": "thinking",
    }


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


@pytest.mark.parametrize("turn", range(3))
def test_matches_real_tool_call_cassettes(turn: int) -> None:
    messages_cassette = read_yaml(
        str(TOOL_CALL_CASSETTES / f"anthropic_messages-stream-{turn:02d}.yaml")
    )
    responses_cassette = read_yaml(
        str(TOOL_CALL_CASSETTES / f"openai_responses-stream-{turn:02d}.yaml")
    )

    match_messages_events(messages_cassette, responses_cassette)

    message_events = _data_events(messages_cassette)
    response_events = _data_events(responses_cassette)
    for event in message_events:
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)

    response_calls = [
        event["item"]
        for event in response_events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "function_call"
    ]
    message_tool_starts = [
        event
        for event in message_events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
    ]
    assert [
        (event["content_block"]["id"], event["content_block"]["name"])
        for event in message_tool_starts
    ] == [(item["call_id"], item["name"]) for item in response_calls]
    for message_start, response_call in zip(message_tool_starts, response_calls, strict=True):
        assert [
            event["delta"]["partial_json"]
            for event in message_events
            if event["type"] == "content_block_delta"
            and event["index"] == message_start["index"]
            and event["delta"]["type"] == "input_json_delta"
        ] == [
            event["delta"]
            for event in response_events
            if event["type"] == "response.function_call_arguments.delta"
            and event["item_id"] == response_call["id"]
        ]

    response_input = responses_cassette["request"]["body"]["input"]
    request_function_calls = [
        item for item in response_input if item.get("type") == "function_call"
    ]
    request_function_outputs = [
        item for item in response_input if item.get("type") == "function_call_output"
    ]
    message_request_blocks = [
        block
        for message in messages_cassette["request"]["body"]["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
    ]
    assert [
        (block["id"], block["name"], block["input"])
        for block in message_request_blocks
        if block["type"] == "tool_use"
    ] == [
        (item["call_id"], item["name"], json_loads(item["arguments"]))
        for item in request_function_calls
    ]
    assert [
        (block["tool_use_id"], block["content"])
        for block in message_request_blocks
        if block["type"] == "tool_result"
    ] == [(item["call_id"], item["output"]) for item in request_function_outputs]


def test_matches_real_two_turn_request_body() -> None:
    messages_cassette = read_yaml(
        str(REASONING_RAW_CASSETTES / "anthropic_messages-stream-01.yaml")
    )
    responses_cassette = read_yaml(
        str(REASONING_RAW_CASSETTES / "openai_responses-stream-01.yaml")
    )
    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette["request"]["body"] == {
        "max_tokens": 2048,
        "messages": [
            {"content": "What is 2+2?", "role": "user"},
            {
                "content": [
                    {
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
