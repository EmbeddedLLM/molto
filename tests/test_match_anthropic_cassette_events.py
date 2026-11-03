from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEventAdapter,
)
from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml
from scripts.match_anthropic_cassette_events import (
    _message_event_units,
    _messages_tool_choice,
    _write_message_event_units,
    main,
    match_messages_events,
)
from tests.cassette_discovery import matcher_pairs

MATCHED_CASSETTES = Path(__file__).parent / "cassettes" / "matched"
REASONING_RAW_CASSETTES = MATCHED_CASSETTES / "reasoning_raw"
MESSAGE_EVENT_ADAPTER = AnthropicMessagesStreamEventAdapter
OPAQUE_STATE_CASSETTES = (
    Path(__file__).parent / "cassettes" / "matched" / "input_image_with_reasoning_encrypted"
)
ENCRYPTED_REASONING_CASSETTES = (
    Path(__file__).parent / "cassettes" / "matched" / "reasoning_encrypted"
)
TOOL_CALL_ENCRYPTED_REASONING_CASSETTES = (
    Path(__file__).parent / "cassettes" / "matched" / "tool_call_with_reasoning_encrypted"
)


def _sse_unit(event: dict[str, Any]) -> list[str]:
    return [
        f"event: {event['type']}\n\n",
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
    events: list[dict[str, Any]] = [
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
        }
    ]

    next_block_index = 0
    if thinking_deltas is not None:
        events.extend(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
                {"type": "ping"},
                *[
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": delta,
                        },
                    }
                    for delta in thinking_deltas
                ],
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "signature_delta",
                        "signature": "signed-thinking",
                    },
                },
                {"type": "content_block_stop", "index": 0},
            ]
        )
        next_block_index = 1

    if text_deltas is not None:
        events.append(
            {
                "type": "content_block_start",
                "index": next_block_index,
                "content_block": {"type": "text", "text": ""},
            }
        )
        events.extend(
            {
                "type": "content_block_delta",
                "index": next_block_index,
                "delta": {"type": "text_delta", "text": delta},
            }
            for delta in text_deltas
        )
        events.append({"type": "content_block_stop", "index": next_block_index})
        next_block_index += 1

    for tool_call in tool_calls or []:
        events.append(
            {
                "type": "content_block_start",
                "index": next_block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": {},
                    "caller": {"type": "direct"},
                },
            }
        )
        events.extend(
            {
                "type": "content_block_delta",
                "index": next_block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": delta,
                },
            }
            for delta in tool_call["argument_deltas"]
        )
        events.append({"type": "content_block_stop", "index": next_block_index})
        next_block_index += 1

    events.extend(
        [
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
            {"type": "message_stop"},
        ]
    )

    sse_entries = [entry for event in events for entry in _sse_unit(event)]
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
    reasoning_added_encrypted_content: str | None = None,
    reasoning_done_encrypted_content: str | None = None,
) -> dict[str, Any]:
    has_reasoning = (
        reasoning_deltas is not None
        or reasoning_summary_deltas is not None
        or reasoning_added_encrypted_content is not None
        or reasoning_done_encrypted_content is not None
    )
    response = {
        "id": "resp_reference",
        "created_at": 123,
        "model": "model-response-reference",
    }
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": response},
    ]
    if has_reasoning:
        added_reasoning_item = {
            "type": "reasoning",
            "content": [],
            "summary": [],
        }
        if reasoning_added_encrypted_content is not None:
            added_reasoning_item["encrypted_content"] = reasoning_added_encrypted_content
        events.append(
            {
                "type": "response.output_item.added",
                "item": added_reasoning_item,
            }
        )
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
        if reasoning_done_encrypted_content is not None:
            reasoning_item["encrypted_content"] = reasoning_done_encrypted_content
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
                    "output_tokens_details": {"reasoning_tokens": 13 if has_reasoning else 0},
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


def test_updates_event_name_when_event_type_changes() -> None:
    original_event = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "answer"},
    }
    cassette = {"response": {"sse": _sse_unit(original_event)}}
    prefix, units = _message_event_units(cassette)
    replacement_event = {"type": "content_block_stop", "index": 0}
    units[0].event = replacement_event

    _write_message_event_units(cassette, prefix, units)

    assert cassette["response"]["sse"] == _sse_unit(replacement_event)


def test_preserves_unchanged_event_formatting() -> None:
    entries = ["event: ping\n\n", 'data: { "type" : "ping" }\n\n', "\n"]
    cassette = {"response": {"sse": entries.copy()}}
    prefix, units = _message_event_units(cassette)

    _write_message_event_units(cassette, prefix, units)

    assert cassette["response"]["sse"] == entries


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
                {
                    "content": [{"text": "canonical question", "type": "text"}],
                    "role": "user",
                },
                {
                    "content": [
                        {"thinking": "canonical reasoning", "type": "thinking"},
                        {"text": "canonical answer", "type": "text"},
                    ],
                    "role": "assistant",
                },
                {
                    "content": [{"text": "canonical follow-up", "type": "text"}],
                    "role": "user",
                },
            ],
            "model": "model-request-reference",
            "output_config": {"effort": "high"},
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
        == "event: content_block_stop"
    )


def test_matches_responses_reasoning_summary_to_messages_thinking() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        reasoning_summary_deltas=["summary", " text"],
        reasoning_added_encrypted_content="initial-signature",
        reasoning_done_encrypted_content="canonical-signature",
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
    thinking_start = next(
        event
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "thinking"
    )
    assert thinking_start["content_block"]["signature"] == "initial-signature"
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


def test_rewrites_messages_stop_reason_from_completed_response() -> None:
    messages_cassette = _messages_cassette(thinking_deltas=None, text_deltas=["answer"])
    responses_cassette = _responses_cassette(reasoning_deltas=None, text_deltas=["answer"])
    for index, entry in enumerate(messages_cassette["response"]["sse"]):
        if not entry.strip().startswith("data:"):
            continue
        event = json_loads(entry.strip().removeprefix("data:").lstrip())
        if event.get("type") == "message_start":
            event["message"]["stop_reason"] = "max_tokens"
            event["message"]["stop_sequence"] = "stale"
        elif event.get("type") == "message_delta":
            event["delta"] = {"stop_reason": "max_tokens", "stop_sequence": "stale"}
        else:
            continue
        messages_cassette["response"]["sse"][index] = f"data: {json_dumps(event)}\n\n"

    match_messages_events(messages_cassette, responses_cassette)

    message_delta = next(
        event for event in _data_events(messages_cassette) if event["type"] == "message_delta"
    )
    message_start = next(
        event["message"]
        for event in _data_events(messages_cassette)
        if event["type"] == "message_start"
    )
    assert message_start["stop_reason"] is None
    assert message_start["stop_sequence"] is None
    assert message_delta["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}


@pytest.mark.parametrize(
    ("incomplete_reason", "stop_reason"),
    [
        ("max_output_tokens", "max_tokens"),
        ("content_filter", "end_turn"),
        ("future_reason", "end_turn"),
    ],
)
def test_maps_incomplete_response_to_messages_stop_reason(
    incomplete_reason: str,
    stop_reason: str,
) -> None:
    messages_cassette = _messages_cassette(thinking_deltas=None, text_deltas=["answer"])
    responses_cassette = _responses_cassette(reasoning_deltas=None, text_deltas=["answer"])
    for index, entry in enumerate(responses_cassette["response"]["sse"]):
        if not entry.strip().startswith("data:"):
            continue
        event = json_loads(entry.strip().removeprefix("data:").lstrip())
        if event.get("type") != "response.completed":
            continue
        event["type"] = "response.incomplete"
        event["response"]["status"] = "incomplete"
        event["response"]["incomplete_details"] = {"reason": incomplete_reason}
        responses_cassette["response"]["sse"][index] = f"data: {json_dumps(event)}\n\n"
        break

    match_messages_events(messages_cassette, responses_cassette)

    message_delta = next(
        event for event in _data_events(messages_cassette) if event["type"] == "message_delta"
    )
    assert message_delta["delta"] == {"stop_reason": stop_reason, "stop_sequence": None}


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
    output = output_path.read_text()
    assert output.index("      - text: canonical prompt") < output.index("        type: text")
    assert read_yaml(str(output_path))["request"]["body"]["messages"] == [
        {
            "content": [{"type": "text", "text": "canonical prompt"}],
            "role": "user",
        }
    ]


def test_matches_user_message_and_part_boundaries() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=["answer"],
        request_input=[
            {
                "content": [
                    {"text": "Part one", "type": "input_text"},
                    {"text": "Part two", "type": "input_text"},
                ],
                "role": "user",
            },
            {
                "content": [{"text": "Follow-up", "type": "input_text"}],
                "role": "user",
            },
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
                {"text": "Part one", "type": "text"},
                {"text": "Part two", "type": "text"},
            ],
            "role": "user",
        },
        {
            "content": [{"text": "Follow-up", "type": "text"}],
            "role": "user",
        },
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
    assert all(event["content_block"]["caller"] == {"type": "direct"} for event in tool_starts)
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
    message_delta = next(event for event in events if event["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert messages_cassette["request"]["body"]["messages"] == [
        {
            "content": [{"text": "Check Paris.", "type": "text"}],
            "role": "user",
        },
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
                    "content": [{"text": "Sunny", "type": "text"}],
                    "tool_use_id": "call_weather",
                    "type": "tool_result",
                },
                {
                    "content": [{"text": "12:00Z", "type": "text"}],
                    "tool_use_id": "call_time",
                    "type": "tool_result",
                },
            ],
            "role": "user",
        },
    ]


def test_matches_function_tools_and_parallel_tool_choice() -> None:
    messages_cassette = _messages_cassette(thinking_deltas=None, text_deltas=["text"])
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=["text"],
    )
    responses_cassette["request"]["body"].update(
        {
            "parallel_tool_calls": False,
            "tools": [
                {
                    "description": "Get the weather.",
                    "name": "Weather",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "type": "object",
                    },
                    "strict": False,
                    "type": "function",
                },
                {
                    "name": "DateTime",
                    "parameters": {"properties": {}, "type": "object"},
                    "strict": True,
                    "type": "function",
                },
            ],
        }
    )

    match_messages_events(messages_cassette, responses_cassette)

    request = messages_cassette["request"]["body"]
    AnthropicMessagesCreateRequest.model_validate(request)
    assert request["tools"] == [
        {
            "description": "Get the weather.",
            "input_schema": {
                "additionalProperties": False,
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "type": "object",
            },
            "name": "Weather",
        },
        {
            "input_schema": {"properties": {}, "type": "object"},
            "name": "DateTime",
            "strict": True,
        },
    ]
    assert request["tool_choice"] == {
        "disable_parallel_tool_use": True,
        "type": "auto",
    }


@pytest.mark.parametrize(
    ("response_choice", "expected_choice"),
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        (
            {"name": "Weather", "type": "function"},
            {"name": "Weather", "type": "tool"},
        ),
    ],
)
def test_maps_function_tool_choice(response_choice: Any, expected_choice: dict[str, Any]) -> None:
    assert _messages_tool_choice({"tool_choice": response_choice}) == expected_choice


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


def test_maps_same_provider_encrypted_content_to_messages_signatures() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=["canonical reasoning"],
        reasoning_added_encrypted_content="initial-signature",
        reasoning_done_encrypted_content="canonical-signature",
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
    thinking_start = next(
        event
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "thinking"
    )
    assert thinking_start["content_block"]["signature"] == "initial-signature"
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


@pytest.mark.parametrize("has_messages_thinking", [False, True])
def test_matches_encrypted_only_reasoning_lifecycle(has_messages_thinking: bool) -> None:
    messages_cassette = _messages_cassette(
        thinking_deltas=["unrelated native reasoning"] if has_messages_thinking else None,
        text_deltas=["template answer"],
    )
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        reasoning_added_encrypted_content="initial-signature",
        reasoning_done_encrypted_content="canonical-signature",
        text_deltas=["canonical answer"],
    )

    match_messages_events(messages_cassette, responses_cassette)

    events = _data_events(messages_cassette)
    for event in events:
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)
    thinking_start = next(
        event
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "thinking"
    )
    assert thinking_start == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "thinking",
            "thinking": "",
            "signature": "initial-signature",
        },
    }
    assert not any(
        event["type"] == "content_block_delta" and event["delta"]["type"] == "thinking_delta"
        for event in events
    )
    assert [
        event["delta"]["signature"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "signature_delta"
    ] == ["canonical-signature"]
    text_start = next(
        event
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "text"
    )
    assert text_start["index"] == 1


def test_maps_encrypted_reasoning_history_without_raw_content() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=["canonical reasoning"],
        reasoning_done_encrypted_content="canonical-signature",
        text_deltas=["canonical answer"],
        request_input=[
            {"content": "question", "role": "user"},
            {
                "encrypted_content": "canonical-history-signature",
                "summary": [{"text": "prior summary", "type": "summary_text"}],
                "type": "reasoning",
            },
            {"content": "prior answer", "role": "assistant"},
            {"content": "follow-up", "role": "user"},
        ],
    )
    messages_cassette = _messages_cassette(
        thinking_deltas=["template reasoning"],
        text_deltas=["template answer"],
    )

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette["request"]["body"]["messages"][1]["content"][0] == {
        "signature": "canonical-history-signature",
        "thinking": "prior summary",
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


def test_uses_standard_service_tier_for_translated_messages() -> None:
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

    match_messages_events(messages_cassette, responses_cassette)

    message = next(
        event["message"]
        for event in _data_events(messages_cassette)
        if event["type"] == "message_start"
    )
    assert message["usage"]["service_tier"] == "standard"


@pytest.mark.parametrize(
    ("messages_path", "responses_path"),
    [(messages, responses) for _, messages, responses in matcher_pairs("tool_call", "messages")],
)
def test_matches_real_tool_call_cassettes(messages_path: Path, responses_path: Path) -> None:
    """Ensures discovered tool turns retain call identities, arguments, and result history."""
    messages_cassette = read_yaml(messages_path)
    responses_cassette = read_yaml(responses_path)

    match_messages_events(messages_cassette, responses_cassette)

    messages_request = messages_cassette["request"]["body"]
    AnthropicMessagesCreateRequest.model_validate(messages_request)
    assert messages_request["tool_choice"] == {
        "disable_parallel_tool_use": True,
        "type": "auto",
    }
    assert messages_request["tools"] == [
        {
            "description": tool["description"],
            "input_schema": tool["parameters"],
            "name": tool["name"],
        }
        for tool in responses_cassette["request"]["body"]["tools"]
    ]

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

    message_delta = next(event for event in message_events if event["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == ("tool_use" if response_calls else "end_turn")

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
    ] == [
        (
            item["call_id"],
            [{"text": part["text"], "type": "text"} for part in item["output"]],
        )
        for item in request_function_outputs
    ]


@pytest.mark.parametrize(
    ("messages_path", "responses_path"),
    [
        (messages, responses)
        for scenario in ("prompt_cache_implicit", "prompt_cache_explicit")
        for _, messages, responses in matcher_pairs(scenario, "messages")
    ],
)
def test_matches_prompt_cache_cassettes(messages_path: Path, responses_path: Path) -> None:
    """Ensures discovered cache turns preserve markers, cache policy, and semantic usage."""
    messages_cassette = read_yaml(messages_path)
    responses_cassette = read_yaml(responses_path)
    original_messages = deepcopy(messages_cassette)
    original_responses = deepcopy(responses_cassette)
    messages_cassette["request"]["body"].update(
        messages=[{"role": "user", "content": "stale prompt"}],
        system="stale system",
        cache_control={"type": "ephemeral", "ttl": "5m"},
    )

    match_messages_events(messages_cassette, responses_cassette)

    request = messages_cassette["request"]["body"]
    reference_request = responses_cassette["request"]["body"]
    AnthropicMessagesCreateRequest.model_validate(request)
    assert request["model"] == reference_request["model"]
    assert request["max_tokens"] == reference_request["max_output_tokens"]
    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": part["text"]} for part in item["content"]],
        }
        for item in reference_request["input"]
        if item["role"] == "user"
    ]
    if reference_request["prompt_cache_options"]["mode"] == "implicit":
        assert request["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "system" not in request
    else:
        assert "cache_control" not in request
        assert request["system"] == [
            {
                "type": "text",
                "text": reference_request["input"][0]["content"][0]["text"],
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
            {"type": "text", "text": reference_request["input"][1]["content"][0]["text"]},
        ]

    events = _data_events(messages_cassette)
    response_events = _data_events(responses_cassette)
    completed = next(
        event["response"] for event in response_events if event["type"] == "response.completed"
    )
    for event in events:
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)
    message = next(event["message"] for event in events if event["type"] == "message_start")
    terminal = next(event for event in events if event["type"] == "message_delta")
    assert message["id"] == completed["id"]
    assert message["model"] == completed["model"]
    assert message["usage"]["output_tokens"] == 0
    assert terminal["usage"]["output_tokens"] == completed["usage"]["output_tokens"]
    for usage in (message["usage"], terminal["usage"]):
        assert (
            usage["cache_read_input_tokens"]
            == completed["usage"]["input_tokens_details"]["cached_tokens"]
        )
        assert (
            usage["cache_creation_input_tokens"]
            == completed["usage"]["input_tokens_details"]["cache_write_tokens"]
        )
        assert (
            usage["input_tokens"]
            + usage["cache_read_input_tokens"]
            + usage["cache_creation_input_tokens"]
            == completed["usage"]["input_tokens"]
        )
    assert message["usage"]["cache_read_input_tokens"] > 0
    assert message["usage"]["cache_creation"] == {
        "ephemeral_5m_input_tokens": 0,
        "ephemeral_1h_input_tokens": 0,
    }
    assert [
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
    ] == [
        event["delta"]
        for event in response_events
        if event["type"] == "response.output_text.delta"
    ]
    assert responses_cassette == original_responses
    assert {
        key: value for key, value in messages_cassette["request"].items() if key != "body"
    } == {key: value for key, value in original_messages["request"].items() if key != "body"}
    assert {
        key: value for key, value in messages_cassette["response"].items() if key != "sse"
    } == {key: value for key, value in original_messages["response"].items() if key != "sse"}
    matched = deepcopy(messages_cassette)
    match_messages_events(messages_cassette, responses_cassette)
    assert messages_cassette == matched


@pytest.mark.parametrize(
    ("options", "marked", "ttl"),
    [
        ({"mode": "implicit"}, False, "5m"),
        ({"mode": "implicit", "ttl": "30m"}, False, "1h"),
        ({"mode": "explicit", "ttl": "30m"}, True, "1h"),
        (None, True, "1h"),
    ],
)
def test_matches_cache_creation_lifetime_breakdown(
    options: dict[str, str] | None, marked: bool, ttl: str
) -> None:
    messages_cassette = _messages_cassette(thinking_deltas=None, text_deltas=["old"])
    part = {"type": "input_text", "text": "cacheable prefix"}
    if marked:
        part["prompt_cache_breakpoint"] = {}
    responses_cassette = _responses_cassette(
        reasoning_deltas=None,
        text_deltas=["new"],
        request_input=[{"role": "user", "content": [part]}],
    )
    if options is not None:
        responses_cassette["request"]["body"]["prompt_cache_options"] = options
    prefix, units = _message_event_units(messages_cassette)
    for unit in units:
        event = unit.event
        if event["type"] in {"message_start", "message_delta"}:
            usage = (
                event["message"]["usage"] if event["type"] == "message_start" else event["usage"]
            )
            usage["cache_creation"] = {
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 200,
            }
    _write_message_event_units(messages_cassette, prefix, units)

    match_messages_events(messages_cassette, responses_cassette)

    for event in _data_events(messages_cassette):
        if event["type"] in {"message_start", "message_delta"}:
            usage = (
                event["message"]["usage"] if event["type"] == "message_start" else event["usage"]
            )
            assert usage["cache_creation_input_tokens"] == 7
            assert usage["cache_creation"] == {
                "ephemeral_5m_input_tokens": 7 if ttl == "5m" else 0,
                "ephemeral_1h_input_tokens": 7 if ttl == "1h" else 0,
            }


def test_matches_real_two_turn_request_body() -> None:
    messages_cassette = read_yaml(str(REASONING_RAW_CASSETTES / "vllm-messages-stream-01.yaml"))
    responses_cassette = read_yaml(str(REASONING_RAW_CASSETTES / "vllm-responses-stream-01.yaml"))
    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette["request"]["body"] == {
        "max_tokens": 2048,
        "messages": [
            {
                "content": [{"text": "What is 2+2?", "type": "text"}],
                "role": "user",
            },
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
            {
                "content": [{"text": "What about 2*2?", "type": "text"}],
                "role": "user",
            },
        ],
        "model": "deepseek-v4-flash",
        "output_config": {"effort": "high"},
        "stream": True,
        "thinking": {"display": "summarized", "type": "adaptive"},
    }
    message_start = next(
        event["message"]
        for event in _data_events(messages_cassette)
        if event["type"] == "message_start"
    )
    assert message_start["usage"]["output_tokens"] == 0


@pytest.mark.parametrize(
    ("messages_path", "responses_path"),
    [
        (messages, responses)
        for _, messages, responses in matcher_pairs(
            "input_image_with_reasoning_encrypted", "messages"
        )
    ],
)
def test_same_provider_opaque_state_cassettes_are_matched(
    messages_path: Path, responses_path: Path
) -> None:
    """Ensures every discovered encrypted turn retains the same-issuer opaque-state mapping."""
    messages_cassette = read_yaml(messages_path)
    responses_cassette = read_yaml(responses_path)
    original_messages_cassette = deepcopy(messages_cassette)

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette == original_messages_cassette


@pytest.mark.parametrize(
    ("messages_path", "responses_path"),
    [
        (messages, responses)
        for _, messages, responses in matcher_pairs("reasoning_encrypted", "messages")
    ],
)
def test_encrypted_reasoning_cassettes_are_matched(
    messages_path: Path, responses_path: Path
) -> None:
    """Ensures every discovered encrypted turn retains the same-issuer opaque-state mapping."""
    messages_cassette = read_yaml(messages_path)
    responses_cassette = read_yaml(responses_path)
    original_messages_cassette = deepcopy(messages_cassette)

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette == original_messages_cassette
    AnthropicMessagesCreateRequest.model_validate(messages_cassette["request"]["body"])
    for event in _data_events(messages_cassette):
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)


@pytest.mark.parametrize(
    ("messages_path", "responses_path"),
    [
        (messages, responses)
        for _, messages, responses in matcher_pairs(
            "tool_call_with_reasoning_encrypted", "messages"
        )
    ],
)
def test_tool_call_encrypted_reasoning_cassettes_are_matched(
    messages_path: Path, responses_path: Path
) -> None:
    """Ensures every discovered encrypted turn retains the same-issuer opaque-state mapping."""
    messages_cassette = read_yaml(messages_path)
    responses_cassette = read_yaml(responses_path)
    original_messages_cassette = deepcopy(messages_cassette)

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette == original_messages_cassette
    AnthropicMessagesCreateRequest.model_validate(messages_cassette["request"]["body"])
    for event in _data_events(messages_cassette):
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)


@pytest.mark.parametrize(
    "cassette_dir",
    [
        OPAQUE_STATE_CASSETTES,
        ENCRYPTED_REASONING_CASSETTES,
        TOOL_CALL_ENCRYPTED_REASONING_CASSETTES,
    ],
    ids=["input-image", "reasoning", "tool-call"],
)
def test_opaque_reasoning_history_uses_prior_done_snapshot(cassette_dir: Path) -> None:
    responses_turn_0 = read_yaml(str(cassette_dir / "openai-responses-stream-00.yaml"))
    responses_turn_1 = read_yaml(str(cassette_dir / "openai-responses-stream-01.yaml"))
    messages_turn_1 = read_yaml(str(cassette_dir / "anthropic-messages-stream-01.yaml"))

    turn_0_events = _data_events(responses_turn_0)
    done_signature = next(
        event["item"]["encrypted_content"]
        for event in turn_0_events
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "reasoning"
    )
    terminal_signature = next(
        item["encrypted_content"]
        for event in turn_0_events
        if event["type"] in {"response.completed", "response.incomplete"}
        for item in event["response"]["output"]
        if item["type"] == "reasoning"
    )
    responses_history_signature = next(
        item["encrypted_content"]
        for item in responses_turn_1["request"]["body"]["input"]
        if isinstance(item, dict) and item.get("type") == "reasoning"
    )
    messages_history_signature = next(
        block["signature"]
        for message in messages_turn_1["request"]["body"]["messages"]
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "thinking"
    )

    assert responses_history_signature == done_signature
    assert messages_history_signature == done_signature
    assert terminal_signature == done_signature


def test_request_only_matching_preserves_recorded_error(tmp_path: Path) -> None:
    cassette_dir = Path(__file__).parent / "cassettes" / "errors" / "context_length_exceeded"
    expected = read_yaml(str(cassette_dir / "anthropic-messages-stream-00.yaml"))
    responses_path = cassette_dir / "openai-responses-stream-00.yaml"
    original_responses = responses_path.read_bytes()
    unmatched = deepcopy(expected)
    unmatched["request"]["body"]["model"] = "unmatched-model"
    unmatched["request"]["body"]["messages"] = [{"role": "user", "content": "Unmatched input"}]
    unmatched["request"]["body"]["max_tokens"] = 99
    input_path = tmp_path / "input.yaml"
    output_path = tmp_path / "matched.yaml"
    dump_yaml(unmatched, str(input_path))

    assert (
        main(
            [str(input_path), str(responses_path), "--request-only", "--output", str(output_path)]
        )
        == 0
    )

    assert read_yaml(str(output_path)) == expected
    assert read_yaml(str(input_path)) == unmatched
    assert responses_path.read_bytes() == original_responses
