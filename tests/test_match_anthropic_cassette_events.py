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
    _write_message_event_units,
    main,
    match_messages_events,
)

MESSAGE_EVENT_ADAPTER = AnthropicMessagesStreamEventAdapter
MATCHED_CASSETTES = Path(__file__).parent / "cassettes" / "matched" / "reasoning_raw"
OPAQUE_STATE_CASSETTES = (
    Path(__file__).parent / "cassettes" / "matched" / "input_image_with_reasoning"
)
ENCRYPTED_REASONING_CASSETTES = (
    Path(__file__).parent / "cassettes" / "matched" / "reasoning_encrypted"
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
    text_deltas: list[str],
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

    text_index = 0
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
        text_index = 1

    events.append(
        {
            "type": "content_block_start",
            "index": text_index,
            "content_block": {"type": "text", "text": ""},
        }
    )
    events.extend(
        {
            "type": "content_block_delta",
            "index": text_index,
            "delta": {"type": "text_delta", "text": delta},
        }
        for delta in text_deltas
    )
    events.extend(
        [
            {"type": "content_block_stop", "index": text_index},
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
    text_deltas: list[str],
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
                            "reasoning_tokens": 13
                            if reasoning_deltas is not None or reasoning_summary_deltas is not None
                            else 0
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


def test_maps_encrypted_reasoning_history_without_raw_content() -> None:
    responses_cassette = _responses_cassette(
        reasoning_deltas=["canonical reasoning"],
        reasoning_encrypted_content="canonical-signature",
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


def test_matches_real_two_turn_request_body() -> None:
    messages_cassette = read_yaml(str(MATCHED_CASSETTES / "vllm-messages-stream-01.yaml"))
    responses_cassette = read_yaml(str(MATCHED_CASSETTES / "vllm-responses-stream-01.yaml"))
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


@pytest.mark.parametrize("turn", range(2))
def test_same_provider_opaque_state_cassettes_are_matched(turn: int) -> None:
    messages_cassette = read_yaml(
        str(OPAQUE_STATE_CASSETTES / f"anthropic-messages-stream-{turn:02d}.yaml")
    )
    responses_cassette = read_yaml(
        str(OPAQUE_STATE_CASSETTES / f"openai-responses-stream-{turn:02d}.yaml")
    )
    original_messages_cassette = deepcopy(messages_cassette)

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette == original_messages_cassette


@pytest.mark.parametrize("turn", range(2))
def test_encrypted_reasoning_cassettes_are_matched(turn: int) -> None:
    messages_cassette = read_yaml(
        str(ENCRYPTED_REASONING_CASSETTES / f"anthropic-messages-stream-{turn:02d}.yaml")
    )
    responses_cassette = read_yaml(
        str(ENCRYPTED_REASONING_CASSETTES / f"openai-responses-stream-{turn:02d}.yaml")
    )
    original_messages_cassette = deepcopy(messages_cassette)

    match_messages_events(messages_cassette, responses_cassette)

    assert messages_cassette == original_messages_cassette
    AnthropicMessagesCreateRequest.model_validate(messages_cassette["request"]["body"])
    for event in _data_events(messages_cassette):
        if event["type"] != "ping":
            MESSAGE_EVENT_ADAPTER.validate_python(event)
