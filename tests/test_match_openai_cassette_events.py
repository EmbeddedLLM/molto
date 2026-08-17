from copy import deepcopy
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionChunk

from molto.utils.io import json_dumps, json_loads, read_yaml
from scripts.match_openai_cassette_events import main, match_chat_completions_events

CASSETTES = Path(__file__).parent / "cassettes" / "matched"
CHAT_CASSETTE = CASSETTES / "simple_text" / "openai_chat_completions-stream-00.yaml"
RESPONSES_CASSETTE = CASSETTES / "simple_text" / "openai_responses-stream-00.yaml"
REASONING_CASSETTES = [
    (
        CASSETTES / "reasoning_raw" / f"openai_chat_completions-stream-{index:02d}.yaml",
        CASSETTES / "reasoning_raw" / f"openai_responses-stream-{index:02d}.yaml",
    )
    for index in range(2)
]


def _data_events(cassette: dict) -> list[dict]:
    return [
        json_loads(entry.strip().removeprefix("data:").lstrip())
        for entry in cassette["response"]["sse"]
        if entry.strip().startswith("data:") and entry.strip() != "data: [DONE]"
    ]


def _chat_content(event: dict) -> str | None:
    choices = event.get("choices")
    if not choices:
        return None
    delta = choices[0]["delta"]
    if "role" in delta:
        return None
    return delta.get("content")


def test_matches_simple_text_stream_cassettes(tmp_path: Path) -> None:
    output_path = tmp_path / "matched.yaml"

    assert main([str(CHAT_CASSETTE), str(RESPONSES_CASSETTE), "--output", str(output_path)]) == 0

    chat_events = _data_events(read_yaml(str(output_path)))
    for event in chat_events:
        ChatCompletionChunk.model_validate(event)
    response_events = _data_events(read_yaml(str(RESPONSES_CASSETTE)))
    completed = next(
        event["response"] for event in response_events if event["type"] == "response.completed"
    )
    response_id = completed["id"]
    created_at = completed["created_at"]
    expected_content = [
        event["delta"]
        for event in response_events
        if event["type"] == "response.output_text.delta"
    ]

    assert {event["id"] for event in chat_events} == {response_id}
    assert {event["created"] for event in chat_events} == {created_at}
    assert {event["model"] for event in chat_events} == {completed["model"]}
    assert {event["service_tier"] for event in chat_events} == {"default"}
    assert [
        event["choices"][0]["delta"]["content"]
        for event in chat_events
        if event["choices"]
        and "content" in event["choices"][0]["delta"]
        and "role" not in event["choices"][0]["delta"]
    ] == expected_content

    usage = next(event["usage"] for event in chat_events if event["usage"] is not None)
    assert usage["prompt_tokens"] == completed["usage"]["input_tokens"]
    assert usage["completion_tokens"] == completed["usage"]["output_tokens"]
    assert usage["total_tokens"] == completed["usage"]["total_tokens"]
    assert (
        usage["prompt_tokens_details"]["cached_tokens"]
        == completed["usage"]["input_tokens_details"]["cached_tokens"]
    )
    assert (
        usage["prompt_tokens_details"]["cache_write_tokens"]
        == completed["usage"]["input_tokens_details"]["cache_write_tokens"]
    )
    assert (
        usage["completion_tokens_details"]["reasoning_tokens"]
        == completed["usage"]["output_tokens_details"]["reasoning_tokens"]
    )


def test_event_matching_preserves_chat_user_message_and_part_boundaries() -> None:
    chat_cassette = read_yaml(str(CHAT_CASSETTE))
    responses_cassette = read_yaml(str(RESPONSES_CASSETTE))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Follow-up"}],
        },
    ]
    chat_cassette["request"]["body"]["messages"] = deepcopy(messages)

    match_chat_completions_events(chat_cassette, responses_cassette)

    assert chat_cassette["request"]["body"]["messages"] == messages


@pytest.mark.parametrize(("chat_path", "responses_path"), REASONING_CASSETTES)
def test_matches_reasoning_text_stream_cassettes(chat_path: Path, responses_path: Path) -> None:
    chat_cassette = read_yaml(str(chat_path))
    responses_cassette = read_yaml(str(responses_path))

    match_chat_completions_events(chat_cassette, responses_cassette)

    assert chat_cassette["request"]["body"]["reasoning_effort"] == "high"
    chat_events = _data_events(chat_cassette)
    response_events = _data_events(responses_cassette)
    for event in chat_events:
        ChatCompletionChunk.model_validate(event)

    for chat_field, response_prefix in (
        ("reasoning_content", "response.reasoning_text"),
        ("content", "response.output_text"),
    ):
        chat_deltas = [
            value
            for event in chat_events
            if event.get("choices")
            and "role" not in event["choices"][0]["delta"]
            and event["choices"][0]["finish_reason"] is None
            and isinstance((value := event["choices"][0]["delta"].get(chat_field)), str)
        ]
        response_deltas = [
            event["delta"]
            for event in response_events
            if event["type"] == f"{response_prefix}.delta"
        ]
        done_text = next(
            event["text"]
            for event in response_events
            if event["type"] == f"{response_prefix}.done"
        )

        assert chat_deltas == response_deltas
        assert "".join(chat_deltas) == done_text

    assert (
        sum(
            bool(event["choices"] and event["choices"][0]["delta"].get("role"))
            for event in chat_events
        )
        == 1
    )
    assert (
        sum(
            bool(event["choices"] and event["choices"][0]["finish_reason"] == "stop")
            for event in chat_events
        )
        == 1
    )
    assert sum(event["usage"] is not None for event in chat_events) == 1
    assert sum(entry.strip() == "data: [DONE]" for entry in chat_cassette["response"]["sse"]) == 1


@pytest.mark.parametrize("adjustment", ["extra", "missing"])
def test_rebuilds_unequal_chat_content_delta_stream(adjustment: str) -> None:
    chat_cassette = read_yaml(str(CHAT_CASSETTE))
    responses_cassette = read_yaml(str(RESPONSES_CASSETTE))
    sse_entries = chat_cassette["response"]["sse"]
    content_indices = [
        index
        for index, entry in enumerate(sse_entries)
        if entry.strip().startswith("data:")
        and entry.strip() != "data: [DONE]"
        and _chat_content(json_loads(entry.strip().removeprefix("data:").lstrip())) is not None
    ]
    first_content_index = content_indices[0]
    second_content_index = content_indices[1]
    first_content_unit = sse_entries[first_content_index:second_content_index]
    if adjustment == "extra":
        sse_entries[first_content_index:first_content_index] = first_content_unit
    else:
        del sse_entries[first_content_index:second_content_index]

    match_chat_completions_events(chat_cassette, responses_cassette)

    chat_content = [
        content
        for event in _data_events(chat_cassette)
        if (content := _chat_content(event)) is not None
    ]
    response_content = [
        event["delta"]
        for event in _data_events(responses_cassette)
        if event["type"] == "response.output_text.delta"
    ]
    assert chat_content == response_content

    chat_events = _data_events(chat_cassette)
    for event in chat_events:
        ChatCompletionChunk.model_validate(event)
    assert len(chat_events) == len(response_content) + 3
    assert (
        sum(
            bool(event["choices"] and event["choices"][0]["delta"].get("role"))
            for event in chat_events
        )
        == 1
    )
    assert (
        sum(
            bool(event["choices"] and event["choices"][0]["finish_reason"] == "stop")
            for event in chat_events
        )
        == 1
    )
    assert sum(event["usage"] is not None for event in chat_events) == 1
    assert sum(entry.strip() == "data: [DONE]" for entry in sse_entries) == 1
    for index, entry in enumerate(sse_entries):
        if entry.strip().startswith("data:"):
            expected_suffix = "\n" if entry.strip() == "data: [DONE]" else "\n\n"
            assert entry.endswith(expected_suffix)
            assert sse_entries[index + 1] == "\n"


def test_rewrites_chat_finish_reason_from_completed_response() -> None:
    chat_cassette = read_yaml(str(CHAT_CASSETTE))
    responses_cassette = read_yaml(str(RESPONSES_CASSETTE))
    for index, entry in enumerate(chat_cassette["response"]["sse"]):
        if not entry.strip().startswith("data:") or entry.strip() == "data: [DONE]":
            continue
        event = json_loads(entry.strip().removeprefix("data:").lstrip())
        if event.get("choices") and event["choices"][0]["finish_reason"] is not None:
            event["choices"][0]["finish_reason"] = "length"
            chat_cassette["response"]["sse"][index] = f"data: {json_dumps(event)}\n\n"
            break

    match_chat_completions_events(chat_cassette, responses_cassette)

    finish_reasons = [
        event["choices"][0]["finish_reason"]
        for event in _data_events(chat_cassette)
        if event.get("choices") and event["choices"][0]["finish_reason"] is not None
    ]
    assert finish_reasons == ["stop"]


@pytest.mark.parametrize(
    ("incomplete_reason", "finish_reason"),
    [
        ("content_filter", "content_filter"),
        ("max_output_tokens", "length"),
        ("future_reason", "stop"),
    ],
)
def test_maps_incomplete_response_to_chat_finish_reason(
    incomplete_reason: str,
    finish_reason: str,
) -> None:
    chat_cassette = read_yaml(str(CHAT_CASSETTE))
    responses_cassette = read_yaml(str(RESPONSES_CASSETTE))
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

    match_chat_completions_events(chat_cassette, responses_cassette)

    finish_reasons = [
        event["choices"][0]["finish_reason"]
        for event in _data_events(chat_cassette)
        if event.get("choices") and event["choices"][0]["finish_reason"] is not None
    ]
    assert finish_reasons == [finish_reason]


def test_uses_default_service_tier_for_translated_chat_completions() -> None:
    chat_cassette = read_yaml(str(CHAT_CASSETTE))
    responses_cassette = read_yaml(str(RESPONSES_CASSETTE))
    for index, entry in enumerate(responses_cassette["response"]["sse"]):
        if not entry.strip().startswith("data:"):
            continue
        event = json_loads(entry.strip().removeprefix("data:").lstrip())
        if event.get("type") not in {"response.created", "response.completed"}:
            continue
        event["response"]["service_tier"] = "priority"
        responses_cassette["response"]["sse"][index] = f"data: {json_dumps(event)}\n\n"

    match_chat_completions_events(chat_cassette, responses_cassette)

    assert {event["service_tier"] for event in _data_events(chat_cassette)} == {"default"}
