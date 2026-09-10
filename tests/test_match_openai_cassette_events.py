from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk

from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml
from scripts.match_openai_cassette_events import main, match_chat_completions_events

CASSETTES = Path(__file__).parent / "cassettes" / "matched"
CHAT_CASSETTE = CASSETTES / "simple_text" / "openai-chat_completions-stream-00.yaml"
RESPONSES_CASSETTE = CASSETTES / "simple_text" / "openai-responses-stream-00.yaml"
REASONING_CASSETTES = [
    (
        CASSETTES / "reasoning_raw" / f"vllm-chat_completions-stream-{index:02d}.yaml",
        CASSETTES / "reasoning_raw" / f"vllm-responses-stream-{index:02d}.yaml",
    )
    for index in range(2)
]
TOOL_CALL_CASSETTES = [
    (
        index,
        CASSETTES / "tool_call" / f"openai-chat_completions-stream-{index:02d}.yaml",
        CASSETTES / "tool_call" / f"openai-responses-stream-{index:02d}.yaml",
    )
    for index in range(3)
]
TOOL_CALL_CHOICE_CASSETTES = (
    CASSETTES / "tool_call_choice" / "openai-chat_completions-stream-00.yaml",
    CASSETTES / "tool_call_choice" / "openai-responses-stream-00.yaml",
)
TOOL_CALL_CHOICE_HISTORY_CASSETTES = (
    CASSETTES / "tool_call_choice" / "openai-chat_completions-stream-01.yaml",
    CASSETTES / "tool_call_choice" / "openai-responses-stream-01.yaml",
)


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


def _response_tool_call_chunks(events: list[dict]) -> list[dict[str, Any]]:
    calls_by_item_id: dict[str, int] = {}
    chunks: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_index = len(calls_by_item_id)
                calls_by_item_id[item["id"]] = call_index
                chunks.append(
                    {
                        "arguments": item["arguments"],
                        "id": item["call_id"],
                        "index": call_index,
                        "name": item["name"],
                        "type": "function",
                    }
                )
        elif event.get("type") == "response.function_call_arguments.delta":
            chunks.append(
                {
                    "arguments": event["delta"],
                    "index": calls_by_item_id[event["item_id"]],
                }
            )
    return chunks


def _chat_tool_call_chunks(events: list[dict]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for event in events:
        if not event.get("choices"):
            continue
        for tool_call in event["choices"][0]["delta"].get("tool_calls") or []:
            function = tool_call["function"]
            chunk = {
                "arguments": function["arguments"],
                "index": tool_call["index"],
            }
            for source, target in (("id", "id"), ("type", "type")):
                if source in tool_call:
                    chunk[target] = tool_call[source]
            if "name" in function:
                chunk["name"] = function["name"]
            chunks.append(chunk)
    return chunks


def _response_message_text(item: dict[str, Any]) -> str:
    content = item["content"]
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content)


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
    user_messages = [
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
    responses_cassette["request"]["body"]["input"] = [
        {
            "content": [
                {"type": "input_text", "text": "Part one"},
                {"type": "input_text", "text": "Part two"},
            ],
            "role": "user",
        },
        {
            "arguments": "{}",
            "call_id": "call_test",
            "name": "TestTool",
            "type": "function_call",
        },
        {
            "call_id": "call_test",
            "output": "done",
            "type": "function_call_output",
        },
        {
            "content": [{"type": "input_text", "text": "Follow-up"}],
            "role": "user",
        },
    ]

    match_chat_completions_events(chat_cassette, responses_cassette)

    assert [
        message
        for message in chat_cassette["request"]["body"]["messages"]
        if message["role"] == "user"
    ] == user_messages


def test_cli_sorts_yaml_mapping_keys(tmp_path: Path) -> None:
    chat_cassette = read_yaml(str(CHAT_CASSETTE))
    content_part = {"type": "text", "text": "ordering sentinel"}
    chat_cassette["request"]["body"]["messages"][0]["content"] = [content_part]
    chat_path = tmp_path / "chat.yaml"
    output_path = tmp_path / "matched.yaml"
    dump_yaml(chat_cassette, str(chat_path), sort_keys=False)

    assert main([str(chat_path), str(RESPONSES_CASSETTE), "--output", str(output_path)]) == 0

    output = output_path.read_text()
    assert output.index("      - text: ordering sentinel") < output.index("        type: text")


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


@pytest.mark.parametrize(("turn", "chat_path", "responses_path"), TOOL_CALL_CASSETTES)
def test_matches_tool_call_stream_cassettes(
    turn: int, chat_path: Path, responses_path: Path
) -> None:
    chat_cassette = read_yaml(str(chat_path))
    responses_cassette = read_yaml(str(responses_path))
    original_chat_request = deepcopy(chat_cassette["request"]["body"])

    match_chat_completions_events(chat_cassette, responses_cassette)

    chat_events = _data_events(chat_cassette)
    response_events = _data_events(responses_cassette)
    for event in chat_events:
        ChatCompletionChunk.model_validate(event)

    assert [content for event in chat_events if (content := _chat_content(event)) is not None] == [
        event["delta"]
        for event in response_events
        if event["type"] == "response.output_text.delta"
    ]
    expected_tool_chunks = _response_tool_call_chunks(response_events)
    assert _chat_tool_call_chunks(chat_events) == expected_tool_chunks
    finish_reasons = [
        event["choices"][0]["finish_reason"]
        for event in chat_events
        if event.get("choices") and event["choices"][0]["finish_reason"] is not None
    ]
    assert finish_reasons == (["tool_calls"] if expected_tool_chunks else ["stop"])

    chat_request = chat_cassette["request"]["body"]
    assert {key: value for key, value in chat_request.items() if key != "messages"} == {
        key: value for key, value in original_chat_request.items() if key != "messages"
    }
    response_input = responses_cassette["request"]["body"]["input"]
    assert isinstance(response_input, list)
    expected_history_calls = [
        {
            "function": {"arguments": item["arguments"], "name": item["name"]},
            "id": item["call_id"],
            "type": "function",
        }
        for item in response_input
        if item.get("type") == "function_call"
    ]
    actual_history_calls = [
        tool_call
        for message in chat_request["messages"]
        for tool_call in message.get("tool_calls", [])
    ]
    assert actual_history_calls == expected_history_calls
    assert [
        {
            "content": [{"text": part["text"], "type": "text"} for part in item["output"]],
            "role": "tool",
            "tool_call_id": item["call_id"],
        }
        for item in response_input
        if item.get("type") == "function_call_output"
    ] == [message for message in chat_request["messages"] if message["role"] == "tool"]
    assert [
        _response_message_text(item)
        for item in response_input
        if item.get("type", "message") == "message" and item.get("role") == "assistant"
    ] == [
        message["content"]
        for message in chat_request["messages"]
        if message["role"] == "assistant"
    ]
    assert len(chat_request["messages"]) == 1 + 2 * turn
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


@pytest.mark.parametrize("adjustment", ["extra", "missing"])
def test_rebuilds_unequal_chat_tool_argument_delta_stream(adjustment: str) -> None:
    _, chat_path, responses_path = TOOL_CALL_CASSETTES[0]
    chat_cassette = read_yaml(str(chat_path))
    responses_cassette = read_yaml(str(responses_path))
    sse_entries = chat_cassette["response"]["sse"]
    tool_indices = []
    for index, entry in enumerate(sse_entries):
        if not entry.strip().startswith("data:") or entry.strip() == "data: [DONE]":
            continue
        event = json_loads(entry.strip().removeprefix("data:").lstrip())
        if event.get("choices") and event["choices"][0]["delta"].get("tool_calls"):
            tool_indices.append(index)

    first_argument_index = tool_indices[1]
    second_argument_index = tool_indices[2]
    first_argument_unit = sse_entries[first_argument_index:second_argument_index]
    if adjustment == "extra":
        sse_entries[first_argument_index:first_argument_index] = first_argument_unit
    else:
        del sse_entries[first_argument_index:second_argument_index]

    match_chat_completions_events(chat_cassette, responses_cassette)

    chat_events = _data_events(chat_cassette)
    response_events = _data_events(responses_cassette)
    assert _chat_tool_call_chunks(chat_events) == _response_tool_call_chunks(response_events)
    for event in chat_events:
        ChatCompletionChunk.model_validate(event)
    assert sum(entry.strip() == "data: [DONE]" for entry in sse_entries) == 1


def test_matches_tool_only_chat_role_and_history() -> None:
    chat_cassette = read_yaml(str(TOOL_CALL_CHOICE_CASSETTES[0]))
    responses_cassette = read_yaml(str(TOOL_CALL_CHOICE_CASSETTES[1]))
    history_chat_cassette = read_yaml(str(TOOL_CALL_CHOICE_HISTORY_CASSETTES[0]))
    history_responses_cassette = read_yaml(str(TOOL_CALL_CHOICE_HISTORY_CASSETTES[1]))

    match_chat_completions_events(chat_cassette, responses_cassette)
    match_chat_completions_events(history_chat_cassette, history_responses_cassette)

    assistant_message = next(
        message
        for message in history_chat_cassette["request"]["body"]["messages"]
        if message["role"] == "assistant"
    )
    assert "content" not in assistant_message

    chat_events = _data_events(chat_cassette)
    assert chat_events[0]["choices"][0]["delta"] == {
        "content": None,
        "refusal": None,
        "role": "assistant",
    }
    assert chat_events[1]["choices"][0]["delta"]["tool_calls"]

    first_match = deepcopy(chat_cassette)
    match_chat_completions_events(chat_cassette, responses_cassette)
    assert chat_cassette == first_match


def test_rejects_tool_call_presence_mismatch() -> None:
    _, chat_path, _ = TOOL_CALL_CASSETTES[0]
    _, _, responses_path = TOOL_CALL_CASSETTES[2]
    chat_cassette = read_yaml(str(chat_path))
    responses_cassette = read_yaml(str(responses_path))

    with pytest.raises(ValueError, match="disagree on the presence of a tool call"):
        match_chat_completions_events(chat_cassette, responses_cassette)


def test_rejects_incomplete_response_tool_call_lifecycle() -> None:
    _, chat_path, responses_path = TOOL_CALL_CASSETTES[0]
    chat_cassette = read_yaml(str(chat_path))
    responses_cassette = read_yaml(str(responses_path))
    responses_cassette["response"]["sse"] = [
        entry
        for entry in responses_cassette["response"]["sse"]
        if not (
            entry.strip().startswith("data:")
            and entry.strip() != "data: [DONE]"
            and json_loads(entry.strip().removeprefix("data:").lstrip()).get("type")
            == "response.function_call_arguments.done"
        )
    ]

    with pytest.raises(
        ValueError,
        match="Expected exactly one 'response.function_call_arguments.done' event",
    ):
        match_chat_completions_events(chat_cassette, responses_cassette)


def test_matches_multiple_ordered_tool_calls() -> None:
    _, chat_path, responses_path = TOOL_CALL_CASSETTES[0]
    chat_cassette = read_yaml(str(chat_path))
    responses_cassette = read_yaml(str(responses_path))
    second_call_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_second",
                "type": "function_call",
                "status": "in_progress",
                "arguments": "",
                "call_id": "call_second",
                "name": "DateTime",
            },
            "output_index": 2,
        },
        {
            "type": "response.function_call_arguments.delta",
            "delta": "{}",
            "item_id": "fc_second",
            "output_index": 2,
        },
        {
            "type": "response.function_call_arguments.done",
            "arguments": "{}",
            "item_id": "fc_second",
            "output_index": 2,
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": "fc_second",
                "type": "function_call",
                "status": "completed",
                "arguments": "{}",
                "call_id": "call_second",
                "name": "DateTime",
            },
            "output_index": 2,
        },
    ]
    sse_entries = responses_cassette["response"]["sse"]
    completed_index = next(
        index
        for index, entry in enumerate(sse_entries)
        if entry.strip().startswith("data:")
        and json_loads(entry.strip().removeprefix("data:").lstrip()).get("type")
        == "response.completed"
    )
    sse_entries[completed_index:completed_index] = [
        entry for event in second_call_events for entry in (f"data: {json_dumps(event)}\n\n", "\n")
    ]

    match_chat_completions_events(chat_cassette, responses_cassette)

    chat_events = _data_events(chat_cassette)
    response_events = _data_events(responses_cassette)
    assert _chat_tool_call_chunks(chat_events) == _response_tool_call_chunks(response_events)
    assert {chunk["index"] for chunk in _chat_tool_call_chunks(chat_events)} == {0, 1}
    for event in chat_events:
        ChatCompletionChunk.model_validate(event)


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


def test_request_only_matching_preserves_recorded_error(tmp_path: Path) -> None:
    cassette_dir = Path(__file__).parent / "cassettes" / "errors" / "context_length_exceeded"
    expected = read_yaml(str(cassette_dir / "openai-chat_completions-stream-00.yaml"))
    responses_path = cassette_dir / "openai-responses-stream-00.yaml"
    original_responses = responses_path.read_bytes()
    unmatched = deepcopy(expected)
    unmatched["request"]["body"]["model"] = "unmatched-model"
    unmatched["request"]["body"]["messages"] = [{"role": "user", "content": "Unmatched input"}]
    unmatched["request"]["body"]["max_completion_tokens"] = 99
    unmatched["request"]["body"].pop("stream_options", None)
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
