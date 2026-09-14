from copy import deepcopy
from pathlib import Path

import pytest

from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesResponse,
)
from molto.utils.io import dump_yaml, read_yaml
from scripts.match_anthropic_cassette_responses import main, match_messages_response

CASSETTES = Path(__file__).parent / "cassettes" / "matched" / "simple_text"


def _cassettes(turn: int = 0) -> tuple[dict, dict]:
    messages = read_yaml(str(CASSETTES / f"anthropic-messages-non_stream-{turn:02d}.yaml"))
    responses = read_yaml(str(CASSETTES / f"openai-responses-non_stream-{turn:02d}.yaml"))
    assert isinstance(messages, dict)
    assert isinstance(responses, dict)
    return messages, responses


@pytest.mark.parametrize("turn", [0, 1])
def test_matches_real_non_streaming_cassette_and_is_idempotent(turn: int) -> None:
    messages, responses = _cassettes(turn)
    transport = deepcopy(messages["response"]["headers"])

    match_messages_response(messages, responses)

    AnthropicMessagesCreateRequest.model_validate(messages["request"]["body"])
    AnthropicMessagesResponse.model_validate(messages["response"]["body"])
    message = messages["response"]["body"]
    response = responses["response"]["body"]
    assert message["id"] == response["id"]
    assert message["model"] == response["model"]
    assert message["content"] == [
        {"text": part["text"], "type": "text"}
        for item in response["output"]
        if item["type"] == "message"
        for part in item["content"]
    ]
    assert message["usage"]["input_tokens"] == response["usage"]["input_tokens"]
    assert messages["response"]["headers"] == transport
    if turn == 1:
        assert messages["request"]["body"]["messages"][1]["content"] == [
            {
                "text": (
                    "The ocean is a vast, mysterious body of water that covers most of "
                    "the Earth's surface."
                ),
                "type": "text",
            }
        ]

    matched = deepcopy(messages)
    match_messages_response(messages, responses)
    assert messages == matched


def test_matches_ordered_content_usage_and_tool_stop() -> None:
    messages, responses = _cassettes()
    response = responses["response"]["body"]
    response["output"] = [
        {
            "content": [],
            "encrypted_content": "opaque-state",
            "summary": [{"text": "considered", "type": "summary_text"}],
            "type": "reasoning",
        },
        {
            "content": [
                {"annotations": [], "logprobs": [], "text": "Calling it.", "type": "output_text"}
            ],
            "role": "assistant",
            "type": "message",
        },
        {
            "arguments": '{"city":"Paris"}',
            "call_id": "call_weather",
            "name": "weather",
            "type": "function_call",
        },
    ]
    response["usage"] = {
        "input_tokens": 11,
        "input_tokens_details": {"cache_write_tokens": 2, "cached_tokens": 3},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": 18,
    }

    match_messages_response(messages, responses)

    message = messages["response"]["body"]
    assert message["content"] == [
        {
            "signature": "opaque-state",
            "thinking": "considered",
            "type": "thinking",
        },
        {"text": "Calling it.", "type": "text"},
        {
            "caller": {"type": "direct"},
            "id": "call_weather",
            "input": {"city": "Paris"},
            "name": "weather",
            "type": "tool_use",
        },
    ]
    assert message["stop_reason"] == "tool_use"
    assert message["usage"] == {
        "cache_creation": {
            "ephemeral_1h_input_tokens": 0,
            "ephemeral_5m_input_tokens": 0,
        },
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 3,
        "inference_geo": "global",
        "input_tokens": 6,
        "output_tokens": 7,
        "output_tokens_details": {"thinking_tokens": 4},
        "service_tier": "standard",
    }
    AnthropicMessagesResponse.model_validate(message)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("max_output_tokens", "max_tokens"), ("content_filter", "end_turn"), ("other", "end_turn")],
)
def test_matches_incomplete_stop_reason(reason: str, expected: str) -> None:
    messages, responses = _cassettes()
    responses["response"]["body"].update(
        {"incomplete_details": {"reason": reason}, "status": "incomplete"}
    )

    match_messages_response(messages, responses)

    assert messages["response"]["body"]["stop_reason"] == expected


def test_preserves_explicit_false_stream() -> None:
    messages, responses = _cassettes()
    responses["request"]["body"]["stream"] = False

    match_messages_response(messages, responses)

    assert messages["request"]["body"]["stream"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda messages, responses: responses["request"]["body"].update(stream=True),
        lambda messages, responses: responses["response"].pop("body"),
        lambda messages, responses: responses["response"]["body"].update(status="queued"),
        lambda messages, responses: responses["response"]["body"].update(
            output=[{"type": "computer_call"}]
        ),
    ],
)
def test_rejects_invalid_non_streaming_cassettes(mutation) -> None:
    messages, responses = _cassettes()
    mutation(messages, responses)

    with pytest.raises(ValueError):
        match_messages_response(messages, responses)


def test_cli_writes_sorted_output_without_mutating_input(tmp_path: Path) -> None:
    messages, responses = _cassettes()
    messages_path = tmp_path / "messages.yaml"
    responses_path = tmp_path / "responses.yaml"
    output_path = tmp_path / "matched.yaml"
    dump_yaml(messages, str(messages_path))
    dump_yaml(responses, str(responses_path))

    assert main([str(messages_path), str(responses_path), "--output", str(output_path)]) == 0

    assert read_yaml(str(messages_path)) == messages
    matched = read_yaml(str(output_path))
    assert matched["response"]["body"]["id"] == responses["response"]["body"]["id"]
    assert output_path.read_text().splitlines()[0].startswith("filename:")
