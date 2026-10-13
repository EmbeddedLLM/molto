from copy import deepcopy
from pathlib import Path

import pytest

from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
)
from molto.utils.io import dump_yaml, read_yaml
from scripts.match_openai_cassette_responses import (
    main,
    match_chat_completions_response,
)

CASSETTES = Path(__file__).parent / "cassettes" / "matched" / "simple_text"


def _cassettes(turn: int = 0) -> tuple[dict, dict]:
    chat = read_yaml(str(CASSETTES / f"openai-chat_completions-non_stream-{turn:02d}.yaml"))
    responses = read_yaml(str(CASSETTES / f"openai-responses-non_stream-{turn:02d}.yaml"))
    assert isinstance(chat, dict)
    assert isinstance(responses, dict)
    return chat, responses


@pytest.mark.parametrize("turn", [0, 1])
def test_matches_real_non_streaming_cassette_and_is_idempotent(turn: int) -> None:
    """Ensures matched responses preserve native schemas and individual input text parts."""
    chat, responses = _cassettes(turn)
    transport = deepcopy(chat["response"]["headers"])

    match_chat_completions_response(chat, responses)

    OpenAIChatCompletionsCreateRequest.model_validate(chat["request"]["body"])
    OpenAIChatCompletionsResponse.model_validate(chat["response"]["body"])
    chat_body = chat["response"]["body"]
    response_body = responses["response"]["body"]
    assert chat_body["id"] == response_body["id"]
    assert chat_body["created"] == response_body["created_at"]
    assert chat_body["model"] == response_body["model"]
    assert chat_body["choices"][0]["message"]["content"] == "".join(
        part["text"]
        for item in response_body["output"]
        if item["type"] == "message"
        for part in item["content"]
    )
    assert chat_body["usage"]["prompt_tokens"] == response_body["usage"]["input_tokens"]
    assert chat["response"]["headers"] == transport
    response_input = responses["request"]["body"]["input"]
    assert chat["request"]["body"]["messages"][0]["content"] == [
        {"type": "text", "text": part["text"]} for part in response_input[0]["content"]
    ]
    if turn == 1:
        assert chat["request"]["body"]["messages"][1]["content"] == (
            "The ocean is a vast, mysterious body of water that covers most of the "
            "Earth's surface."
        )

    matched = deepcopy(chat)
    match_chat_completions_response(chat, responses)
    assert chat == matched


@pytest.mark.parametrize("details_state", ["missing", "null"])
@pytest.mark.parametrize("token_count", [0, 3])
def test_populates_optional_usage_details(details_state: str, token_count: int) -> None:
    chat, responses = _cassettes()
    chat_usage = chat["response"]["body"]["usage"]
    for field in ("prompt_tokens_details", "completion_tokens_details"):
        if details_state == "missing":
            chat_usage.pop(field)
        else:
            chat_usage[field] = None
    response_usage = responses["response"]["body"]["usage"]
    response_usage["input_tokens_details"] = {
        "cache_write_tokens": token_count,
        "cached_tokens": token_count,
    }
    response_usage["output_tokens_details"] = {"reasoning_tokens": token_count}

    match_chat_completions_response(chat, responses)

    assert chat_usage["prompt_tokens_details"] == {
        "cache_write_tokens": token_count,
        "cached_tokens": token_count,
    }
    assert chat_usage["completion_tokens_details"] == {"reasoning_tokens": token_count}
    OpenAIChatCompletionsResponse.model_validate(chat["response"]["body"])


@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
def test_matches_reasoning_tools_usage_and_finish_reason(reasoning_field: str) -> None:
    """Ensures raw reasoning and tool history retain the provider's native field."""
    chat, responses = _cassettes()
    chat["response"]["body"]["choices"][0]["message"][reasoning_field] = "unmatched"
    alternate_field = (
        "reasoning" if reasoning_field == "reasoning_content" else "reasoning_content"
    )
    chat["response"]["body"]["choices"][0]["message"][alternate_field] = None
    responses["request"]["body"]["input"] = [
        {
            "content": [{"text": "Check Paris.", "type": "input_text"}],
            "role": "user",
        },
        {
            "content": [{"text": "I should use weather.", "type": "reasoning_text"}],
            "summary": [],
            "type": "reasoning",
        },
        {
            "arguments": '{"city":"Paris"}',
            "call_id": "call_weather",
            "name": "weather",
            "type": "function_call",
        },
        {
            "call_id": "call_weather",
            "output": "sunny",
            "type": "function_call_output",
        },
    ]
    response = responses["response"]["body"]
    response["output"] = [
        {
            "content": [{"text": "Use the weather tool.", "type": "reasoning_text"}],
            "summary": [],
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
    chat_usage = chat["response"]["body"]["usage"]
    chat_usage["prompt_tokens_details"]["cache_write_tokens"] = 0
    chat_usage["prompt_tokens_details"]["audio_tokens"] = 1
    chat_usage["completion_tokens_details"]["audio_tokens"] = 2

    match_chat_completions_response(chat, responses)

    choice = chat["response"]["body"]["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"] | {"annotations": [], "refusal": None} == {
        alternate_field: None,
        "annotations": [],
        "content": "Calling it.",
        reasoning_field: "Use the weather tool.",
        "refusal": None,
        "role": "assistant",
        "tool_calls": [
            {
                "function": {"arguments": '{"city":"Paris"}', "name": "weather"},
                "id": "call_weather",
                "type": "function",
            }
        ],
    }
    assert chat_usage["prompt_tokens_details"]["cache_write_tokens"] == 2
    assert chat_usage["prompt_tokens_details"]["cached_tokens"] == 3
    assert chat_usage["completion_tokens_details"]["reasoning_tokens"] == 4
    assert chat_usage["prompt_tokens_details"]["audio_tokens"] == 1
    assert chat_usage["completion_tokens_details"]["audio_tokens"] == 2
    assert chat["request"]["body"]["messages"] == [
        {
            "content": [{"text": "Check Paris.", "type": "text"}],
            "role": "user",
        },
        {
            reasoning_field: "I should use weather.",
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {"arguments": '{"city":"Paris"}', "name": "weather"},
                    "id": "call_weather",
                    "type": "function",
                }
            ],
        },
        {
            "content": [{"text": "sunny", "type": "text"}],
            "role": "tool",
            "tool_call_id": "call_weather",
        },
    ]
    OpenAIChatCompletionsCreateRequest.model_validate(chat["request"]["body"])
    OpenAIChatCompletionsResponse.model_validate(chat["response"]["body"])


@pytest.mark.parametrize(
    "image_url", ["https://example.com/image.png", "data:image/png;base64,aW1hZ2U="]
)
def test_matches_image_input_without_losing_order_or_provider_extensions(image_url: str) -> None:
    """Ensures matched image requests and provider response extensions retain fidelity."""
    chat, responses = _cassettes()
    responses["request"]["body"]["input"] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Before the image."},
                {"type": "input_image", "image_url": image_url, "detail": "high"},
                {"type": "input_text", "text": "After the image."},
            ],
        }
    ]
    chat["request"]["body"]["chat_template_kwargs"] = {"enable_thinking": False}
    chat["response"]["body"]["provider_trace"] = "retained"
    chat["response"]["body"]["choices"][0]["message"]["provider_message"] = "retained"
    native_transport = deepcopy(
        {key: value for key, value in chat["response"].items() if key != "body"}
    )
    canonical = deepcopy(responses)

    match_chat_completions_response(chat, responses)

    assert chat["request"]["body"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Before the image."},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                {"type": "text", "text": "After the image."},
            ],
        }
    ]
    assert chat["request"]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat["response"]["body"]["provider_trace"] == "retained"
    assert chat["response"]["body"]["choices"][0]["message"]["provider_message"] == "retained"
    assert {
        key: value for key, value in chat["response"].items() if key != "body"
    } == native_transport
    assert responses == canonical
    OpenAIChatCompletionsResponse.model_validate(chat["response"]["body"])
    matched = deepcopy(chat)
    match_chat_completions_response(chat, responses)
    assert chat == matched


@pytest.mark.parametrize("token_field", ["max_tokens", "max_completion_tokens"])
def test_preserves_recorded_non_streaming_token_limit_field(token_field: str) -> None:
    """Ensures non-streaming matching retains the provider's recorded token field."""
    chat, responses = _cassettes()
    chat["request"]["body"].pop("max_completion_tokens", None)
    chat["request"]["body"][token_field] = 17
    responses["request"]["body"]["max_output_tokens"] = 512

    match_chat_completions_response(chat, responses)

    assert chat["request"]["body"][token_field] == 512
    other_field = (
        "max_tokens" if token_field == "max_completion_tokens" else "max_completion_tokens"
    )
    assert other_field not in chat["request"]["body"]


def test_matches_parallel_image_tool_history_in_non_streaming_request() -> None:
    """Ensures non-streaming history also relocates images after every tool response."""
    chat, responses = _cassettes()
    image_history = read_yaml(
        str(
            CASSETTES.parent
            / "tool_call_image_result_parallel"
            / "openai-responses-stream-01.yaml"
        )
    )["request"]["body"]["input"]
    responses["request"]["body"]["input"] = image_history

    match_chat_completions_response(chat, responses)

    messages = chat["request"]["body"]["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "user",
        "user",
    ]
    results = [item for item in image_history if item.get("type") == "function_call_output"]
    for index, result in enumerate(results):
        call_id = result["call_id"]
        assert messages[2 + index]["content"] == [
            {
                "type": "text",
                "text": (
                    f"The image result of tool call {call_id!r} will be "
                    "attached as user prompt after all tool responses."
                ),
            }
        ]
        assert messages[4 + index]["content"] == [
            {"type": "text", "text": f"Image returned by tool call {call_id!r}:"},
            {"type": "image_url", "image_url": {"url": result["output"][0]["image_url"]}},
        ]
    OpenAIChatCompletionsCreateRequest.model_validate(chat["request"]["body"])
    OpenAIChatCompletionsResponse.model_validate(chat["response"]["body"])


@pytest.mark.parametrize(
    "reasoning",
    [
        {
            "content": [],
            "summary": [{"text": "summary", "type": "summary_text"}],
            "type": "reasoning",
        },
        {
            "content": [{"text": "raw", "type": "reasoning_text"}],
            "encrypted_content": "opaque",
            "summary": [],
            "type": "reasoning",
        },
    ],
)
def test_rejects_unrepresentable_chat_reasoning(reasoning: dict) -> None:
    chat, responses = _cassettes()
    responses["response"]["body"]["output"] = [reasoning]

    with pytest.raises(ValueError, match="cannot map to Chat Completions"):
        match_chat_completions_response(chat, responses)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("max_output_tokens", "length"), ("content_filter", "content_filter"), ("other", "stop")],
)
def test_matches_incomplete_finish_reason(reason: str, expected: str) -> None:
    chat, responses = _cassettes()
    responses["response"]["body"].update(
        {"incomplete_details": {"reason": reason}, "status": "incomplete"}
    )

    match_chat_completions_response(chat, responses)

    assert chat["response"]["body"]["choices"][0]["finish_reason"] == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda chat, responses: responses["request"]["body"].update(stream=True),
        lambda chat, responses: responses["response"].pop("body"),
        lambda chat, responses: responses["response"]["body"].update(status="queued"),
        lambda chat, responses: chat["response"]["body"].update(choices=[]),
    ],
)
def test_rejects_invalid_non_streaming_cassettes(mutation) -> None:
    chat, responses = _cassettes()
    mutation(chat, responses)

    with pytest.raises(ValueError):
        match_chat_completions_response(chat, responses)


def test_cli_writes_sorted_output_without_mutating_input(tmp_path: Path) -> None:
    chat, responses = _cassettes()
    chat_path = tmp_path / "chat.yaml"
    responses_path = tmp_path / "responses.yaml"
    output_path = tmp_path / "matched.yaml"
    dump_yaml(chat, str(chat_path))
    dump_yaml(responses, str(responses_path))

    assert main([str(chat_path), str(responses_path), "--output", str(output_path)]) == 0

    assert read_yaml(str(chat_path)) == chat
    matched = read_yaml(str(output_path))
    assert matched["response"]["body"]["id"] == responses["response"]["body"]["id"]
    assert output_path.read_text().splitlines()[0].startswith("filename:")
