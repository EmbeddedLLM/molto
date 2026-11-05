from copy import deepcopy
from pathlib import Path

import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import TypeAdapter

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


def test_matches_reasoning_tools_usage_and_finish_reason() -> None:
    chat, responses = _cassettes()
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
        "annotations": [],
        "content": "Calling it.",
        "reasoning_content": "Use the weather tool.",
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
            "reasoning_content": "I should use weather.",
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
        {
            "content": [],
            "encrypted_content": "opaque",
            "summary": [],
            "type": "reasoning",
        },
    ],
)
def test_rejects_unrepresentable_chat_reasoning(reasoning: dict) -> None:
    """Ensures ordinary Chat matching rejects reasoning with summaries or signatures."""
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


def test_matching_discards_stale_raw_response_bytes() -> None:
    """Ensures deliberately matched JSON replaces raw bytes without mutating the canonical recording."""
    source, canonical = _cassettes()
    source["response"]["body_bytes"] = b"old recorded response"
    canonical["response"]["body_bytes"] = b"canonical original bytes"
    original = deepcopy(canonical)
    match_chat_completions_response(source, canonical)
    assert "body_bytes" not in source["response"]
    assert canonical == original


def _search_cassettes() -> tuple[dict, dict]:
    directory = CASSETTES.parent / "web_search"
    return (
        read_yaml(str(directory / "openai-chat_completions-non_stream-00.yaml")),
        read_yaml(str(directory / "openai-responses-non_stream-00.yaml")),
    )


def test_search_response_rejects_signature_only_reasoning() -> None:
    """Ensures search never silently discards signature-only reasoning."""
    chat, responses = _search_cassettes()
    responses["response"]["body"]["output"].insert(
        0,
        {"type": "reasoning", "content": [], "summary": [], "encrypted_content": "opaque"},
    )

    with pytest.raises(ValueError, match="Encrypted Responses reasoning"):
        match_chat_completions_response(chat, responses)


@pytest.mark.parametrize("field", ["content", "summary"])
def test_search_response_still_rejects_signed_reasoning_text(field: str) -> None:
    """Ensures search never discards actual reasoning text together with its signature."""
    chat, responses = _search_cassettes()
    reasoning = {
        "type": "reasoning",
        "content": [],
        "summary": [],
        "encrypted_content": "opaque",
    }
    reasoning[field] = [
        {"type": "reasoning_text" if field == "content" else "summary_text", "text": "Think."}
    ]
    responses["response"]["body"]["output"].insert(0, reasoning)

    with pytest.raises(ValueError, match="reasoning"):
        match_chat_completions_response(chat, responses)


def test_matches_web_search_response_without_mutating_recordings(tmp_path: Path) -> None:
    """Ensures reasoning-free search recordings yield SDK-valid Chat output and stay intact."""
    from hashlib import sha256

    directory = CASSETTES.parent / "web_search"
    paths = [
        directory / f"openai-{endpoint}-non_stream-00.yaml"
        for endpoint in ("chat_completions", "responses")
    ]
    originals = {path: sha256(path.read_bytes()).digest() for path in paths}
    output_path = tmp_path / "matched-search.yaml"

    assert main([*(str(path) for path in paths), "--output", str(output_path)]) == 0

    chat = read_yaml(str(output_path))
    responses = read_yaml(str(paths[1]))
    TypeAdapter(CompletionCreateParamsNonStreaming).validate_python(chat["request"]["body"])
    parsed = OpenAIChatCompletionsResponse.model_validate(chat["response"]["body"])
    assert parsed.choices[0].message.content == "".join(
        part["text"]
        for item in responses["response"]["body"]["output"]
        if item["type"] == "message"
        for part in item["content"]
    )
    assert parsed.choices[0].finish_reason == "stop"
    assert "annotations" not in chat["response"]["body"]["choices"][0]["message"]
    assert parsed.choices[0].message.tool_calls is None
    assert chat["request"]["body"]["web_search_options"] == {}
    assert chat["request"]["body"]["model"] == responses["request"]["body"]["model"]
    assert "tool_choice" not in chat["request"]["body"]
    assert chat["response"]["headers"] == read_yaml(str(paths[0]))["response"]["headers"]
    matched = deepcopy(chat)
    match_chat_completions_response(chat, responses)
    assert chat == matched
    assert originals == {path: sha256(path.read_bytes()).digest() for path in originals}


def test_search_request_maps_context_location_and_client_tools() -> None:
    """Ensures explicit hosted-only selection preserves client functions and search options."""
    chat, responses = _search_cassettes()
    request = responses["request"]["body"]
    request["instructions"] = "Search before answering."
    request["tools"][0].update(
        search_context_size="low",
        user_location={"type": "approximate", "city": "Paris", "country": "FR"},
    )
    request["tools"].append({"type": "function", "name": "web_search", "parameters": {}})
    request["tool_choice"] = {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "web_search"}],
    }

    match_chat_completions_response(chat, responses)

    assert chat["request"]["body"]["web_search_options"] == {
        "search_context_size": "low",
        "user_location": {
            "type": "approximate",
            "approximate": {"city": "Paris", "country": "FR"},
        },
    }
    assert chat["request"]["body"]["tools"] == [
        {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    ]
    assert chat["request"]["body"]["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "Search before answering."}],
    }


@pytest.mark.parametrize("choice", [None, "auto", "none", {"type": "function", "name": "lookup"}])
def test_search_request_rejects_optional_or_conflicting_search(choice: object) -> None:
    """Ensures Chat's mandatory search never silently changes canonical tool selection."""
    chat, responses = _search_cassettes()
    responses["request"]["body"]["tool_choice"] = choice
    with pytest.raises(ValueError, match="requires hosted search"):
        match_chat_completions_response(chat, responses)


def test_search_request_rejects_generic_required_with_client_functions() -> None:
    """Ensures generic required cannot stand in for mandatory hosted search with mixed tools."""
    chat, responses = _search_cassettes()
    responses["request"]["body"]["tools"].append({"type": "function", "name": "lookup"})
    with pytest.raises(ValueError, match="requires hosted search"):
        match_chat_completions_response(chat, responses)


def test_search_request_rejects_required_chat_function_choice() -> None:
    """Ensures mandatory hosted search cannot discard a required client-function call."""
    chat, responses = _search_cassettes()
    chat["request"]["body"].update(
        tool_choice="required",
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
    )
    responses["request"]["body"].update(
        tools=[{"type": "web_search"}, {"type": "function", "name": "lookup", "parameters": {}}],
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "web_search"}],
        },
    )

    with pytest.raises(ValueError, match="Chat tool_choice conflicts"):
        match_chat_completions_response(chat, responses)


@pytest.mark.parametrize("action", ["search", "open_page", "find_in_page"])
def test_search_failure_and_browsing_without_text_stay_empty(action: str) -> None:
    """Ensures omitted hosted operations do not become client calls or fabricated narration."""
    chat, responses = _search_cassettes()
    responses["response"]["body"]["output"] = [
        {
            "id": "ws_failed",
            "type": "web_search_call",
            "status": "failed",
            "action": {"type": action},
        }
    ]

    match_chat_completions_response(chat, responses)

    choice = chat["response"]["body"]["choices"][0]
    assert choice["message"]["content"] is None
    assert "annotations" not in choice["message"]
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"


@pytest.mark.parametrize("has_search", [False, True])
@pytest.mark.parametrize(
    "incomplete_fields",
    [
        pytest.param({}, id="absent-details"),
        pytest.param({"incomplete_details": None}, id="null-details"),
        pytest.param({"incomplete_details": {}}, id="absent-reason"),
        pytest.param({"incomplete_details": {"reason": None}}, id="null-reason"),
    ],
)
def test_unknown_incomplete_response_maps_to_stop(
    has_search: bool, incomplete_fields: dict
) -> None:
    """Ensures missing reasons coerce to stop without inferring a search pause."""
    chat, responses = _search_cassettes() if has_search else _cassettes()
    response = responses["response"]["body"]
    response.pop("incomplete_details", None)
    response.update(status="incomplete", **incomplete_fields)

    match_chat_completions_response(chat, responses)

    assert chat["response"]["body"]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("incomplete_details", [[], "invalid"])
def test_malformed_incomplete_response_details_still_rejected(incomplete_details: object) -> None:
    """Ensures accepting unknown reasons does not accept malformed details containers."""
    chat, responses = _cassettes()
    responses["response"]["body"].update(
        status="incomplete", incomplete_details=incomplete_details
    )
    with pytest.raises(ValueError, match="incomplete_details"):
        match_chat_completions_response(chat, responses)


@pytest.mark.parametrize("field", ["filters", "max_tool_calls"])
def test_search_request_rejects_unrepresentable_chat_options(field: str) -> None:
    """Ensures fixture matching cannot silently drop source domain restrictions or budgets."""
    chat, responses = _search_cassettes()
    request = responses["request"]["body"]
    if field == "filters":
        request["tools"][0][field] = {"allowed_domains": ["example.com"]}
    else:
        request[field] = 2

    with pytest.raises(ValueError, match="cannot map to Chat web_search_options"):
        match_chat_completions_response(chat, responses)
