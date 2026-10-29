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
            "ephemeral_5m_input_tokens": 2,
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


def test_matches_explicit_null_cache_control() -> None:
    """Ensures an explicit null cache control matches like an omitted control."""
    messages, responses = _cassettes()
    messages["request"]["body"]["cache_control"] = None
    AnthropicMessagesCreateRequest.model_validate(messages["request"]["body"])

    match_messages_response(messages, responses)

    AnthropicMessagesResponse.model_validate(messages["response"]["body"])
    matched = deepcopy(messages)
    match_messages_response(messages, responses)
    assert messages == matched


@pytest.mark.parametrize("ttl", ["5m", "1h"])
@pytest.mark.parametrize("cache_location", ["automatic", "block", "block-null"])
def test_matches_positive_cache_writes_by_effective_lifetime(
    ttl: str, cache_location: str
) -> None:
    """Ensures automatic and block-only caching attribute positive writes to their actual TTL."""
    messages, responses = _cassettes()
    request = messages["request"]["body"]
    control = {"type": "ephemeral", "ttl": ttl}
    if cache_location == "automatic":
        request["cache_control"] = control
    else:
        if cache_location == "block-null":
            request["cache_control"] = None
        request["system"] = [{"type": "text", "text": "Cached prefix", "cache_control": control}]
        responses["request"]["body"]["instructions"] = "Cached prefix"
    responses["response"]["body"]["usage"]["input_tokens_details"]["cache_write_tokens"] = 2
    AnthropicMessagesCreateRequest.model_validate(request)

    match_messages_response(messages, responses)

    usage = messages["response"]["body"]["usage"]
    assert usage["cache_creation_input_tokens"] == 2
    assert usage["cache_creation"] == {
        "ephemeral_5m_input_tokens": 2 if ttl == "5m" else 0,
        "ephemeral_1h_input_tokens": 2 if ttl == "1h" else 0,
    }
    AnthropicMessagesResponse.model_validate(messages["response"]["body"])
    matched = deepcopy(messages)
    match_messages_response(messages, responses)
    assert messages == matched


@pytest.mark.parametrize("cache_write_tokens", [0, 3, 4])
def test_mixed_cache_lifetimes_preserve_only_observed_distribution(
    cache_write_tokens: int,
) -> None:
    """Ensures mixed lifetimes retain an observed split and reject an unknowable positive split."""
    messages, responses = _cassettes()
    messages["request"]["body"]["system"] = [
        {"type": "text", "text": "First", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "Second", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
    ]
    responses["request"]["body"]["instructions"] = "First\nSecond"
    usage = messages["response"]["body"]["usage"]
    usage["cache_creation_input_tokens"] = 3
    usage["cache_creation"] = {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 2}
    responses["response"]["body"]["usage"]["input_tokens_details"]["cache_write_tokens"] = (
        cache_write_tokens
    )
    AnthropicMessagesCreateRequest.model_validate(messages["request"]["body"])
    AnthropicMessagesResponse.model_validate(messages["response"]["body"])

    if cache_write_tokens == 4:
        with pytest.raises(ValueError, match="mixed cache lifetimes"):
            match_messages_response(messages, responses)
        return

    match_messages_response(messages, responses)

    assert usage["cache_creation_input_tokens"] == cache_write_tokens
    assert usage["cache_creation"] == {
        "ephemeral_5m_input_tokens": 1 if cache_write_tokens else 0,
        "ephemeral_1h_input_tokens": 2 if cache_write_tokens else 0,
    }
    AnthropicMessagesResponse.model_validate(messages["response"]["body"])
    matched = deepcopy(messages)
    match_messages_response(messages, responses)
    assert messages == matched


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


def _search_cassettes() -> tuple[dict, dict]:
    directory = CASSETTES.parent / "web_search"
    return (
        read_yaml(str(directory / "anthropic-messages-non_stream-00.yaml")),
        read_yaml(str(directory / "openai-responses-non_stream-00.yaml")),
    )


def test_search_recording_requires_observed_sources() -> None:
    """Ensures missing recorded source data is rejected without modifying either cassette."""
    messages, responses = _search_cassettes()
    responses["response"]["body"]["output"][0]["action"].pop("sources")
    originals = deepcopy((messages, responses))
    with pytest.raises(ValueError, match="include=.*web_search_call.action.sources"):
        match_messages_response(messages, responses)
    assert (messages, responses) == originals


@pytest.mark.parametrize("sources", [[], [{"type": "url", "url": "https://example.org"}]])
def test_search_response_maps_sources_citations_and_usage(sources: list) -> None:
    """Ensures native SDK-valid search blocks retain only observed sources and attribution."""
    from anthropic.types import Message
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from pydantic import TypeAdapter

    messages, responses = _search_cassettes()
    response = responses["response"]["body"]
    response["output"][0]["action"]["sources"] = sources
    response["output"][1]["content"] = [
        {
            "type": "output_text",
            "text": "aé xyz!",
            "annotations": [
                {
                    "type": "url_citation",
                    "start_index": 1,
                    "end_index": 5,
                    "url": "https://example.org/a",
                    "title": "A",
                },
                {
                    "type": "url_citation",
                    "start_index": 3,
                    "end_index": 6,
                    "url": "https://example.org/b",
                    "title": None,
                },
            ],
        }
    ]
    responses["request"]["body"]["max_tool_calls"] = 2
    match_messages_response(messages, responses)
    body = messages["response"]["body"]
    Message.model_validate(body)
    assert body["id"] == response["id"]
    assert body["model"] == response["model"]
    TypeAdapter(MessageCreateParamsNonStreaming).validate_python(messages["request"]["body"])
    assert messages["request"]["body"]["tool_choice"] == {"type": "tool", "name": "web_search"}
    assert messages["request"]["body"]["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 2,
        }
    ]
    call, result, *text = body["content"]
    assert call["type"] == "server_tool_use"
    assert call["input"] == {"query": response["output"][0]["action"]["query"]}
    assert result["tool_use_id"] == call["id"]
    assert result["content"] == (
        [
            {
                "type": "web_search_result",
                "url": "https://example.org",
                "title": "",
                "encrypted_content": "N/A",
            }
        ]
        if sources
        else []
    )
    assert "".join(block["text"] for block in text) == "aé xyz!"
    assert [len(block.get("citations", [])) for block in text] == [0, 1, 2, 1, 0]
    assert text[1]["citations"][0]["cited_text"] == "é xy"
    assert text[2]["citations"][1]["cited_text"] == "xyz"
    assert text[2]["citations"][1]["title"] == ""
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 1, "web_fetch_requests": 0}
    assert body["stop_reason"] == "end_turn"
    matched = deepcopy(messages)
    match_messages_response(messages, responses)
    assert messages == matched


@pytest.mark.parametrize("offset", [0, 3, 6])
def test_search_response_preserves_point_citations(offset: int) -> None:
    """Ensures point citations keep one block and use its text as their fallback excerpt."""
    text = "answer"
    messages, responses = _search_cassettes()
    annotations = [
        {
            "type": "url_citation",
            "url": "https://example.org/point",
            "start_index": offset,
            "end_index": offset,
        },
        {
            "type": "url_citation",
            "url": "https://example.org/range",
            "title": "Range",
            "start_index": 0,
            "end_index": len(text),
        },
    ]
    responses["response"]["body"]["output"][1]["content"] = [
        {"type": "output_text", "text": text, "annotations": annotations}
    ]

    match_messages_response(messages, responses)

    body = messages["response"]["body"]
    AnthropicMessagesResponse.model_validate(body)
    blocks = [block for block in body["content"] if block["type"] == "text"]
    assert [block["text"] for block in blocks] == [text]
    assert [
        citation
        for citation in blocks[0]["citations"]
        if citation["url"] == "https://example.org/point"
    ] == [
        {
            "type": "web_search_result_location",
            "url": "https://example.org/point",
            "title": "",
            "encrypted_index": "N/A",
            "cited_text": text,
        }
    ]
    assert [[citation["url"] for citation in block["citations"]] for block in blocks] == [
        ["https://example.org/point", "https://example.org/range"]
    ]
    AnthropicMessagesCreateRequest.model_validate(
        {
            "model": body["model"],
            "max_tokens": 100,
            "messages": [{"role": "assistant", "content": body["content"]}],
        }
    )


@pytest.mark.parametrize("text", ["", " \t"])
@pytest.mark.parametrize("cited", [False, True])
def test_search_response_rejects_blank_text(text: str, cited: bool) -> None:
    """Ensures the matcher rejects wholly blank text instead of generating invalid blocks."""
    messages, responses = _search_cassettes()
    responses["response"]["body"]["output"][1]["content"] = [
        {
            "type": "output_text",
            "text": text,
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://example.org/point",
                    "start_index": 0,
                    "end_index": 0,
                }
            ]
            if cited
            else [],
        }
    ]
    with pytest.raises(ValueError, match="non-whitespace"):
        match_messages_response(messages, responses)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ({"query": ""}, {"query": ""}),
        ({"query": "hi"}, {"query": "hi"}),
        ({}, {}),
        ({"query": None}, {}),
        ({"queries": []}, {}),
        ({"queries": ["preferred"], "query": "other"}, {"query": "preferred"}),
    ],
)
def test_search_query_presence(action: dict, expected: dict) -> None:
    """Ensures output and history distinguish absent queries from explicitly empty queries."""
    messages, responses = _search_cassettes()
    responses["response"]["body"]["output"][0]["action"] = {
        "type": "search",
        "sources": [],
        **action,
    }
    responses["request"]["body"]["input"] = [
        deepcopy(responses["response"]["body"]["output"][0]),
        {"role": "user", "content": "Continue."},
    ]
    match_messages_response(messages, responses)
    assert messages["response"]["body"]["content"][0]["input"] == expected
    assert (
        messages["request"]["body"]["messages"][0]["content"]
        == messages["response"]["body"]["content"][:2]
    )


@pytest.mark.parametrize("queries", [["a", "b"], ["a", "a"], ["", "b"]])
def test_search_batch_response(queries: list[str]) -> None:
    """Ensures output query batches preserve order, distinct IDs, and one pooled result."""
    messages, responses = _search_cassettes()
    search = responses["response"]["body"]["output"][0]
    search["action"]["queries"] = queries
    match_messages_response(messages, responses)

    AnthropicMessagesResponse.model_validate(messages["response"]["body"])
    output = messages["response"]["body"]["content"]
    calls = [block for block in output if block["type"] == "server_tool_use"]
    results = [block for block in output if block["type"] == "web_search_tool_result"]
    assert [call["input"] for call in calls] == [{"query": query} for query in queries]
    assert calls[0]["id"] == search["id"]
    assert len({call["id"] for call in calls}) == len(queries)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == search["id"]
    assert [result["url"] for result in results[0]["content"]] == [
        source["url"] for source in search["action"]["sources"]
    ]
    assert output[: len(queries)] == calls
    matched = deepcopy(messages)
    match_messages_response(messages, responses)
    assert messages == matched


@pytest.mark.parametrize("queries", [["a", "b"], ["a", "a"], ["", "b"]])
def test_search_history_rejects_query_batches(queries: list[str]) -> None:
    """Ensures request matching rejects pooled query history without changing either cassette."""
    messages, responses = _search_cassettes()
    search = deepcopy(responses["response"]["body"]["output"][0])
    search["action"]["queries"] = queries
    responses["request"]["body"]["input"] = [
        search,
        {"role": "user", "content": "Continue."},
    ]
    originals = deepcopy((messages, responses))

    with pytest.raises(ValueError, match="cannot preserve multi-query search history"):
        match_messages_response(messages, responses)

    assert (messages, responses) == originals


def test_search_batch_validates_every_query() -> None:
    """Ensures invalid later queries cannot become native Messages search arguments."""
    messages, responses = _search_cassettes()
    responses["response"]["body"]["output"][0]["action"]["queries"] = ["a", 2]
    original = deepcopy(messages)
    with pytest.raises(ValueError, match="query must be a string"):
        match_messages_response(messages, responses)
    assert messages == original


@pytest.mark.parametrize("action", ["open_page", "find_in_page"])
def test_search_browsing_is_omitted(action: str) -> None:
    """Ensures unsupported browsing operations produce no invented Messages search call."""
    messages, responses = _search_cassettes()
    responses["response"]["body"]["output"][0]["action"] = {
        "type": action,
        "url": "https://example.org",
    }
    match_messages_response(messages, responses)
    assert all(block["type"] == "text" for block in messages["response"]["body"]["content"])


@pytest.mark.parametrize("count", [None, 0, 7])
def test_search_counter_is_observed_not_inferred(count: int | None) -> None:
    """Ensures missing usage remains unknown and explicit counts survive unchanged."""
    messages, responses = _search_cassettes()
    response = responses["response"]["body"]
    response["output"][0]["action"]["sources"] = []
    response["tool_usage"] = {"web_search": {"num_requests": count}}
    match_messages_response(messages, responses)
    usage = messages["response"]["body"]["usage"]
    assert usage.get("server_tool_use") == (
        None if count is None else {"web_search_requests": count, "web_fetch_requests": 0}
    )


def test_search_failed_and_unfinished_states() -> None:
    """Ensures failed searches use unavailable and unfinished searches do not gain results."""
    messages, responses = _search_cassettes()
    response = responses["response"]["body"]
    response["output"][0]["status"] = "failed"
    match_messages_response(messages, responses)
    assert messages["response"]["body"]["content"][1]["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "unavailable",
    }
    response.update(status="incomplete", incomplete_details=None)
    response["output"][0]["status"] = "in_progress"
    match_messages_response(messages, responses)
    assert messages["response"]["body"]["stop_reason"] == "end_turn"
    assert not any(
        block["type"] == "web_search_tool_result"
        for block in messages["response"]["body"]["content"]
    )


@pytest.mark.parametrize("size", ["low", "medium", "high"])
def test_search_context_size_rejected(size: str) -> None:
    """Ensures explicit context size is rejected rather than silently discarded."""
    messages, responses = _search_cassettes()
    responses["request"]["body"]["tools"][0]["search_context_size"] = size
    with pytest.raises(ValueError, match="search_context_size"):
        match_messages_response(messages, responses)


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
def test_unknown_incomplete_response_maps_to_end_turn(
    has_search: bool, incomplete_fields: dict
) -> None:
    """Ensures missing reasons coerce to end_turn without inferring a search pause."""
    messages, responses = _search_cassettes() if has_search else _cassettes()
    response = responses["response"]["body"]
    response.pop("incomplete_details", None)
    response.update(status="incomplete", **incomplete_fields)

    match_messages_response(messages, responses)

    assert messages["response"]["body"]["stop_reason"] == "end_turn"


@pytest.mark.parametrize("incomplete_details", [[], "invalid"])
def test_malformed_incomplete_response_details_still_rejected(incomplete_details: object) -> None:
    """Ensures accepting unknown reasons does not accept malformed details containers."""
    messages, responses = _cassettes()
    responses["response"]["body"].update(
        status="incomplete", incomplete_details=incomplete_details
    )
    with pytest.raises(ValueError, match="incomplete_details"):
        match_messages_response(messages, responses)
