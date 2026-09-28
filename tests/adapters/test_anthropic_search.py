"""Messages hosted-search projection and request regressions."""

import pytest

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.normalized import (
    InputTokenDetails,
    OutputTokenDetails,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    SearchCall,
    SearchResult,
    TextDelta,
    TextOutput,
    Usage,
    WebCitation,
    WebSearchSource,
    WebSearchTool,
    WebSearchToolChoice,
)
from molto.utils.web_search import messages_text_blocks


def setup_adapter():
    adapter = AnthropicMessagesAdapter(PROVIDER_CAPABILITIES["anthropic"])
    request = adapter.normalize_request(
        AnthropicMessagesCreateRequest(
            model="m",
            max_tokens=100,
            messages=[{"role": "user", "content": "Search"}],
            stream=True,
        )
    )
    return adapter, request


async def iterate(events):
    for event in events:
        yield event


def test_search_configuration_and_history():
    """Ensures hosted selection, budgets and opaque placeholders survive normalization."""
    adapter, _ = setup_adapter()
    request = adapter.normalize_request(
        AnthropicMessagesCreateRequest(
            model="m",
            max_tokens=100,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s",
                            "name": "web_search",
                            "input": {"query": ""},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "s",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://x.test",
                                    "title": "X",
                                    "encrypted_content": "N/A",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": "Answer",
                            "citations": [
                                {
                                    "type": "web_search_result_location",
                                    "url": "https://x.test",
                                    "title": None,
                                    "cited_text": "Source",
                                    "encrypted_index": "N/A",
                                }
                            ],
                        },
                    ],
                }
            ],
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                    "allowed_domains": ["x.test"],
                }
            ],
            tool_choice={"type": "tool", "name": "web_search"},
        )
    )
    assert request.tools == [WebSearchTool(allowed_domains=["x.test"])]
    assert request.max_tool_calls == 3
    assert request.tool_choice == WebSearchToolChoice()
    call, result, text = request.input[0].content
    assert call.query == ""
    assert result.sources[0].encrypted_content == "N/A"
    assert text.citations[0].end_index == 6
    assert text.citations[0].encrypted_index == "N/A"


@pytest.mark.parametrize(
    ("query", "queries", "expected"),
    [
        (None, None, {}),
        (None, [], {}),
        ("", None, {"query": ""}),
        ("fallback", ["one"], {"query": "one"}),
    ],
)
def test_missing_empty_and_single_queries(query, queries, expected):
    """Ensures missing query metadata differs from an explicitly empty query."""
    assert SearchCall(id="s", query=query, queries=queries).to_messages()["input"] == expected


def test_overlapping_unicode_citations():
    """Ensures overlapping codepoint ranges preserve text and full answer excerpts."""
    blocks = messages_text_blocks(
        TextOutput(
            content="a😀cdef",
            citations=[
                WebCitation(url="https://a", title=None, start_index=0, end_index=4),
                WebCitation(url="https://b", start_index=2, end_index=6),
            ],
        )
    )
    assert [block["text"] for block in blocks] == ["a😀", "cd", "ef"]
    assert [[c["url"] for c in block["citations"]] for block in blocks] == [
        ["https://a"],
        ["https://a", "https://b"],
        ["https://b"],
    ]
    assert blocks[1]["citations"][0] == {
        "type": "web_search_result_location",
        "url": "https://a",
        "title": "",
        "cited_text": "a😀cd",
        "encrypted_index": "N/A",
    }


async def test_buffered_citations_search_omission_and_collection():
    """Ensures delayed citation splits follow linked search results with dense lifecycles."""
    adapter, request = setup_adapter()
    parts = [
        SearchCall(id="browse", action="open_page"),
        SearchResult(tool_call_id="browse", sources=[]),
        SearchCall(id="search", query="q", status="completed"),
        SearchResult(tool_call_id="search", sources=[WebSearchSource(url="https://a")]),
    ]
    events = [ResponseStartEvent(id="r", model="m")]
    for index, part in enumerate(parts):
        events += [PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)]
    text = TextOutput(
        content="Alpha. Beta.",
        defer_citations=True,
        citations=[WebCitation(url="https://a", start_index=7, end_index=12)],
    )
    events += [
        PartStartEvent(index=4, part=TextOutput(content="", defer_citations=True)),
        PartDeltaEvent(index=4, delta=TextDelta(content_delta=text.content)),
        PartEndEvent(index=4, part=text),
        ResponseEndEvent(id="r", model="m", created_at=0, termination_reason="pause"),
    ]
    projected = [event async for event in adapter.project_events(iterate(events), request)]
    starts = [event for event in projected if event.type == "content_block_start"]
    assert [event.index for event in starts] == [0, 1, 2, 3]
    assert [event.content_block.type for event in starts] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
        "text",
    ]
    assert starts[0].content_block.id == starts[1].content_block.tool_use_id == "search"
    assert starts[1].content_block.content[0].encrypted_content == "N/A"
    response = await adapter.project_response(iterate(projected), request)
    assert response.stop_reason == "pause_turn"
    assert (
        "".join(block.text for block in response.content if block.type == "text") == text.content
    )
    assert response.content[-1].citations[0].cited_text == "Beta."


async def test_multi_query_failure_closes_source():
    """Ensures a translation failure identifies the call and deterministically closes input."""
    adapter, request = setup_adapter()
    closed = False
    part = SearchCall(id="multi", query="one", queries=["one", "two"])

    async def source():
        nonlocal closed
        try:
            yield ResponseStartEvent(id="r", model="m")
            yield PartStartEvent(index=0, part=part)
            yield PartEndEvent(index=0, part=part)
        finally:
            closed = True

    with pytest.raises(ValueError, match="multi.*multiple single-query searches"):
        _ = [event async for event in adapter.project_events(source(), request)]
    assert closed


@pytest.mark.parametrize("count", [None, 0, 2])
@pytest.mark.parametrize("has_token_usage", [False, True])
async def test_search_usage_observation(count, has_token_usage):
    """Ensures unknown usage remains unknown while an explicitly reported zero survives."""
    adapter, request = setup_adapter()
    usage = Usage(
        input_tokens=1,
        input_tokens_details=InputTokenDetails(),
        output_tokens=1,
        output_tokens_details=OutputTokenDetails(),
        total_tokens=2,
    )
    if count is not None:
        usage.extra_fields["web_search_requests"] = count
    end = ResponseEndEvent(
        id="r",
        model="m",
        created_at=0,
        termination_reason="stop",
        usage=usage if has_token_usage else None,
    )
    if not has_token_usage and count is not None:
        end.extra_fields["web_search_requests"] = count
    events = [ResponseStartEvent(id="r", model="m"), end]
    response = await adapter.project_response(
        adapter.project_events(iterate(events), request), request
    )
    if count is None:
        assert response.usage.server_tool_use is None
    else:
        assert response.usage.server_tool_use.web_search_requests == count
        assert response.usage.server_tool_use.web_fetch_requests == 0


def test_point_citation_preserves_attribution():
    """Ensures an empty range keeps its source without associating unrelated text."""
    blocks = messages_text_blocks(
        TextOutput(
            content="ab", citations=[WebCitation(url="https://x", start_index=1, end_index=1)]
        )
    )
    assert [block["text"] for block in blocks] == ["a", "", "b"]
    assert blocks[1]["citations"][0]["cited_text"] == ""
    assert "citations" not in blocks[0] and "citations" not in blocks[2]


@pytest.mark.parametrize("kind", ["search", "deferred_text"])
async def test_early_close_releases_source_after_buffered_projection(kind):
    """Ensures closing a buffered block start deterministically releases upstream input."""
    adapter, request = setup_adapter()
    closed = False
    part = (
        SearchCall(id="s", query="q")
        if kind == "search"
        else TextOutput(content="answer", defer_citations=True)
    )

    async def source():
        nonlocal closed
        try:
            yield ResponseStartEvent(id="r", model="m")
            yield PartStartEvent(index=0, part=part)
            yield PartEndEvent(index=0, part=part)
            raise AssertionError("Projection must not read ahead after yielding its block start.")
        finally:
            closed = True

    stream = adapter.project_events(source(), request)
    assert (await anext(stream)).type == "message_start"
    assert (await anext(stream)).type == "content_block_start"
    await stream.aclose()
    assert closed


def test_dynamic_search_caller_is_outside_basic_scope():
    """Ensures a code-execution-owned search is not relabeled as a basic direct search."""
    from molto.types.exceptions import UnsupportedInput

    adapter, _ = setup_adapter()
    request = AnthropicMessagesCreateRequest(
        model="m",
        max_tokens=100,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "name": "web_search",
                        "id": "s",
                        "input": {"query": "q"},
                        "caller": {"type": "code_execution_20250825", "tool_id": "exec"},
                    }
                ],
            }
        ],
    )
    with pytest.raises(UnsupportedInput, match="directly invoked"):
        adapter.normalize_request(request)
