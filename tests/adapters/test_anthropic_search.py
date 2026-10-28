"""Messages hosted-search projection and request regressions."""

import pytest

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    FunctionToolChoice,
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
from molto.utils.web_search import (
    messages_text_blocks,
    search_calls_to_messages,
)


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


@pytest.mark.parametrize("hosted_first", [True, False])
def test_rejects_ambiguous_named_search_selection(hosted_first):
    """Ensures a shared name cannot silently select hosted search over a client function."""
    tools = [
        {"type": "web_search_20250305", "name": "web_search"},
        {"name": "web_search", "input_schema": {"type": "object"}},
    ]
    if not hosted_first:
        tools.reverse()
    adapter, _ = setup_adapter()
    with pytest.raises(UnsupportedInput, match="ambiguous.*Rename the client function"):
        adapter.normalize_request(
            AnthropicMessagesCreateRequest(
                model="m",
                max_tokens=100,
                messages=[{"role": "user", "content": "Search"}],
                tools=tools,
                tool_choice={"type": "tool", "name": "web_search"},
            )
        )


@pytest.mark.parametrize(
    ("hosted", "function_names", "choice", "expected"),
    [
        (True, ("different",), {"type": "tool", "name": "web_search"}, WebSearchToolChoice()),
        (
            False,
            ("web_search",),
            {"type": "tool", "name": "web_search"},
            FunctionToolChoice(name="web_search"),
        ),
        (
            True,
            ("web_search", "different"),
            {"type": "tool", "name": "different"},
            FunctionToolChoice(name="different"),
        ),
        (True, ("web_search",), None, None),
        (True, ("web_search",), {"type": "auto"}, None),
        (True, ("web_search",), {"type": "any"}, "required"),
        (True, ("web_search",), {"type": "none"}, "none"),
    ],
)
def test_unambiguous_search_selection(hosted, function_names, choice, expected):
    """Ensures distinct named choices and general selection retain their execution semantics."""
    tools = [{"name": name, "input_schema": {"type": "object"}} for name in function_names]
    if hosted:
        tools.insert(0, {"type": "web_search_20250305", "name": "web_search"})
    adapter, _ = setup_adapter()
    request = adapter.normalize_request(
        AnthropicMessagesCreateRequest(
            model="m",
            max_tokens=100,
            messages=[{"role": "user", "content": "Search"}],
            tools=tools,
            tool_choice=choice,
        )
    )
    assert request.tool_choice == expected
    assert len(request.tools) == len(tools)


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
    assert (
        search_calls_to_messages(SearchCall(id="s", query=query, queries=queries))[0]["input"]
        == expected
    )


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


async def test_multi_query_expansion_closes_source():
    """Ensures each query has a unique call while the original batch result appears once."""
    adapter, request = setup_adapter()
    closed = False
    part = SearchCall(id="multi", query="fallback", queries=["one", "", "one"])
    result = SearchResult(
        tool_call_id="multi", sources=[WebSearchSource(url="https://python.org/")]
    )

    async def source():
        nonlocal closed
        try:
            yield ResponseStartEvent(id="r", model="m")
            yield PartStartEvent(index=0, part=part)
            yield PartEndEvent(index=0, part=part)
            yield PartStartEvent(index=1, part=result)
            yield PartEndEvent(index=1, part=result)
            yield ResponseEndEvent(id="r", model="m", created_at=0, termination_reason="stop")
        finally:
            closed = True

    projected = [event async for event in adapter.project_events(source(), request)]
    starts = [event for event in projected if event.type == "content_block_start"]
    assert [event.index for event in starts] == list(range(4))
    calls = [event.content_block for event in starts[:3]]
    assert [call.input for call in calls] == [{"query": "one"}, {"query": ""}, {"query": "one"}]
    assert len({call.id for call in calls}) == 3
    assert calls[0].id == starts[-1].content_block.tool_use_id == "multi"
    assert [source.url for source in starts[-1].content_block.content] == ["https://python.org/"]
    assert [event.index for event in projected if event.type == "content_block_stop"] == list(
        range(4)
    )
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


@pytest.mark.parametrize(
    "text, ranges, expected_text, expected_sources",
    [
        ("abc de f", [(0, 5), (6, 7)], ["abc d", "e ", "f"], [[0], [1], []]),
        ("abc de f", [(0, 6), (7, 8)], ["abc de ", "f"], [[0], [1]]),
        (" \ta", [(2, 3)], [" \ta"], [[0]]),
        ("a \t", [(0, 1)], ["a \t"], [[0]]),
        ("a b", [(0, 2), (1, 3)], ["a ", "b"], [[0, 1], [1]]),
        ("a \t b", [(0, 1), (2, 3), (4, 5)], ["a \t ", "b"], [[0, 1], [2]]),
        ("a \tb", [(0, 1), (3, 4), (1, 1)], ["a \t", "b"], [[0, 2], [1]]),
        ("a \tb", [(0, 1), (3, 4), (2, 2)], ["a \t", "b"], [[0, 2], [1]]),
        ("a \tb", [(0, 1), (3, 4), (3, 3)], ["a \t", "b"], [[0], [1, 2]]),
    ],
)
def test_citation_boundaries_preserve_nonblank_blocks(
    text, ranges, expected_text, expected_sources
):
    """Ensures cited whitespace and blank gaps keep exact text and attribution in valid blocks."""
    part = TextOutput(
        content=text,
        citations=[
            WebCitation(url=f"https://source/{index}", start_index=start, end_index=end)
            for index, (start, end) in enumerate(ranges)
        ],
    )
    blocks = messages_text_blocks(part)

    assert [block["text"] for block in blocks] == expected_text
    assert "".join(block["text"] for block in blocks) == text
    assert all(block["text"].strip() for block in blocks)
    assert [[citation["url"] for citation in block.get("citations", [])] for block in blocks] == [
        [f"https://source/{index}" for index in sources] for sources in expected_sources
    ]
    for block, sources in zip(blocks, expected_sources, strict=True):
        assert [citation["cited_text"] for citation in block.get("citations", [])] == [
            text[ranges[index][0] : ranges[index][1]] for index in sources
        ]
    AnthropicMessagesCreateRequest.model_validate(
        {"model": "m", "max_tokens": 100, "messages": [{"role": "assistant", "content": blocks}]}
    )


@pytest.mark.parametrize(
    "text, offset, expected_text, citation_index",
    [
        ("abc", 0, ["abc"], 0),
        ("abc", 1, ["a", "bc"], 1),
        ("abc", 3, ["abc"], 0),
    ],
)
def test_point_citation_preserves_attribution(text, offset, expected_text, citation_index):
    """Ensures point attribution survives on a nonblank block, including at the answer's end."""
    blocks = messages_text_blocks(
        TextOutput(
            content=text,
            citations=[WebCitation(url="https://x", start_index=offset, end_index=offset)],
        )
    )
    assert [block["text"] for block in blocks] == expected_text
    assert "".join(block["text"] for block in blocks) == text
    assert all(block["text"].strip() for block in blocks)
    assert blocks[citation_index]["citations"][0]["url"] == "https://x"
    assert blocks[citation_index]["citations"][0]["cited_text"] == ""
    assert all(
        "citations" not in block for index, block in enumerate(blocks) if index != citation_index
    )
    AnthropicMessagesCreateRequest.model_validate(
        {"model": "m", "max_tokens": 100, "messages": [{"role": "assistant", "content": blocks}]}
    )


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
@pytest.mark.parametrize("cited", [False, True])
def test_blank_text_cannot_be_projected_losslessly(text, cited):
    """Ensures an entirely blank part fails explicitly instead of emitting invalid Messages text."""
    part = TextOutput(
        content=text,
        citations=[WebCitation(url="https://x", start_index=0, end_index=0)] if cited else [],
    )
    with pytest.raises(ValueError, match="non-whitespace"):
        messages_text_blocks(part)


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
