"""Messages hosted-search projection and request regressions."""

import pytest

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    CitationDelta,
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
    search_call_from_messages,
    search_calls_to_messages,
)


def setup_adapter(citation_policy="drop"):
    adapter = AnthropicMessagesAdapter(
        PROVIDER_CAPABILITIES["anthropic"], citation_policy=citation_policy
    )
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


@pytest.mark.parametrize("citation_policy", ["buffer", "drop", "auto"])
async def test_citation_policy_controls_text_latency_without_search(citation_policy):
    """Ensures buffer waits for ordinary text while auto and drop preserve immediate deltas."""
    adapter, request = setup_adapter(citation_policy)
    completed = False

    async def source():
        nonlocal completed
        yield ResponseStartEvent(id="r", model="m")
        yield PartStartEvent(index=0, part=TextOutput(content=""))
        yield PartDeltaEvent(index=0, delta=TextDelta(content_delta="Alpha. "))
        yield PartDeltaEvent(index=0, delta=TextDelta(content_delta="Beta."))
        completed = True
        yield PartEndEvent(index=0, part=TextOutput(content="Alpha. Beta."))
        yield ResponseEndEvent(id="r", model="m", created_at=0, termination_reason="stop")

    stream = adapter.project_events(source(), request)
    assert (await anext(stream)).type == "message_start"
    start = await anext(stream)
    assert start.type == "content_block_start"
    assert completed is (citation_policy == "buffer")
    remaining = [event async for event in stream]
    deltas = [event.delta.text for event in remaining if event.type == "content_block_delta"]
    assert start.content_block.text == ""
    if citation_policy == "buffer":
        assert deltas == ["Alpha. Beta."]
    else:
        assert deltas == ["Alpha. ", "Beta."]


@pytest.mark.parametrize("citation_policy", ["buffer", "drop", "auto"])
@pytest.mark.parametrize("citation_delivery", ["start", "delta", "end"])
async def test_explicit_citation_policy_handles_all_citation_delivery(
    citation_policy, citation_delivery
):
    """Ensures buffer preserves attribution while drop and unsignaled auto omit unexpected citations."""
    adapter, request = setup_adapter(citation_policy)
    citation = WebCitation(url="https://a", start_index=7, end_index=12)
    events = [
        ResponseStartEvent(id="r", model="m"),
        PartStartEvent(
            index=0,
            part=TextOutput(
                content="", citations=[citation] if citation_delivery == "start" else []
            ),
        ),
        PartDeltaEvent(index=0, delta=TextDelta(content_delta="Alpha. Beta.")),
    ]
    if citation_delivery == "delta":
        events.append(PartDeltaEvent(index=0, delta=CitationDelta(citation=citation)))
    events += [
        PartEndEvent(index=0, part=TextOutput(content="Alpha. Beta.", citations=[citation])),
        ResponseEndEvent(id="r", model="m", created_at=0, termination_reason="stop"),
    ]
    response = await adapter.project_response(
        adapter.project_events(iterate(events), request), request
    )
    assert "".join(block.text for block in response.content) == "Alpha. Beta."
    if citation_policy == "buffer":
        assert [block.text for block in response.content] == ["Alpha. ", "Beta."]
        assert response.content[-1].citations[0].cited_text == "Beta."
    else:
        assert len(response.content) == 1
        assert not response.content[0].citations


def test_citation_policy_rejects_unknown_mode():
    """Ensures an unsupported policy cannot silently discard requested attribution."""
    with pytest.raises(ValueError, match="Unsupported citation policy"):
        setup_adapter(citation_policy="invalid")


@pytest.mark.parametrize("signal", ["tool", "search_call", "search_result", "citation"])
async def test_auto_buffers_text_when_request_announces_citations(signal):
    """Ensures tools or history select auto buffering before text and late citations arrive."""
    adapter, request = setup_adapter("auto")
    citation = WebCitation(url="https://a", start_index=7, end_index=12)
    if signal == "tool":
        request.tools = [WebSearchTool()]
    else:
        part = {
            "search_call": SearchCall(id="s", query="q"),
            "search_result": SearchResult(tool_call_id="s", sources=[]),
            "citation": TextOutput(content="Alpha. Beta.", citations=[citation]),
        }[signal]
        request.input = [AssistantInput(content=[part])]
    completed = False

    async def source():
        nonlocal completed
        yield ResponseStartEvent(id="r", model="m")
        yield PartStartEvent(index=0, part=TextOutput(content=""))
        yield PartDeltaEvent(index=0, delta=TextDelta(content_delta="Alpha. "))
        yield PartDeltaEvent(index=0, delta=TextDelta(content_delta="Beta."))
        yield PartDeltaEvent(index=0, delta=CitationDelta(citation=citation))
        completed = True
        yield PartEndEvent(index=0, part=TextOutput(content="Alpha. Beta.", citations=[citation]))
        yield ResponseEndEvent(id="r", model="m", created_at=0, termination_reason="stop")

    stream = adapter.project_events(source(), request)
    assert (await anext(stream)).type == "message_start"
    start = await anext(stream)
    assert start.type == "content_block_start"
    assert completed
    events = [start, *[event async for event in stream]]
    blocks = [event.content_block for event in events if event.type == "content_block_start"]
    assert len(blocks) == 2
    assert not blocks[0].citations
    assert blocks[1].citations[0].cited_text == "Beta."
    assert [event.delta.text for event in events if event.type == "content_block_delta"] == [
        "Alpha. ",
        "Beta.",
    ]


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
    adapter, request = setup_adapter(citation_policy="buffer")
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
        citations=[WebCitation(url="https://a", start_index=7, end_index=12)],
    )
    events += [
        PartStartEvent(index=4, part=TextOutput(content="")),
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
        web_search_requests=count,
        input_tokens=1,
        input_tokens_details=InputTokenDetails(),
        output_tokens=1,
        output_tokens_details=OutputTokenDetails(),
        total_tokens=2,
    )
    end = ResponseEndEvent(
        id="r",
        model="m",
        created_at=0,
        termination_reason="stop",
        usage=usage if has_token_usage else None,
    )
    events = [ResponseStartEvent(id="r", model="m"), end]
    response = await adapter.project_response(
        adapter.project_events(iterate(events), request), request
    )
    if not has_token_usage or count is None:
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
            text[ranges[index][0] : ranges[index][1]] or block["text"] for index in sources
        ]
    AnthropicMessagesCreateRequest.model_validate(
        {"model": "m", "max_tokens": 100, "messages": [{"role": "assistant", "content": blocks}]}
    )


@pytest.mark.parametrize(
    "text, offset, expected_text, citation_index",
    [
        ("abc", 0, ["abc"], 0),
        ("abc", 1, ["abc"], 0),
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
    assert blocks[citation_index]["citations"][0]["cited_text"] == blocks[citation_index]["text"]
    assert all(
        "citations" not in block for index, block in enumerate(blocks) if index != citation_index
    )
    AnthropicMessagesCreateRequest.model_validate(
        {"model": "m", "max_tokens": 100, "messages": [{"role": "assistant", "content": blocks}]}
    )


@pytest.mark.parametrize("offset", [-1, 4])
def test_point_citation_rejects_offsets_outside_text(offset):
    """Ensures excluding point citations from splitting does not bypass offset validation."""
    part = TextOutput(
        content="abc",
        citations=[WebCitation(url="https://x", start_index=offset, end_index=offset)],
    )
    with pytest.raises(ValueError, match="Citation offsets fall outside"):
        messages_text_blocks(part)


def test_point_citation_preserves_observed_source_state():
    """Ensures a point's observed source excerpt and encrypted reference beat answer fallbacks."""
    blocks = messages_text_blocks(
        TextOutput(
            content="abc",
            citations=[
                WebCitation(
                    url="https://x",
                    start_index=1,
                    end_index=1,
                    cited_text="source excerpt",
                    encrypted_index="original-reference",
                )
            ],
        )
    )
    assert len(blocks) == 1
    assert blocks[0]["text"] == "abc"
    assert blocks[0]["citations"][0]["cited_text"] == "source excerpt"
    assert blocks[0]["citations"][0]["encrypted_index"] == "original-reference"


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
    adapter, request = setup_adapter(citation_policy="buffer")
    closed = False
    part = SearchCall(id="s", query="q") if kind == "search" else TextOutput(content="answer")

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


@pytest.mark.parametrize("search_input", [["a"], None, {"query": "q", "extra": True}])
def test_search_history_conversion_remains_strict(search_input):
    """Ensures request-side conversion rejects non-object inputs and unknown argument keys."""
    with pytest.raises(UnsupportedInput, match="web search input fields"):
        search_call_from_messages({"id": "s", "input": search_input})


def test_search_history_rejects_unsupported_input():
    """Ensures the request adapter rejects unknown search arguments with UnsupportedInput."""
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
                        "input": {"query": "q", "extra": True},
                    }
                ],
            }
        ],
    )
    with pytest.raises(UnsupportedInput, match="web search input fields"):
        adapter.normalize_request(request)


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
