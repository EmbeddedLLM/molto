"""Focused OpenAI hosted-search request and normalization regressions."""

from json import dumps, loads

import httpx
import pytest

from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    NormalizedRequest,
    PartEndEvent,
    PartStartEvent,
    SearchCall,
    SearchResult,
    TextOutput,
    WebCitation,
    WebSearchTool,
)
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient

CAPABILITIES = PROVIDER_CAPABILITIES["openai"]


def _response(output=None, **extra):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "output": output or [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        **extra,
    }


def _event(kind, **values):
    return {"type": kind, "sequence_number": 0, **values}


async def _normalize(wire, normalized_request=None, extra_body=None, expected_include=None):
    def replay(request):
        if expected_include is not None:
            assert loads(request.content)["include"] == expected_include
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {dumps(event)}\n\n" for event in wire),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(replay)) as http_client:
        client = OpenAIResponsesClient(
            http_client, capabilities=CAPABILITIES, api_key="test", extra_body=extra_body
        )
        return [
            event
            async for event in client.events(
                normalized_request
                or NormalizedRequest(model="m", tools=[WebSearchTool()], stream=True)
            )
        ]


@pytest.mark.parametrize(
    "action",
    [
        {"type": "search", "query": ""},
        {"type": "search"},
        {"type": "search", "queries": ["one", "two"]},
        {"type": "open_page", "url": "https://example.com"},
        {"type": "find_in_page", "url": "https://example.com", "pattern": "word"},
    ],
)
async def test_search_classification_waits_for_completed_action(action):
    """Ensures unclassified item-added snapshots never become speculative search operations."""
    initial = {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}
    final = {**initial, "status": "completed", "action": action}
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.output_item.added", item=initial, output_index=0),
            _event("response.web_search_call.in_progress", item_id="ws_1", output_index=0),
            _event("response.web_search_call.searching", item_id="ws_1", output_index=0),
            _event("response.output_item.done", item=final, output_index=0),
            _event("response.completed", response=_response([final])),
        ]
    )
    calls = [
        event.part
        for event in events
        if isinstance(event, PartStartEvent) and isinstance(event.part, SearchCall)
    ]
    assert len(calls) == 1
    assert calls[0].action == action["type"]
    assert calls[0].query == action.get("query")
    assert calls[0].queries == action.get("queries")
    results = [
        event.part
        for event in events
        if isinstance(event, PartEndEvent) and isinstance(event.part, SearchResult)
    ]
    assert len(results) == (action["type"] == "search")
    if results:
        assert results[0].sources is None


@pytest.mark.parametrize(
    "sources",
    [
        None,
        [],
        [{"type": "url", "url": "https://example.com", "title": "Source", "page_age": "today"}],
    ],
)
async def test_search_preserves_absent_empty_and_populated_sources(sources):
    """Ensures unavailable source data never becomes an invented empty successful result."""
    action = {"type": "search", "query": "q"}
    if sources is not None:
        action["sources"] = sources
    item = {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": action}
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.output_item.done", item=item, output_index=0),
            _event("response.completed", response=_response([item])),
        ]
    )
    result = next(
        event.part
        for event in events
        if isinstance(event, PartEndEvent) and isinstance(event.part, SearchResult)
    )
    assert (result.sources is None) == (sources is None)
    if sources:
        assert result.sources[0].title == "Source"
        assert result.sources[0].page_age == "today"
    elif sources == []:
        assert result.sources == []


@pytest.mark.parametrize("refusal_first", [False, True])
async def test_text_parts_keep_local_citation_offsets(refusal_first: bool):
    """Ensures text after refusal snapshots keeps dense, balanced indices and local citations."""
    wire = [
        _event("response.created", response=_response()),
        _event(
            "response.output_item.added",
            output_index=0,
            item={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        ),
    ]
    parts = []
    if refusal_first:
        refusal = {"type": "refusal", "refusal": "Cannot answer that part."}
        parts.append(refusal)
        fields = {"output_index": 0, "content_index": 0, "item_id": "msg_1"}
        wire.extend(
            [
                _event("response.content_part.added", **fields, part=refusal),
                _event("response.content_part.done", **fields, part=refusal),
            ]
        )
    for index, text in enumerate(["first", "second"], start=int(refusal_first)):
        citation = {
            "type": "url_citation",
            "url": "https://example.com",
            "title": "Source",
            "start_index": 0,
            "end_index": len(text),
        }
        part = {"type": "output_text", "text": text, "annotations": [citation], "logprobs": []}
        parts.append(part)
        fields = {"output_index": 0, "content_index": index, "item_id": "msg_1"}
        wire.extend(
            [
                _event(
                    "response.content_part.added",
                    **fields,
                    part={**part, "text": "", "annotations": []},
                ),
                _event("response.output_text.delta", **fields, delta=text, logprobs=[]),
                _event(
                    "response.output_text.annotation.added",
                    **fields,
                    annotation_index=0,
                    annotation=citation,
                ),
                _event("response.content_part.done", **fields, part=part),
            ]
        )
    item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": parts,
    }
    wire.extend(
        [
            _event("response.output_item.done", output_index=0, item=item),
            _event("response.completed", response=_response([item])),
        ]
    )
    events = await _normalize(wire)
    assert [event.index for event in events if isinstance(event, PartStartEvent)] == [0, 1]
    texts = [
        event
        for event in events
        if isinstance(event, PartEndEvent) and isinstance(event.part, TextOutput)
    ]
    assert [event.index for event in texts] == [0, 1]
    assert [event.part.content for event in texts] == ["first", "second"]
    assert [event.part.citations[0].start_index for event in texts] == [0, 0]
    assert [event.part.citations[0].end_index for event in texts] == [5, 6]
    assert all(event.part.defer_citations for event in texts)


@pytest.mark.parametrize("count", [None, 0, 3])
async def test_search_usage_retains_reported_count_only(count):
    """Ensures search counters distinguish unknown from observed zero without item counting."""
    response = _response(
        usage={
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3,
        }
    )
    if count is not None:
        response["tool_usage"] = {"web_search": {"num_requests": count}}
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.completed", response=response),
        ]
    )
    assert events[-1].usage.extra_fields.get("web_search_requests") == count
    assert ("web_search_requests" in events[-1].usage.extra_fields) == (count is not None)


@pytest.mark.parametrize("stream", [False, True])
async def test_native_chat_search_rejected_before_transport(stream):
    """Ensures native search Chat routes fail before constructing SDK transport."""
    async with httpx.AsyncClient() as http_client:
        client = OpenAIChatCompletionsClient(http_client, capabilities=CAPABILITIES)
        request = OpenAIChatCompletionsCreateRequest(
            model="gpt-5-search-api", messages=[], web_search_options={}, stream=stream
        )
        with pytest.raises(UnsupportedInput, match="Chat Completions upstream"):
            if stream:
                await anext(client.native_events(request))
            else:
                await client.native_response(request)


@pytest.mark.parametrize(
    "normalized_request",
    [
        NormalizedRequest(tools=[WebSearchTool()]),
        NormalizedRequest(input=[SearchCall(id="ws_1")]),
        NormalizedRequest(input=[SearchResult(tool_call_id="ws_1", sources=[])]),
        NormalizedRequest(
            input=[
                AssistantInput(
                    content=[
                        TextOutput(
                            content="answer",
                            citations=[
                                WebCitation(url="https://example.com", start_index=0, end_index=6)
                            ],
                        )
                    ]
                )
            ]
        ),
    ],
)
def test_normalized_chat_search_rejected(normalized_request):
    """Ensures hosted declarations and replayed search state cannot select Chat upstream."""
    with pytest.raises(UnsupportedInput, match="Chat Completions upstream"):
        OpenAIChatCompletionsClient._request_body(normalized_request)


def test_responses_rejects_explicit_domain_blocking():
    """Ensures even an explicit empty blocklist is rejected rather than silently discarded."""
    with pytest.raises(UnsupportedInput, match="blocked_domains"):
        OpenAIResponsesClient._request_body(
            NormalizedRequest(tools=[WebSearchTool(blocked_domains=[])])
        )


def test_standalone_result_history_defers_validity_to_api():
    """Ensures replayed search results retain their linked identity without local state validation."""
    body = OpenAIResponsesClient._request_body(
        NormalizedRequest(input=[SearchResult(tool_call_id="ws_1", sources=[])])
    )
    assert body["input"] == [
        {
            "id": "ws_1",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "sources": []},
        }
    ]


@pytest.mark.parametrize("split_at", [1, 2, 3, 4])
def test_split_search_history_preserves_assistant_text_boundaries(split_at):
    """Ensures split search history links results without merging assistant text runs or messages."""
    parts = [
        TextOutput(content="Before"),
        SearchCall(id="ws_1", query="x"),
        TextOutput(content="During"),
        SearchResult(tool_call_id="ws_1", sources=[]),
        TextOutput(content="After"),
    ]
    tail = AssistantInput(content=[TextOutput(content="Tail")])
    combined = OpenAIResponsesClient._request_body(
        NormalizedRequest(input=[AssistantInput(content=parts), tail])
    )
    split = OpenAIResponsesClient._request_body(
        NormalizedRequest(
            input=[
                AssistantInput(content=parts[:split_at]),
                AssistantInput(content=parts[split_at:]),
                tail,
            ]
        )
    )
    assert split == combined
    assert [item["type"] for item in split["input"]] == [
        "message",
        "web_search_call",
        "message",
        "message",
        "message",
    ]
    assert split["input"][1] == {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": "x", "sources": []},
    }
    assert [
        [part["text"] for part in item["content"]]
        for item in split["input"]
        if item["type"] == "message"
    ] == [["Before"], ["During"], ["After"], ["Tail"]]


@pytest.mark.parametrize("nested_call", [False, True])
@pytest.mark.parametrize("nested_result", [False, True])
def test_search_history_links_top_level_and_assistant_parts(nested_call, nested_result):
    """Ensures call/result linkage spans top-level parts and explicit assistant messages."""
    call = SearchCall(id="ws_1", query="x")
    result = SearchResult(tool_call_id="ws_1", error="unavailable")
    body = OpenAIResponsesClient._request_body(
        NormalizedRequest(
            input=[
                AssistantInput(content=[call]) if nested_call else call,
                AssistantInput(content=[result]) if nested_result else result,
            ]
        )
    )
    assert body["input"] == [
        {
            "id": "ws_1",
            "type": "web_search_call",
            "status": "failed",
            "action": {"type": "search", "query": "x"},
        }
    ]


async def test_unclassified_partial_search_does_not_invent_invocation():
    """Ensures an interrupted actionless item remains unclassified on translated routes."""
    item = {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.output_item.added", output_index=0, item=item),
            _event(
                "response.incomplete",
                response=_response([item], status="incomplete", incomplete_details=None),
            ),
        ]
    )
    assert not any(isinstance(event, PartStartEvent) for event in events)
    assert events[-1].termination_reason == "pause"


async def test_later_text_cannot_overtake_unclassified_search():
    """Ensures interleaved later output waits for an earlier hosted action's classification."""
    initial = {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}
    search = {
        **initial,
        "status": "completed",
        "action": {"type": "search", "query": "q", "sources": []},
    }
    text = {"type": "output_text", "text": "answer", "annotations": []}
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [text],
    }
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.output_item.added", output_index=0, item=initial),
            _event(
                "response.output_item.added",
                output_index=1,
                item={**message, "status": "in_progress", "content": []},
            ),
            _event(
                "response.content_part.added",
                output_index=1,
                content_index=0,
                item_id="msg_1",
                part={**text, "text": ""},
            ),
            _event(
                "response.output_text.delta",
                output_index=1,
                content_index=0,
                item_id="msg_1",
                delta="answer",
                logprobs=[],
            ),
            _event("response.output_item.done", output_index=0, item=search),
            _event(
                "response.content_part.done",
                output_index=1,
                content_index=0,
                item_id="msg_1",
                part=text,
            ),
            _event("response.output_item.done", output_index=1, item=message),
            _event("response.completed", response=_response([search, message])),
        ]
    )
    starts = [event for event in events if isinstance(event, PartStartEvent)]
    assert [type(event.part) for event in starts] == [SearchCall, SearchResult, TextOutput]
    assert [event.index for event in starts] == [0, 1, 2]


@pytest.mark.parametrize(
    "include",
    [
        ["reasoning.encrypted_content"],
        ["web_search_call.action.sources", "reasoning.encrypted_content"],
    ],
)
async def test_translated_search_merges_transport_include_without_duplicates(include):
    """Ensures SDK extra_body cannot remove required search sources or caller include values."""
    await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.completed", response=_response()),
        ],
        extra_body={"include": include},
        expected_include=["web_search_call.action.sources", "reasoning.encrypted_content"],
    )


@pytest.mark.parametrize(
    "history",
    [
        [SearchCall(id="ws_1")],
        [
            AssistantInput(
                content=[
                    TextOutput(
                        content="old",
                        citations=[
                            WebCitation(url="https://example.com", start_index=0, end_index=3)
                        ],
                    )
                ]
            )
        ],
    ],
)
async def test_search_history_without_tool_declaration_still_defers_citations(history):
    """Ensures cited follow-ups remain buffered even when the current turn omits tool declarations."""
    part = {"type": "output_text", "text": "answer", "annotations": []}
    item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [part],
    }
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.output_item.added", output_index=0, item={**item, "content": []}),
            _event(
                "response.content_part.added",
                output_index=0,
                content_index=0,
                item_id="msg_1",
                part={**part, "text": ""},
            ),
            _event(
                "response.content_part.done",
                output_index=0,
                content_index=0,
                item_id="msg_1",
                part=part,
            ),
            _event("response.output_item.done", output_index=0, item=item),
            _event("response.completed", response=_response([item])),
        ],
        normalized_request=NormalizedRequest(model="m", input=history, stream=True),
    )
    assert all(
        event.part.defer_citations
        for event in events
        if isinstance(event, (PartStartEvent, PartEndEvent)) and isinstance(event.part, TextOutput)
    )


async def test_reported_search_usage_without_token_usage_is_retained():
    """Ensures an independent search counter survives without inventing token usage."""
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event(
                "response.completed",
                response=_response(usage=None, tool_usage={"web_search": {"num_requests": 2}}),
            ),
        ]
    )
    assert events[-1].usage is None
    assert events[-1].extra_fields["web_search_requests"] == 2


async def test_null_search_usage_remains_unknown():
    """Ensures an explicitly null search-usage object never crashes or becomes zero."""
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.completed", response=_response(tool_usage={"web_search": None})),
        ]
    )
    assert "web_search_requests" not in events[-1].extra_fields


@pytest.mark.parametrize("stream", [False, True])
async def test_native_chat_extra_body_search_rejected(stream):
    """Ensures transport extensions cannot bypass the Chat search upstream exclusion."""
    async with httpx.AsyncClient() as http_client:
        client = OpenAIChatCompletionsClient(
            http_client, capabilities=CAPABILITIES, extra_body={"web_search_options": {}}
        )
        request = OpenAIChatCompletionsCreateRequest(model="m", messages=[], stream=stream)
        with pytest.raises(UnsupportedInput, match="Chat Completions upstream"):
            if stream:
                await anext(client.native_events(request))
            else:
                await client.native_response(request)


async def test_item_done_text_snapshot_rebases_local_citations():
    """Ensures completion-only snapshots rebase ranges when joining several text parts."""
    parts = [
        {
            "type": "output_text",
            "text": text,
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Source",
                    "start_index": 0,
                    "end_index": len(text),
                }
            ],
        }
        for text in ["first", "second"]
    ]
    item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": parts,
    }
    events = await _normalize(
        [
            _event("response.created", response=_response()),
            _event("response.output_item.added", output_index=0, item={**item, "content": []}),
            _event("response.output_item.done", output_index=0, item=item),
            _event("response.completed", response=_response([item])),
        ]
    )
    text = next(
        event.part
        for event in events
        if isinstance(event, PartEndEvent) and isinstance(event.part, TextOutput)
    )
    assert text.content == "firstsecond"
    assert [(citation.start_index, citation.end_index) for citation in text.citations] == [
        (0, 5),
        (5, 11),
    ]
