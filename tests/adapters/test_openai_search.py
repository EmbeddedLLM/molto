"""Focused hosted-search request and output projection regressions."""

import pytest

from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    CitationDelta,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    SearchCall,
    SearchResult,
    TextOutput,
    WebCitation,
    WebSearchSource,
    WebSearchTool,
    WebSearchToolChoice,
)
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.openai import OpenAIResponsesClient

CAPABILITIES = PROVIDER_CAPABILITIES["openai"]


async def _events(parts, reason="stop"):
    yield ResponseStartEvent(id="resp_1", model="m")
    for index, part in enumerate(parts):
        yield PartStartEvent(index=index, part=part)
        yield PartEndEvent(index=index, part=part)
    yield ResponseEndEvent(id="resp_1", created_at=1, model="m", termination_reason=reason)


@pytest.mark.parametrize("mixed", [False, True])
def test_chat_search_requires_only_hosted_tool(mixed):
    """Ensures client functions cannot satisfy Chat's required hosted-search selection."""
    request = OpenAIChatCompletionsCreateRequest(
        model="ordinary-search-capable-model",
        messages=[{"role": "user", "content": "Find"}],
        web_search_options={
            "search_context_size": "low",
            "user_location": {
                "type": "approximate",
                "approximate": {"city": "Boston", "country": "US"},
            },
        },
        tools=[{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
        if mixed
        else None,
    )
    normalized = OpenAIChatCompletionsAdapter(CAPABILITIES).normalize_request(request)
    assert isinstance(normalized.tool_choice, WebSearchToolChoice)
    body = OpenAIResponsesClient._native_request(normalized).model_dump()
    assert body["model"] == request.model
    assert body["tools"][-1] == {
        "type": "web_search",
        "search_context_size": "low",
        "user_location": {"type": "approximate", "city": "Boston", "country": "US"},
    }
    assert body["include"] == ["web_search_call.action.sources"]
    assert body["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "web_search"}],
    }


@pytest.mark.parametrize(
    "choice", ["auto", "none", {"type": "function", "function": {"name": "client"}}]
)
def test_chat_search_rejects_conflicting_choice(choice):
    """Ensures explicit incompatible tool controls fail before upstream transport."""
    with pytest.raises(UnsupportedInput, match="conflicts with tool_choice"):
        OpenAIChatCompletionsAdapter(CAPABILITIES).normalize_request(
            OpenAIChatCompletionsCreateRequest(
                model="m", messages=[], web_search_options={}, tool_choice=choice
            )
        )


@pytest.mark.parametrize("anthropic_extra_content", [False, True])
def test_responses_search_normalizes_options_budget_and_history(anthropic_extra_content):
    """Ensures search history and Anthropic extra_content normalize into the expected IR."""
    request = OpenAIResponsesCreateRequest(
        model="m",
        max_tool_calls=3,
        tools=[{"type": "web_search", "filters": {"allowed_domains": ["example.com"]}}],
        input=[
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "",
                    "sources": [{"type": "url", "url": "https://example.com"}],
                },
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com",
                                "title": "Source",
                                "start_index": 0,
                                "end_index": 6,
                            }
                        ],
                    }
                ],
            },
        ],
    )
    if anthropic_extra_content:
        history = request.model_dump()["input"]
        history[0]["action"]["sources"][0]["extra_content"] = {
            "anthropic": {
                "encrypted_content": "opaque-source-state",
                "title": "Original source title",
                "page_age": "2026-10-01",
            }
        }
        history[1]["content"][0]["annotations"][0]["extra_content"] = {
            "anthropic": {
                "encrypted_index": "opaque-citation-index",
                "cited_text": "Original source excerpt",
            }
        }
        request = OpenAIResponsesCreateRequest(**{**request.model_dump(), "input": history})

    normalized = OpenAIResponsesAdapter(CAPABILITIES).normalize_request(request)
    assert normalized.input == [
        SearchCall(id="ws_1", query="", status="completed"),
        SearchResult(
            tool_call_id="ws_1",
            sources=[
                WebSearchSource(
                    url="https://example.com",
                    encrypted_content="opaque-source-state" if anthropic_extra_content else None,
                    title="Original source title" if anthropic_extra_content else None,
                    page_age="2026-10-01" if anthropic_extra_content else None,
                )
            ],
        ),
        AssistantInput(
            content=[
                TextOutput(
                    content="answer",
                    citations=[
                        WebCitation(
                            url="https://example.com",
                            title="Source",
                            start_index=0,
                            end_index=6,
                            encrypted_index="opaque-citation-index"
                            if anthropic_extra_content
                            else None,
                            cited_text="Original source excerpt"
                            if anthropic_extra_content
                            else None,
                        )
                    ],
                )
            ]
        ),
    ]
    assert normalized.tools == [WebSearchTool(allowed_domains=["example.com"])]
    assert normalized.max_tool_calls == 3
    body = OpenAIResponsesClient._native_request(normalized).model_dump()
    assert body["input"][0]["action"]["query"] == ""
    assert body["input"][1]["content"][0]["annotations"][0]["end_index"] == 6


@pytest.mark.parametrize(
    "source",
    [
        {"type": "api", "name": "oai-weather"},
        {"type": "api", "name": "oai-weather", "url": "https://example.com"},
        {"type": "url"},
        {"type": "url", "url": 1},
        {"type": "url", "url": "https://example.com", "provider_field": True},
        {"name": "untyped"},
    ],
)
def test_responses_search_history_rejects_untranslatable_sources(source):
    """Ensures translated history rejects sources it cannot convert losslessly.

    Native passthrough leaves source validation to the upstream, but translated input is
    lossless-or-rejected. A non-URL type, a missing or malformed URL, or an unconverted
    field would otherwise be dropped or misreported, so each raises `UnsupportedInput`.
    """
    request = OpenAIResponsesCreateRequest(
        model="m",
        input=[
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "weather: Taiwan",
                    "sources": [source],
                },
            }
        ],
    )

    with pytest.raises(UnsupportedInput, match="Only URL web search sources"):
        OpenAIResponsesAdapter(CAPABILITIES).normalize_request(request)


@pytest.mark.parametrize(
    "source",
    [
        {"type": "url", "url": "https://example.com"},
        {"url": "https://example.com"},
        {"type": "url", "url": "https://example.com", "title": "Example", "page_age": "1 day"},
    ],
)
def test_responses_search_history_translates_convertible_url_sources(source):
    """Ensures URL sources translate with or without `type`, keeping every converted field."""
    request = OpenAIResponsesCreateRequest(
        model="m",
        input=[
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "news", "sources": [source]},
            }
        ],
    )

    normalized = OpenAIResponsesAdapter(CAPABILITIES).normalize_request(request)

    (result,) = [part for part in normalized.input if isinstance(part, SearchResult)]
    (converted,) = result.sources
    assert converted.url == source["url"]
    assert (converted.title, converted.page_age) == (source.get("title"), source.get("page_age"))


@pytest.mark.parametrize("error", [None, "max_uses_exceeded", "too_many_requests"])
async def test_search_result_completes_linked_response_item(error):
    """Ensures hosted results close their invocation with observed success or failure."""
    adapter = OpenAIResponsesAdapter(CAPABILITIES)
    request = adapter.normalize_request(
        OpenAIResponsesCreateRequest(model="m", tools=[{"type": "web_search"}])
    )
    events = [
        event
        async for event in adapter.project_events(
            _events(
                [
                    SearchCall(id="ws_1", query="q"),
                    SearchResult(
                        tool_call_id="ws_1", sources=[] if error is None else None, error=error
                    ),
                    TextOutput(content="answer"),
                ]
            ),
            request,
        )
    ]
    searches = [
        event.item
        for event in events
        if event.type == "response.output_item.done" and event.item.type == "web_search_call"
    ]
    assert len(searches) == 1
    assert searches[0].id == "ws_1"
    assert searches[0].status == ("completed" if error is None else "failed")
    assert [
        event.output_index for event in events if event.type == "response.output_item.added"
    ] == [0, 1]
    assert sum(event.type == "response.web_search_call.completed" for event in events) == (
        error is None
    )
    assert events[-1].response.status == "completed"


async def test_paused_search_remains_incomplete_without_fake_result():
    """Ensures pause_turn preserves partial searches and null Responses incomplete details."""
    adapter = OpenAIResponsesAdapter(CAPABILITIES)
    request = adapter.normalize_request(
        OpenAIResponsesCreateRequest(model="m", tools=[{"type": "web_search"}])
    )
    events = [
        event
        async for event in adapter.project_events(
            _events([SearchCall(id="ws_1", query="q")], "pause"), request
        )
    ]
    assert events[-1].type == "response.incomplete"
    assert events[-1].response.incomplete_details is None
    assert events[-1].response.output[0].status == "in_progress"
    assert not any(event.type == "response.web_search_call.completed" for event in events)


@pytest.mark.parametrize("answer", ["answer", None])
async def test_chat_search_failure_omits_hosted_blocks_and_maps_pause(answer):
    """Ensures Chat preserves real answer text or absent content after a failed paused search."""
    adapter = OpenAIChatCompletionsAdapter(CAPABILITIES)
    request = adapter.normalize_request(
        OpenAIChatCompletionsCreateRequest(model="m", messages=[], web_search_options={})
    )
    parts = [
        SearchCall(id="ws_1", query="q"),
        SearchResult(tool_call_id="ws_1", error="unavailable"),
    ]
    if answer is not None:
        parts.append(TextOutput(content=answer))
    response = await adapter.project_response(
        adapter.project_events(_events(parts, "pause"), request), request
    )
    assert response.choices[0].message.content == answer
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].message.annotations is None
    assert response.choices[0].finish_reason == "length"


async def test_responses_citation_delta_matches_final_annotation():
    """Ensures streamed citation indices, missing titles, and terminal annotations agree."""
    adapter = OpenAIResponsesAdapter(CAPABILITIES)
    request = adapter.normalize_request(OpenAIResponsesCreateRequest(model="m"))
    citation = WebCitation(url="https://example.com", start_index=0, end_index=3)

    async def source():
        yield ResponseStartEvent(id="resp_1", model="m")
        yield PartStartEvent(index=0, part=TextOutput(content="abc"))
        yield PartDeltaEvent(index=0, delta=CitationDelta(citation=citation))
        yield PartEndEvent(index=0, part=TextOutput(content="abc", citations=[citation]))
        yield ResponseEndEvent(id="resp_1", created_at=1, model="m", termination_reason="stop")

    events = [event async for event in adapter.project_events(source(), request)]
    annotations = [
        event for event in events if event.type == "response.output_text.annotation.added"
    ]
    assert len(annotations) == 1
    assert annotations[0].annotation_index == 0
    assert annotations[0].annotation["title"] == ""
    assert events[-1].response.output[0].content[0].annotations[0].title == ""


def test_responses_history_rejects_non_web_annotations():
    """Ensures translation cannot silently discard a caller's file citation semantics."""
    request = OpenAIResponsesCreateRequest(
        model="m",
        input=[
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [
                            {
                                "type": "file_citation",
                                "file_id": "file_1",
                                "filename": "source.txt",
                                "index": 0,
                            }
                        ],
                    }
                ],
            }
        ],
    )
    with pytest.raises(UnsupportedInput, match="Only web URL citations"):
        OpenAIResponsesAdapter(CAPABILITIES).normalize_request(request)
