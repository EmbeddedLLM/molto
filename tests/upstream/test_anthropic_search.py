"""Hosted-search Messages request and stream normalization regressions."""

from pathlib import Path

import httpx
import pytest

from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesStreamEventAdapter
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    CitationDelta,
    FunctionTool,
    FunctionToolChoice,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    SearchCall,
    SearchLocation,
    SearchResult,
    TextInput,
    TextOutput,
    UserInput,
    WebCitation,
    WebSearchSource,
    WebSearchTool,
    WebSearchToolChoice,
)
from molto.upstream.anthropic import AnthropicMessagesClient
from tests.core.test_interface import _get_http_client, _read_cassette


async def test_recorded_search_citation_ir_has_nonblank_text():
    """Ensures completed recorded citation IR contains nonblank text and nonblank ranges."""
    path = (
        Path(__file__).parents[1]
        / "cassettes"
        / "anthropic"
        / "web_search"
        / "anthropic-messages-stream-00.yaml"
    )
    native = _read_cassette(path)["request"]["body"]
    request = NormalizedRequest(
        model=native["model"],
        input=[
            UserInput(content=[TextInput(content=native["messages"][0]["content"][0]["text"])])
        ],
        tools=[WebSearchTool()],
        tool_choice="required",
        max_tool_calls=1,
        max_output_tokens=native["max_tokens"],
        stream=True,
    )
    async with _get_http_client([path], "messages") as http_client:
        client = AnthropicMessagesClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["anthropic"], api_key="test"
        )
        events = [event async for event in client.events(request)]

    cited_parts = [
        event.part
        for event in events
        if isinstance(event, PartEndEvent)
        and isinstance(event.part, TextOutput)
        and event.part.citations
    ]
    assert cited_parts
    for part in cited_parts:
        assert part.content.strip()
        for citation in part.citations:
            assert 0 <= citation.start_index <= citation.end_index <= len(part.content)
            if citation.start_index != citation.end_index:
                assert part.content[citation.start_index : citation.end_index].strip()


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
@pytest.mark.parametrize("cited", [False, True])
def test_blank_text_history_is_an_input_error(text, cited):
    """Ensures unrepresentable blank history fails before sending an invalid Messages request."""
    part = TextOutput(
        content=text,
        citations=[WebCitation(url="https://example.com", start_index=0, end_index=len(text))]
        if cited
        else [],
    )
    request = NormalizedRequest(
        model="m", max_output_tokens=100, input=[AssistantInput(content=[part])]
    )
    with pytest.raises(UnsupportedInput, match="non-whitespace"):
        AnthropicMessagesClient._request_body(request)


@pytest.mark.parametrize(
    "part",
    [
        TextOutput(
            content="answer",
            citations=[WebCitation(url="https://example.com", start_index=0, end_index=99)],
        ),
        TextOutput(
            content="answer",
            citations=[WebCitation(url="https://example.com", start_index=4, end_index=2)],
        ),
    ],
)
def test_unrepresentable_search_history_is_an_input_error(part):
    """Ensures known history restrictions raise UnsupportedInput directly without wrapping."""
    request = NormalizedRequest(
        model="m", max_output_tokens=100, input=[AssistantInput(content=[part])]
    )
    with pytest.raises(UnsupportedInput) as caught:
        AnthropicMessagesClient._request_body(request)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("queries", [["first", "second"], ["first", "", "first"]])
def test_multi_query_history_is_rejected(nested: bool, queries: list[str]):
    """Ensures completed query batches cannot become unmatched server calls in follow-up history."""
    parts = [
        SearchCall(id="ws_1", query="fallback", queries=queries, status="completed"),
        SearchResult(tool_call_id="ws_1", sources=[]),
    ]
    history = [AssistantInput(content=parts)] if nested else parts
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[*history, UserInput(content=[TextInput(content="Please continue.")])],
    )
    with pytest.raises(UnsupportedInput, match="multi-query search history"):
        AnthropicMessagesClient._request_body(request)


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize(
    ("queries", "expected_query"),
    [(None, "fallback"), ([], "fallback"), (["first"], "first"), ([""], "")],
)
def test_single_query_history_preserves_result_pairing(nested, queries, expected_query):
    """Ensures singular and empty query metadata preserve one completed call and its result."""
    parts = [
        SearchCall(id="ws_1", query="fallback", queries=queries, status="completed"),
        SearchResult(tool_call_id="ws_1", sources=[]),
    ]
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[AssistantInput(content=parts)] if nested else parts,
    )
    body = AnthropicMessagesClient._request_body(request)
    assert body["messages"][0]["content"] == [
        {
            "type": "server_tool_use",
            "id": "ws_1",
            "name": "web_search",
            "input": {"query": expected_query},
        },
        {"type": "web_search_tool_result", "tool_use_id": "ws_1", "content": []},
    ]


def test_history_projection_preserves_unexpected_value_error(monkeypatch):
    """Ensures unexpected helper failures are not relabeled as unsupported caller input."""
    error = ValueError("unexpected projection failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("molto.upstream.anthropic.messages_text_blocks", fail)
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[AssistantInput(content=[TextOutput(content="answer")])],
    )
    with pytest.raises(ValueError) as caught:
        AnthropicMessagesClient._request_body(request)
    assert caught.value is error


def test_request_search_options_and_history():
    """Ensures target-native tools, budget, selection and reduced history are preserved."""
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        max_tool_calls=3,
        tools=[
            WebSearchTool(
                allowed_domains=["x.test"],
                user_location=SearchLocation(city="Paris", country="FR", timezone="Europe/Paris"),
            )
        ],
        tool_choice=WebSearchToolChoice(),
        input=[
            AssistantInput(
                content=[
                    SearchCall(id="s", query="q"),
                    SearchResult(
                        tool_call_id="s", sources=[WebSearchSource(url="https://x.test")]
                    ),
                    TextOutput(
                        content="Answer",
                        citations=[
                            WebCitation(
                                url="https://x.test",
                                start_index=0,
                                end_index=6,
                                encrypted_index="N/A",
                            )
                        ],
                    ),
                ]
            )
        ],
    )
    body = AnthropicMessagesClient._request_body(request)
    assert body["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
            "allowed_domains": ["x.test"],
            "user_location": {
                "type": "approximate",
                "city": "Paris",
                "country": "FR",
                "timezone": "Europe/Paris",
            },
        }
    ]
    assert body["tool_choice"] == {"type": "tool", "name": "web_search"}
    call, result, text = body["messages"][0]["content"]
    assert call["input"] == {"query": "q"}
    assert result["content"][0]["encrypted_content"] == "N/A"
    assert text["citations"][0]["encrypted_index"] == "N/A"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "vllm"])
@pytest.mark.parametrize("state", [None, "", "N/A", "original-opaque-state"])
@pytest.mark.parametrize("part_kind", ["citation", "source", "nested_source"])
def test_search_history_requires_state_only_for_anthropic(provider, state, part_kind):
    """Ensures only Anthropic rejects missing state while preserving opaque values and compatible routes."""
    if part_kind == "citation":
        part = AssistantInput(
            content=[
                TextOutput(
                    content="answer",
                    citations=[
                        WebCitation(
                            url="https://example.com",
                            start_index=0,
                            end_index=6,
                            encrypted_index=state,
                        )
                    ],
                )
            ]
        )
        field = "encrypted_index"
    else:
        part = SearchResult(
            tool_call_id="ws_1",
            sources=[WebSearchSource(url="https://example.com", encrypted_content=state)],
        )
        if part_kind == "nested_source":
            part = AssistantInput(content=[part])
        field = "encrypted_content"
    request = NormalizedRequest(model="m", max_output_tokens=100, input=[part])
    capabilities = PROVIDER_CAPABILITIES[provider]
    if provider == "anthropic" and state in (None, "", "N/A"):
        with pytest.raises(UnsupportedInput, match=field):
            AnthropicMessagesClient._request_body(request, capabilities=capabilities)
    else:
        body = AnthropicMessagesClient._request_body(request, capabilities=capabilities)
        block = body["messages"][0]["content"][0]
        value = (
            block["citations"][0][field] if part_kind == "citation" else block["content"][0][field]
        )
        assert value == ("N/A" if state is None else state)


@pytest.mark.parametrize("sources,error", [(None, None), ([], None), (None, "unavailable")])
def test_anthropic_search_history_without_sources_needs_no_ciphertext(sources, error):
    """Ensures unknown, empty, and failed searches do not require invented continuation state."""
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[
            SearchCall(id="ws_1", query="example"),
            SearchResult(tool_call_id="ws_1", sources=sources, error=error),
        ],
    )
    body = AnthropicMessagesClient._request_body(
        request, capabilities=PROVIDER_CAPABILITIES["anthropic"]
    )
    blocks = body["messages"][0]["content"]
    assert len(blocks) == (1 if sources is None and error is None else 2)
    if sources == []:
        assert blocks[1]["content"] == []
    elif error is not None:
        assert blocks[1]["content"]["error_code"] == error


@pytest.mark.parametrize("context", ["low", "medium", "high"])
def test_rejects_explicit_context_size(context):
    """Ensures an explicit context size cannot silently disappear on Messages targets."""
    with pytest.raises(UnsupportedInput, match="tools.web_search.search_context_size.*Messages"):
        AnthropicMessagesClient._request_body(
            NormalizedRequest(
                model="m",
                max_output_tokens=100,
                input=[],
                tools=[WebSearchTool(search_context_size=context)],
            )
        )


@pytest.mark.parametrize(
    "choice", [None, "auto", "required", FunctionToolChoice(name="different")]
)
def test_function_name_collision_without_ambiguous_choice(choice):
    """Ensures distinct server and client tools may share names unless selection is ambiguous."""
    tools = [
        WebSearchTool(),
        FunctionTool(name="web_search", parameters={}),
        FunctionTool(name="different", parameters={}),
    ]
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            model="m", max_output_tokens=100, input=[], tools=tools, tool_choice=choice
        )
    )
    assert len(body["tools"]) == 3


@pytest.mark.parametrize("choice", [WebSearchToolChoice(), FunctionToolChoice(name="web_search")])
def test_rejects_ambiguous_named_tool_selection(choice):
    """Ensures Messages named choice never accidentally selects the wrong execution owner."""
    with pytest.raises(UnsupportedInput, match="ambiguous"):
        AnthropicMessagesClient._request_body(
            NormalizedRequest(
                model="m",
                max_output_tokens=100,
                input=[],
                tools=[WebSearchTool(), FunctionTool(name="web_search", parameters={})],
                tool_choice=choice,
            )
        )


@pytest.mark.parametrize("error", [None, "max_uses_exceeded", "too_many_requests"])
@pytest.mark.parametrize("delta_count", [None, 0, 3])
async def test_search_stream_normalization(monkeypatch, error, delta_count):
    """Ensures hosted ownership, error identity, full-block citations and pause usage survive."""
    raws = [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "model",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 0,
                    "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0},
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "name": "web_search",
                "id": "s",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"q"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "s",
                "content": []
                if error is None
                else {"type": "web_search_tool_result_error", "error_code": error},
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": "A"},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "url": "https://x.test",
                    "title": None,
                    "cited_text": "Source",
                    "encrypted_index": "opaque",
                },
            },
        },
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "😀"}},
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "pause_turn", "stop_sequence": None},
            "usage": {
                "output_tokens": 2,
                "server_tool_use": None
                if delta_count is None
                else {"web_search_requests": delta_count, "web_fetch_requests": 0},
            },
        },
        {"type": "message_stop"},
    ]

    async def native_events(request):
        for raw in raws:
            yield AnthropicMessagesStreamEventAdapter.validate_python(raw)

    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["anthropic"], api_key="test"
        )
        monkeypatch.setattr(client, "native_events", native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(model="m", max_output_tokens=100, input=[], stream=True)
            )
        ]
    calls = [
        event.part
        for event in events
        if isinstance(event, PartStartEvent) and isinstance(event.part, SearchCall)
    ]
    assert len(calls) == 1 and calls[0].query == "q" and calls[0].status == "in_progress"
    results = [
        event.part
        for event in events
        if isinstance(event, PartEndEvent) and isinstance(event.part, SearchResult)
    ]
    assert results[0].error == error
    assert results[0].sources == ([] if error is None else None)
    citations = [
        event.delta.citation
        for event in events
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, CitationDelta)
    ]
    assert len(citations) == 1 and citations[0].end_index == 2
    terminal = next(event for event in events if isinstance(event, ResponseEndEvent))
    assert terminal.termination_reason == "pause"
    assert terminal.usage.web_search_requests == (2 if delta_count is None else delta_count)
    assert terminal.usage.extra_fields == {}
    assert terminal.extra_fields == {}


@pytest.mark.parametrize("nested", [False, True])
def test_browsing_history_omits_linked_results(nested):
    """Ensures reduced browsing history never produces an orphan Messages result."""
    parts = [
        SearchCall(id="browse", action="open_page", url="https://x"),
        SearchResult(tool_call_id="browse", sources=[]),
        SearchCall(id="search", query="q"),
        SearchResult(tool_call_id="search", sources=[]),
    ]
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[AssistantInput(content=parts)] if nested else parts,
    )
    body = AnthropicMessagesClient._request_body(request)
    assert [part["type"] for part in body["messages"][0]["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
    ]
    assert body["messages"][0]["content"][1]["tool_use_id"] == "search"


@pytest.mark.parametrize("stop_reason", ["pause_turn", "max_tokens"])
@pytest.mark.parametrize(
    ("initial_input", "fragments", "expected_query"),
    [
        ({}, ["{"], None),
        ({}, ['{"query": "Par'], None),
        ({"query": "initial"}, [], "initial"),
        ({"query": "initial"}, [""], "initial"),
        ({"query": "initial"}, ['{"query":', '"replacement"}'], "replacement"),
        ({"query": "initial"}, ["{"], None),
        ({"query": "initial", "extra": True}, [], "initial"),
        ({"query": "initial"}, ['{"query":"replacement","extra":true}'], "replacement"),
        ({"query": "initial"}, ['{"extra":true}'], None),
        ({"query": "initial"}, ['["a"'], "initial"),
        ({}, ['["a"'], None),
        (None, ['["a"'], None),
        (None, [], None),
    ],
)
async def test_interrupted_search_arguments_match_sdk(
    stop_reason, initial_input, fragments, expected_query
):
    """Ensures SDK object inputs retain queries and non-objects fall back without fake results."""
    from json import dumps

    from anthropic import AsyncAnthropic
    from anthropic.lib.streaming._messages import accumulate_event

    wire = [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "model",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "s",
                "name": "web_search",
                "input": initial_input,
            },
        },
        *[
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": fragment},
            }
            for fragment in fragments
        ],
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"event: {event['type']}\ndata: {dumps(event)}\n\n" for event in wire),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        sdk = AsyncAnthropic(http_client=http_client, api_key="test")
        reference = None
        async with await sdk.messages.create(
            model="m",
            max_tokens=100,
            messages=[{"role": "user", "content": "Search"}],
            stream=True,
        ) as stream:
            async for event in stream:
                reference = accumulate_event(event=event, current_snapshot=reference)
        client = AnthropicMessagesClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["anthropic"], api_key="test"
        )
        events = [
            event
            async for event in client.events(
                NormalizedRequest(model="m", max_output_tokens=100, input=[], stream=True)
            )
        ]
    completed = [event.part for event in events if isinstance(event, PartEndEvent)]
    assert len(completed) == 1 and isinstance(completed[0], SearchCall)
    assert completed[0].query == expected_query
    if isinstance(reference.content[0].input, dict):
        assert completed[0].query == reference.content[0].input.get("query")
    assert completed[0].status == "in_progress"
    assert not any(
        isinstance(event, (PartStartEvent, PartEndEvent)) and isinstance(event.part, SearchResult)
        for event in events
    )
    assert events[-1].termination_reason == (
        "pause" if reference.stop_reason == "pause_turn" else "max_output_tokens"
    )
