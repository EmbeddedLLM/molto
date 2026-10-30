"""Public hosted-search replay parity and explicitly synthetic boundary regressions."""

import asyncio
from contextlib import aclosing
from copy import deepcopy
from itertools import product
from typing import Any

import pytest
from anthropic import AsyncAnthropic
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic.lib.streaming._messages import accumulate_event
from httpx import AsyncByteStream, AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI
from openai import BadRequestError as OpenAIBadRequestError
from openai import RateLimitError as OpenAIRateLimitError
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import BaseModel

from molto import Molto
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEventAdapter,
)
from molto.types.exceptions import UnsupportedInput
from molto.upstream.openai import OpenAIResponsesClient
from molto.utils.io import dumps_sse, json_loads, load_sse
from molto.utils.test import reconstructable_chat_completions_chunks
from tests.core.test_interface import (
    PROTOCOLS,
    RESPONSES_MESSAGES_PROTOCOLS,
    Protocol,
    _assert_responses_stream_invariants,
    _coordinator,
    _get_http_client,
    _matched_cassette_path,
    _read_cassette,
    _sse_records,
)

SEARCH_ROUTES = tuple(product(PROTOCOLS, RESPONSES_MESSAGES_PROTOCOLS))


def _path(protocol: Protocol, stream: bool = True):
    mode = "stream" if stream else "non_stream"
    return _matched_cassette_path("web_search", protocol, f"{mode}-00.yaml")


def _body(protocol: Protocol, stream: bool = True) -> dict[str, Any]:
    return deepcopy(_read_cassette(_path(protocol, stream))["request"]["body"])


def _request_pair(downstream: Protocol, upstream: Protocol, stream: bool):
    # Translated collection still executes an upstream stream, so both output
    # modes share the streaming recording's answer and native request semantics.
    body = _body(downstream, stream if downstream == upstream else True)
    body["stream"] = stream
    expected = _body(upstream, stream if downstream == upstream else True)
    if downstream == upstream:
        return body, expected
    if upstream == "messages":
        body["max_output_tokens" if downstream == "responses" else "max_completion_tokens"] = 4096
        if downstream == "responses":
            expected["tool_choice"] = {"type": "any"}
    else:
        expected["tool_choice"] = {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "web_search"}],
        }
        prompt = expected["input"]
        expected["input"] = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        if downstream == "messages":
            expected["max_output_tokens"] = 4096
        else:
            expected.pop("stream_options")
    return body, expected


async def _sdk_result(protocol: Protocol, stream: bool):
    path = _path(protocol, stream)
    async with _get_http_client([path], protocol) as http_client:
        if protocol == "messages":
            client = AsyncAnthropic(api_key="test-api-key", http_client=http_client)
            result = await client.messages.create(**_body(protocol, stream))
        else:
            client = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
            create = (
                client.responses.create
                if protocol == "responses"
                else client.chat.completions.create
            )
            result = await create(**_body(protocol, stream))
        if not stream:
            return result
        async with result:
            return [event async for event in result]


def _collect(protocol: Protocol, events: list[BaseModel]) -> BaseModel:
    if protocol == "responses":
        assert events[-1].type in {"response.completed", "response.incomplete"}
        return events[-1].response
    if protocol == "chat/completions":
        state = ChatCompletionStreamState()
        for event in events:
            state.handle_chunk(event.model_copy(deep=True))
        return state.current_completion_snapshot
    snapshot = None
    for event in events:
        snapshot = accumulate_event(event=event.model_copy(deep=True), current_snapshot=snapshot)
        # The SDK accumulator does not copy this compatible-provider extension,
        # although its parsed message_delta preserves the observed value.
        if (
            event.type == "message_delta"
            and getattr(event.usage, "output_tokens_details", None) is not None
        ):
            snapshot.usage = snapshot.usage.model_copy(
                update={"output_tokens_details": event.usage.output_tokens_details}
            )
    assert snapshot is not None
    return snapshot


def _assert_lifecycle(protocol: Protocol, events: list[BaseModel]) -> None:
    if protocol == "responses":
        _assert_responses_stream_invariants([dumps_sse(event) for event in events])
        assert events[0].type == "response.created"
        assert events[-1].type in {"response.completed", "response.incomplete"}
        added = [
            event.output_index for event in events if event.type == "response.output_item.added"
        ]
        done = [
            event.output_index for event in events if event.type == "response.output_item.done"
        ]
        assert added == done == list(range(len(added)))
    elif protocol == "messages":
        assert events[0].type == "message_start"
        assert events[-1].type == "message_stop"
        opened = set()
        starts = []
        for event in events:
            if event.type == "content_block_start":
                assert event.index == len(starts)
                starts.append(event.index)
                opened.add(event.index)
            elif event.type == "content_block_delta":
                assert event.index in opened
            elif event.type == "content_block_stop":
                assert event.index in opened
                opened.remove(event.index)
        assert not opened
        assert sum(event.type == "message_delta" for event in events) == 1
    else:
        assert (
            sum(choice.finish_reason is not None for event in events for choice in event.choices)
            == 1
        )
        assert all(not choice.delta.tool_calls for event in events for choice in event.choices)
        assert all(
            not getattr(choice.delta, "annotations", None)
            for event in events
            for choice in event.choices
        )


def _responses_semantics(response: BaseModel) -> dict[str, Any]:
    """Rebase citation offsets across preserved text parts; retain all search values."""
    data = response.model_dump(mode="json")
    text = ""
    citations = []
    searches = []
    thinking = []
    for item in data["output"]:
        if item["type"] == "message":
            assert item["role"] == "assistant"
            assert item["status"] == "completed"
            for part in item["content"]:
                assert part["type"] == "output_text"
                for citation in part["annotations"]:
                    citations.append(
                        {
                            **citation,
                            "start_index": citation["start_index"] + len(text),
                            "end_index": citation["end_index"] + len(text),
                        }
                    )
                text += part["text"]
        elif item["type"] == "web_search_call":
            action = item["action"]
            searches.append(
                {
                    "id": item["id"],
                    "status": item["status"],
                    "type": action["type"],
                    "query": action.get("query"),
                    "sources": [source["url"] for source in action.get("sources", [])],
                }
            )
        elif item["type"] == "reasoning":
            thinking.append(
                {"summary": item["summary"], "encrypted_content": item["encrypted_content"]}
            )
        else:
            raise AssertionError(f"Unexpected output item {item['type']}")
    usage = data["usage"]
    return {
        "id": data["id"],
        "model": data["model"],
        "status": data["status"],
        "text": text,
        "citations": citations,
        "searches": searches,
        "thinking": thinking,
        "usage": {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "input_tokens_details": usage["input_tokens_details"],
            "output_tokens_details": usage["output_tokens_details"],
        },
    }


def _messages_semantics(response: BaseModel) -> dict[str, Any]:
    data = response.model_dump(mode="json")
    content = []
    for block in data["content"]:
        block.pop("parsed_output", None)
        if block["type"] == "web_search_tool_result" and isinstance(block["content"], list):
            for source in block["content"]:
                source.setdefault("page_age", None)
        content.append(block)
    return {
        "id": data["id"],
        "model": data["model"],
        "role": data["role"],
        "type": data["type"],
        "stop_reason": data["stop_reason"],
        "stop_sequence": data["stop_sequence"],
        "content": content,
        # Geography/cache-lifetime breakdowns are provider-only; token counts,
        # reasoning counts, and observed search counters remain exact.
        "usage": {
            key: data["usage"].get(key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens_details",
                "server_tool_use",
            )
        },
    }


@pytest.mark.parametrize(
    ("protocol", "search_in_extra_body"),
    [(protocol, False) for protocol in PROTOCOLS] + [("chat/completions", True)],
)
@pytest.mark.parametrize("stream", [True, False])
async def test_native_search_matches_sdk(
    protocol: Protocol, stream: bool, search_in_extra_body: bool
):
    """Ensures native search, including Chat body extensions, preserves SDK requests and output."""
    expected = await _sdk_result(protocol, stream)
    body = _body(protocol, stream)
    extra_body = (
        {"web_search_options": body.pop("web_search_options")} if search_in_extra_body else None
    )
    async with _get_http_client([_path(protocol, stream)], protocol) as http_client:
        result = await _coordinator(Molto(http_client=http_client), protocol)(
            body=body, endpoint=protocol, api_key="test-api-key", extra_body=extra_body
        )
        actual = [event async for event in result] if stream else result
    if stream:
        if protocol != "chat/completions":
            _assert_lifecycle(protocol, actual)
        assert [type(event) for event in actual] == [type(event) for event in expected]
        assert [event.model_dump(mode="json", exclude_unset=True) for event in actual] == [
            event.model_dump(mode="json", exclude_unset=True) for event in expected
        ]
    else:
        assert type(actual) is type(expected)
        assert actual.model_dump(mode="json", exclude_unset=True) == expected.model_dump(
            mode="json", exclude_unset=True
        )


@pytest.mark.parametrize(
    ("downstream", "upstream"),
    [route for route in SEARCH_ROUTES if route[0] != route[1] and route[0] != "chat/completions"],
)
@pytest.mark.parametrize("stream", [True, False])
async def test_translated_search_matches_sdk(
    downstream: Protocol, upstream: Protocol, stream: bool
):
    """Ensures exact translated requests, SDK-collected search semantics, and balanced lifecycle."""
    body, expected_request = _request_pair(downstream, upstream, stream)
    expected_events = await _sdk_result(downstream, True)
    expected = _collect(downstream, expected_events)
    async with _get_http_client(
        [_path(upstream)], upstream, expected_body=expected_request
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body,
            endpoint=upstream,
            api_key="test-api-key",
            **({"citation_policy": "buffer"} if downstream == "messages" else {}),
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle(downstream, events)
            actual = _collect(downstream, events)
        else:
            actual = result
    if downstream == "responses":
        # Messages retains the item-completion signature/sources, not the later
        # Responses terminal snapshots. Neither value is inferred or fabricated.
        expected = expected.model_copy(deep=True)
        completed = {
            event.item.id: event.item
            for event in expected_events
            if event.type == "response.output_item.done"
        }
        expected.output = [completed.get(item.id, item) for item in expected.output]
        actual_semantics = _responses_semantics(actual)
        expected_semantics = _responses_semantics(expected)
        native = _collect("messages", await _sdk_result("messages", True))
        # Canonical Responses has no Anthropic state. Compare the extension with
        # the Messages SDK before reducing only this unavailable field locally.
        assert [citation.pop("extra_content") for citation in actual_semantics["citations"]] == [
            {
                "anthropic": citation.model_dump(
                    include={"encrypted_index", "cited_text"}, exclude_none=True
                )
            }
            for block in native.content
            if block.type == "text"
            for citation in block.citations or []
        ]
        assert [
            [source.extra_content for source in item.action.sources]
            for item in actual.output
            if item.type == "web_search_call"
        ] == [
            [
                {
                    "anthropic": source.model_dump(
                        include={"encrypted_content", "title", "page_age"}, exclude_none=True
                    )
                }
                for source in block.content
            ]
            for block in native.content
            if block.type == "web_search_tool_result"
        ]
        assert actual_semantics == expected_semantics
    else:
        assert _messages_semantics(actual) == _messages_semantics(expected)


@pytest.mark.parametrize("stream", [True, False])
async def test_messages_ambiguous_search_selection_rejected_before_transport(stream: bool):
    """Ensures named search collisions fail before HTTP in both public output modes."""
    body = _body("messages")
    body["stream"] = stream
    body["tools"].append({"name": "web_search", "input_schema": {"type": "object"}})
    body["tool_choice"] = {"type": "tool", "name": "web_search"}
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        raise AssertionError("Ambiguous selection must fail before transport")

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        with pytest.raises(UnsupportedInput, match="ambiguous"):
            await Molto(http_client=http_client).messages(
                body=body, endpoint="responses", api_key="test-api-key"
            )
    assert requests == []


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("choice_type", [None, "auto", "any", "tool"])
async def test_messages_search_selection_matches_sdk(stream: bool, choice_type: str | None):
    """Ensures accepted search choices preserve exact requests and SDK-backed search results."""
    body, expected_request = _request_pair("messages", "responses", stream)
    if choice_type is None:
        body.pop("tool_choice", None)
    else:
        body["tool_choice"] = {"type": choice_type}
        if choice_type == "tool":
            body["tool_choice"]["name"] = "web_search"
    if choice_type != "tool":
        # Synthetic declarations exercise selection; recorded search output is unchanged.
        body["tools"].append({"name": "web_search", "input_schema": {"type": "object"}})
        expected_request["tools"].append(
            {
                "type": "function",
                "name": "web_search",
                "parameters": {"type": "object"},
                "strict": False,
            }
        )
        expected_request.pop("tool_choice")
        if choice_type == "any":
            expected_request["tool_choice"] = "required"
    expected = _collect("messages", await _sdk_result("messages", True))
    async with _get_http_client(
        [_path("responses")], "responses", expected_body=expected_request
    ) as http_client:
        result = await Molto(http_client=http_client).messages(
            body=body, endpoint="responses", api_key="test-api-key", citation_policy="buffer"
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle("messages", events)
            actual = _collect("messages", events)
        else:
            actual = result
    assert _messages_semantics(actual) == _messages_semantics(expected)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("annotation_model", [True, False])
async def test_messages_citation_annotation_matches_sdk(
    stream: bool, annotation_model: bool, monkeypatch
):
    """Ensures dictionary and SDK model annotation deltas preserve citation associations and search parity."""
    native_events = OpenAIResponsesClient.native_events
    converted = []

    async def annotation_events(client, request):
        async with aclosing(native_events(client, request)) as events:
            async for event in events:
                if event.type == "response.output_text.annotation.added":
                    annotation = event.annotation
                    if isinstance(annotation, BaseModel):
                        annotation = annotation.model_dump()
                    citation = AnnotationURLCitation.model_validate(annotation)
                    converted.append(citation)
                    event = event.model_copy(
                        update={
                            "annotation": citation if annotation_model else citation.model_dump()
                        }
                    )
                yield event

    monkeypatch.setattr(OpenAIResponsesClient, "native_events", annotation_events)
    body, expected_request = _request_pair("messages", "responses", stream)
    expected = _collect("messages", await _sdk_result("messages", True))
    async with _get_http_client(
        [_path("responses")], "responses", expected_body=expected_request
    ) as http_client:
        result = await Molto(http_client=http_client).messages(
            body=body, endpoint="responses", api_key="test-api-key", citation_policy="buffer"
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle("messages", events)
            actual = _collect("messages", events)
        else:
            actual = result
    assert converted
    assert _messages_semantics(actual) == _messages_semantics(expected)


@pytest.mark.parametrize(
    "include", [None, ["web_search_call.action.sources"], ["reasoning.encrypted_content"]]
)
@pytest.mark.parametrize("stream", [True, False])
async def test_messages_search_extra_body_matches_sdk(stream: bool, include: list[str] | None):
    """Ensures opaque search is forwarded verbatim and explicit buffering preserves SDK attribution."""
    body, expected_request = _request_pair("messages", "responses", stream)
    body.pop("tools")
    body.pop("tool_choice")
    extra_body = {
        "tools": [{"type": "web_search"}],
        "tool_choice": "required",
    }
    if include is not None:
        extra_body["include"] = include
    expected_request.pop("include")
    expected_request.update(extra_body)
    expected = _collect("messages", await _sdk_result("messages", True))
    async with _get_http_client(
        [_path("responses")], "responses", expected_body=expected_request
    ) as http_client:
        result = await Molto(http_client=http_client).messages(
            body=body,
            endpoint="responses",
            api_key="test-api-key",
            citation_policy="buffer",
            extra_body=extra_body,
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle("messages", events)
            actual = _collect("messages", events)
        else:
            actual = result
    assert _messages_semantics(actual) == _messages_semantics(expected)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("upstream", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize("search_in_extra_body", [False, True])
async def test_chat_search_rejects_recorded_signatures(
    upstream: Protocol, stream: bool, search_in_extra_body: bool
):
    """Ensures recorded SDK reasoning signatures fail Chat projection regardless of search declaration."""
    reference = _collect(upstream, await _sdk_result(upstream, True))
    if upstream == "responses":
        assert any(
            item.type == "reasoning" and item.encrypted_content for item in reference.output
        )
    else:
        assert any(block.type == "thinking" and block.signature for block in reference.content)
    body, expected_request = _request_pair("chat/completions", upstream, stream)
    extra_body = None
    if search_in_extra_body:
        body.pop("web_search_options")
        extra_body = {
            "tools": expected_request["tools"],
            "tool_choice": "required"
            if upstream == "responses"
            else {"type": "tool", "name": "web_search"},
        }
        expected_request.update(extra_body)
        if upstream == "responses":
            expected_request.pop("include")
    async with _get_http_client(
        [_path(upstream)], upstream, expected_body=expected_request
    ) as http_client:
        with pytest.raises(
            ValueError, match="Chat Completions cannot preserve reasoning signatures"
        ):
            result = await Molto(http_client=http_client).chat_completions(
                body=body, endpoint=upstream, api_key="test-api-key", extra_body=extra_body
            )
            if stream:
                async with result:
                    async for _ in result:
                        pass


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("upstream", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize("replacement", ["null", "empty", "function"])
async def test_chat_search_extra_body_replacement_rejects_signatures(
    upstream: Protocol, stream: bool, replacement: str
):
    """Ensures replacing hosted tools preserves signature rejection, even for same-named functions."""
    body, expected_request = _request_pair("chat/completions", upstream, stream)
    function = (
        {"type": "function", "name": "web_search", "parameters": {}}
        if upstream == "responses"
        else {"name": "web_search", "input_schema": {"type": "object"}}
    )
    extra_body = {
        "tools": None if replacement == "null" else [] if replacement == "empty" else [function],
        "tool_choice": "auto" if upstream == "responses" else {"type": "auto"},
    }
    expected_request.update(extra_body)
    async with _get_http_client(
        [_path(upstream)], upstream, expected_body=expected_request
    ) as http_client:
        with pytest.raises(
            ValueError, match="Chat Completions cannot preserve reasoning signatures"
        ):
            result = await Molto(http_client=http_client).chat_completions(
                body=body, endpoint=upstream, api_key="test-api-key", extra_body=extra_body
            )
            if stream:
                async with result:
                    async for _ in result:
                        pass


async def test_chat_search_raw_reasoning_starts_with_assistant_role():
    """Ensures search-enabled raw reasoning preserves the recorded SDK chunk order."""
    chat_path = _matched_cassette_path("reasoning_raw", "chat/completions", "stream-00.yaml")
    responses_path = _matched_cassette_path("reasoning_raw", "responses", "stream-00.yaml")
    chat_body = deepcopy(_read_cassette(chat_path)["request"]["body"])
    async with _get_http_client([chat_path], "chat/completions") as http_client:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        async with await sdk.chat.completions.create(**chat_body) as stream:
            expected = [event async for event in stream]

    # Replay observed raw reasoning with a synthetic search declaration; this
    # checks projection, not whether the recorded provider supports hosted search.
    chat_body["web_search_options"] = {}
    expected_request = deepcopy(_read_cassette(responses_path)["request"]["body"])
    expected_request.update(
        tools=[{"type": "web_search"}],
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "web_search"}],
        },
        include=["web_search_call.action.sources"],
    )
    async with _get_http_client(
        [responses_path], "responses", expected_body=expected_request
    ) as http_client:
        stream = await Molto(http_client=http_client).chat_completions(
            body=chat_body, endpoint="responses", provider="vllm", api_key="test-api-key"
        )
        actual = [event async for event in stream]

    assert actual[0].choices[0].delta.role == "assistant"
    assert sum(choice.delta.role is not None for event in actual for choice in event.choices) == 1
    assert [
        chunk
        for event in actual
        for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=chat_body)
    ] == [
        chunk
        for event in expected
        for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=chat_body)
    ]


@pytest.mark.parametrize("downstream", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize("stream", [True, False])
async def test_chat_upstream_search_rejected_before_transport(downstream: Protocol, stream: bool):
    """Ensures normalized hosted search still fails before opening Chat transport."""
    requests = []
    body = _body(downstream, stream)

    def handler(request: Request) -> Response:
        requests.append(request)
        raise AssertionError("Hosted search must fail before transport")

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        with pytest.raises(UnsupportedInput, match="[Ss]earch"):
            result = await _coordinator(Molto(http_client=http_client), downstream)(
                body=body, endpoint="chat/completions", api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
    assert requests == []


@pytest.mark.parametrize(
    ("downstream", "citation_policy"),
    [("responses", None), ("messages", "drop"), ("messages", "buffer"), ("messages", "auto")],
)
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("synthetic_annotations", [False, True])
@pytest.mark.parametrize("extra_search", [False, True])
async def test_chat_upstream_extra_search_reaches_transport(
    downstream: Protocol,
    citation_policy: str | None,
    stream: bool,
    synthetic_annotations: bool,
    extra_search: bool,
):
    """Ensures Chat preserves SDK text chunks under every policy, including opaque search."""
    body = {"model": "test-model", "stream": stream}
    if downstream == "responses":
        body["input"] = "Search"
    else:
        body.update(max_tokens=4096, messages=[{"role": "user", "content": "Search"}])
    expected = {
        "model": "test-model",
        "stream": True,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Search"}]}],
        "stream_options": {"include_usage": True},
    }
    extra_body = {"web_search_options": {}} if extra_search else None
    if extra_body:
        expected.update(extra_body)
    if downstream == "messages":
        expected["max_completion_tokens"] = 4096
        expected["stream_options"]["include_obfuscation"] = False
    wire = [
        load_sse(record)
        for record in _sse_records(_read_cassette(_path("chat/completions")), "chat/completions")
    ]
    if synthetic_annotations:
        # The recording has no annotations. Add an in-memory provider extension
        # to verify its documented omission without changing the cassette.
        delta = next(
            chunk["choices"][0]["delta"]
            for chunk in wire
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content")
        )
        delta["annotations"] = [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com",
                    "title": "Source",
                    "start_index": 0,
                    "end_index": 1,
                },
            }
        ]
    payload = "".join(dumps_sse(chunk) for chunk in wire) + "data: [DONE]\n\n"
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        assert json_loads(request.content) == expected
        return Response(200, content=payload, headers={"content-type": "text/event-stream"})

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        async with await sdk.chat.completions.create(**expected) as sdk_stream:
            sdk_events = [event async for event in sdk_stream]
        if synthetic_annotations:
            assert any(
                getattr(choice.delta, "annotations", None)
                for chunk in sdk_events
                for choice in chunk.choices
            )
        reference = _collect("chat/completions", sdk_events)
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body,
            endpoint="chat/completions",
            api_key="test-api-key",
            extra_body=extra_body,
            **({"citation_policy": citation_policy} if downstream == "messages" else {}),
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle(downstream, events)
            if downstream == "messages":
                assert [
                    event.delta.text
                    for event in events
                    if event.type == "content_block_delta" and event.delta.type == "text_delta"
                ] == [
                    choice.delta.content
                    for chunk in sdk_events
                    for choice in chunk.choices
                    if choice.delta.content
                ]
            result = _collect(downstream, events)
    assert len(requests) == 2
    if downstream == "responses":
        assert result.output_text == reference.choices[0].message.content
        assert all(
            not part.annotations
            for item in result.output
            if item.type == "message"
            for part in item.content
        )
    else:
        assert (
            "".join(block.text for block in result.content if block.type == "text")
            == reference.choices[0].message.content
        )
        assert all(not block.citations for block in result.content if block.type == "text")


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("downstream", "upstream"), [("responses", "messages"), ("messages", "responses")]
)
@pytest.mark.parametrize("budget", [1, 7])
async def test_search_budget_exact_request(
    downstream: Protocol, upstream: Protocol, stream: bool, budget: int
):
    """Ensures caller budgets map one-to-one in exact public upstream requests."""
    body, expected = _request_pair(downstream, upstream, stream)
    if downstream == "responses":
        body["max_tool_calls"] = budget
        expected["tools"][0]["max_uses"] = budget
    else:
        body["tools"][0]["max_uses"] = budget
        expected["max_tool_calls"] = budget
    async with _get_http_client(
        [_path(upstream)], upstream, expected_body=expected
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body, endpoint=upstream, api_key="test-api-key"
        )
        if stream:
            _ = [event async for event in result]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("downstream", "upstream", "option"),
    [
        ("responses", "messages", "search_context_size"),
        ("chat/completions", "messages", "search_context_size"),
        ("messages", "responses", "blocked_domains"),
    ],
)
async def test_untranslatable_search_options_fail_before_transport(
    downstream: Protocol, upstream: Protocol, option: str, stream: bool
):
    """Ensures explicit unsupported search settings, including medium context, are never dropped."""
    body, _ = _request_pair(downstream, upstream, stream)
    target = body["web_search_options"] if downstream == "chat/completions" else body["tools"][0]
    target[option] = ["example.com"] if option == "blocked_domains" else "medium"
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        raise AssertionError("Unsupported options must fail before HTTP")

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        with pytest.raises(UnsupportedInput, match=option):
            result = await _coordinator(Molto(http_client=http_client), downstream)(
                body=body, endpoint=upstream, api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
    assert requests == []


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("upstream", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize("name", ["lookup", "web_search"])
async def test_chat_search_required_with_client_function(
    upstream: Protocol, stream: bool, name: str
):
    """Ensures hosted search remains required and ambiguous Messages names are rejected."""
    body, expected = _request_pair("chat/completions", upstream, stream)
    function = {
        "name": name,
        "description": "Look up a local record.",
        "parameters": {"type": "object", "properties": {}},
    }
    body["tools"] = [{"type": "function", "function": function}]
    if upstream == "responses":
        expected["tools"] = [{"type": "function", "strict": False, **function}, *expected["tools"]]
        expected["tool_choice"] = {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "web_search"}],
        }
    else:
        expected["tools"] = [
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            },
            *expected["tools"],
        ]
    async with _get_http_client(
        [_path(upstream)], upstream, expected_body=expected
    ) as http_client:
        if upstream == "messages" and name == "web_search":
            with pytest.raises(UnsupportedInput, match="ambiguous"):
                await Molto(http_client=http_client).chat_completions(
                    body=body, endpoint=upstream, api_key="test-api-key"
                )
            return
        with pytest.raises(
            ValueError, match="Chat Completions cannot preserve reasoning signatures"
        ):
            result = await Molto(http_client=http_client).chat_completions(
                body=body, endpoint=upstream, api_key="test-api-key"
            )
            if stream:
                async with result:
                    async for _ in result:
                        pass


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("upstream", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize("choice", ["none", {"type": "function", "function": {"name": "lookup"}}])
async def test_chat_search_conflicting_choice_rejected(upstream: Protocol, stream: bool, choice):
    """Ensures explicit conflicting Chat tool selection fails rather than weakening required search."""
    body, _ = _request_pair("chat/completions", upstream, stream)
    body["tool_choice"] = choice
    body["tools"] = [
        {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    ]
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        raise AssertionError("Conflicting selection must fail before HTTP")

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        with pytest.raises(UnsupportedInput, match="[Ss]earch"):
            result = await Molto(http_client=http_client).chat_completions(
                body=body, endpoint=upstream, api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
    assert requests == []


def _history(protocol: Protocol, search: bool) -> list[dict[str, Any]]:
    text = "Python 3.13 was released."
    url = "https://www.python.org/downloads/release/python-3130/"
    if protocol == "responses":
        history = []
        if search:
            history.append(
                {
                    "type": "web_search_call",
                    "id": "ws_previous",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "Python release date",
                        "sources": [{"type": "url", "url": url}],
                    },
                }
            )
        history.append(
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": url,
                                "title": "Python",
                                "start_index": 0,
                                "end_index": len(text),
                            }
                        ]
                        if search
                        else [],
                        "logprobs": [],
                    }
                ],
            }
        )
        history.append(
            {"role": "user", "content": [{"type": "input_text", "text": "What was the date?"}]}
        )
        return history
    blocks = []
    if search:
        blocks.extend(
            [
                {
                    "type": "server_tool_use",
                    "id": "ws_previous",
                    "name": "web_search",
                    "input": {"query": "Python release date"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "ws_previous",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": url,
                            "title": "",
                            "encrypted_content": "N/A",
                        }
                    ],
                },
            ]
        )
    blocks.append(
        {
            "type": "text",
            "text": text,
            **(
                {
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": url,
                            "title": "Python",
                            "cited_text": text,
                            "encrypted_index": "N/A",
                        }
                    ]
                }
                if search
                else {}
            ),
        }
    )
    return [
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": [{"type": "text", "text": "What was the date?"}]},
    ]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("downstream", "upstream", "search"),
    [
        (downstream, upstream, search)
        for downstream, upstream in SEARCH_ROUTES
        for search in (False, True)
        if not (search and downstream == "chat/completions")
    ],
)
async def test_fixed_route_followup_history(
    downstream: Protocol, upstream: Protocol, stream: bool, search: bool
):
    """Ensures native and compatible-provider followups preserve reduced history and placeholders."""
    body, expected = _request_pair(downstream, upstream, stream)
    body_key = "input" if downstream == "responses" else "messages"
    expected_key = "input" if upstream == "responses" else "messages"
    if isinstance(body[body_key], str):
        body[body_key] = [
            {"role": "user", "content": [{"type": "input_text", "text": body[body_key]}]}
        ]
    if isinstance(expected[expected_key], str):
        expected[expected_key] = [
            {"role": "user", "content": [{"type": "input_text", "text": expected[expected_key]}]}
        ]
    body[body_key].extend(_history(downstream, search))
    expected[expected_key].extend(_history(upstream, search))
    async with _get_http_client(
        [_path(upstream, stream if downstream == upstream else True)],
        upstream,
        expected_body=expected,
    ) as http_client:
        if downstream == "chat/completions":
            with pytest.raises(
                ValueError, match="Chat Completions cannot preserve reasoning signatures"
            ):
                result = await Molto(http_client=http_client).chat_completions(
                    body=body, endpoint=upstream, api_key="test-api-key"
                )
                if stream:
                    async with result:
                        async for _ in result:
                            pass
        else:
            result = await _coordinator(Molto(http_client=http_client), downstream)(
                body=body,
                endpoint=upstream,
                api_key="test-api-key",
                provider="openai"
                if downstream == "responses" and upstream == "messages" and search
                else None,
            )
            if stream:
                _ = [event async for event in result]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("result_kind", ["sources", "empty", "error"])
async def test_split_assistant_search_history_matches_combined_sdk(stream: bool, result_kind: str):
    """Ensures split and combined history preserve linked calls and OpenAI/Messages SDK parity."""
    body, expected_request = _request_pair("messages", "responses", stream)
    history = _history("messages", True)
    expected_history = _history("responses", True)
    if result_kind == "empty":
        history[0]["content"][1]["content"] = []
        expected_history[0]["action"]["sources"] = []
    elif result_kind == "error":
        history[0]["content"][1]["content"] = {
            "type": "web_search_tool_result_error",
            "error_code": "unavailable",
        }
        expected_history[0]["status"] = "failed"
        expected_history[0]["action"].pop("sources")
    history.insert(-1, {"role": "assistant", "content": [{"type": "text", "text": "More"}]})
    expected_history.insert(
        -1,
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "More", "annotations": [], "logprobs": []}
            ],
        },
    )
    expected_request["input"].extend(expected_history)
    expected = _collect("messages", await _sdk_result("messages", True))
    reference = _collect("responses", await _sdk_result("responses", True))
    actual_results = []
    for split in (False, True):
        request = deepcopy(body)
        assistant = history[0]
        request["messages"].extend(
            [
                {"role": "assistant", "content": assistant["content"][:1]},
                {"role": "assistant", "content": assistant["content"][1:]},
                *history[1:],
            ]
            if split
            else history
        )
        async with _get_http_client(
            [_path("responses")], "responses", expected_body=expected_request
        ) as http_client:
            result = await Molto(http_client=http_client).messages(
                body=request,
                endpoint="responses",
                api_key="test-api-key",
                citation_policy="buffer",
            )
            if stream:
                events = [event async for event in result]
                _assert_lifecycle("messages", events)
                result = _collect("messages", events)
        assert result.id == reference.id
        assert result.model == reference.model
        assert (
            "".join(block.text for block in result.content if block.type == "text")
            == reference.output_text
        )
        assert result.usage.output_tokens == reference.usage.output_tokens
        actual_results.append(_messages_semantics(result))
    assert actual_results == [_messages_semantics(expected)] * 2


@pytest.mark.parametrize("stream", [True, False])
async def test_multi_query_history_rejected_before_messages_transport(stream: bool):
    """Ensures completed query batches cannot send unmatched server calls on a user follow-up."""
    body, _ = _request_pair("responses", "messages", stream)
    history = _history("responses", True)
    history[0]["action"]["queries"] = ["Python release date", "Python release notes"]
    history[0]["action"]["sources"] = []
    history[1]["content"][0]["annotations"] = []
    body["input"] = [
        {"role": "user", "content": [{"type": "input_text", "text": body["input"]}]},
        *history,
    ]
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        raise AssertionError("Unsupported history must fail before transport.")

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        with pytest.raises(UnsupportedInput, match="multi-query search history"):
            await Molto(http_client=http_client).responses(
                body=body, endpoint="messages", api_key="test-api-key"
            )
    assert requests == []


class _SearchStream(AsyncByteStream):
    def __init__(self, payload: str, *, pause: bool = False):
        self.records = payload.split("\n\n")
        self.pause = pause
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for index, record in enumerate(self.records):
            if self.pause and index == len(self.records) - 2:
                self.waiting.set()
                await asyncio.Event().wait()
            yield f"{record}\n\n".encode()

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize(
    ("downstream", "upstream", "cancel"),
    [
        (downstream, upstream, cancel)
        for downstream, upstream in SEARCH_ROUTES
        for cancel in (False, True)
        if downstream != "chat/completions" or not cancel
    ],
)
async def test_search_stream_close_releases_provider(
    downstream: Protocol, upstream: Protocol, cancel: bool
):
    """Ensures close, cancellation, and signature rejection release search streams without closing shared HTTP."""
    body, expected = _request_pair(downstream, upstream, True)
    payload = "".join(_read_cassette(_path(upstream))["response"]["sse"])
    provider_stream = _SearchStream(payload, pause=cancel)

    def handler(request: Request) -> Response:
        assert json_loads(request.content) == expected
        return Response(200, stream=provider_stream, headers={"content-type": "text/event-stream"})

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        if downstream == "chat/completions":
            with pytest.raises(
                ValueError, match="Chat Completions cannot preserve reasoning signatures"
            ):
                result = await Molto(http_client=http_client).chat_completions(
                    body=body, endpoint=upstream, api_key="test-api-key"
                )
                async with result:
                    async for _ in result:
                        pass
            assert provider_stream.closed
            assert not http_client.is_closed
            return
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body, endpoint=upstream, api_key="test-api-key"
        )
        if cancel:

            async def consume():
                async for _ in result:
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(provider_stream.waiting.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await result.aclose()
        assert provider_stream.closed
        assert not http_client.is_closed


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("sources", [None, [], [{"type": "url", "url": "https://python.org/"}]])
@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_multi_query_projection_matches_sdk(stream: bool, sources, status: str):
    """Ensures split queries retain SDK batch results, citations, usage, and balanced lifecycles."""
    events = _synthetic_events("responses")
    queries = ["first", "", "first"]
    for item in _search_snapshots(events):
        if "action" in item:
            item["action"].update(query="fallback", queries=queries, sources=sources)
            item["status"] = status
    for event in events:
        if event.get("response", {}).get("usage") is not None:
            event["response"]["tool_usage"] = {"web_search": {"num_requests": 1}}
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    search = next(item for item in reference.output if item.type == "web_search_call")
    calls = [block for block in result.content if block.type == "server_tool_use"]
    assert [block.input for block in calls] == [
        {"query": query} for query in search.action.queries
    ]
    assert len({block.id for block in calls}) == len(queries)
    assert calls[0].id == search.id
    results = [block for block in result.content if block.type == "web_search_tool_result"]
    if status == "failed":
        assert len(results) == 1
        assert results[0].content.type == "web_search_tool_result_error"
        assert results[0].content.error_code == "unavailable"
    elif search.action.sources is None:
        assert results == []
    else:
        assert len(results) == 1
        assert [source.url for source in results[0].content] == [
            source.url for source in search.action.sources
        ]
    if results:
        assert results[0].tool_use_id == search.id
    assert (
        "".join(block.text for block in result.content if block.type == "text")
        == reference.output_text
    )
    assert sum(len(block.citations or []) for block in result.content if block.type == "text") == 1
    assert (
        result.usage.server_tool_use.web_search_requests
        == reference.tool_usage["web_search"]["num_requests"]
        == 1
    )


def _synthetic_events(protocol: Protocol) -> list[dict[str, Any]]:
    """Copy recorded events for synthetic cases without modifying cassette evidence."""
    return [load_sse(record) for record in _sse_records(_read_cassette(_path(protocol)), protocol)]


def _without_reasoning(events: list[dict[str, Any]], protocol: Protocol) -> list[dict[str, Any]]:
    """Model a synthetic reasoning-free turn while retaining search and text lifecycles."""
    if protocol == "responses":
        reasoning_indices = {
            event["output_index"]
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "reasoning"
        }
        index_key = "output_index"
    else:
        reasoning_indices = {
            event["index"]
            for event in events
            if event["type"] == "content_block_start"
            and event["content_block"]["type"] == "thinking"
        }
        index_key = "index"
    retained = []
    for event in events:
        if event.get(index_key) in reasoning_indices:
            continue
        if index_key in event:
            event[index_key] -= sum(index < event[index_key] for index in reasoning_indices)
        if protocol == "responses" and "response" in event:
            event["response"]["output"] = [
                item for item in event["response"].get("output", []) if item["type"] != "reasoning"
            ]
        if "sequence_number" in event:
            event["sequence_number"] = len(retained)
        retained.append(event)
    return retained


async def _synthetic_replay(
    downstream: Protocol,
    upstream: Protocol,
    stream: bool,
    events: list[dict[str, Any]],
    *,
    request_pair=None,
    extra_body=None,
    citation_policy="buffer",
):
    payload = "".join(dumps_sse(event, event_type=event["type"]) for event in events)
    body, expected_request = request_pair or _request_pair(downstream, upstream, stream)

    # A parsed SDK reference proves that mutations still use the native SDK's
    # wire boundary; synthetic provider behavior is not claimed as a recording.
    def sdk_handler(request: Request) -> Response:
        assert json_loads(request.content) == expected_request
        return Response(
            200, content=payload.encode(), headers={"content-type": "text/event-stream"}
        )

    async with AsyncClient(transport=MockTransport(sdk_handler)) as http_client:
        if upstream == "messages":
            sdk = AsyncAnthropic(api_key="test-api-key", http_client=http_client)
            reference = await sdk.messages.create(**expected_request)
        else:
            sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
            reference = await sdk.responses.create(**expected_request)
        async with reference:
            sdk_events = [event async for event in reference]
    provider_stream = _SearchStream(payload)
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        assert json_loads(request.content) == expected_request
        return Response(200, stream=provider_stream, headers={"content-type": "text/event-stream"})

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body,
            endpoint=upstream,
            api_key="test-api-key",
            extra_body=extra_body,
            **({"citation_policy": citation_policy} if downstream == "messages" else {}),
        )
        if stream:
            projected = [event async for event in result]
            _assert_lifecycle(downstream, projected)
            result = _collect(downstream, projected)
        assert provider_stream.closed
        assert len(requests) == 1
    return result, _collect(upstream, sdk_events)


def _search_snapshots(events: list[dict[str, Any]]):
    for event in events:
        item = event.get("item")
        if item and item["type"] == "web_search_call":
            yield item
        for item in event.get("response", {}).get("output", []):
            if item["type"] == "web_search_call":
                yield item


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("query_fields", "expected_input"),
    [
        ({}, {}),
        ({"query": None}, {}),
        ({"queries": []}, {}),
        ({"query": ""}, {"query": ""}),
        ({"query": "one", "queries": ["one"]}, {"query": "one"}),
    ],
)
async def test_synthetic_query_cardinality(stream: bool, query_fields, expected_input):
    """Ensures missing/null/empty queries and matching singular-array queries retain distinct wire inputs."""
    events = _synthetic_events("responses")
    for item in _search_snapshots(events):
        if "action" in item:
            item["action"].pop("query", None)
            item["action"].pop("queries", None)
            item["action"].update(query_fields)
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    calls = [block for block in result.content if block.type == "server_tool_use"]
    assert len(calls) == 1
    assert calls[0].input == expected_input
    assert calls[0].id == next(
        item.id for item in reference.output if item.type == "web_search_call"
    )
    assert (
        "".join(block.text for block in result.content if block.type == "text")
        == reference.output_text
    )


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("action", ["open_page", "find_in_page"])
@pytest.mark.parametrize("mixed", [False, True])
async def test_synthetic_browsing_omission(stream: bool, action: str, mixed: bool):
    """Ensures browsing-only and mixed search/browsing streams retain answers without invented search calls."""
    events = _synthetic_events("responses")
    search_id = next(item["id"] for item in _search_snapshots(events))
    browsing = {"type": action, "url": "https://www.python.org/"}
    if action == "find_in_page":
        browsing["pattern"] = "Python"
    if mixed:
        original_done = next(
            event
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "web_search_call"
        )
        insertion = events.index(original_done) + 1
        browse_item = {
            "id": "ws_browse",
            "type": "web_search_call",
            "status": "completed",
            "action": browsing,
        }
        for event in events[insertion:]:
            if "output_index" in event:
                event["output_index"] += 1
            response = event.get("response", {})
            if response.get("output"):
                response["output"].insert(2, deepcopy(browse_item))
        events[insertion:insertion] = [
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {**browse_item, "status": "in_progress"},
            },
            {"type": "response.output_item.done", "output_index": 2, "item": browse_item},
        ]
    else:
        for item in _search_snapshots(events):
            if "action" in item:
                item["action"] = deepcopy(browsing)
    for index, event in enumerate(events):
        event["sequence_number"] = index
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    calls = [block for block in result.content if block.type == "server_tool_use"]
    results = [block for block in result.content if block.type == "web_search_tool_result"]
    assert [block.id for block in calls] == ([search_id] if mixed else [])
    assert [block.tool_use_id for block in results] == ([search_id] if mixed else [])
    assert (
        "".join(block.text for block in result.content if block.type == "text")
        == reference.output_text
    )
    assert sum(len(block.citations or []) for block in result.content if block.type == "text") == 1


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("downstream", ["messages", "chat/completions"])
async def test_synthetic_responses_failed_search(stream: bool, downstream: Protocol):
    """Ensures a failed Responses search maps to linked unavailable output or Chat omission, retaining its answer."""
    events = _synthetic_events("responses")
    for item in _search_snapshots(events):
        if item["status"] == "completed":
            item["status"] = "failed"
            item["action"].pop("sources", None)
    if downstream == "chat/completions":
        events = _without_reasoning(events, "responses")
    result, reference = await _synthetic_replay(downstream, "responses", stream, events)
    if downstream == "messages":
        call = next(block for block in result.content if block.type == "server_tool_use")
        error = next(block for block in result.content if block.type == "web_search_tool_result")
        assert error.tool_use_id == call.id
        assert error.content.type == "web_search_tool_result_error"
        assert error.content.error_code == "unavailable"
        assert result.stop_reason == "end_turn"
        text = "".join(block.text for block in result.content if block.type == "text")
    else:
        assert not result.choices[0].message.tool_calls
        assert result.choices[0].finish_reason == "stop"
        text = result.choices[0].message.content
    assert text == reference.output_text


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
@pytest.mark.parametrize("error_code", ["too_many_requests", "max_uses_exceeded"])
@pytest.mark.parametrize("answer", [False, True])
async def test_synthetic_messages_search_error(
    stream: bool, downstream: Protocol, error_code: str, answer: bool
):
    """Ensures HTTP-200 search errors remain tool results, preserving actual answers or genuinely empty content."""
    events = _synthetic_events("messages")
    for event in events:
        block = event.get("content_block", {})
        if block.get("type") == "web_search_tool_result":
            block["content"] = {"type": "web_search_tool_result_error", "error_code": error_code}
    if not answer:
        events = [event for event in events if event.get("index", -1) < 3]
    if downstream == "chat/completions":
        events = _without_reasoning(events, "messages")
    result, reference = await _synthetic_replay(downstream, "messages", stream, events)
    expected_text = "".join(block.text for block in reference.content if block.type == "text")
    if downstream == "responses":
        call = next(item for item in result.output if item.type == "web_search_call")
        assert call.status == "failed"
        assert call.id == next(
            block.id for block in reference.content if block.type == "server_tool_use"
        )
        assert result.status == "completed"
        assert result.output_text == expected_text
    else:
        assert result.choices[0].message.content == (expected_text or None)
        assert result.choices[0].finish_reason == "stop"
        assert not result.choices[0].message.tool_calls


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
async def test_synthetic_pause_has_no_invented_result(stream: bool, downstream: Protocol):
    """Ensures caller-managed pause emits the exact downstream terminal state without a fabricated result or retry."""
    events = [event for event in _synthetic_events("messages") if event.get("index", -1) < 2]
    for event in events:
        if event["type"] == "message_delta":
            event["delta"]["stop_reason"] = "pause_turn"
    if downstream == "chat/completions":
        events = _without_reasoning(events, "messages")
    result, reference = await _synthetic_replay(downstream, "messages", stream, events)
    assert reference.stop_reason == "pause_turn"
    assert not any(block.type == "web_search_tool_result" for block in reference.content)
    if downstream == "responses":
        assert result.status == "incomplete"
        assert result.incomplete_details is None
        call = next(item for item in result.output if item.type == "web_search_call")
        assert call.status == "in_progress"
        assert getattr(call.action, "sources", None) is None
    else:
        assert result.choices[0].finish_reason == "length"
        assert result.choices[0].message.content is None
        assert not result.choices[0].message.tool_calls


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("timing", ["initial", "delta", "final"])
@pytest.mark.parametrize(
    ("text", "ranges", "expected_text", "associations"),
    [
        ("A😀βé Z", [(0, 4), (2, 6)], ["A😀", "βé", " Z"], [[0], [0, 1], [1]]),
        ("abc de f", [(0, 6), (7, 8)], ["abc de ", "f"], [[0], [1]]),
        ("abc", [(0, 0)], ["abc"], [[0]]),
        ("abc", [(1, 1)], ["abc"], [[0]]),
        ("abc", [(3, 3)], ["abc"], [[0]]),
    ],
    ids=["unicode-overlap", "whitespace-gap", "point-start", "point-middle", "point-end"],
)
async def test_synthetic_citation_boundaries_match_sdk_and_history(
    stream: bool, timing: str, text, ranges, expected_text, associations
):
    """Ensures SDK attribution and exact text survive output and compatible-provider history projection."""
    citations = [
        {
            "type": "url_citation",
            "url": f"https://example.com/{index}",
            "title": None if index == 0 else "B",
            "start_index": start,
            "end_index": end,
        }
        for index, (start, end) in enumerate(ranges)
    ]
    events = []
    sent_text = False
    for event in _synthetic_events("responses"):
        if event["type"] == "response.output_text.delta":
            if sent_text or timing == "initial":
                continue
            sent_text = True
            event["delta"] = text
        elif event["type"] == "response.output_text.annotation.added":
            if timing != "delta":
                continue
            for index, citation in enumerate(citations):
                events.append(
                    {**event, "annotation_index": index, "annotation": deepcopy(citation)}
                )
            continue
        elif event["type"] == "response.output_text.done":
            event["text"] = text
        elif event["type"] == "response.content_part.done" or (
            event["type"] == "response.content_part.added" and timing == "initial"
        ):
            event["part"]["text"] = text
            event["part"]["annotations"] = deepcopy(citations)
        item = event.get("item", {})
        if item.get("type") == "message" and item.get("content"):
            item["content"][0]["text"] = text
            item["content"][0]["annotations"] = deepcopy(citations)
        for item in event.get("response", {}).get("output", []):
            if item["type"] == "message":
                item["content"][0]["text"] = text
                item["content"][0]["annotations"] = deepcopy(citations)
        events.append(event)
    for index, event in enumerate(events):
        event["sequence_number"] = index
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    blocks = [block for block in result.content if block.type == "text"]
    assert "".join(block.text for block in blocks) == reference.output_text == text
    assert all(block.text.strip() for block in blocks)
    assert [block.text for block in blocks] == expected_text
    message = next(item for item in reference.output if item.type == "message")
    sources = message.content[0].annotations
    assert [[citation.url for citation in block.citations or []] for block in blocks] == [
        [sources[index].url for index in indices] for indices in associations
    ]
    for block in blocks:
        for citation in block.citations or []:
            source = next(source for source in sources if source.url == citation.url)
            assert citation.cited_text == (
                text[source.start_index : source.end_index] or block.text
            )
            assert citation.encrypted_index == "N/A"
            assert citation.title == (source.title or "")

    body, expected_request = _request_pair("responses", "messages", stream)
    history = message.model_dump(exclude_none=True)
    for citation in history["content"][0]["annotations"]:
        citation.setdefault("title", "")
    body["input"] = [
        {"role": "user", "content": [{"type": "input_text", "text": body["input"]}]},
        history,
        {"role": "user", "content": [{"type": "input_text", "text": "Continue."}]},
    ]
    expected_request["messages"].extend(
        [
            {
                "role": "assistant",
                "content": [block.model_dump(exclude_none=True) for block in blocks],
            },
            {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
        ]
    )
    requests = []
    async with _get_http_client(
        [_path("messages")],
        "messages",
        expected_body=expected_request,
        requests=requests,
    ) as http_client:
        followup = await Molto(http_client=http_client).responses(
            body=body, endpoint="messages", provider="openai", api_key="test-api-key"
        )
        if stream:
            _ = [event async for event in followup]
    assert len(requests) == 1
    outgoing = json_loads(requests[0].content)
    AnthropicMessagesCreateRequest.model_validate(outgoing)
    history_blocks = outgoing["messages"][-2]["content"]
    assert all(block["text"].strip() for block in history_blocks)
    assert "".join(block["text"] for block in history_blocks) == reference.output_text


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("upstream", RESPONSES_MESSAGES_PROTOCOLS)
async def test_synthetic_messages_usage_accounting(upstream: Protocol, stream: bool):
    """Ensures translated and native Messages preserve exact token, cache, and search counts."""
    expected_usage = {
        "input_tokens": 105,
        "output_tokens": 6039,
        "cache_read_input_tokens": 7123,
        "cache_creation_input_tokens": 7345,
        "output_tokens_details": {"thinking_tokens": 4},
        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
    }
    if upstream == "responses":
        expected_usage["service_tier"] = "standard"
    events = _synthetic_events(upstream)
    for event in events:
        if event.get("response", {}).get("usage") is not None:
            event["response"]["usage"] = {
                "input_tokens": 14573,
                "output_tokens": 6039,
                "total_tokens": 20612,
                "input_tokens_details": {"cached_tokens": 7123, "cache_write_tokens": 7345},
                "output_tokens_details": {"reasoning_tokens": 4},
            }
        elif event["type"] == "message_start":
            event["message"]["usage"] = {**expected_usage, "output_tokens": 0}
        elif event["type"] == "message_delta":
            event["usage"] = deepcopy(expected_usage)

    if upstream == "responses" or stream:
        result, reference = await _synthetic_replay("messages", upstream, stream, events)
        if upstream == "responses":
            assert reference.usage.input_tokens == 14573
            assert reference.usage.output_tokens == expected_usage["output_tokens"]
            assert reference.model_dump()["tool_usage"]["web_search"]["num_requests"] == 1
        else:
            assert (
                reference.usage.model_dump(exclude_unset=True, exclude_none=True) == expected_usage
            )
    else:
        reference = _collect(
            "messages",
            [AnthropicMessagesStreamEventAdapter.validate_python(event) for event in events],
        )
        body = _body("messages", False)

        def handler(request: Request) -> Response:
            assert json_loads(request.content) == body
            return Response(200, json=reference.model_dump(exclude_unset=True))

        async with AsyncClient(transport=MockTransport(handler)) as http_client:
            result = await Molto(http_client=http_client).messages(
                body=body, endpoint="messages", api_key="test-api-key"
            )
    assert result.usage.model_dump(exclude_unset=True, exclude_none=True) == expected_usage


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("count", ["missing", None, 0, 2])
@pytest.mark.parametrize("token_usage", [True, False])
async def test_synthetic_observed_search_usage(stream: bool, count, token_usage: bool):
    """Ensures search counts require token usage and preserve observed zero versus unknown."""
    events = _synthetic_events("responses")
    for event in events:
        response = event.get("response", {})
        if response.get("usage") is not None:
            if count == "missing":
                response["tool_usage"] = None
            else:
                response["tool_usage"] = {"web_search": {"num_requests": count}}
            if not token_usage:
                response["usage"] = None
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    if not token_usage or count in ("missing", None):
        assert result.usage.server_tool_use is None
    else:
        assert reference.tool_usage["web_search"]["num_requests"] == count
        assert result.usage.server_tool_use.web_search_requests == count
        assert result.usage.server_tool_use.web_fetch_requests == 0


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "sources",
    [
        "missing",
        None,
        [],
        [{"type": "url", "url": "https://example.com/", "title": "Source", "page_age": "2 days"}],
    ],
)
async def test_synthetic_missing_empty_and_metadata_sources(stream: bool, sources):
    """Ensures unavailable sources never become fabricated empty results and observed metadata remains exact."""
    events = _synthetic_events("responses")
    for item in _search_snapshots(events):
        if "action" in item:
            if sources == "missing":
                item["action"].pop("sources", None)
            else:
                item["action"]["sources"] = deepcopy(sources)
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    assert reference.status == "completed"
    calls = [block for block in result.content if block.type == "server_tool_use"]
    results = [block for block in result.content if block.type == "web_search_tool_result"]
    assert len(calls) == 1
    if sources in ("missing", None):
        assert results == []
    else:
        assert len(results) == 1
        assert results[0].tool_use_id == calls[0].id
        assert [source.model_dump() for source in results[0].content] == [
            {
                "type": "web_search_result",
                "url": source["url"],
                "title": source["title"],
                "page_age": source["page_age"],
                "encrypted_content": "N/A",
            }
            for source in sources
        ]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(("downstream", "upstream"), SEARCH_ROUTES)
async def test_synthetic_search_http_failure_matches_sdk(
    downstream: Protocol, upstream: Protocol, stream: bool
):
    """Ensures an actual HTTP 429 remains the SDK request error instead of becoming a search-result error."""
    body, expected_request = _request_pair(downstream, upstream, stream)
    error_body = {
        "error": {"type": "rate_limit_error", "message": "Synthetic request limit reached."}
    }
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        assert json_loads(request.content) == expected_request
        return Response(429, json=error_body, headers={"x-should-retry": "false"})

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        sdk = (
            AsyncAnthropic(api_key="test-api-key", http_client=http_client)
            if upstream == "messages"
            else AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        )
        create = sdk.messages.create if upstream == "messages" else sdk.responses.create
        with pytest.raises(
            AnthropicRateLimitError if upstream == "messages" else OpenAIRateLimitError
        ) as sdk_error:
            await create(**expected_request)
        with pytest.raises(type(sdk_error.value)) as actual_error:
            result = await _coordinator(Molto(http_client=http_client), downstream)(
                body=body, endpoint=upstream, api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
        assert actual_error.value.status_code == sdk_error.value.status_code == 429
        assert actual_error.value.body == sdk_error.value.body
        assert actual_error.value.response.json() == error_body
        assert actual_error.value.response.is_closed
        assert len(requests) == 2


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("downstream", "upstream"), [("messages", "responses"), ("responses", "messages")]
)
async def test_synthetic_optional_search_can_return_answer_only(
    downstream: Protocol, upstream: Protocol, stream: bool
):
    """Ensures automatic search selection permits a normal answer without invented search activity."""
    body, expected = _request_pair(downstream, upstream, stream)
    body["tool_choice"] = "auto" if downstream == "responses" else {"type": "auto"}
    if upstream == "responses":
        expected.pop("tool_choice")
    else:
        expected["tool_choice"] = {"type": "auto"}
    events = _synthetic_events(upstream)
    if upstream == "responses":
        events = [event for event in events if event.get("output_index") != 1]
        for index, event in enumerate(events):
            event["sequence_number"] = index
            if event.get("output_index", -1) > 1:
                event["output_index"] -= 1
            response = event.get("response", {})
            if "output" in response:
                response["output"] = [
                    item for item in response["output"] if item["type"] != "web_search_call"
                ]
    else:
        events = [event for event in events if event.get("index") not in {1, 2}]
        for event in events:
            if event.get("index", -1) > 2:
                event["index"] -= 2
    result, reference = await _synthetic_replay(
        downstream, upstream, stream, events, request_pair=(body, expected)
    )
    if downstream == "messages":
        assert not any(
            block.type in {"server_tool_use", "web_search_tool_result"} for block in result.content
        )
        assert (
            "".join(block.text for block in result.content if block.type == "text")
            == reference.output_text
        )
    else:
        assert not any(item.type == "web_search_call" for item in result.output)
        assert result.output_text == "".join(
            block.text for block in reference.content if block.type == "text"
        )


@pytest.mark.parametrize("stream", [True, False])
async def test_synthetic_repeated_searches_preserve_identity(stream: bool):
    """Ensures repeated provider searches retain ordered distinct invocation/result pairs without client calls."""
    events = _synthetic_events("responses")
    second_search = deepcopy([event for event in events if event.get("output_index") == 1])
    for event in second_search:
        event["output_index"] = 2
        if "item_id" in event:
            event["item_id"] = "ws_second"
        if "item" in event:
            event["item"]["id"] = "ws_second"
            if "action" in event["item"]:
                event["item"]["action"]["query"] = "second query"
                event["item"]["action"]["queries"] = ["second query"]
    second_item = deepcopy(second_search[-1]["item"])
    insertion = (
        max(index for index, event in enumerate(events) if event.get("output_index") == 1) + 1
    )
    for event in events[insertion:]:
        if "output_index" in event:
            event["output_index"] += 1
        if event.get("response", {}).get("output"):
            event["response"]["output"].insert(2, deepcopy(second_item))
            event["response"]["tool_usage"]["web_search"]["num_requests"] = 2
    events[insertion:insertion] = second_search
    for index, event in enumerate(events):
        event["sequence_number"] = index
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    calls = [block for block in result.content if block.type == "server_tool_use"]
    results = [block for block in result.content if block.type == "web_search_tool_result"]
    assert [block.id for block in calls] == [
        item.id for item in reference.output if item.type == "web_search_call"
    ]
    assert [block.tool_use_id for block in results] == [block.id for block in calls]
    assert calls[1].input == {"query": "second query"}
    assert len(calls) == len(results) == 2
    assert result.usage.server_tool_use.web_search_requests == 2
    assert (
        "".join(block.text for block in result.content if block.type == "text")
        == reference.output_text
    )


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("new_search", [False, True])
@pytest.mark.parametrize("citation_policy", ["drop", "buffer", "auto"])
async def test_search_history_uses_citation_policy(
    stream: bool, override: bool, new_search: bool, citation_policy: str
):
    """Ensures auto buffers typed history only while buffer also preserves opaque-history citations."""
    body, expected = _request_pair("messages", "responses", stream)
    for request in (body, expected):
        request.pop("tools")
        request.pop("tool_choice")
    expected["input"].extend(_history("responses", True))
    expected.pop("include")
    if override:
        extra_body = {"input": expected["input"]}
    else:
        body["messages"].extend(_history("messages", True))
        extra_body = None
    events = _synthetic_events("responses")
    if not new_search:
        # A follow-up may cite past search results without performing another search.
        events = [event for event in events if event.get("output_index") != 1]
        for index, event in enumerate(events):
            event["sequence_number"] = index
            if event.get("output_index", -1) > 1:
                event["output_index"] -= 1
            response = event.get("response", {})
            if "output" in response:
                response["output"] = [
                    item for item in response["output"] if item["type"] != "web_search_call"
                ]
            response.pop("tool_usage", None)
    result, reference = await _synthetic_replay(
        "messages",
        "responses",
        stream,
        events,
        request_pair=(body, expected),
        extra_body=extra_body,
        citation_policy=citation_policy,
    )
    text_blocks = [block for block in result.content if block.type == "text"]
    assert "".join(block.text for block in text_blocks) == reference.output_text
    citations = [citation for block in text_blocks for citation in block.citations or []]
    if citation_policy == "drop" or (citation_policy == "auto" and override):
        assert citations == []
        return
    annotation = (
        next(item for item in reference.output if item.type == "message").content[0].annotations[0]
    )
    assert len(citations) == 1
    assert citations[0].url == annotation.url
    assert citations[0].encrypted_index == "N/A"
    assert (
        citations[0].cited_text
        == reference.output_text[annotation.start_index : annotation.end_index]
    )


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("refusal_first", [False, True])
async def test_synthetic_multiple_responses_text_parts_have_local_offsets(
    stream: bool, refusal_first: bool
):
    """Ensures text after refusals retains local citation offsets in nonblank, balanced blocks."""
    events = _synthetic_events("responses")
    if refusal_first:
        # Snapshot-only refusals were already omitted; their index must not
        # leave the initial normalized text part open.
        refusal = {"type": "refusal", "refusal": "Cannot answer that part."}
        added_index = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "response.content_part.added"
            and event["part"]["type"] == "output_text"
        )
        refusal_added = deepcopy(events[added_index])
        refusal_added["part"] = refusal
        for event in events:
            if "content_index" in event:
                event["content_index"] += 1
            item = event.get("item", {})
            if item.get("type") == "message" and item.get("content"):
                item["content"].insert(0, deepcopy(refusal))
            for item in event.get("response", {}).get("output", []):
                if item["type"] == "message":
                    item["content"].insert(0, deepcopy(refusal))
        events[added_index:added_index] = [
            refusal_added,
            {**deepcopy(refusal_added), "type": "response.content_part.done"},
        ]
    first_content_index = int(refusal_first)
    text = " Ω "
    citation = {
        "type": "url_citation",
        "url": "https://example.com/second",
        "title": "Second",
        "start_index": 1,
        "end_index": 2,
    }
    added_events = []
    sent_text = False
    for event in deepcopy(events):
        if event.get("content_index") != first_content_index:
            continue
        event["content_index"] = first_content_index + 1
        if event["type"] == "response.output_text.delta":
            if sent_text:
                continue
            event["delta"] = text
            sent_text = True
        elif event["type"] == "response.output_text.done":
            event["text"] = text
        elif event["type"] == "response.output_text.annotation.added":
            event["annotation"] = deepcopy(citation)
        elif event["type"] == "response.content_part.done":
            event["part"]["text"] = text
            event["part"]["annotations"] = [deepcopy(citation)]
        added_events.append(event)
    item_done_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "message"
    )
    part = {"type": "output_text", "text": text, "annotations": [citation], "logprobs": []}
    events[item_done_index]["item"]["content"].append(deepcopy(part))
    for item in events[-1]["response"]["output"]:
        if item["type"] == "message":
            item["content"].append(deepcopy(part))
    events[item_done_index:item_done_index] = added_events
    for index, event in enumerate(events):
        event["sequence_number"] = index
    result, reference = await _synthetic_replay("messages", "responses", stream, events)
    blocks = [block for block in result.content if block.type == "text"]
    assert "".join(block.text for block in blocks) == reference.output_text
    assert all(block.text.strip() for block in blocks)
    assert blocks[-1].text == text
    source = next(item for item in reference.output if item.type == "message").content[-1]
    annotation = source.annotations[0]
    assert (annotation.start_index, annotation.end_index) == (1, 2)
    assert len(blocks[-1].citations) == 1
    assert blocks[-1].citations[0].url == annotation.url
    assert (
        blocks[-1].citations[0].cited_text
        == source.text[annotation.start_index : annotation.end_index]
        == "Ω"
    )
    assert blocks[-1].citations[0].encrypted_index == "N/A"


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("downstream", "upstream"), list(product(RESPONSES_MESSAGES_PROTOCOLS, repeat=2))
)
async def test_search_history_reaches_sdk_error_boundary(
    downstream: Protocol, upstream: Protocol, stream: bool
):
    """Ensures native and compatible-provider reduced history retains recorded SDK error parity."""
    body, expected = _request_pair(downstream, upstream, stream)
    for request, protocol in ((body, downstream), (expected, upstream)):
        key = "input" if protocol == "responses" else "messages"
        if isinstance(request[key], str):
            request[key] = [
                {"role": "user", "content": [{"type": "input_text", "text": request[key]}]}
            ]
        request[key].extend(_history(protocol, True))
    error_path = _matched_cassette_path("error-invalid_request_error", upstream, "stream-00.yaml")
    error_type = AnthropicBadRequestError if upstream == "messages" else OpenAIBadRequestError
    async with _get_http_client([error_path], upstream, expected_body=expected) as http_client:
        sdk = (
            AsyncAnthropic(api_key="test-api-key", http_client=http_client)
            if upstream == "messages"
            else AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        )
        create = sdk.messages.create if upstream == "messages" else sdk.responses.create
        with pytest.raises(error_type) as sdk_error:
            await create(**expected)
    requests = []
    async with _get_http_client(
        [error_path], upstream, expected_body=expected, requests=requests
    ) as http_client:
        with pytest.raises(error_type) as actual_error:
            result = await _coordinator(Molto(http_client=http_client), downstream)(
                body=body,
                endpoint=upstream,
                api_key="test-api-key",
                provider="openai"
                if downstream == "responses" and upstream == "messages"
                else None,
            )
            if stream:
                _ = [event async for event in result]
    assert len(requests) == 1
    assert actual_error.value.status_code == sdk_error.value.status_code == 400
    assert actual_error.value.body == sdk_error.value.body
    assert actual_error.value.response.json() == sdk_error.value.response.json()
