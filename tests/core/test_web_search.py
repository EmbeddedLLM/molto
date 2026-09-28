"""Public hosted-search replay parity and explicitly synthetic boundary regressions."""

import asyncio
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
from pydantic import BaseModel

from molto import Molto
from molto.types.exceptions import UnsupportedInput
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


def _chat_semantics(response: BaseModel, upstream: Protocol) -> dict[str, Any]:
    data = response.model_dump(mode="json")
    choice = data["choices"][0]
    message = choice["message"]
    assert not message.get("tool_calls")
    assert not message.get("annotations")
    assert not message.get("reasoning_content")
    return {
        "id": data["id"],
        "model": data["model"],
        "created": data["created"] if upstream == "responses" else None,
        "content": message["content"],
        "finish_reason": choice["finish_reason"],
        "usage": {
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "total_tokens": data["usage"]["total_tokens"],
            "cached_tokens": data["usage"]["prompt_tokens_details"]["cached_tokens"],
            "cache_write_tokens": data["usage"]["prompt_tokens_details"].get(
                "cache_write_tokens", 0
            ),
            "reasoning_tokens": data["usage"]["completion_tokens_details"]["reasoning_tokens"],
        },
    }


@pytest.mark.parametrize("protocol", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize("stream", [True, False])
async def test_native_search_matches_sdk(protocol: Protocol, stream: bool):
    """Ensures native hosted-search fields and SDK event types remain byte-value equivalent."""
    expected = await _sdk_result(protocol, stream)
    async with _get_http_client([_path(protocol, stream)], protocol) as http_client:
        result = await _coordinator(Molto(http_client=http_client), protocol)(
            body=_body(protocol, stream), endpoint=protocol, api_key="test-api-key"
        )
        actual = [event async for event in result] if stream else result
    if stream:
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
    ("downstream", "upstream"), [route for route in SEARCH_ROUTES if route[0] != route[1]]
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
            body=body, endpoint=upstream, api_key="test-api-key"
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
        assert actual_semantics == expected_semantics
    elif downstream == "messages":
        assert _messages_semantics(actual) == _messages_semantics(expected)
    else:
        assert _chat_semantics(actual, upstream) == _chat_semantics(expected, upstream)


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


@pytest.mark.parametrize("downstream", PROTOCOLS)
@pytest.mark.parametrize("stream", [True, False])
async def test_chat_upstream_search_rejected_before_transport(downstream: Protocol, stream: bool):
    """Ensures every caller rejects hosted search on a Chat upstream before opening HTTP."""
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        raise AssertionError("Hosted search must fail before transport")

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        with pytest.raises(UnsupportedInput, match="[Ss]earch"):
            result = await _coordinator(Molto(http_client=http_client), downstream)(
                body=_body(downstream, stream), endpoint="chat/completions", api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
    assert requests == []


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
        result = await Molto(http_client=http_client).chat_completions(
            body=body, endpoint=upstream, api_key="test-api-key"
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle("chat/completions", events)
        else:
            assert not result.choices[0].message.tool_calls


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
    """Ensures ordinary followups and reduced search history reach the fixed upstream, including N/A placeholders."""
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
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body, endpoint=upstream, api_key="test-api-key"
        )
        if stream:
            _ = [event async for event in result]


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


@pytest.mark.parametrize(("downstream", "upstream"), SEARCH_ROUTES)
@pytest.mark.parametrize("cancel", [False, True])
async def test_search_stream_close_releases_provider(
    downstream: Protocol, upstream: Protocol, cancel: bool
):
    """Ensures early close and cancellation release a hosted-search stream without closing shared HTTP."""
    body, expected = _request_pair(downstream, upstream, True)
    payload = "".join(_read_cassette(_path(upstream))["response"]["sse"])
    provider_stream = _SearchStream(payload, pause=cancel)

    def handler(request: Request) -> Response:
        assert json_loads(request.content) == expected
        return Response(200, stream=provider_stream, headers={"content-type": "text/event-stream"})

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
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
async def test_multi_query_projection_failure_closes_provider(stream: bool):
    """Ensures a synthetic multi-query search rejects Messages projection and releases its HTTP stream."""
    body, expected = _request_pair("messages", "responses", stream)
    records = []
    for record in _sse_records(_read_cassette(_path("responses")), "responses"):
        event = load_sse(record)
        if (
            event["type"] == "response.output_item.done"
            and event["item"]["type"] == "web_search_call"
        ):
            event["item"]["action"]["queries"] = ["first", "second"]
        records.append(dumps_sse(event))
    provider_stream = _SearchStream("".join(records))

    def handler(request: Request) -> Response:
        assert json_loads(request.content) == expected
        return Response(200, stream=provider_stream, headers={"content-type": "text/event-stream"})

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        with pytest.raises(UnsupportedInput, match="Search item .*single.query") as caught:
            result = await Molto(http_client=http_client).messages(
                body=body, endpoint="responses", api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
        assert type(caught.value) is UnsupportedInput
        assert provider_stream.closed
        assert not http_client.is_closed


def _synthetic_events(protocol: Protocol) -> list[dict[str, Any]]:
    """Copy recorded events for synthetic cases without modifying cassette evidence."""
    return [load_sse(record) for record in _sse_records(_read_cassette(_path(protocol)), protocol)]


async def _synthetic_replay(
    downstream: Protocol,
    upstream: Protocol,
    stream: bool,
    events: list[dict[str, Any]],
    *,
    request_pair=None,
):
    payload = "".join(dumps_sse(event, event_type=event["type"]) for event in events)
    body, expected_request = request_pair or _request_pair(downstream, upstream, stream)

    # A parsed SDK reference proves that mutations still use the native SDK's
    # wire boundary; synthetic provider behavior is not claimed as a recording.
    def sdk_handler(request: Request) -> Response:
        return Response(
            200, content=payload.encode(), headers={"content-type": "text/event-stream"}
        )

    async with AsyncClient(transport=MockTransport(sdk_handler)) as http_client:
        if upstream == "messages":
            sdk = AsyncAnthropic(api_key="test-api-key", http_client=http_client)
            reference = await sdk.messages.create(**_body(upstream))
        else:
            sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
            reference = await sdk.responses.create(**_body(upstream))
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
            body=body, endpoint=upstream, api_key="test-api-key"
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
        assert result.choices[0].message.content == expected_text
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
        assert result.choices[0].message.content == ""
        assert not result.choices[0].message.tool_calls


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("timing", ["initial", "delta", "final"])
async def test_synthetic_unicode_overlapping_late_citations(stream: bool, timing: str):
    """Ensures code-point offsets, overlaps, and late citations preserve exact Unicode answer associations."""
    text = "A😀βé Z"
    citations = [
        {
            "type": "url_citation",
            "url": "https://example.com/a",
            "title": None,
            "start_index": 0,
            "end_index": 4,
        },
        {
            "type": "url_citation",
            "url": "https://example.com/b",
            "title": "B",
            "start_index": 2,
            "end_index": 6,
        },
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
    assert reference.output_text == text
    assert [block.text for block in blocks] == ["A😀", "βé", " Z"]
    assert [[citation.url for citation in block.citations] for block in blocks] == [
        [citations[0]["url"]],
        [citations[0]["url"], citations[1]["url"]],
        [citations[1]["url"]],
    ]
    for block in blocks:
        for citation in block.citations:
            source = next(source for source in citations if source["url"] == citation.url)
            assert citation.cited_text == text[source["start_index"] : source["end_index"]]
            assert citation.encrypted_index == "N/A"
            assert citation.title == (source["title"] or "")


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("count", ["missing", None, 0, 2])
@pytest.mark.parametrize("token_usage", [True, False])
async def test_synthetic_observed_search_usage(stream: bool, count, token_usage: bool):
    """Ensures observed zero and nonzero counts survive while missing/null counts remain unknown."""
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
    if count in ("missing", None):
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
async def test_search_history_without_tools_still_buffers_citations(stream: bool):
    """Ensures fixed-route followups keep late citations even when no search tool is redeclared."""
    body, expected = _request_pair("messages", "responses", stream)
    for request in (body, expected):
        request.pop("tools")
        request.pop("tool_choice")
    body["messages"].extend(_history("messages", True))
    expected["input"].extend(_history("responses", True))
    expected.pop("include")
    events = _synthetic_events("responses")
    result, reference = await _synthetic_replay(
        "messages", "responses", stream, events, request_pair=(body, expected)
    )
    text_blocks = [block for block in result.content if block.type == "text"]
    assert "".join(block.text for block in text_blocks) == reference.output_text
    citations = [citation for block in text_blocks for citation in block.citations or []]
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
async def test_synthetic_multiple_responses_text_parts_have_local_offsets(stream: bool):
    """Ensures a later Responses content part uses local citation offsets without losing spaces or Unicode."""
    events = _synthetic_events("responses")
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
        if event.get("content_index") != 0:
            continue
        event["content_index"] = 1
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
    assert [block.text for block in blocks[-3:]] == [" ", "Ω", " "]
    assert not blocks[-3].citations
    assert not blocks[-1].citations
    assert len(blocks[-2].citations) == 1
    assert blocks[-2].citations[0].url == citation["url"]
    assert blocks[-2].citations[0].cited_text == "Ω"
    assert blocks[-2].citations[0].encrypted_index == "N/A"


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("downstream", "upstream"), list(product(RESPONSES_MESSAGES_PROTOCOLS, repeat=2))
)
async def test_search_history_reaches_sdk_error_boundary(
    downstream: Protocol, upstream: Protocol, stream: bool
):
    """Ensures reduced history reaches transport and a recorded SDK rejection propagates without local placeholder validation."""
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
                body=body, endpoint=upstream, api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
    assert len(requests) == 1
    assert actual_error.value.status_code == sdk_error.value.status_code == 400
    assert actual_error.value.body == sdk_error.value.body
    assert actual_error.value.response.json() == sdk_error.value.response.json()
