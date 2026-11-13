"""Independent Anthropic recordings for hosted search and metadata-preserving continuation."""

from copy import deepcopy
from pathlib import Path

import pytest
from anthropic import AsyncAnthropic
from httpx import AsyncClient, MockTransport, Request, Response
from openai.types.responses import Response as OpenAIResponse

from molto import Molto
from molto.adapters.openai import OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import PartEndEvent, ResponseEndEvent, SearchCall, TextOutput
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.utils.io import json_loads
from molto.utils.test import assert_stream_lifecycle, collect_sdk_response
from tests.core.test_interface import _get_http_client, _read_cassette
from tests.core.test_web_search import (
    _history,
    _messages_semantics,
    _path,
    _request_pair,
    _sdk_result,
)

SEARCH = (
    Path(__file__).parents[1]
    / "cassettes"
    / "anthropic"
    / "web_search"
    / "anthropic-messages-stream-00.yaml"
)
PRESERVED_CONTINUATION = SEARCH.parent / "anthropic-messages-stream-01.yaml"


def _responses_body(stream: bool):
    native = _read_cassette(SEARCH)["request"]["body"]
    return {
        "model": native["model"],
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": native["messages"][0]["content"][0]["text"]}
                ],
            }
        ],
        "tools": [{"type": "web_search"}],
        "tool_choice": "required",
        "max_tool_calls": 1,
        "max_output_tokens": 1024,
        "stream": stream,
    }


async def _sdk_events(path: Path = SEARCH):
    async with _get_http_client([path], "messages") as http_client:
        sdk = AsyncAnthropic(api_key="test-api-key", http_client=http_client)
        async with await sdk.messages.create(**_read_cassette(path)["request"]["body"]) as stream:
            return [event async for event in stream]


async def test_recorded_anthropic_search_preserves_native_sdk_events():
    """Ensures genuine search argument/citation deltas and terminal usage survive native replay."""
    expected = await _sdk_events()
    async with _get_http_client([SEARCH], "messages") as http_client:
        stream = await Molto(http_client=http_client).messages(
            body=_read_cassette(SEARCH)["request"]["body"],
            endpoint="messages",
            api_key="test-api-key",
        )
        actual = [event async for event in stream]
    assert_stream_lifecycle("messages", actual)
    assert [type(event) for event in actual] == [type(event) for event in expected]
    assert [event.model_dump(exclude_unset=True) for event in actual] == [
        event.model_dump(exclude_unset=True) for event in expected
    ]
    search_indices = {
        event.index
        for event in expected
        if event.type == "content_block_start" and event.content_block.type == "server_tool_use"
    }
    assert any(
        event.type == "content_block_delta"
        and event.index in search_indices
        and event.delta.type == "input_json_delta"
        and event.delta.partial_json
        for event in expected
    )
    assert any(
        event.type == "content_block_delta" and event.delta.type == "citations_delta"
        for event in expected
    )
    terminal_usage = next(event.usage for event in expected if event.type == "message_delta")
    assert terminal_usage.input_tokens != expected[0].message.usage.input_tokens
    assert terminal_usage.server_tool_use.web_search_requests == 1


@pytest.mark.parametrize("stream", [True, False])
async def test_recorded_anthropic_search_translates_sdk_semantics(stream: bool):
    """Ensures both Responses modes preserve observed search inputs, citations, sources, and usage."""
    reference = collect_sdk_response("messages", await _sdk_events())
    async with _get_http_client([SEARCH], "messages") as http_client:
        result = await Molto(http_client=http_client).responses(
            body=_responses_body(stream), endpoint="messages", api_key="test-api-key"
        )
        if stream:
            events = [event async for event in result]
            assert_stream_lifecycle("responses", events)
            result = collect_sdk_response("responses", events)

    native_text = [block for block in reference.content if block.type == "text"]
    text_parts = [
        part for item in result.output if item.type == "message" for part in item.content
    ]
    assert result.id == reference.id
    assert result.model == reference.model
    assert result.status == "completed" and reference.stop_reason == "end_turn"
    assert result.output_text == "".join(block.text for block in native_text)
    assert [
        (citation.url, citation.title, part.text[citation.start_index : citation.end_index])
        for part in text_parts
        for citation in part.annotations
    ] == [
        (citation.url, citation.title, block.text)
        for block in native_text
        for citation in block.citations or []
    ]
    native_calls = [block for block in reference.content if block.type == "server_tool_use"]
    native_results = [
        block for block in reference.content if block.type == "web_search_tool_result"
    ]
    calls = [item for item in result.output if item.type == "web_search_call"]
    assert [(call.id, call.action.query) for call in calls] == [
        (block.id, block.input["query"]) for block in native_calls
    ]
    assert [call.id for call in calls] == [block.tool_use_id for block in native_results]
    assert all(call.status == "completed" for call in calls)
    assert [[source.url for source in call.action.sources] for call in calls] == [
        [source.url for source in block.content] for block in native_results
    ]
    assert [
        source.model_dump().get("extra_content")
        for call in calls
        for source in call.action.sources
    ] == [
        {
            "anthropic": {
                key: value
                for key, value in source.model_dump(exclude_none=True).items()
                if key in {"encrypted_content", "title", "page_age"}
            }
        }
        for block in native_results
        for source in block.content
    ]
    assert [
        citation.model_dump().get("extra_content")
        for part in text_parts
        for citation in part.annotations
    ] == [
        {
            "anthropic": {
                "encrypted_index": citation.encrypted_index,
                "cited_text": citation.cited_text,
            }
        }
        for block in native_text
        for citation in block.citations or []
    ]
    if stream:
        assert [
            event.model_dump()["annotation"]
            for event in events
            if event.type == "response.output_text.annotation.added"
        ] == [citation.model_dump() for part in text_parts for citation in part.annotations]
        assert [
            event.item.model_dump()
            for event in events
            if event.type == "response.output_item.done"
        ] == [item.model_dump() for item in result.output]
    usage = reference.usage
    assert result.usage.input_tokens == (
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    )
    assert result.usage.output_tokens == usage.output_tokens
    assert result.usage.total_tokens == result.usage.input_tokens + usage.output_tokens
    assert result.usage.input_tokens_details.cached_tokens == usage.cache_read_input_tokens
    assert (
        result.usage.input_tokens_details.cache_write_tokens == usage.cache_creation_input_tokens
    )


async def test_recorded_anthropic_search_preserves_ir_usage_and_citation_state():
    """Ensures native search counts and opaque citation state remain available before reduction."""
    reference = collect_sdk_response("messages", await _sdk_events())
    capabilities = PROVIDER_CAPABILITIES["anthropic"]
    request = OpenAIResponsesAdapter(capabilities).normalize_request(
        OpenAIResponsesCreateRequest(**_responses_body(True))
    )
    async with _get_http_client([SEARCH], "messages") as http_client:
        client = AnthropicMessagesClient(
            http_client, api_key="test-api-key", capabilities=capabilities
        )
        events = [event async for event in client.events(request)]
    assert isinstance(events[-1], ResponseEndEvent)
    assert events[-1].usage.web_search_requests == (
        reference.usage.server_tool_use.web_search_requests
    )
    parts = [event.part for event in events if isinstance(event, PartEndEvent)]
    assert [(part.id, part.query) for part in parts if isinstance(part, SearchCall)] == [
        (block.id, block.input["query"])
        for block in reference.content
        if block.type == "server_tool_use"
    ]
    assert [
        (citation.cited_text, citation.encrypted_index)
        for part in parts
        if isinstance(part, TextOutput)
        for citation in part.citations
    ] == [
        (citation.cited_text, citation.encrypted_index)
        for block in reference.content
        if block.type == "text"
        for citation in block.citations or []
    ]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("state", [None, "", "N/A"])
@pytest.mark.parametrize("part_kind", ["source", "citation"])
async def test_recorded_history_rejects_missing_search_state_before_transport(
    stream: bool, state: str | None, part_kind: str
):
    """Ensures SDK-serialized history with lost provider state fails before an Anthropic followup."""
    requests = []
    body = _responses_body(stream)
    async with _get_http_client([SEARCH], "messages", requests=requests) as http_client:
        molto = Molto(http_client=http_client)
        result = await molto.responses(body=body, endpoint="messages", api_key="test-api-key")
        if stream:
            result = collect_sdk_response("responses", [event async for event in result])
        restored = OpenAIResponse.model_validate_json(result.model_dump_json(exclude_none=True))
        history = [item.model_dump(exclude_none=True) for item in restored.output]
        if part_kind == "source":
            target = next(
                source
                for item in history
                if item["type"] == "web_search_call"
                for source in item["action"]["sources"]
            )
            field = "encrypted_content"
        else:
            target = next(
                citation
                for item in history
                if item["type"] == "message"
                for part in item["content"]
                for citation in part["annotations"]
            )
            field = "encrypted_index"
        if state is None:
            del target["extra_content"]
        else:
            target["extra_content"]["anthropic"][field] = state
        body["input"].extend(history)
        body["input"].append({"role": "user", "content": "Continue."})
        # Continuation state is required even when no new search is declared.
        for key in ("tools", "tool_choice", "max_tool_calls"):
            body.pop(key)
        with pytest.raises(UnsupportedInput, match=field):
            await molto.responses(body=body, endpoint="messages", api_key="test-api-key")
    assert len(requests) == 1


@pytest.mark.parametrize("stream", [True, False])
async def test_recorded_search_history_preserves_provider_state(stream: bool):
    """Ensures SDK-serialized Responses history restores the exact native blocks and accepted continuation."""
    native_reference = collect_sdk_response("messages", await _sdk_events())
    followup_reference = collect_sdk_response(
        "messages", await _sdk_events(PRESERVED_CONTINUATION)
    )
    native_followup = _read_cassette(PRESERVED_CONTINUATION)["request"]["body"]
    requests = []
    body = _responses_body(stream)
    async with _get_http_client(
        [SEARCH, PRESERVED_CONTINUATION], "messages", requests=requests
    ) as http_client:
        molto = Molto(http_client=http_client)
        result = await molto.responses(body=body, endpoint="messages", api_key="test-api-key")
        if stream:
            result = collect_sdk_response("responses", [event async for event in result])
        # A caller can persist the SDK response as JSON and append its output
        # later; both SDK loading and Molto request validation must retain state.
        restored = OpenAIResponse.model_validate_json(result.model_dump_json(exclude_none=True))
        history = [item.model_dump(exclude_none=True) for item in restored.output]
        assert history == [item.model_dump(exclude_none=True) for item in result.output]
        body["input"].extend(history)
        body["input"].append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": native_followup["messages"][-1]["content"][0]["text"],
                    }
                ],
            }
        )
        body["tool_choice"] = "none"
        original_body = deepcopy(body)
        validated = OpenAIResponsesCreateRequest.model_validate(body)
        assert validated.model_dump() == original_body
        assert validated.model_dump() == original_body
        result = await molto.responses(body=validated, endpoint="messages", api_key="test-api-key")
        if stream:
            events = [event async for event in result]
            assert_stream_lifecycle("responses", events)
            result = collect_sdk_response("responses", events)

    assert body == original_body
    assert len(requests) == 2
    sent = json_loads(requests[1].content)
    assert sent == native_followup
    expected_blocks = [block.model_dump(exclude_none=True) for block in native_reference.content]
    for block in expected_blocks:
        # Direct-call provenance is outside the translated history contract.
        assert block.pop("caller", {"type": "direct"}) == {"type": "direct"}
    assert sent["messages"][1]["content"] == expected_blocks
    assert result.id == followup_reference.id
    assert result.model == followup_reference.model
    assert result.status == "completed" and followup_reference.stop_reason == "end_turn"
    assert result.output_text == "".join(
        block.text for block in followup_reference.content if block.type == "text"
    )
    assert result.usage.input_tokens == (
        followup_reference.usage.input_tokens
        + followup_reference.usage.cache_read_input_tokens
        + followup_reference.usage.cache_creation_input_tokens
    )
    assert result.usage.output_tokens == followup_reference.usage.output_tokens
    assert [
        (
            citation.url,
            citation.title,
            part.text[citation.start_index : citation.end_index],
            citation.extra_content,
        )
        for item in result.output
        if item.type == "message"
        for part in item.content
        for citation in part.annotations
    ] == [
        (
            citation.url,
            citation.title,
            block.text,
            {
                "anthropic": {
                    "encrypted_index": citation.encrypted_index,
                    "cited_text": citation.cited_text,
                }
            },
        )
        for block in followup_reference.content
        if block.type == "text"
        for citation in block.citations or []
    ]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("history_type", ["web_search_call", "message"])
@pytest.mark.parametrize("history_container", [list, tuple, iter])
@pytest.mark.parametrize(
    ("endpoint", "provider"),
    [
        ("responses", None),
        ("responses", "anthropic"),
        ("chat/completions", None),
        ("messages", "openai"),
        ("messages", "vllm"),
    ],
)
async def test_search_metadata_rejects_other_upstream_routes(
    endpoint: str, provider: str | None, history_type: str, stream: bool, history_container
):
    """Ensures either source or citation state fails before transport when its Anthropic route changes."""
    body = _responses_body(False)
    async with _get_http_client([SEARCH], "messages") as http_client:
        result = await Molto(http_client=http_client).responses(
            body=body, endpoint="messages", api_key="test-api-key"
        )
    body["stream"] = stream
    body["input"].extend(
        item.model_dump(exclude_none=True) for item in result.output if item.type == history_type
    )
    if endpoint == "responses":
        # Native routing checks only Molto-owned metadata, even with provider extensions.
        body["provider_option"] = True
    for item in body["input"]:
        if item.get("type") == "web_search_call":
            item["action"]["sources"] = history_container(item["action"].get("sources", []))
        elif item.get("type") == "message":
            for part in item["content"]:
                if part["type"] == "output_text":
                    part["annotations"] = history_container(part["annotations"])
            item["content"] = history_container(item["content"])
    body["input"] = history_container(body["input"])
    requests = []

    def handle(request: Request) -> Response:
        requests.append(request)
        raise AssertionError("Provider-specific search state must fail before HTTP")

    async with AsyncClient(transport=MockTransport(handle)) as http_client:
        with pytest.raises(UnsupportedInput, match="extra_content"):
            result = await Molto(http_client=http_client).responses(
                body=body, endpoint=endpoint, provider=provider, api_key="test-api-key"
            )
            if stream:
                _ = [event async for event in result]
    assert requests == []


@pytest.mark.parametrize("stream", [True, False])
async def test_messages_search_history_does_not_extend_openai_requests(stream: bool):
    """Ensures synthetic native Messages state does not leak into an actual Responses upstream body."""
    body, expected_request = _request_pair("messages", "responses", stream)
    history = _history("messages", True)
    source = history[0]["content"][1]["content"][0]
    source.update(
        encrypted_content="original-provider-content", title="Original title", page_age="2024"
    )
    citation = history[0]["content"][2]["citations"][0]
    citation.update(encrypted_index="original-provider-index", cited_text="Original excerpt")
    body["messages"].extend(history)
    expected_request["input"].extend(_history("responses", True))
    reference = collect_sdk_response("messages", await _sdk_result("messages", True))
    requests = []
    async with _get_http_client(
        [_path("responses")], "responses", expected_body=expected_request, requests=requests
    ) as http_client:
        result = await Molto(http_client=http_client).messages(
            body=body, endpoint="responses", api_key="test-api-key", citation_policy="buffer"
        )
        if stream:
            result = collect_sdk_response("messages", [event async for event in result])
    assert len(requests) == 1
    assert json_loads(requests[0].content) == expected_request
    assert b"extra_content" not in requests[0].content
    assert b"original-provider" not in requests[0].content
    assert _messages_semantics(result) == _messages_semantics(reference)
