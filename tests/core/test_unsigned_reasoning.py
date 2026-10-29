"""SDK parity for synthetic unsigned reasoning derived from recorded streams."""

from itertools import permutations

import pytest
from anthropic import AsyncAnthropic
from anthropic.lib.streaming._messages import accumulate_event
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import BaseModel

from molto import Molto
from molto.types.exceptions import UnsupportedInput
from molto.utils.io import dumps_sse, json_loads, load_sse
from tests.core.test_interface import (
    PROTOCOLS,
    Protocol,
    _assert_responses_stream_invariants,
    _coordinator,
    _matched_cassette_path,
    _read_cassette,
    _sse_records,
)


def _collect(protocol: Protocol, events: list[BaseModel]) -> BaseModel:
    if protocol == "responses":
        assert events[-1].type == "response.completed"
        return events[-1].response
    if protocol == "chat/completions":
        state = ChatCompletionStreamState()
        for event in events:
            state.handle_chunk(event.model_copy(deep=True))
        return state.current_completion_snapshot
    snapshot = None
    for event in events:
        snapshot = accumulate_event(event=event.model_copy(deep=True), current_snapshot=snapshot)
    assert snapshot is not None
    return snapshot


def _assert_lifecycle(protocol: Protocol, events: list[BaseModel]) -> None:
    if protocol == "responses":
        _assert_responses_stream_invariants([dumps_sse(event) for event in events])
        assert events[0].type == "response.created"
        assert events[-1].type == "response.completed"
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


def _unsigned_events(protocol: Protocol) -> list[dict]:
    scenario = "reasoning_raw" if protocol == "chat/completions" else "reasoning_encrypted"
    cassette = _read_cassette(_matched_cassette_path(scenario, protocol, "stream-00.yaml"))
    events = [load_sse(record) for record in _sse_records(cassette, protocol)]

    def remove_signatures(value):
        if isinstance(value, dict):
            value.pop("encrypted_content", None)
            if "signature" in value:
                value["signature"] = ""
            for child in value.values():
                remove_signatures(child)
        elif isinstance(value, list):
            for child in value:
                remove_signatures(child)

    # These mutations model synthetic unsigned output, not observed provider behavior.
    # Chat has no signature field; its unchanged wire reasoning is classified by
    # the OpenAI summary capability selected by the public call below.
    events = [
        event
        for event in events
        if event.get("type") != "content_block_delta"
        or event["delta"]["type"] != "signature_delta"
    ]
    remove_signatures(events)
    return events


def _body(protocol: Protocol, stream: bool) -> dict:
    body = {"model": "test-model", "stream": stream}
    if protocol == "responses":
        body.update(input="Question", max_output_tokens=2048)
    else:
        body["messages"] = [{"role": "user", "content": "Question"}]
        body["max_tokens" if protocol == "messages" else "max_completion_tokens"] = 2048
    return body


def _expected_request(downstream: Protocol, upstream: Protocol) -> dict:
    body = _body(upstream, True)
    if upstream == "responses":
        body["input"] = [{"role": "user", "content": [{"type": "input_text", "text": "Question"}]}]
    else:
        body["messages"][0]["content"] = [{"type": "text", "text": "Question"}]
    if upstream == "chat/completions":
        body["stream_options"] = {"include_usage": True}
    if downstream == "messages":
        body.setdefault("stream_options", {})["include_obfuscation"] = False
    return body


def _reasoning_and_text(protocol: Protocol, response) -> tuple[str, str]:
    if protocol == "responses":
        reasoning = [item for item in response.output if item.type == "reasoning"]
        assert all(not item.encrypted_content for item in reasoning)
        return "".join(
            part.text for item in reasoning for part in item.summary
        ), response.output_text
    if protocol == "messages":
        reasoning = [block for block in response.content if block.type == "thinking"]
        assert all(not block.signature for block in reasoning)
        return (
            "".join(block.thinking for block in reasoning),
            "".join(block.text for block in response.content if block.type == "text"),
        )
    message = response.choices[0].message
    return message.reasoning_content, message.content


@pytest.mark.parametrize(("downstream", "upstream"), list(permutations(PROTOCOLS, 2)))
@pytest.mark.parametrize("stream", [True, False])
async def test_unsigned_reasoning_output_matches_sdk(
    downstream: Protocol, upstream: Protocol, stream: bool
):
    """Ensures unsigned summaries survive all translated routes but remain invalid replay history."""
    events = _unsigned_events(upstream)
    payload = "".join(dumps_sse(event, event.get("type", "")) for event in events)
    expected_request = _expected_request(downstream, upstream)
    requests = []

    def handler(request: Request) -> Response:
        requests.append(request)
        assert json_loads(request.content) == expected_request
        return Response(200, text=payload, headers={"content-type": "text/event-stream"})

    async with AsyncClient(transport=MockTransport(handler)) as http_client:
        if upstream == "messages":
            sdk = AsyncAnthropic(api_key="test-api-key", http_client=http_client)
            create = sdk.messages.create
        else:
            sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
            create = (
                sdk.responses.create if upstream == "responses" else sdk.chat.completions.create
            )
        async with await create(**expected_request) as sdk_stream:
            reference = _collect(upstream, [event async for event in sdk_stream])

        client = Molto(http_client=http_client)
        result = await _coordinator(client, downstream)(
            body=_body(downstream, stream),
            endpoint=upstream,
            provider="openai",
            api_key="test-api-key",
        )
        if stream:
            projected = [event async for event in result]
            _assert_lifecycle(downstream, projected)
            result = _collect(downstream, projected)

        expected = _reasoning_and_text(upstream, reference)
        assert all(expected)
        assert _reasoning_and_text(downstream, result) == expected
        assert len(requests) == 2

        if downstream != "chat/completions":
            history_body = _body(downstream, stream)
            if downstream == "responses":
                history_body["input"] = [
                    item.model_dump(exclude_none=True) for item in result.output
                ]
            else:
                history_body["messages"] = [
                    {
                        "role": "assistant",
                        "content": [
                            block.model_dump(exclude_none=True) for block in result.content
                        ],
                    }
                ]
            with pytest.raises(UnsupportedInput, match="summary requires a signature"):
                await _coordinator(client, downstream)(
                    body=history_body,
                    endpoint=upstream,
                    provider="openai",
                    api_key="test-api-key",
                )
            assert len(requests) == 2
