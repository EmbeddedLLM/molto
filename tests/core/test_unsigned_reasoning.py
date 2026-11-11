"""SDK parity for synthetic unsigned reasoning derived from recorded streams."""

from itertools import permutations

import pytest
from anthropic import AsyncAnthropic
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI

from molto import Molto
from molto.types.exceptions import UnsupportedInput
from molto.utils.io import dumps_sse, json_loads, load_sse
from molto.utils.test import assert_stream_lifecycle, collect_sdk_response
from tests.core.test_interface import (
    PROTOCOLS,
    Protocol,
    _coordinator,
    _matched_cassette_path,
    _read_cassette,
    _sse_records,
)


def _unsigned_events(protocol: Protocol, ciphertext: str | None = None) -> list[dict]:
    scenario = "reasoning_raw" if protocol == "chat/completions" else "reasoning_encrypted"
    cassette = _read_cassette(_matched_cassette_path(scenario, protocol, "stream-00.yaml"))
    events = [load_sse(record) for record in _sse_records(cassette, protocol)]

    def remove_signatures(value):
        if isinstance(value, dict):
            if ciphertext is not None and "encrypted_content" in value:
                value["encrypted_content"] = ciphertext
            else:
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
        assert response.status == "completed"
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


@pytest.mark.parametrize(
    ("downstream", "upstream", "ciphertext", "provider"),
    [
        (downstream, upstream, ciphertext, provider)
        for downstream, upstream in permutations(PROTOCOLS, 2)
        for ciphertext in ((None, "") if upstream == "responses" else (None,))
        for provider in {
            "responses": ("openai", "openrouter"),
            "messages": ("anthropic",),
            "chat/completions": ("openai",),
        }[upstream]
    ],
)
@pytest.mark.parametrize("stream", [True, False])
async def test_unsigned_reasoning_output_and_history_match_sdk(
    downstream: Protocol, upstream: Protocol, ciphertext: str | None, provider: str, stream: bool
):
    """Ensures unsigned output matches the SDK and history is validated by its destination."""
    events = _unsigned_events(upstream, ciphertext)
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
            reference = collect_sdk_response(upstream, [event async for event in sdk_stream])

        client = Molto(http_client=http_client)
        result = await _coordinator(client, downstream)(
            body=_body(downstream, stream),
            endpoint=upstream,
            provider=provider,
            api_key="test-api-key",
        )
        if stream:
            projected = [event async for event in result]
            assert_stream_lifecycle(downstream, projected)
            result = collect_sdk_response(downstream, projected)

        expected = _reasoning_and_text(upstream, reference)
        assert all(expected)
        assert _reasoning_and_text(downstream, result) == expected
        assert len(requests) == 2

        history_body = _body(downstream, stream)
        if downstream == "responses":
            history_body["input"] = [
                {"role": "user", "content": "Question"},
                *(item.model_dump(exclude_none=True) for item in result.output),
                {"role": "user", "content": "Follow up"},
            ]
        else:
            assistant = (
                {
                    "role": "assistant",
                    "content": [block.model_dump(exclude_none=True) for block in result.content],
                }
                if downstream == "messages"
                else {
                    "role": "assistant",
                    "content": expected[1],
                    "reasoning_content": expected[0],
                }
            )
            history_body["messages"].extend([assistant, {"role": "user", "content": "Follow up"}])

        if upstream != "chat/completions":
            with pytest.raises(UnsupportedInput, match="summary requires a signature"):
                await _coordinator(client, downstream)(
                    body=history_body,
                    endpoint=upstream,
                    provider=provider,
                    api_key="test-api-key",
                )
            assert len(requests) == 2
            return

        expected_request["messages"].extend(
            [
                {
                    "role": "assistant",
                    "reasoning_content": expected[0],
                    "content": expected[1],
                },
                {"role": "user", "content": [{"type": "text", "text": "Follow up"}]},
            ]
        )
        async with await create(**expected_request) as sdk_stream:
            reference = collect_sdk_response(upstream, [event async for event in sdk_stream])
        followup = await _coordinator(client, downstream)(
            body=history_body, endpoint=upstream, provider=provider, api_key="test-api-key"
        )
        if stream:
            projected = [event async for event in followup]
            assert_stream_lifecycle(downstream, projected)
            followup = collect_sdk_response(downstream, projected)
        assert _reasoning_and_text(downstream, followup) == _reasoning_and_text(
            upstream, reference
        )
        assert len(requests) == 4

        # Chat can replay summary text, but must not silently discard opaque state.
        if downstream == "responses":
            thinking = next(
                item for item in history_body["input"] if item.get("type") == "reasoning"
            )
            thinking["encrypted_content"] = "opaque-state"
        else:
            thinking = next(block for block in assistant["content"] if block["type"] == "thinking")
            thinking["signature"] = "opaque-state"
        with pytest.raises(UnsupportedInput, match="cannot represent thinking signatures"):
            await _coordinator(client, downstream)(
                body=history_body, endpoint=upstream, provider=provider, api_key="test-api-key"
            )
        assert len(requests) == 4
