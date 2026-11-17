"""Coordinator selection checks; synthetic flags do not establish provider support."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from molto import Molto
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import AssistantInput, Thinking
from molto.upstream.openai import ChatCompletionsClient
from molto.upstream.registry import CLIENTS
from molto.utils.io import json_loads, read_yaml
from molto.utils.test import assert_stream_lifecycle, collect_sdk_response


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("openrouter", "responses"),
        ("fireworks", "responses"),
        ("fireworks", "chat/completions"),
        ("togetherai", "responses"),
        ("togetherai", "chat/completions"),
        ("vllm", "responses"),
        ("vllm", "chat/completions"),
        ("bedrock", "chat/completions"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", ["", "Prior thought"])
async def test_raw_history_rejects_signatures_before_http(
    provider, endpoint, stream, thinking, monkeypatch
):
    """Ensures fixed and caller-selected raw profiles cannot silently discard signed history."""
    if provider == "bedrock":
        monkeypatch.setattr(CLIENTS[provider, endpoint], "REQUIRES_REASONING_REPRESENTATION", True)
    body = {
        "model": "synthetic",
        "max_tokens": 32,
        "stream": stream,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thinking, "signature": "synthetic-state"}
                ],
            }
        ],
    }
    before = deepcopy(body)
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("Signed raw history reached HTTP"))
        )
    ) as client:
        with pytest.raises(UnsupportedInput, match="Raw reasoning cannot preserve.*signature"):
            await client.messages(
                body,
                provider=provider,
                endpoint=endpoint,
                reasoning_representation="raw",
                api_key="test-key",
                base_url="https://example.test",
            )
    assert body == before


@pytest.mark.parametrize(
    ("downstream", "upstream"),
    [
        ("responses", "chat/completions"),
        ("messages", "chat/completions"),
        ("chat_completions", "responses"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("selection", [None, "invalid"])
async def test_flagged_translation_requires_selection_before_normalization(
    downstream, upstream, stream, selection, monkeypatch
):
    """Ensures every translation boundary requires a valid selection before validation or HTTP."""
    monkeypatch.setattr(CLIENTS["vllm", upstream], "REQUIRES_REASONING_REPRESENTATION", True)
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("Missing selection reached HTTP"))
        )
    ) as client:
        with pytest.raises(ValueError, match="reasoning_representation='raw' or 'summary'"):
            await getattr(client, downstream)(
                {"stream": stream},
                endpoint=upstream,
                provider="vllm",
                reasoning_representation=selection,
            )


@pytest.mark.parametrize("downstream", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_selected_profile_controls_history_and_sdk_backed_projection(
    downstream, stream, monkeypatch
):
    """Ensures successive selections reach adapters and clients without leaking into HTTP or globals."""
    recording = read_yaml(
        Path(__file__).parents[1]
        / "cassettes/matched/reasoning_raw/vllm-chat_completions-stream-00.yaml"
    )
    profiles = []
    histories = []

    class AmbiguousChatClient(ChatCompletionsClient):
        REQUIRES_REASONING_REPRESENTATION = True

        @classmethod
        def resolve_capabilities(cls, capabilities):
            return replace(capabilities, thinking="summary")

        async def _events(self, request):
            profiles.append(self._capabilities)
            histories.append(request.input)
            async for event in super()._events(request):
                yield event

    monkeypatch.setitem(CLIENTS, ("vllm", "chat/completions"), AmbiguousChatClient)
    original_profile = PROVIDER_CAPABILITIES["vllm"]
    requests = []

    def response():
        return httpx.Response(
            200,
            content="".join(recording["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    def handle(request):
        requests.append(json_loads(request.content))
        return response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response())) as transport:
        sdk = AsyncOpenAI(api_key="test-key", http_client=transport)
        async with await sdk.chat.completions.create(**recording["request"]["body"]) as events:
            expected = collect_sdk_response("chat/completions", [event async for event in events])
    sdk_message = expected.choices[0].message

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        for selection in ("raw", "summary", "raw"):
            body = {"model": recording["request"]["body"]["model"], "stream": stream}
            if downstream == "responses":
                body["input"] = [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Prior thought"}]
                        if selection == "summary"
                        else [],
                        "content": [{"type": "reasoning_text", "text": "Prior thought"}]
                        if selection == "raw"
                        else [],
                    }
                ]
            else:
                body.update(
                    max_tokens=2048,
                    messages=[
                        {
                            "role": "assistant",
                            "content": [{"type": "thinking", "thinking": "Prior thought"}],
                        }
                    ],
                    thinking={"type": "disabled"},
                )
            before = deepcopy(body)
            result = await getattr(client, downstream)(
                body,
                provider="vllm",
                endpoint="chat/completions",
                base_url="https://example.test/v1",
                reasoning_representation=selection,
            )
            if stream:
                events = [event async for event in result]
                assert_stream_lifecycle(downstream, events)
                result = collect_sdk_response(downstream, events)
            if downstream == "responses":
                reasoning = next(item for item in result.output if item.type == "reasoning")
                parts = reasoning.content if selection == "raw" else reasoning.summary
                assert "".join(part.text for part in parts) == sdk_message.reasoning_content
                assert not (reasoning.summary if selection == "raw" else reasoning.content)
                assert result.output_text == sdk_message.content
            else:
                reasoning = next(block for block in result.content if block.type == "thinking")
                assert reasoning.thinking == sdk_message.reasoning_content
                assert (
                    "".join(block.text for block in result.content if block.type == "text")
                    == sdk_message.content
                )
            assert body == before
            assert profiles[-1] == replace(original_profile, thinking=selection)
            history = histories[-1][0]
            thought = history.content[0] if isinstance(history, AssistantInput) else history
            assert isinstance(thought, Thinking)
            assert thought.content == ("Prior thought" if selection == "raw" else "")
            assert thought.summary == ("Prior thought" if selection == "summary" else "")
            assert "reasoning_representation" not in requests[-1]
            assert requests[-1]["messages"] == [
                {"role": "assistant", "reasoning_content": "Prior thought"}
            ]
            assert PROVIDER_CAPABILITIES["vllm"] is original_profile


@pytest.mark.parametrize(
    "provider_endpoint",
    [
        pair
        for pair, client_type in CLIENTS.items()
        if not client_type.REQUIRES_REASONING_REPRESENTATION
    ],
)
async def test_unflagged_clients_keep_fixed_profiles(provider_endpoint):
    """Ensures the public option cannot override any currently fixed client profile."""
    provider, endpoint = provider_endpoint
    client_type = CLIENTS[provider_endpoint]
    expected = client_type.resolve_capabilities(PROVIDER_CAPABILITIES[provider])
    async with Molto() as client:
        for selection in ("raw", "summary"):
            upstream, capabilities = client._upstream_client(
                endpoint,
                provider,
                downstream="messages" if endpoint != "messages" else "responses",
                reasoning_representation=selection,
                api_key="test-key",
                base_url="https://example.test",
                extra_body=None,
                extra_headers=None,
            )
            assert capabilities == upstream._capabilities == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("selection", [None, "raw", "summary"])
async def test_native_flagged_client_preserves_sdk_parity(stream, selection, monkeypatch):
    """Ensures native Chat bypasses reasoning selection and preserves parsed SDK output and body."""
    mode = "stream" if stream else "non_stream"
    recording = read_yaml(
        Path(__file__).parents[1]
        / f"cassettes/matched/simple_text/openai-chat_completions-{mode}-00.yaml"
    )
    monkeypatch.setattr(
        CLIENTS["openai", "chat/completions"], "REQUIRES_REASONING_REPRESENTATION", True
    )
    body = recording["request"]["body"]
    requests = []

    def handle(request):
        requests.append(json_loads(request.content))
        if stream:
            return httpx.Response(
                200,
                content="".join(recording["response"]["sse"]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=recording["response"]["body"])

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        actual = await client.chat_completions(
            body, api_key="test-key", reasoning_representation=selection
        )
        actual = [event.model_dump() async for event in actual] if stream else actual.model_dump()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        sdk = AsyncOpenAI(api_key="test-key", http_client=transport)
        expected = await sdk.chat.completions.create(**body)
        if stream:
            async with expected:
                expected = [event.model_dump() async for event in expected]
        else:
            expected = expected.model_dump()
    assert actual == expected
    assert requests == [body, body]
