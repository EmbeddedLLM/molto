"""Replay and synthetic coverage for Together's explicitly selected Chat dialect."""

from copy import deepcopy
from json import dumps, loads
from pathlib import Path

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI, BadRequestError, OpenAIError
from openai import AuthenticationError as SDKAuthenticationError
from pydantic import SecretStr

from molto import Molto
from molto.adapters.openai import OpenAIChatCompletionsAdapter
from molto.configs import ENV_CONFIG
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import ContextLengthExceededError, ModelNotFoundError, UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    NormalizedRequest,
    PartEndEvent,
    PartStartEvent,
    TextOutput,
    Thinking,
    ToolCall,
)
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.upstream.openai import OpenAIChatCompletionsClient
from molto.upstream.together import TogetherAIChatCompletionsClient
from molto.utils.io import read_yaml

CAPABILITIES = PROVIDER_CAPABILITIES["togetherai"]
CASSETTES = Path(__file__).parent / "cassettes" / "matched"


def _chunk(delta, finish_reason=None):
    return {
        "id": "chatcmpl_synthetic",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "synthetic",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _stream(*chunks):
    return httpx.Response(
        200,
        text="".join(f"data: {dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("base_url", [None, "https://custom.example/prefix"])
@pytest.mark.parametrize("explicit_key", [None, "explicit-together-key"])
async def test_together_native_cassette_sdk_parity(monkeypatch, stream, base_url, explicit_key):
    """Ensures native Together routing retains independent SDK-parsed matched fixture output."""
    # These are existing OpenAI matched fixtures, not evidence of live Together support.
    mode = "stream" if stream else "non_stream"
    cassette = read_yaml(CASSETTES / "simple_text" / f"openai-chat_completions-{mode}-00.yaml")
    expected_body = {**cassette["request"]["body"], "reasoning": {"enabled": False}}
    monkeypatch.setattr(ENV_CONFIG, "together_ai_api_key", SecretStr("configured-together-key"))
    key = explicit_key or "configured-together-key"
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == (base_url or "https://api.together.ai/v1") + "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {key}"
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        assert loads(request.content) == expected_body
        if stream:
            return httpx.Response(
                200,
                text="".join(cassette["response"]["sse"]),
                headers={"content-type": "text/event-stream", "x-provider": "together"},
            )
        return httpx.Response(200, json=cassette["response"]["body"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        sdk = AsyncOpenAI(
            api_key=key, base_url=base_url or "https://api.together.ai/v1", http_client=transport
        )
        expected = await sdk.chat.completions.create(
            **cassette["request"]["body"], extra_body={"reasoning": {"enabled": False}}
        )
        if stream:
            async with expected:
                expected = [event.model_dump() async for event in expected]
        else:
            expected = expected.model_dump()
        client = Molto(http_client=transport)
        actual = await client.chat_completions(
            body=cassette["request"]["body"],
            provider="togetherai",
            base_url=base_url,
            api_key=explicit_key,
            extra_body={"reasoning": {"enabled": False}},
            extra_headers={"authorization": "Bearer unrelated", "openai-project": "unrelated"},
        )
        if stream:
            assert actual.response_headers["x-provider"] == "together"
            actual = [event.model_dump() async for event in actual]
        else:
            actual = actual.model_dump()
        assert actual == expected
    assert len(requests) == 2


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("coordinator", ["responses", "chat_completions", "messages"])
@pytest.mark.parametrize("api_key", [None, ""])
async def test_missing_together_key_never_uses_openai_credentials(
    monkeypatch, stream, coordinator, api_key
):
    """Ensures all downstream routes reject missing Together credentials before HTTP."""
    monkeypatch.setattr(ENV_CONFIG, "together_ai_api_key", SecretStr(""))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "unrelated-admin-key")
    requests = []

    def handler(request):
        requests.append(request)
        raise AssertionError("Missing credentials must fail locally")

    body = {"model": "synthetic", "stream": stream}
    if coordinator == "responses":
        body["input"] = "Hello"
    else:
        body["messages"] = [{"role": "user", "content": "Hello"}]
    if coordinator == "messages":
        body["max_tokens"] = 32
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        # Messages projects into Anthropic's namespace, the OpenAI interfaces into OpenAI's.
        from anthropic import AuthenticationError as AnthropicAuthenticationError

        error_class = (
            AnthropicAuthenticationError if coordinator == "messages" else SDKAuthenticationError
        )
        with pytest.raises(error_class, match="TOGETHER_AI_API_KEY") as error:
            await getattr(client, coordinator)(
                body=body, endpoint="chat/completions", provider="togetherai", api_key=api_key
            )
        assert str(error.value.request.url) == "https://api.together.ai/v1/chat/completions"
        assert isinstance(error.value.__cause__.__cause__, OpenAIError)
    assert requests == []


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"reasoning": "  first\n"}, ["  first\n"]),
        ({"reasoning": ["", " second ", ""]}, ["", " second ", ""]),
        ({"reasoning": []}, []),
        ({"reasoning": ""}, [""]),
        ({"reasoning": "", "reasoning_content": "other"}, ["other"]),
        ({"reasoning": [], "reasoning_content": "other"}, ["other"]),
        ({"reasoning": [""], "reasoning_content": ""}, [""]),
    ],
)
def test_reasoning_alias_normalization_and_history(fields, expected):
    """Ensures ordered reasoning blocks and empty boundaries survive normalized history."""
    request = OpenAIChatCompletionsCreateRequest(
        model="synthetic", messages=[{"role": "assistant", "content": None, **fields}]
    )
    normalized = OpenAIChatCompletionsAdapter(CAPABILITIES).normalize_request(request)
    assert normalized.input == [
        AssistantInput(content=[Thinking(content=part, summary="") for part in expected])
    ]
    body = TogetherAIChatCompletionsClient._request_body(normalized)
    message = body["messages"][0]
    assert message.get("reasoning") == ("\n".join(expected) if expected else None)
    assert "reasoning_content" not in message
    assert request.model_dump()["messages"][0] == {"role": "assistant", "content": None, **fields}


@pytest.mark.parametrize("reasoning", ["one", [""]])
def test_reasoning_alias_collision_is_rejected(reasoning):
    """Ensures populated alternate representations never silently override each other."""
    request = OpenAIChatCompletionsCreateRequest(
        model="synthetic",
        messages=[{"role": "assistant", "reasoning": reasoning, "reasoning_content": "two"}],
    )
    with pytest.raises(UnsupportedInput, match="both reasoning aliases"):
        OpenAIChatCompletionsAdapter(CAPABILITIES).normalize_request(request)


@pytest.mark.parametrize("following", [{"content": "\n\n"}, {"reasoning": " think "}])
async def test_tool_lifecycle_closes_before_following_content(following):
    """Ensures actual nonempty content finishes the preceding call before starting a new part."""
    first = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{"},
    }
    chunks = [
        _chunk({"tool_calls": [first]}),
        _chunk(
            {
                "content": "",
                "reasoning": "",
                "tool_calls": [{"index": 0, "function": {"arguments": "}"}}],
            }
        ),
        _chunk(following),
        _chunk({}, "tool_calls"),
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _stream(*chunks))
    ) as transport:
        client = TogetherAIChatCompletionsClient(
            transport, capabilities=CAPABILITIES, api_key="test"
        )
        events = [
            event
            async for event in client.events(NormalizedRequest(model="synthetic", stream=True))
        ]
    starts = [event for event in events if isinstance(event, PartStartEvent)]
    ends = [event for event in events if isinstance(event, PartEndEvent)]
    assert [event.index for event in starts] == [0, 1]
    assert [event.index for event in ends] == [0, 1]
    assert ends[0].part == ToolCall(tool_call_id="call_1", tool_name="lookup", args="{}")
    assert events.index(ends[0]) < events.index(starts[1])
    assert ends[1].part == (
        TextOutput(content="\n\n")
        if "content" in following
        else Thinking(content=" think ", summary="")
    )


@pytest.mark.parametrize(
    "following",
    [
        {"content": "text"},
        {"reasoning": "think"},
        {
            "tool_calls": [
                {"index": 1, "id": "call_2", "function": {"name": "second", "arguments": "{}"}}
            ]
        },
    ],
)
async def test_closed_tool_call_cannot_resume(following):
    """Ensures intervening text, reasoning, or another call makes later fragments fail explicitly."""
    chunks = [
        _chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "first", "arguments": "{"}}
                ]
            }
        ),
        _chunk(following),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
        _chunk({}, "stop"),
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _stream(*chunks))
    ) as transport:
        client = TogetherAIChatCompletionsClient(
            transport, capabilities=CAPABILITIES, api_key="test"
        )
        with pytest.raises(ValueError, match="contiguous"):
            [
                event
                async for event in client.events(NormalizedRequest(model="synthetic", stream=True))
            ]


@pytest.mark.parametrize(
    ("sdk_type", "body", "normalized_type"),
    [
        (
            BadRequestError,
            {
                "message": "Input is longer than the model's context length",
                "code": "invalid_request",
            },
            ContextLengthExceededError,
        ),
        (
            APIStatusError,
            {"message": "Unavailable", "code": "model_not_available"},
            ModelNotFoundError,
        ),
    ],
)
def test_together_semantic_error_recognition_is_provider_specific(sdk_type, body, normalized_type):
    """Ensures Together-only signals retain original SDK evidence without changing OpenAI classification."""
    response = httpx.Response(
        400,
        json=body,
        request=httpx.Request("POST", "https://api.together.ai/v1/chat/completions"),
    )
    original = sdk_type(body["message"], response=response, body=body)
    client = TogetherAIChatCompletionsClient(None, capabilities=CAPABILITIES, api_key="test")
    with pytest.raises(normalized_type) as error:
        client._handle_error(original)
    assert error.value.__cause__ is original
    assert error.value.response is response
    assert error.value.body == body
    openai_client = OpenAIChatCompletionsClient(None, capabilities=CAPABILITIES, api_key="test")
    with pytest.raises(sdk_type) as unchanged:
        openai_client._handle_error(original)
    assert unchanged.value is original


@pytest.mark.parametrize("placement", ["message", "part", "call"])
def test_translated_history_rejects_google_state(placement):
    """Ensures opaque native continuation state is rejected instead of dropped in translation."""
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ],
    }
    extension = {"google": {"thought_signature": "opaque"}}
    target = (
        message
        if placement == "message"
        else message["content"][0]
        if placement == "part"
        else message["tool_calls"][0]
    )
    target["extra_content"] = extension
    original = deepcopy(message)
    request = OpenAIChatCompletionsCreateRequest(model="synthetic", messages=[message])
    with pytest.raises(UnsupportedInput, match="extra_content"):
        OpenAIChatCompletionsAdapter(CAPABILITIES).normalize_request(request)
    assert message == original


@pytest.mark.parametrize("placement", ["message", "call"])
@pytest.mark.parametrize("coordinator", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_translated_output_rejects_google_state_and_closes(placement, coordinator, stream):
    """Ensures unsupported output state fails explicitly and releases the upstream stream."""
    delta = {
        "tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}
        ]
    }
    target = delta if placement == "message" else delta["tool_calls"][0]
    target["extra_content"] = {"google": {"thought_signature": "opaque"}}
    responses = []

    def handler(request):
        response = _stream(_chunk(delta), _chunk({}, "tool_calls"))
        responses.append(response)
        return response

    body = {"model": "synthetic", "stream": stream}
    if coordinator == "responses":
        body["input"] = "Hello"
    else:
        body.update(messages=[{"role": "user", "content": "Hello"}], max_tokens=32)
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        with pytest.raises(ValueError, match="extra_content"):
            result = await getattr(client, coordinator)(
                body=body, endpoint="chat/completions", provider="togetherai", api_key="test"
            )
            if stream:
                async with result:
                    [event async for event in result]
        assert responses[0].is_closed


@pytest.mark.parametrize(
    "part",
    [
        Thinking(content="", summary="summary"),
        Thinking(content="", summary="", signature="opaque"),
    ],
)
def test_chat_projection_rejects_nonportable_reasoning(part):
    """Ensures Chat history never converts summaries or signatures into unsigned raw text."""
    request = NormalizedRequest(model="synthetic", input=[AssistantInput(content=[part])])
    for client in (OpenAIChatCompletionsClient, TogetherAIChatCompletionsClient):
        with pytest.raises(UnsupportedInput, match="cannot represent"):
            client._request_body(request)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("is longer than the model's context length"),
        OpenAIError("is longer than the model's context length"),
    ],
)
def test_together_context_recognition_requires_sdk_bad_request(error):
    """Ensures unrelated exception classes are never reclassified by a matching message."""
    client = TogetherAIChatCompletionsClient(None, capabilities=CAPABILITIES, api_key="test")
    with pytest.raises(type(error)) as result:
        client._handle_error(error)
    assert result.value is error


async def test_generic_chat_missing_index_fails_instead_of_losing_call():
    """Ensures unsupported missing indices cannot produce a successful empty translated response."""
    chunk = _chunk(
        {
            "tool_calls": [
                {"index": None, "id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _stream(chunk, _chunk({}, "tool_calls")))
    ) as transport:
        client = OpenAIChatCompletionsClient(transport, capabilities=CAPABILITIES, api_key="test")
        with pytest.raises(ValueError, match="require an index"):
            [
                event
                async for event in client.events(NormalizedRequest(model="synthetic", stream=True))
            ]


@pytest.mark.parametrize("coordinator", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_together_alternative_reasoning_field_rejects_without_silent_loss(
    coordinator, stream
):
    """Ensures a model using a different reasoning dialect cannot silently lose its reasoning."""
    body = {"model": "synthetic", "stream": stream}
    if coordinator == "responses":
        body["input"] = "Hello"
    else:
        body.update(messages=[{"role": "user", "content": "Hello"}], max_tokens=32)
    transport = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: _stream(
                _chunk({"reasoning_content": "actual alternate reasoning", "content": "Answer"}),
                _chunk({}, "stop"),
            )
        )
    )
    async with Molto(http_client=transport) as client:
        with pytest.raises(ValueError, match="reasoning_content.*reasoning.*dialect"):
            result = await getattr(client, coordinator)(
                body=body, endpoint="chat/completions", provider="togetherai", api_key="test"
            )
            if stream:
                async with result:
                    [event async for event in result]


@pytest.mark.parametrize("coordinator", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_recorded_together_qwen_alternate_dialect_fails_explicitly(coordinator, stream):
    """Ensures the captured Qwen reasoning_content dialect fails without discarding reasoning."""
    root = (
        Path(__file__).parent
        / "cassettes"
        / "provider_recordings"
        / "2026-11-03"
        / "together-text-stream"
    )
    paths = list(root.glob("*.yaml"))
    assert len(paths) == 1
    cassette = read_yaml(paths[0])
    recorded = cassette["request"]["body"]
    responses = []

    def handler(request):
        response = httpx.Response(
            200,
            text="".join(cassette["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )
        responses.append(response)
        return response

    # Parse the source independently first; the actual model emits the alternative key.
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        sdk = AsyncOpenAI(api_key="test", http_client=transport)
        native = await sdk.chat.completions.create(**recorded)
        async with native:
            events = [event async for event in native]
    assert any(
        getattr(choice.delta, "reasoning_content", None)
        for event in events
        for choice in event.choices
    )
    body = {"model": recorded["model"], "stream": stream}
    if coordinator == "responses":
        body.update(
            input=[
                {
                    "role": message["role"],
                    "content": [
                        {"type": "input_text", "text": part["text"]} for part in message["content"]
                    ],
                }
                for message in recorded["messages"]
            ],
            max_output_tokens=recorded["max_tokens"],
        )
    else:
        body.update(messages=recorded["messages"], max_tokens=recorded["max_tokens"])
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        with pytest.raises(ValueError, match="reasoning_content.*reasoning.*dialect"):
            result = await getattr(client, coordinator)(
                body=body, endpoint="chat/completions", provider="togetherai", api_key="test"
            )
            if stream:
                async with result:
                    [event async for event in result]
        assert responses[-1].is_closed
