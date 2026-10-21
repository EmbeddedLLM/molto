"""Together-specific request validation, transport, and stream normalization."""

from json import dumps, loads
from typing import Any

import httpx
import pytest
from openai import APIError, APIStatusError, BadRequestError, NotFoundError, OpenAIError

from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ResourceNotFoundError,
    UnsupportedInput,
)
from molto.types.normalized import (
    AssistantInput,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolReturn,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsStreamChunk,
)
from molto.upstream.openai import OpenAIChatCompletionsClient
from molto.upstream.together import TogetherAIChatCompletionsClient

TOGETHER_CAPABILITIES = PROVIDER_CAPABILITIES["togetherai"]


@pytest.mark.parametrize(
    ("sdk_error_type", "code", "message", "normalized_type"),
    [
        (APIStatusError, "model_not_available", "Unavailable model", ModelNotFoundError),
        (
            BadRequestError,
            None,
            "The input (27 tokens) is longer than the model's context length (16 tokens).",
            ContextLengthExceededError,
        ),
        (
            APIStatusError,
            "context_length_exceeded",
            "Context exceeded",
            ContextLengthExceededError,
        ),
        (APIStatusError, "model_not_found", "Missing model", ModelNotFoundError),
        (NotFoundError, None, "Missing route", ResourceNotFoundError),
    ],
)
async def test_together_error_recognition_preserves_sdk_evidence(
    sdk_error_type, code, message, normalized_type
):
    """Ensures provider rules and inherited fallbacks retain SDK subclasses and transport evidence."""

    class ProviderError(sdk_error_type):
        pass

    request = httpx.Request("POST", "https://together.example/v1/chat/completions")
    # Recognition depends on SDK class and semantic evidence, not HTTP status.
    response = httpx.Response(422, request=request)
    body = {"code": code, "message": message}
    error = ProviderError(message, response=response, body=body)
    async with httpx.AsyncClient() as http_client:
        client = TogetherAIChatCompletionsClient(http_client, capabilities=TOGETHER_CAPABILITIES)
        with pytest.raises(normalized_type) as exc_info:
            client._handle_error(error)
    assert exc_info.value.__cause__ is error
    assert exc_info.value.message == message
    assert exc_info.value.body is body
    assert exc_info.value.response is response
    assert exc_info.value.request is request


@pytest.mark.parametrize(
    ("sdk_error_type", "message"),
    [
        (BadRequestError, "max_tokens must be positive"),
        (BadRequestError, "Invalid context length option"),
        (APIStatusError, "The input is longer than the model's context length"),
        (APIError, "The input is longer than the model's context length"),
        (ValueError, "The input is longer than the model's context length"),
    ],
)
async def test_together_unknown_errors_pass_through(sdk_error_type, message):
    """Ensures the overflow message rule requires BadRequestError and a specific phrase."""
    request = httpx.Request("POST", "https://together.example/v1/chat/completions")
    if issubclass(sdk_error_type, APIStatusError):
        error = sdk_error_type(message, response=httpx.Response(400, request=request), body={})
    elif issubclass(sdk_error_type, APIError):
        error = sdk_error_type(message, request=request, body={})
    else:
        error = sdk_error_type(message)
    async with httpx.AsyncClient() as http_client:
        client = TogetherAIChatCompletionsClient(http_client, capabilities=TOGETHER_CAPABILITIES)
        with pytest.raises(sdk_error_type) as exc_info:
            client._handle_error(error)
    assert exc_info.value is error


@pytest.mark.parametrize(
    ("sdk_error_type", "code", "message", "expected_type"),
    [
        (NotFoundError, "model_not_available", "Unavailable model", ResourceNotFoundError),
        (
            BadRequestError,
            None,
            "The input is longer than the model's context length",
            BadRequestError,
        ),
    ],
)
async def test_together_recognition_does_not_affect_openai(
    sdk_error_type, code, message, expected_type
):
    """Ensures Together-only signals do not change the shared OpenAI client's error categories."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = sdk_error_type(
        message, response=httpx.Response(400, request=request), body={"code": code}
    )
    async with httpx.AsyncClient() as http_client:
        client = OpenAIChatCompletionsClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["openai"]
        )
        with pytest.raises(expected_type) as exc_info:
            client._handle_error(error)
    if expected_type is ResourceNotFoundError:
        assert exc_info.value.__cause__ is error
    else:
        assert exc_info.value is error


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl_123",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        "created": 1,
        "model": "m",
        "object": "chat.completion.chunk",
    }


def _sse_response(*chunks: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        text="".join(f"data: {dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize(
    ("blocks", "reasoning"),
    [
        ([""], ""),
        (["  Think\n\ncarefully.\t "], "  Think\n\ncarefully.\t "),
        (["First", "Second"], "First\nSecond"),
        (["", "Second"], "\nSecond"),
        (["First", ""], "First\n"),
        (["", ""], "\n"),
        ([" First\n", "\tSecond "], " First\n\n\tSecond "),
    ],
)
def test_together_client_preserves_raw_reasoning_history(
    blocks: list[str], reasoning: str
) -> None:
    """Ensures raw blocks join with newlines while preserving whitespace, turns, and tool links."""
    request = NormalizedRequest(
        model="m",
        input=[
            AssistantInput(
                content=[
                    *(Thinking(content=block, summary="") for block in blocks),
                    TextOutput(content="Before tools"),
                    ToolCall(tool_name="lookup", tool_call_id="call_1", args=' {"x": 1} '),
                ]
            ),
            ToolReturn(tool_call_id="call_1", content=[TextInput(content="Result")]),
            AssistantInput(content=[Thinking(content="Next turn", summary="")]),
        ],
    )

    assert TogetherAIChatCompletionsClient._request_body(request)["messages"] == [
        {
            "role": "assistant",
            "reasoning": reasoning,
            "content": "Before tools",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "lookup", "arguments": ' {"x": 1} '},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "Result"}],
        },
        {"role": "assistant", "reasoning": "Next turn"},
    ]


@pytest.mark.parametrize(
    "content",
    [
        [Thinking(content="", summary="Summary")],
        [Thinking(content="", summary="", signature="signed")],
        [TextOutput(content="Text"), Thinking(content="Later", summary="")],
        [
            ToolCall(tool_name="lookup", tool_call_id="call_1", args="{}"),
            Thinking(content="Later", summary=""),
        ],
        [
            ToolCall(tool_name="lookup", tool_call_id="call_1", args="{}"),
            TextOutput(content="Later"),
        ],
    ],
)
def test_together_client_rejects_unrepresentable_reasoning_history(content) -> None:
    """Ensures history still rejects summaries, opaque state, and reordered parts."""
    request = NormalizedRequest(model="m", input=[AssistantInput(content=content)])

    with pytest.raises(UnsupportedInput):
        TogetherAIChatCompletionsClient._request_body(request)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("base_url", [None, "https://together.example/v1"])
async def test_together_native_transport_uses_url_key_and_sdk_models(
    stream: bool, base_url: str | None
) -> None:
    """Ensures native transport uses Together's URL, key, extensions, and SDK result models."""
    body = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "m",
        "stream": stream,
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == (base_url or "https://api.together.ai/v1") + "/chat/completions"
        assert request.headers["authorization"] == "Bearer together-key"
        assert loads(request.content) == {**body, "reasoning": {"enabled": True}}
        if stream:
            return _sse_response(_chunk({"reasoning": "Thought", "content": "Answer"}, "stop"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_123",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning": "Thought",
                            "content": "Answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "m",
                "object": "chat.completion",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TogetherAIChatCompletionsClient(
            http_client,
            capabilities=TOGETHER_CAPABILITIES,
            api_key="together-key",
            base_url=base_url,
            extra_body={"reasoning": {"enabled": True}},
        )
        request = OpenAIChatCompletionsCreateRequest.model_validate(body)
        if stream:
            events = [event async for event in client.native_events(request)]
            assert len(events) == 1
            assert type(events[0]) is OpenAIChatCompletionsStreamChunk
            message = events[0].choices[0].delta
        else:
            response = await client.native_response(request)
            assert type(response) is OpenAIChatCompletionsResponse
            message = response.choices[0].message

    assert len(requests) == 1
    assert message.reasoning == "Thought"
    assert message.content == "Answer"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("api_key", [None, ""])
async def test_together_missing_key_blocks_openai_admin_credentials(
    monkeypatch: pytest.MonkeyPatch, stream: bool, api_key: str | None
) -> None:
    """Ensures missing Together credentials stop native transport despite OpenAI environment keys."""
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "openai-admin-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("Missing Together credentials must fail before HTTP")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TogetherAIChatCompletionsClient(
            http_client, capabilities=TOGETHER_CAPABILITIES, api_key=api_key
        )
        request = OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": "Hi"}], model="m", stream=stream
        )
        with pytest.raises(AuthenticationError, match="TOGETHER_AI_API_KEY") as exc_info:
            if stream:
                [event async for event in client.native_events(request)]
            else:
                await client.native_response(request)

    assert type(exc_info.value.__cause__) is OpenAIError
    assert str(exc_info.value.request.url) == "https://api.together.ai/v1/chat/completions"
    assert exc_info.value.response is None
    assert requests == []


async def test_together_events_preserve_raw_reasoning_and_text_after_tools() -> None:
    """Ensures raw reasoning and trailing whitespace produce ordered, balanced part lifecycles."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _chunk({"role": "assistant", "content": "", "reasoning": ""}),
            _chunk({"reasoning": "  Think\n"}),
            _chunk({"reasoning": "\t "}),
            _chunk(
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": ""},
                        }
                    ],
                }
            ),
            _chunk(
                {"content": "", "tool_calls": [{"index": 0, "function": {"arguments": '{"x":1}'}}]}
            ),
            _chunk({"content": "\n\n"}),
            _chunk({}, "tool_calls"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TogetherAIChatCompletionsClient(
            http_client, capabilities=TOGETHER_CAPABILITIES, api_key="together-key"
        )
        events = [
            event async for event in client.events(NormalizedRequest(model="m", stream=True))
        ]

    assert events == [
        ResponseStartEvent(id="chatcmpl_123", created_at=1, model="m"),
        PartStartEvent(index=0, part=Thinking(content="", summary="")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="  Think\n")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="\t ")),
        PartEndEvent(index=0, part=Thinking(content="  Think\n\t ", summary="")),
        PartStartEvent(index=1, part=ToolCall(tool_call_id="call_1", tool_name="lookup", args="")),
        PartDeltaEvent(index=1, delta=ToolCallDelta(args_delta='{"x":1}')),
        PartEndEvent(
            index=1, part=ToolCall(tool_call_id="call_1", tool_name="lookup", args='{"x":1}')
        ),
        PartStartEvent(index=2, part=TextOutput(content="")),
        PartDeltaEvent(index=2, delta=TextDelta(content_delta="\n\n")),
        PartEndEvent(index=2, part=TextOutput(content="\n\n")),
        ResponseEndEvent(
            id="chatcmpl_123", created_at=1, model="m", termination_reason="tool_call"
        ),
    ]
