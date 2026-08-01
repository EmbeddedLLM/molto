import inspect
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError
from pydantic import ValidationError

from molto import Molto
from molto.providers import OPENAI_RESPONSES_CAPABILITIES, validate_provider_request
from molto.providers.openai import OpenAIProvider, map_openai_error
from molto.providers.openai.output_translation import (
    chat_stream_to_output_events,
    responses_stream_to_output_events,
)
from molto.providers.openai.request_translation import build_openai_request
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.responses import OpenAIResponsesRequest
from molto.utils.exceptions import BadInputError, ModelOverloadError


def _canned_response(endpoint: str) -> httpx.Response:
    """Return a minimal canned SDK response so a request can be sent and captured."""
    if endpoint == "chat_completions":
        body = {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    else:
        body = {
            "id": "x",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    return httpx.Response(200, json=body)


async def _assert_sdk_serializes_same(endpoint: str, body: dict) -> None:
    """Anchor a native request body to the official SDK's wire serialization."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _canned_response(endpoint)

    client = AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    create = (
        client.chat.completions.create
        if endpoint == "chat_completions"
        else client.responses.create
    )
    accepted = set(inspect.signature(create).parameters)
    kwargs = {key: value for key, value in body.items() if key in accepted}
    passthrough = {key: value for key, value in body.items() if key not in kwargs}
    if passthrough:
        kwargs["extra_body"] = passthrough
    await create(**kwargs)
    assert captured["body"] == body


def _responses_payload(text: str) -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4.1-nano",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
    }


def _chat_payload(text: str) -> dict:
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4.1-nano",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"role": "assistant", "content": text, "refusal": None},
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


async def _dict_frames(*frames: dict):
    for frame in frames:
        yield frame


def test_chat_request_cannot_select_responses_native_endpoint() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "developer", "content": "Use JSON."},
            {"role": "user", "content": "Say hi."},
        ],
        max_completion_tokens=20,
        temperature=0.4,
        response_format={"type": "json_object"},
    )

    with pytest.raises(
        BadInputError,
        match="Chat requests must execute through the Chat Completions endpoint",
    ):
        build_openai_request(request.to_ir(), "responses")


def test_removed_chat_request_fields_fail_validation() -> None:
    with pytest.raises(ValidationError, match="function_call"):
        OpenAIChatCompletionRequest(
            model="openai/gpt-4.1-nano",
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "legacy", "arguments": "{}"},
                }
            ],
        )

    with pytest.raises(ValidationError, match="functions"):
        OpenAIChatCompletionRequest(
            model="openai/gpt-4.1-nano",
            messages=[{"role": "user", "content": "hi"}],
            functions=[],
        )


def test_responses_background_execution_is_rejected_locally() -> None:
    with pytest.raises(ValidationError, match="background"):
        OpenAIResponsesRequest(
            model="openai/gpt-4.1-nano",
            input="hi",
            background=True,
        )


def test_native_responses_request_preserves_mcp_tool() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input="Use the server.",
        tools=[
            {
                "type": "mcp",
                "server_label": "docs",
                "server_url": "https://example.com/mcp",
                "require_approval": "never",
            }
        ],
    )

    ir = request.to_ir()
    validate_provider_request(OPENAI_RESPONSES_CAPABILITIES, ir)
    assert build_openai_request(ir, "responses") == {
        "model": "gpt-4.1-nano",
        "input": "Use the server.",
        "tools": [
            {
                "type": "mcp",
                "server_label": "docs",
                "server_url": "https://example.com/mcp",
                "require_approval": "never",
            }
        ],
    }


def test_responses_request_cannot_select_chat_native_endpoint() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input="Use one function.",
        tools=[
            {"type": "function", "name": "first", "parameters": {}},
            {"type": "function", "name": "second", "parameters": {}},
        ],
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "function", "name": "first"},
                {"type": "function", "name": "second"},
            ],
        },
    )

    with pytest.raises(
        BadInputError,
        match="Responses requests must execute through the Responses endpoint",
    ):
        build_openai_request(request.to_ir(), "chat_completions")


async def test_native_chat_non_stream_call_is_normalized_after_sdk_response() -> None:
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(await request.aread()))
        return httpx.Response(200, request=request, json=_chat_payload("hello"))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Say hello."}],
        stream=False,
    )

    response = await Molto(http_client=http_client).chat_completions(request, api_key="test")

    assert len(captured) == 1
    assert captured[0]["stream"] is False
    assert response.choices[0].message.content == "hello"
    await http_client.aclose()


async def test_native_responses_non_stream_call_is_normalized_after_sdk_response() -> None:
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(await request.aread()))
        return httpx.Response(200, request=request, json=_responses_payload("hello"))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input="Say hello.",
        stream=False,
    )

    response = await Molto(http_client=http_client).responses(request, api_key="test")

    assert len(captured) == 1
    assert captured[0]["stream"] is False
    assert response.output[0].content[0].text == "hello"
    await http_client.aclose()


async def test_responses_method_uses_chat_request_endpoint_and_returns_responses() -> None:
    captured_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(200, request=request, json=_chat_payload("hello"))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Say hello."}],
        stream=False,
    )

    response = await Molto(http_client=http_client).responses(request, api_key="test")

    assert captured_paths == ["/v1/chat/completions"]
    assert response.object == "response"
    assert response.output[0].content[0].text == "hello"
    assert response.metadata["system_fingerprint"] == "fp_test"
    await http_client.aclose()


async def test_chat_method_uses_responses_request_endpoint_and_returns_chat() -> None:
    captured_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(200, request=request, json=_responses_payload("hello"))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input="Say hello.",
        stream=False,
    )

    response = await Molto(http_client=http_client).chat_completions(request, api_key="test")

    assert captured_paths == ["/v1/responses"]
    assert response.object == "chat.completion"
    assert response.choices[0].message.content == "hello"
    await http_client.aclose()


async def test_responses_method_stream_uses_chat_request_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints: list[str] = []
    frames = (
        {
            "id": "chatcmpl_cross_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_cross_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    class FakeProvider:
        def __init__(self, endpoint: str, **_kwargs) -> None:
            endpoints.append(endpoint)

        def run(self, _request):
            return chat_stream_to_output_events(_dict_frames(*frames))

    monkeypatch.setattr("molto.core.OpenAIProvider", FakeProvider)
    stream = await Molto().responses(
        OpenAIChatCompletionRequest(
            model="openai/gpt-4.1-nano",
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
        )
    )
    events = [event async for event in stream]

    assert endpoints == ["chat_completions"]
    assert events[-1].type == "response.completed"
    assert events[-1].response.output[0].content[0].text == "hello"


async def test_chat_method_stream_uses_responses_request_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints: list[str] = []

    def response(status: str, usage: dict | None = None) -> dict:
        return {
            "id": "resp_cross_stream",
            "object": "response",
            "created_at": 1,
            "model": "gpt-4.1-nano",
            "status": status,
            "output": [],
            "usage": usage,
        }

    frames = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": response("in_progress"),
        },
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": response(
                "completed",
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        },
    )

    class FakeProvider:
        def __init__(self, endpoint: str, **_kwargs) -> None:
            endpoints.append(endpoint)

        def run(self, _request):
            return responses_stream_to_output_events(_dict_frames(*frames))

    monkeypatch.setattr("molto.core.OpenAIProvider", FakeProvider)
    stream = await Molto().chat_completions(
        OpenAIResponsesRequest(
            model="openai/gpt-4.1-nano",
            input="Say hello.",
            stream=True,
        )
    )
    chunks = [chunk async for chunk in stream]

    assert endpoints == ["responses"]
    assert any(choice.finish_reason == "stop" for chunk in chunks for choice in chunk.choices)
    assert chunks[-1].usage.total_tokens == 2


def test_generated_audio_request_survives_the_input_ir() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4o-audio-preview",
        messages=[{"role": "user", "content": "Say hello."}],
        modalities=["text", "audio"],
        audio={"format": "mp3", "voice": "alloy"},
    )

    assert build_openai_request(request.to_ir(), "chat_completions") == {
        "model": "gpt-4o-audio-preview",
        "audio": {"format": "mp3", "voice": "alloy"},
        "modalities": ["text", "audio"],
        "messages": [{"role": "user", "content": "Say hello."}],
    }


async def test_mcp_rejects_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIResponsesRequest(
        model="openai/gpt-5.1",
        input="List the available tools.",
        tools=[
            {
                "type": "mcp",
                "server_label": "test_server",
                "server_url": "https://example.test/mcp",
                "require_approval": "never",
            }
        ],
    )

    with pytest.raises(BadInputError, match="function tools only"):
        await Molto().chat_completions(request)


async def test_vllm_audio_rejects_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIChatCompletionRequest(
        model="vllm/audio-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "UklGRg==", "format": "wav"},
                    }
                ],
            }
        ],
    )

    with pytest.raises(BadInputError, match="input.audio"):
        await Molto(base_url="https://example.test/v1").chat_completions(request)


async def test_vllm_image_generation_rejects_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIResponsesRequest(
        model="vllm/image-model",
        input="Draw a red circle.",
        tools=[{"type": "image_generation"}],
    )

    with pytest.raises(BadInputError, match="function tools only"):
        await Molto(base_url="https://example.test/v1").chat_completions(request)


async def test_chat_audio_to_responses_rejects_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4o-audio-preview",
        messages=[{"role": "user", "content": "Say hello."}],
        modalities=["text", "audio"],
        audio={"format": "mp3", "voice": "alloy"},
    )

    with pytest.raises(BadInputError, match="cannot represent generated Chat audio"):
        await Molto().responses(request)


async def test_responses_reasoning_to_chat_rejects_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIResponsesRequest(
        model="openai/gpt-5-nano",
        input="Think carefully.",
        reasoning={"effort": "high", "summary": "concise"},
    )

    with pytest.raises(BadInputError, match="cannot represent Responses reasoning or state"):
        await Molto().chat_completions(request)


async def test_provider_closes_owned_client_when_sdk_call_fails() -> None:
    provider = OpenAIProvider(
        "chat_completions",
        max_retries=0,
        base_url="http://127.0.0.1:1/v1",
    )
    close = AsyncMock()
    provider._client.close = close
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Say hello."}],
        stream=False,
    )

    with pytest.raises(ModelOverloadError):
        await anext(provider.run(request.to_ir()))

    close.assert_awaited_once_with()


def test_sdk_error_mapping_preserves_provider_context() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": "req_test"},
    )
    provider_body = {"error": {"message": "bad request", "code": "bad_parameter"}}
    sdk_error = BadRequestError("bad request", response=response, body=provider_body)

    mapped = map_openai_error(sdk_error)

    assert isinstance(mapped, BadInputError)
    assert mapped.status_code == 400
    assert mapped.request_id == "req_test"
    assert mapped.provider_body == provider_body
    assert mapped.provider_code == "bad_parameter"


async def test_molto_does_not_close_caller_owned_http_client() -> None:
    http_client = httpx.AsyncClient()
    molto = Molto(http_client=http_client)

    await molto.close()

    assert not http_client.is_closed
    await http_client.aclose()


def _chat_stream_sse() -> str:
    chunks = [
        {
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hi"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("injected", [True, False], ids=["injected-client", "owned-client"])
async def test_openai_client_close_matrix(injected: bool, stream: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if stream:
            return httpx.Response(
                200,
                content=_chat_stream_sse().encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=_chat_payload("hi"))

    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if injected:
        provider = OpenAIProvider(
            "chat_completions",
            api_key="test",
            http_client=transport_client,
            max_retries=0,
        )
    else:
        provider = OpenAIProvider("chat_completions", api_key="test", max_retries=0)
        provider._client._client = transport_client

    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Say hello."}],
        stream=stream,
    )
    async for _ in provider.run(request.to_ir()):
        pass

    if injected:
        assert not transport_client.is_closed
        await transport_client.aclose()
    else:
        assert transport_client.is_closed
