import inspect
import json
from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError
from pydantic import ValidationError

from molto import Molto
from molto.providers import OPENAI_RESPONSES_CAPABILITIES, validate_provider_request
from molto.providers.openai import OpenAIProvider, map_openai_error
from molto.providers.openai.output_translation import chat_response_to_output_events
from molto.providers.openai.request_translation import build_openai_request
from molto.types.ir.portable import (
    AssistantMessageItem,
    ContentPartAddedEvent,
    ContentPartDeltaEvent,
    ContentPartDoneEvent,
    ErrorDetail,
    ItemStatus,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextPart,
    PreservedOpenAIStreamEvent,
    ReasoningItem,
    ReasoningSummaryPart,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseMetadata,
    ResponseStartedEvent,
    Usage,
)
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.responses import (
    OpenAIResponsesRequest,
    OpenAIResponsesRestrictedRequest,
)
from molto.utils.exceptions import BadInputError, ModelCapabilityError, ModelOverloadError


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


async def test_chat_length_normalizes_to_responses_incomplete_reason() -> None:
    payload = _chat_payload("partial")
    payload["choices"][0]["finish_reason"] = "length"

    events = [event async for event in chat_response_to_output_events(payload)]

    assert isinstance(events[-1], ResponseIncompleteEvent)
    assert events[-1].reason == "max_output_tokens"


async def _reasoning_output_events() -> AsyncIterator:
    started = ReasoningItem(id="rs_1", status=ItemStatus.IN_PROGRESS, parts=())
    done = ReasoningItem(
        id="rs_1",
        status=ItemStatus.COMPLETED,
        parts=(ReasoningSummaryPart(text="Careful reasoning."),),
    )
    message_started = AssistantMessageItem(id="msg_1", status=ItemStatus.IN_PROGRESS, content=())
    message_done = AssistantMessageItem(
        id="msg_1",
        status=ItemStatus.COMPLETED,
        content=(OutputTextPart(text="Hello"),),
    )
    yield ResponseStartedEvent(
        response_id="resp_1",
        sequence_number=0,
        metadata=ResponseMetadata(model="gpt-test", object="response"),
    )
    yield OutputItemAddedEvent(response_id="resp_1", sequence_number=1, item_index=0, item=started)
    yield ContentPartAddedEvent(
        response_id="resp_1",
        sequence_number=2,
        item_id="rs_1",
        item_index=0,
        content_index=0,
        part=ReasoningSummaryPart(text=""),
    )
    yield ContentPartDeltaEvent(
        response_id="resp_1",
        sequence_number=3,
        item_id="rs_1",
        item_index=0,
        content_index=0,
        delta="Careful reasoning.",
    )
    yield ContentPartDoneEvent(
        response_id="resp_1",
        sequence_number=4,
        item_id="rs_1",
        item_index=0,
        content_index=0,
        part=done.parts[0],
    )
    yield OutputItemDoneEvent(response_id="resp_1", sequence_number=5, item_index=0, item=done)
    yield OutputItemAddedEvent(
        response_id="resp_1",
        sequence_number=6,
        item_index=1,
        item=message_started,
    )
    yield ContentPartAddedEvent(
        response_id="resp_1",
        sequence_number=7,
        item_id="msg_1",
        item_index=1,
        content_index=0,
        part=OutputTextPart(text=""),
    )
    yield ContentPartDeltaEvent(
        response_id="resp_1",
        sequence_number=8,
        item_id="msg_1",
        item_index=1,
        content_index=0,
        delta="Hello",
    )
    yield ContentPartDoneEvent(
        response_id="resp_1",
        sequence_number=9,
        item_id="msg_1",
        item_index=1,
        content_index=0,
        part=message_done.content[0],
    )
    yield OutputItemDoneEvent(
        response_id="resp_1",
        sequence_number=10,
        item_index=1,
        item=message_done,
    )
    yield ResponseCompletedEvent(
        response_id="resp_1",
        sequence_number=11,
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


async def _failed_output_events() -> AsyncIterator:
    yield ResponseStartedEvent(
        response_id="resp_1",
        sequence_number=0,
        metadata=ResponseMetadata(model="gpt-test", object="response"),
    )
    yield ResponseFailedEvent(
        response_id="resp_1",
        sequence_number=1,
        error=ErrorDetail(code="failed", message="Provider failed."),
        usage=Usage(input_tokens=1, output_tokens=0, total_tokens=1),
    )


async def _unknown_incomplete_output_events() -> AsyncIterator:
    yield ResponseStartedEvent(
        response_id="resp_1",
        sequence_number=0,
        metadata=ResponseMetadata(model="gpt-test", object="response"),
    )
    yield ResponseIncompleteEvent(
        response_id="resp_1",
        sequence_number=1,
        reason="unknown_reason",
        usage=Usage(input_tokens=1, output_tokens=0, total_tokens=1),
    )


def test_chat_request_translates_to_responses_endpoint() -> None:
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

    assert build_openai_request(request.to_ir(), "responses") == {
        "model": "gpt-4.1-nano",
        "temperature": 0.4,
        "max_output_tokens": 20,
        "input": [
            {"role": "system", "content": "Be concise."},
            {"role": "developer", "content": "Use JSON."},
            {"role": "user", "content": "Say hi."},
        ],
        "text": {"format": {"type": "json_object"}},
    }


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


@pytest.mark.parametrize(
    "unsupported",
    [
        {"service_tier": "default"},
        {"stop": "END"},
    ],
)
def test_chat_fields_without_ir_reject(unsupported: dict) -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        **unsupported,
    )

    with pytest.raises(BadInputError, match="without an IR representation"):
        request.to_ir()


def test_chat_n_one_uses_the_single_completion_ir_contract() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        n=1,
    )

    assert request.to_ir().model == "openai/gpt-4.1-nano"


def test_chat_multiple_choices_reject_without_an_ir_representation() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        n=2,
    )

    with pytest.raises(BadInputError, match="n values other than 1"):
        request.to_ir()


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "hi", "name": "named-user"},
        {"role": "assistant", "content": None, "refusal": "refused"},
    ],
)
def test_chat_history_fields_without_ir_reject(message: dict) -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[message],
    )

    with pytest.raises(BadInputError, match="no IR representation"):
        request.to_ir()


def test_responses_background_execution_is_rejected_locally() -> None:
    with pytest.raises(ValidationError, match="background"):
        OpenAIResponsesRequest(
            model="openai/gpt-4.1-nano",
            input="hi",
            background=True,
        )


def test_responses_service_tier_without_ir_rejects() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input="hi",
        service_tier="default",
    )

    with pytest.raises(BadInputError, match="service_tier"):
        request.to_ir()


def test_responses_text_verbosity_without_ir_rejects() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input="hi",
        text={"verbosity": "low"},
    )

    with pytest.raises(BadInputError, match="verbosity"):
        request.to_ir()


def test_responses_function_output_status_without_ir_rejects() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "done",
                "status": "completed",
            },
        ],
    )

    with pytest.raises(BadInputError, match="status has no IR representation"):
        request.to_ir()


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


def test_responses_request_translates_to_chat_endpoint() -> None:
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

    assert build_openai_request(request.to_ir(), "chat_completions") == {
        "model": "gpt-4.1-nano",
        "messages": [{"role": "user", "content": "Use one function."}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "first", "parameters": {}, "strict": True},
            },
            {
                "type": "function",
                "function": {"name": "second", "parameters": {}, "strict": True},
            },
        ],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "function", "function": {"name": "first"}},
                {"type": "function", "function": {"name": "second"}},
            ],
        },
    }


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

    with pytest.raises(BadInputError, match="tool.hosted"):
        await Molto(strict=True).responses(request, api_key="test", endpoint="chat_completions")


async def test_public_payloads_attach_outer_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)

    with pytest.raises(BadInputError, match="Unsupported endpoint"):
        await Molto().responses(
            {"input": "hi"},
            api_key="test",
            model="openai/gpt-5.6",
            endpoint="invalid",  # type: ignore[arg-type]
        )

    with pytest.raises(BadInputError, match="Unsupported endpoint"):
        await Molto().chat_completions(
            {"messages": [{"role": "user", "content": "hi"}]},
            api_key="test",
            model="openai/gpt-5.6",
            endpoint="invalid",  # type: ignore[arg-type]
        )


def test_public_payloads_validate_dicts_with_endpoint_schema() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpenAIResponsesRequest.model_validate(
            {"model": "openai/gpt-5.6", "input": "hi", "endpoint": "responses"}
        )

    body = OpenAIChatCompletionRequest.model_validate(
        {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert body.messages[0].content == "hi"


def test_complete_request_models_validate_provider_qualified_models() -> None:
    with pytest.raises(ValidationError, match="provider prefix"):
        OpenAIResponsesRequest(model="gpt-5.6", input="hi")

    with pytest.raises(ValidationError, match="unsupported provider prefix"):
        OpenAIChatCompletionRequest(
            model="other/gpt-5.6",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_public_methods_require_api_key() -> None:
    required = inspect.Parameter.empty

    assert inspect.signature(Molto.responses).parameters["api_key"].default is required
    assert inspect.signature(Molto.chat_completions).parameters["api_key"].default is required


async def test_molto_uses_init_http_client_and_per_call_retry_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    http_client = httpx.AsyncClient()

    class CapturingProvider:
        def __init__(self, endpoint, **kwargs) -> None:
            captured.append({"endpoint": endpoint, **kwargs})

        def run(self, _request) -> AsyncIterator:
            return chat_response_to_output_events(_chat_payload("Hello"))

    monkeypatch.setattr("molto.core.OpenAIProvider", CapturingProvider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
    )
    gateway = Molto(http_client=http_client, max_retries=5)

    try:
        await gateway.chat_completions(request, api_key="test")
        await gateway.chat_completions(request, api_key="test", max_retries=1)
        await gateway.close()

        assert captured[0]["api_key"] == "test"
        assert captured[0]["http_client"] is http_client
        assert captured[0]["max_retries"] == 5
        assert captured[1]["http_client"] is http_client
        assert captured[1]["max_retries"] == 1
        assert not http_client.is_closed
    finally:
        await http_client.aclose()


def test_responses_request_models_reject_duplicate_tools() -> None:
    duplicate_tools = [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    with pytest.raises(ValidationError, match="Duplicate tool names"):
        OpenAIResponsesRequest(
            model="openai/gpt-4.1-nano",
            input="hi",
            tools=duplicate_tools,
        )

    with pytest.raises(ValidationError, match="Duplicate tool names"):
        OpenAIResponsesRestrictedRequest(input="hi", tools=duplicate_tools)


async def test_responses_state_rejects_chat_endpoint_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input="Continue.",
        previous_response_id="resp_1",
    )

    with pytest.raises(ModelCapabilityError, match="state.response"):
        await Molto(strict=True).responses(request, api_key="test", endpoint="chat_completions")


@pytest.mark.parametrize(
    ("include_usage", "expected_usage"),
    [(True, True), (False, False)],
)
async def test_chat_stream_usage_option_is_not_forwarded_to_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    include_usage: bool,
    expected_usage: bool,
) -> None:
    captured = {}

    class _ResponsesProvider:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, request):
            captured["stream_options"] = request.stream_options
            captured["generation"] = request.generation
            captured["payload"] = build_openai_request(request, "responses")

            async def events():
                payload = {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": {
                        "id": "resp_1",
                        "object": "response",
                        "created_at": 1,
                        "model": "gpt-5.6-luna",
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "total_tokens": 3,
                        },
                    },
                }
                yield PreservedOpenAIStreamEvent(
                    response_id="resp_1",
                    sequence_number=1,
                    endpoint="responses",
                    event_type="response.completed",
                    payload_json=json.dumps(payload),
                )

            return events()

    monkeypatch.setattr("molto.core.OpenAIProvider", _ResponsesProvider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": include_usage},
        frequency_penalty=0,
        presence_penalty=0,
    )

    result = await Molto().chat_completions(request, api_key="test", endpoint="responses")
    chunks = [chunk async for chunk in result]

    assert captured["stream_options"] is None
    assert captured["generation"].frequency_penalty == 0
    assert captured["generation"].presence_penalty == 0
    assert "frequency_penalty" not in captured["payload"]
    assert "presence_penalty" not in captured["payload"]
    assert chunks[0].choices[0].finish_reason == "stop"
    assert (chunks[0].usage is not None) is expected_usage
    assert len(chunks) == 1


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
async def test_chat_non_neutral_penalty_rejects_responses_endpoint_before_construction(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hi"}],
        **{field: 0.5},
    )

    with pytest.raises(ModelCapabilityError, match=field):
        await Molto(strict=True).chat_completions(request, api_key="test", endpoint="responses")


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
async def test_non_strict_drops_unsupported_penalty_for_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    captured = {}

    class _ResponsesProvider:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, request):
            captured["payload"] = build_openai_request(request, "responses")
            captured["generation"] = request.generation

            async def events():
                payload = {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": {
                        "id": "resp_1",
                        "object": "response",
                        "created_at": 1,
                        "model": "gpt-5.6-luna",
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "total_tokens": 3,
                        },
                    },
                }
                yield PreservedOpenAIStreamEvent(
                    response_id="resp_1",
                    sequence_number=1,
                    endpoint="responses",
                    event_type="response.completed",
                    payload_json=json.dumps(payload),
                )

            return events()

    monkeypatch.setattr("molto.core.OpenAIProvider", _ResponsesProvider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        **{field: 0.5},
    )

    response = await Molto().chat_completions(request, api_key="test", endpoint="responses")
    chunks = [chunk async for chunk in response]

    assert getattr(captured["generation"], field) == 0.5
    assert field not in captured["payload"]
    assert chunks[0].choices[0].finish_reason == "stop"


async def test_strict_rejects_unsupported_penalty_for_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hi"}],
        frequency_penalty=0.5,
    )

    with pytest.raises(ModelCapabilityError, match="frequency_penalty"):
        await Molto(strict=True).chat_completions(request, api_key="test", endpoint="responses")


async def test_previous_chat_audio_rejects_responses_endpoint_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "assistant",
                "content": "Earlier audio.",
                "audio": {"id": "audio_1"},
            }
        ],
    )

    with pytest.raises(ModelCapabilityError, match="input.audio"):
        await Molto(strict=True).chat_completions(request, api_key="test", endpoint="responses")


@pytest.mark.parametrize("public_endpoint", ["chat_completions", "responses"])
async def test_anthropic_defaults_to_chat_completions_endpoint(
    public_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class AnthropicChatProvider:
        def __init__(self, endpoint, **_kwargs):
            captured["endpoint"] = endpoint

        def run(self, _request) -> AsyncIterator:
            return chat_response_to_output_events(_chat_payload("Hello"))

    monkeypatch.setattr("molto.core.OpenAIProvider", AnthropicChatProvider)
    if public_endpoint == "chat_completions":
        request = OpenAIChatCompletionRequest(
            model="anthropic/claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
        )
        response = await Molto().chat_completions(request, api_key="test")
        assert response.choices[0].message.content == "Hello"
    else:
        request = OpenAIResponsesRequest(model="anthropic/claude-sonnet-5", input="hi")
        response = await Molto().responses(request, api_key="test")
        assert response.output[0].content[0].text == "Hello"

    assert captured["endpoint"] == "chat_completions"


async def test_anthropic_responses_to_chat_projects_endpoint_specific_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class AnthropicChatProvider:
        def __init__(self, endpoint, **_kwargs):
            captured["endpoint"] = endpoint

        def run(self, request) -> AsyncIterator:
            captured["request"] = request
            return chat_response_to_output_events(_chat_payload("Hello"))

    monkeypatch.setattr("molto.core.OpenAIProvider", AnthropicChatProvider)
    request = OpenAIResponsesRequest(
        model="anthropic/claude-sonnet-5",
        input="hi",
        include=["reasoning.encrypted_content"],
        reasoning={"effort": "high", "summary": "auto"},
        store=False,
    )

    response = await Molto().responses(request, api_key="test")

    assert response.output[0].content[0].text == "Hello"
    assert captured["endpoint"] == "chat_completions"
    assert captured["request"].include is None
    assert captured["request"].generation.reasoning_summary is None
    assert captured["request"].generation.reasoning_effort == "high"
    assert captured["request"].store is False


async def test_responses_web_search_projects_to_chat_provider_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class ChatProvider:
        def __init__(self, endpoint, **_kwargs):
            captured["endpoint"] = endpoint

        def run(self, request) -> AsyncIterator:
            captured["payload"] = build_openai_request(request, "chat_completions")
            return chat_response_to_output_events(_chat_payload("Hello"))

    monkeypatch.setattr("molto.core.OpenAIProvider", ChatProvider)
    response = await Molto(strict=True).responses(
        OpenAIResponsesRequest(
            model="openai/gpt-test",
            input="Search the web.",
            tools=[{"type": "web_search"}],
        ),
        api_key="test",
        endpoint="chat_completions",
    )

    assert response.output[0].content[0].text == "Hello"
    assert captured["endpoint"] == "chat_completions"
    assert captured["payload"]["web_search_options"] == {}


async def test_anthropic_responses_web_search_rejects_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    with pytest.raises(ModelCapabilityError, match="tool.web_search"):
        await Molto(strict=True).responses(
            OpenAIResponsesRequest(
                model="anthropic/claude-test",
                input="Search the web.",
                tools=[{"type": "web_search"}],
            ),
            api_key="test",
            endpoint="chat_completions",
        )


@pytest.mark.parametrize("public_endpoint", ["chat_completions", "responses"])
async def test_anthropic_responses_endpoint_rejects_before_provider_construction(
    public_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    if public_endpoint == "chat_completions":
        request = OpenAIChatCompletionRequest(
            model="anthropic/claude-test",
            messages=[{"role": "user", "content": "hi"}],
        )
        call = Molto().chat_completions(request, api_key="test", endpoint="responses")
    else:
        request = OpenAIResponsesRequest(
            model="anthropic/claude-test",
            input="hi",
        )
        call = Molto().responses(request, api_key="test", endpoint="responses")

    with pytest.raises(ValueError, match="does not support endpoint 'responses'"):
        await call


def test_endpoint_is_not_part_of_openai_request_models() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpenAIChatCompletionRequest(
            model="openai/gpt-4.1-nano",
            messages=[{"role": "user", "content": "hi"}],
            endpoint="responses",
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpenAIResponsesRequest(
            model="openai/gpt-4.1-nano",
            input="hi",
            endpoint="chat_completions",
        )


async def test_invalid_endpoint_keyword_rejects_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args, **_kwargs):
        raise AssertionError("OpenAIProvider must not be constructed")

    monkeypatch.setattr("molto.core.OpenAIProvider", unexpected_provider)
    request = OpenAIResponsesRequest(model="openai/gpt-4.1-nano", input="hi")

    with pytest.raises(BadInputError, match="Unsupported endpoint"):
        await Molto().responses(request, api_key="test", endpoint="invalid")


async def test_public_methods_reject_the_other_request_type() -> None:
    chat_request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
    )
    responses_request = OpenAIResponsesRequest(model="openai/gpt-4.1-nano", input="hi")

    with pytest.raises(BadInputError, match="OpenAIResponsesRequest"):
        await Molto().responses(chat_request, api_key="test")
    with pytest.raises(BadInputError, match="OpenAIChatCompletionRequest"):
        await Molto().chat_completions(responses_request, api_key="test")


async def test_endpoint_is_keyword_only() -> None:
    request = OpenAIResponsesRequest(model="openai/gpt-4.1-nano", input="hi")

    with pytest.raises(TypeError, match="positional"):
        await Molto().responses(request, "test", "chat_completions")


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
        await Molto(strict=True).chat_completions(
            request,
            api_key="test",
            base_url="https://example.test/v1",
        )


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

    with pytest.raises(BadInputError, match="tool.hosted"):
        await Molto(strict=True).responses(
            request,
            api_key="test",
            endpoint="chat_completions",
            base_url="https://example.test/v1",
        )


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

    with pytest.raises(BadInputError, match="audio"):
        await Molto(strict=True).chat_completions(request, api_key="test", endpoint="responses")


@pytest.mark.parametrize("stream", [False, True])
async def test_responses_reasoning_to_chat_is_intentionally_omitted(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    class ReasoningProvider:
        endpoint: str | None = None

        def __init__(self, endpoint, **kwargs) -> None:
            del kwargs
            self.__class__.endpoint = endpoint

        def run(self, request) -> AsyncIterator:
            assert "endpoint" not in request.model_dump()
            return _reasoning_output_events()

    monkeypatch.setattr("molto.core.OpenAIProvider", ReasoningProvider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "Think carefully."}],
        reasoning_effort="high",
        stream=stream,
    )

    response = await Molto().chat_completions(request, api_key="test", endpoint="responses")

    assert ReasoningProvider.endpoint == "responses"
    if stream:
        chunks = [chunk async for chunk in response]
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"
    else:
        assert response.choices[0].message.content == "Hello"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("events", "message"),
    [
        (_failed_output_events, "failed Response"),
        (_unknown_incomplete_output_events, "unknown_reason"),
    ],
)
async def test_responses_terminal_state_rejects_unrepresentable_chat_output(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    events: Callable[[], AsyncIterator],
    message: str,
) -> None:
    class TerminalProvider:
        def __init__(self, endpoint, **kwargs) -> None:
            del endpoint, kwargs

        def run(self, request) -> AsyncIterator:
            del request
            return events()

    monkeypatch.setattr("molto.core.OpenAIProvider", TerminalProvider)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
    )

    if stream:
        result = await Molto().chat_completions(request, api_key="test", endpoint="responses")
        with pytest.raises(BadInputError, match=message):
            _ = [chunk async for chunk in result]
    else:
        with pytest.raises(BadInputError, match=message):
            await Molto().chat_completions(request, api_key="test", endpoint="responses")


async def test_provider_closes_owned_client_when_sdk_call_fails() -> None:
    provider = OpenAIProvider(
        "chat_completions",
        api_key="test",
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


async def test_molto_does_not_close_caller_owned_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    http_client = httpx.AsyncClient()

    class CapturingProvider:
        def __init__(self, _endpoint, **kwargs) -> None:
            captured.update(kwargs)

        def run(self, _request) -> AsyncIterator:
            return chat_response_to_output_events(_chat_payload("Hello"))

    monkeypatch.setattr("molto.core.OpenAIProvider", CapturingProvider)
    molto = Molto(http_client=http_client)
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Say hello."}],
    )

    try:
        await molto.chat_completions(request, api_key="test")
        await molto.close()

        assert captured["http_client"] is http_client
        assert not http_client.is_closed
    finally:
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
