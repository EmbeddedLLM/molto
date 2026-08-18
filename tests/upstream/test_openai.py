from collections.abc import AsyncIterator

import httpx
import pytest
from openai import AuthenticationError

from molto import Molto
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.core import VLLM_DUMMY_API_KEY
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.normalized import (
    FunctionTool,
    FunctionToolChoice,
    ModelResponseStreamEvent,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TerminationReason,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolReturn,
    UserInput,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEvent,
    OpenAIResponsesStreamEventAdapter,
)
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient

OPENAI_CAPABILITIES = PROVIDER_CAPABILITIES["openai"]
VLLM_CAPABILITIES = PROVIDER_CAPABILITIES["vllm"]


def _unexpected_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"events() performed independent HTTP I/O: {request.url}")


def _sse_response(*records: str) -> httpx.Response:
    return httpx.Response(
        200,
        text="".join(f"data: {record}\n\n" for record in records),
        headers={"content-type": "text/event-stream"},
    )


def _response(*, usage: bool = False) -> dict[str, object]:
    response: dict[str, object] = {
        "id": "resp_123",
        "created_at": 1.9,
        "model": "reasoning-model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "service_tier": "default",
        "tool_choice": "auto",
        "tools": [],
    }
    if usage:
        response["usage"] = {
            "input_tokens": 11,
            "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 18,
        }
    return response


def _response_event(data: dict[str, object]) -> OpenAIResponsesStreamEvent:
    return OpenAIResponsesStreamEventAdapter.validate_python(data)


def _chat_chunk(
    *,
    choices: list[dict[str, object]],
    usage: dict[str, object] | None = None,
) -> OpenAIChatCompletionsStreamChunk:
    return OpenAIChatCompletionsStreamChunk.model_validate(
        {
            "id": "chatcmpl_123",
            "choices": choices,
            "created": 2,
            "model": "reasoning-model",
            "object": "chat.completion.chunk",
            "service_tier": "default",
            "usage": usage,
        }
    )


async def test_responses_adapter_uses_target_default_service_tier():
    request = OpenAIResponsesCreateRequest(input="Hi", model="m", stream=True)
    adapter = OpenAIResponsesAdapter(OPENAI_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="resp_123", model="m", service_tier="priority")
        yield ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="m",
            service_tier="priority",
            termination_reason="stop",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    assert {event.response.service_tier for event in events} == {"default"}
    assert events[-1].type == "response.completed"
    assert events[-1].response.status == "completed"
    assert events[-1].response.incomplete_details is None


async def test_responses_adapter_does_not_synthesize_output_phase():
    request = OpenAIResponsesCreateRequest(input="Hi", model="m", stream=True)
    adapter = OpenAIResponsesAdapter(OPENAI_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="resp_123", model="m")
        yield PartStartEvent(index=0, part=Thinking(content="", summary=""))
        yield PartEndEvent(index=0, part=Thinking(content="Thought", summary=""))
        yield PartStartEvent(index=1, part=TextOutput(content=""))
        yield PartEndEvent(index=1, part=TextOutput(content="Answer"))
        yield ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="m",
            termination_reason="stop",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    message_items = [
        event.item.model_dump(exclude_unset=True)
        for event in events
        if event.type in {"response.output_item.added", "response.output_item.done"}
        and event.item.type == "message"
    ]
    message_items.extend(
        item.model_dump(exclude_unset=True)
        for item in events[-1].response.output
        if item.type == "message"
    )
    assert message_items
    assert all("phase" not in item for item in message_items)


def test_chat_completions_adapter_preserves_user_message_and_part_boundaries():
    request = OpenAIChatCompletionsCreateRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Follow-up"}]},
            {"role": "assistant", "content": "Answer"},
        ],
        model="m",
        stream=True,
    )

    normalized = OpenAIChatCompletionsAdapter(VLLM_CAPABILITIES).normalize_request(request)

    assert normalized.input == [
        UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")]),
        UserInput(content=[TextInput(content="Follow-up")]),
        TextOutput(content="Answer"),
    ]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("vllm", Thinking(content="Reasoning", summary="")),
        ("openai", Thinking(content="", summary="Reasoning")),
    ],
)
def test_chat_completions_adapter_classifies_ambiguous_reasoning_history(
    provider: str,
    expected: Thinking,
) -> None:
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "assistant", "content": "Answer", "reasoning_content": "Reasoning"}],
        model="m",
        stream=True,
    )

    normalized = OpenAIChatCompletionsAdapter(PROVIDER_CAPABILITIES[provider]).normalize_request(
        request
    )

    assert normalized.input == [expected, TextOutput(content="Answer")]


def test_responses_adapter_preserves_user_message_and_part_boundaries():
    request = OpenAIResponsesCreateRequest(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Part one"},
                    {"type": "input_text", "text": "Part two"},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Follow-up"}],
            },
            {"role": "assistant", "content": "Answer"},
        ],
        model="m",
        stream=True,
    )

    normalized = OpenAIResponsesAdapter(OPENAI_CAPABILITIES).normalize_request(request)

    assert normalized.input == [
        UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")]),
        UserInput(content=[TextInput(content="Follow-up")]),
        TextOutput(content="Answer"),
    ]


def test_responses_client_projects_typed_text_history_without_phase():
    body = OpenAIResponsesClient._request_body(
        NormalizedRequest(
            input=[
                UserInput(
                    content=[
                        TextInput(content="Question part one"),
                        TextInput(content="Question part two"),
                    ]
                ),
                UserInput(content=[TextInput(content="Follow-up")]),
                TextOutput(content="Answer"),
            ],
            model="m",
        )
    )

    assert body["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Question part one"},
                {"type": "input_text", "text": "Question part two"},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Follow-up"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Answer",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
    ]


def test_chat_completions_client_preserves_user_message_and_part_boundaries():
    messages = OpenAIChatCompletionsClient._messages(
        NormalizedRequest(
            input=[
                UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")]),
                UserInput(content=[TextInput(content="Follow-up")]),
                TextOutput(content="Answer"),
            ]
        )
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "Follow-up"}]},
        {"role": "assistant", "content": "Answer"},
    ]


def test_openai_adapters_normalize_function_tools_and_history():
    responses = OpenAIResponsesAdapter(OPENAI_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(
            input=[
                {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "lookup",
                    "arguments": "{}",
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "result",
                },
            ],
            model="m",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                }
            ],
            tool_choice={"type": "function", "name": "lookup"},
        )
    )
    chat = OpenAIChatCompletionsAdapter(OPENAI_CAPABILITIES).normalize_request(
        OpenAIChatCompletionsCreateRequest(
            messages=[
                {
                    "role": "assistant",
                    "content": "Calling lookup.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "result"},
            ],
            model="m",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a value.",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": False,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "lookup"}},
        )
    )

    tool = FunctionTool(
        name="lookup",
        description="Look up a value.",
        parameters={"type": "object", "properties": {}},
    )
    assert responses.tools == chat.tools == [tool]
    assert responses.tool_choice == chat.tool_choice == FunctionToolChoice(name="lookup")
    assert responses.input == [
        ToolCall(id="fc_123", tool_call_id="call_123", tool_name="lookup", args="{}"),
        ToolReturn(tool_call_id="call_123", tool_name=None, content="result"),
    ]
    assert chat.input == [
        TextOutput(content="Calling lookup."),
        ToolCall(tool_call_id="call_123", tool_name="lookup", args="{}"),
        ToolReturn(tool_call_id="call_123", tool_name=None, content="result"),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("allowed_callers", ["programmatic"], "directly callable"),
        ("defer_loading", True, "Deferred"),
        ("output_schema", {"type": "object"}, "output schemas"),
    ],
)
def test_responses_adapter_rejects_nonportable_function_tool_fields(
    field,
    value,
    message,
):
    tool = {
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object"},
        "strict": False,
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        OpenAIResponsesAdapter(OPENAI_CAPABILITIES).normalize_request(
            OpenAIResponsesCreateRequest(model="m", tools=[tool])
        )


@pytest.mark.parametrize(
    ("item_type", "extra", "message"),
    [
        ("function_call", {"caller": {"type": "program", "caller_id": "program_1"}}, "direct"),
        ("function_call", {"namespace": "tools"}, "Namespaced"),
        ("function_call", {"status": "incomplete"}, "status"),
        (
            "function_call_output",
            {"caller": {"type": "program", "caller_id": "program_1"}},
            "direct",
        ),
        ("function_call_output", {"namespace": "tools"}, "Namespaced"),
        ("function_call_output", {"status": "incomplete"}, "status"),
    ],
)
def test_responses_adapter_rejects_nonportable_function_history(
    item_type,
    extra,
    message,
):
    item = (
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "lookup",
            "arguments": "{}",
        }
        if item_type == "function_call"
        else {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "result",
        }
    )
    item.update(extra)

    with pytest.raises(ValueError, match=message):
        OpenAIResponsesAdapter(OPENAI_CAPABILITIES).normalize_request(
            OpenAIResponsesCreateRequest(input=[item], model="m")
        )


def test_responses_adapter_normalizes_text_part_function_output():
    normalized = OpenAIResponsesAdapter(OPENAI_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": [{"type": "input_text", "text": "result"}],
                }
            ],
            model="m",
        )
    )

    assert normalized.input == [
        ToolReturn(
            tool_call_id="call_123",
            content=[TextInput(content="result")],
        )
    ]


def test_openai_clients_project_function_tools_and_history():
    request = NormalizedRequest(
        input=[
            TextOutput(content="Calling lookup."),
            ToolCall(tool_call_id="call_123", tool_name="lookup", args="{}"),
            ToolReturn(tool_call_id="call_123", tool_name=None, content="result"),
        ],
        model="m",
        tools=[FunctionTool(name="lookup", parameters={"type": "object"})],
        tool_choice=FunctionToolChoice(name="lookup"),
    )

    responses = OpenAIResponsesClient._request_body(request)
    chat = OpenAIChatCompletionsClient._request_body(request)

    assert responses["input"] == [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Calling lookup.",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
        {
            "type": "function_call",
            "status": "completed",
            "arguments": "{}",
            "call_id": "call_123",
            "name": "lookup",
        },
        {"type": "function_call_output", "call_id": "call_123", "output": "result"},
    ]
    assert responses["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]
    assert responses["tool_choice"] == {"type": "function", "name": "lookup"}
    assert chat["messages"] == [
        {
            "role": "assistant",
            "content": "Calling lookup.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_123"},
    ]
    assert chat["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
                "strict": False,
            },
        }
    ]
    assert chat["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }


@pytest.mark.parametrize(
    "client",
    [OpenAIResponsesClient, OpenAIChatCompletionsClient],
)
def test_openai_clients_reject_failed_tool_results(client):
    request = NormalizedRequest(
        input=[ToolReturn(content="failed", tool_call_id="call_123", outcome="failed")],
        model="m",
    )

    with pytest.raises(ValueError, match="cannot preserve non-successful"):
        client._request_body(request)


@pytest.mark.parametrize("termination_reason", ["content_filter", "max_output_tokens"])
async def test_responses_adapter_projects_incomplete_termination(
    termination_reason: TerminationReason,
):
    request = OpenAIResponsesCreateRequest(input="Hi", model="m", stream=True)
    adapter = OpenAIResponsesAdapter(OPENAI_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="resp_123", model="m")
        yield ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="m",
            termination_reason=termination_reason,
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    terminal = events[-1]
    assert terminal.type == "response.incomplete"
    assert terminal.response.status == "incomplete"
    assert terminal.response.incomplete_details is not None
    assert terminal.response.incomplete_details.reason == termination_reason
    assert terminal.response.service_tier == "default"


@pytest.mark.parametrize(
    "termination_reason",
    [
        "model_context_window_exceeded",
        "pause",
        "refusal",
        "stop_sequence",
        "tool_call",
    ],
)
async def test_responses_adapter_coerces_unrepresentable_termination_to_completed(
    termination_reason: TerminationReason,
):
    request = OpenAIResponsesCreateRequest(input="Hi", model="m", stream=True)
    adapter = OpenAIResponsesAdapter(OPENAI_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="resp_123", model="m")
        yield ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="m",
            termination_reason=termination_reason,
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    terminal = events[-1]
    assert terminal.type == "response.completed"
    assert terminal.response.status == "completed"
    assert terminal.response.incomplete_details is None


@pytest.mark.parametrize(
    "termination_reason",
    ["model_context_window_exceeded", "pause", "refusal"],
)
async def test_chat_completions_adapter_coerces_unrepresentable_termination_to_stop(
    termination_reason: TerminationReason,
):
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
    )
    adapter = OpenAIChatCompletionsAdapter(VLLM_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="chatcmpl_123", model="m")
        yield ResponseEndEvent(
            id="chatcmpl_123",
            created_at=1,
            model="m",
            termination_reason=termination_reason,
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    assert events[-1].choices[0].finish_reason == "stop"


async def test_chat_completions_adapter_uses_target_default_service_tier():
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
    )
    adapter = OpenAIChatCompletionsAdapter(VLLM_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="chatcmpl_123", model="m", service_tier="priority")
        yield ResponseEndEvent(
            id="chatcmpl_123",
            created_at=1,
            model="m",
            service_tier="priority",
            termination_reason="stop",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    assert {event.service_tier for event in events} == {"default"}


async def test_chat_completions_adapter_collapses_reasoning_summary() -> None:
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
    )
    adapter = OpenAIChatCompletionsAdapter(OPENAI_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events() -> AsyncIterator[ModelResponseStreamEvent]:
        yield ResponseStartEvent(id="chatcmpl_123", model="m")
        yield PartStartEvent(index=0, part=Thinking(content="", summary=""))
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Summary"))
        yield PartEndEvent(index=0, part=Thinking(content="", summary="Summary"))
        yield ResponseEndEvent(
            id="chatcmpl_123",
            created_at=1,
            model="m",
            termination_reason="stop",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    reasoning_chunk = next(
        event
        for event in events
        if event.choices and event.choices[0].delta.reasoning_content == "Summary"
    )
    assert reasoning_chunk.choices[0].delta.reasoning_content == "Summary"


async def test_responses_adapter_maps_same_provider_opaque_reasoning_state():
    request = OpenAIResponsesCreateRequest(
        input=[
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Prior thought"}],
                "encrypted_content": "request-state",
                "summary": [],
            },
            {"role": "assistant", "content": "Prior answer"},
        ],
        model="m",
        stream=True,
    )
    adapter = OpenAIResponsesAdapter(OPENAI_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)
    assert isinstance(normalized_request.input, list)
    assert isinstance(normalized_request.input[0], Thinking)
    assert normalized_request.input[0].signature == "request-state"

    async def normalized_events():
        yield ResponseStartEvent(id="resp_123", model="m")
        yield PartStartEvent(
            index=0,
            part=Thinking(content="", summary="", signature="initial-state"),
        )
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="New thought"))
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="response-state"))
        yield PartEndEvent(
            index=0,
            part=Thinking(content="New thought", summary="", signature="response-state"),
        )
        yield ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="m",
            termination_reason="stop",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]
    added = next(event for event in events if event.type == "response.output_item.added")
    done = next(event for event in events if event.type == "response.output_item.done")
    completed = next(event for event in events if event.type == "response.completed")
    assert added.item.encrypted_content == "initial-state"
    assert done.item.encrypted_content == "response-state"
    assert completed.response.output[0].encrypted_content == "response-state"


async def test_responses_events_translate_native_sdk_events(monkeypatch: pytest.MonkeyPatch):
    native_events = [
        _response_event(
            {
                "type": "response.created",
                "response": _response(),
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "summary": [],
                    "content": [],
                    "encrypted_content": "initial-state",
                },
                "output_index": 0,
                "sequence_number": 1,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_text.delta",
                "content_index": 0,
                "delta": "Think",
                "item_id": "rs_123",
                "output_index": 0,
                "sequence_number": 2,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Summary"}],
                    "content": [{"type": "reasoning_text", "text": "Think"}],
                    "encrypted_content": "response-state",
                },
                "output_index": 0,
                "sequence_number": 3,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
                "output_index": 1,
                "sequence_number": 4,
            }
        ),
        _response_event(
            {
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "Answer",
                "item_id": "msg_123",
                "logprobs": [],
                "output_index": 1,
                "sequence_number": 5,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Answer",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                "output_index": 1,
                "sequence_number": 6,
            }
        ),
        _response_event(
            {
                "type": "response.completed",
                "response": _response(usage=True),
                "sequence_number": 7,
            }
        ),
    ]
    captured_requests: list[OpenAIResponsesCreateRequest] = []

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        captured_requests.append(native_request)
        for event in native_events:
            yield event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        events = [event async for event in client.events(NormalizedRequest(input="Hi", model="m"))]

    assert len(captured_requests) == 1
    assert isinstance(captured_requests[0], OpenAIResponsesCreateRequest)
    assert captured_requests[0].model_dump() == {
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hi"}],
            }
        ],
        "model": "m",
    }
    assert events == [
        ResponseStartEvent(
            id="resp_123", created_at=1, model="reasoning-model", service_tier="default"
        ),
        PartStartEvent(
            index=0,
            part=Thinking(content="", summary="", id="rs_123", signature="initial-state"),
        ),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="Think")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="response-state")),
        PartEndEvent(
            index=0,
            part=Thinking(
                content="Think",
                summary="Summary",
                id="rs_123",
                signature="response-state",
            ),
        ),
        PartStartEvent(index=1, part=TextOutput(content="", id="msg_123")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="Answer")),
        PartEndEvent(index=1, part=TextOutput(content="Answer", id="msg_123")),
        ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="reasoning-model",
            service_tier="default",
            termination_reason="stop",
            usage={
                "input_tokens": 11,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 18,
            },
        ),
    ]


async def test_responses_events_normalize_explicit_reasoning_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_events = [
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {"id": "rs_123", "type": "reasoning", "content": [], "summary": []},
                "output_index": 0,
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "Summary",
                "item_id": "rs_123",
                "output_index": 0,
                "sequence_number": 1,
                "summary_index": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [],
                    "summary": [{"type": "summary_text", "text": "Summary"}],
                },
                "output_index": 0,
                "sequence_number": 2,
            }
        ),
    ]

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        for event in native_events:
            yield event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [event async for event in client.events(NormalizedRequest(model="m"))]

    assert events == [
        PartStartEvent(index=0, part=Thinking(content="", summary="", id="rs_123")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Summary")),
        PartEndEvent(
            index=0,
            part=Thinking(content="", summary="Summary", id="rs_123"),
        ),
    ]


async def test_responses_adapter_projects_reasoning_summary_lifecycle() -> None:
    adapter = OpenAIResponsesAdapter(OPENAI_CAPABILITIES)
    request = adapter.normalize_request(
        OpenAIResponsesCreateRequest(input="Question", model="m", stream=True)
    )

    async def normalized_events() -> AsyncIterator[ModelResponseStreamEvent]:
        yield ResponseStartEvent(id="resp_123", created_at=1, model="m")
        yield PartStartEvent(index=0, part=Thinking(content="", summary=""))
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Summary"))
        yield PartEndEvent(
            index=0,
            part=Thinking(content="", summary="Summary", signature="opaque-state"),
        )
        yield ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="m",
            termination_reason="stop",
        )

    events = [event async for event in adapter.project_events(normalized_events(), request)]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    done = next(event for event in events if event.type == "response.output_item.done")
    assert done.item.content == []
    assert [part.text for part in done.item.summary] == ["Summary"]
    assert done.item.encrypted_content == "opaque-state"


@pytest.mark.parametrize(
    ("capabilities", "added_type", "done_type"),
    [
        (VLLM_CAPABILITIES, "response.content_part.added", "response.content_part.done"),
        (
            OPENAI_CAPABILITIES,
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        ),
    ],
)
async def test_responses_adapter_balances_empty_reasoning_part_lifecycle(
    capabilities: ProviderCapabilities,
    added_type: str,
    done_type: str,
) -> None:
    adapter = OpenAIResponsesAdapter(capabilities)
    request = adapter.normalize_request(
        OpenAIResponsesCreateRequest(input="Question", model="m", stream=True)
    )

    async def normalized_events() -> AsyncIterator[ModelResponseStreamEvent]:
        yield ResponseStartEvent(id="resp_123", created_at=1, model="m")
        yield PartStartEvent(index=0, part=Thinking(content="", summary=""))
        yield PartEndEvent(index=0, part=Thinking(content="", summary=""))
        yield ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="m",
            termination_reason="stop",
        )

    events = [event async for event in adapter.project_events(normalized_events(), request)]
    event_types = [event.type for event in events]

    assert event_types.index(added_type) < event_types.index(done_type)


async def test_responses_events_forward_same_provider_opaque_reasoning_state(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_requests: list[OpenAIResponsesCreateRequest] = []

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        captured_requests.append(native_request)
        if False:
            yield

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(
                    input=[
                        Thinking(
                            content="Prior thought",
                            summary="",
                            signature="request-state",
                        ),
                        TextOutput(content="Prior answer"),
                    ],
                    model="m",
                )
            )
        ]

    assert events == []
    assert captured_requests[0].model_dump()["input"][0]["encrypted_content"] == "request-state"


@pytest.mark.parametrize(
    "native_event",
    [
        _response_event(
            {
                "type": "response.in_progress",
                "response": _response(),
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.content_part.added",
                "content_index": 0,
                "item_id": "msg_123",
                "output_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.content_part.done",
                "content_index": 0,
                "item_id": "msg_123",
                "output_index": 0,
                "part": {"type": "output_text", "text": "Answer", "annotations": []},
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_text.done",
                "content_index": 0,
                "item_id": "rs_123",
                "output_index": 0,
                "text": "Think",
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_text.done",
                "content_index": 0,
                "item_id": "msg_123",
                "logprobs": [],
                "output_index": 0,
                "text": "Answer",
                "sequence_number": 0,
            }
        ),
    ],
)
async def test_responses_events_ignore_native_sdk_events(
    monkeypatch: pytest.MonkeyPatch,
    native_event: OpenAIResponsesStreamEvent,
):
    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield native_event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        assert [event async for event in client.events(NormalizedRequest())] == []


async def test_responses_events_retain_incomplete_reason(monkeypatch: pytest.MonkeyPatch):
    incomplete_response = _response()
    incomplete_response.update(
        {
            "incomplete_details": {"reason": "max_output_tokens"},
            "status": "incomplete",
        }
    )
    native_events = [
        _response_event(
            {
                "type": "response.created",
                "response": _response(),
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.incomplete",
                "response": incomplete_response,
                "sequence_number": 1,
            }
        ),
    ]

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        for event in native_events:
            yield event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [event async for event in client.events(NormalizedRequest(model="m"))]

    response_end = next(event for event in events if isinstance(event, ResponseEndEvent))
    assert response_end.termination_reason == "max_output_tokens"


async def test_responses_events_translate_function_call_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
):
    function_call = {
        "id": "fc_123",
        "type": "function_call",
        "status": "completed",
        "arguments": "{}",
        "call_id": "call_123",
        "name": "lookup",
    }
    completed_response = _response()
    completed_response["output"] = [function_call]
    completed_response["status"] = "completed"
    native_events = [
        _response_event(
            {"type": "response.created", "response": _response(), "sequence_number": 0}
        ),
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {**function_call, "status": "in_progress", "arguments": ""},
                "output_index": 0,
                "sequence_number": 1,
            }
        ),
        _response_event(
            {
                "type": "response.function_call_arguments.delta",
                "delta": "{}",
                "item_id": "fc_123",
                "output_index": 0,
                "sequence_number": 2,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": function_call,
                "output_index": 0,
                "sequence_number": 3,
            }
        ),
        _response_event(
            {
                "type": "response.completed",
                "response": completed_response,
                "sequence_number": 4,
            }
        ),
    ]

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        for native_event in native_events:
            yield native_event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        events = [event async for event in client.events(NormalizedRequest(model="m"))]

    assert events == [
        ResponseStartEvent(
            id="resp_123", created_at=1, model="reasoning-model", service_tier="default"
        ),
        PartStartEvent(
            index=0,
            part=ToolCall(id="fc_123", tool_call_id="call_123", tool_name="lookup", args=""),
        ),
        PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta="{}")),
        PartEndEvent(
            index=0,
            part=ToolCall(id="fc_123", tool_call_id="call_123", tool_name="lookup", args="{}"),
        ),
        ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="reasoning-model",
            service_tier="default",
            termination_reason="tool_call",
        ),
    ]


async def test_responses_events_reject_programmatic_function_call(
    monkeypatch: pytest.MonkeyPatch,
):
    native_event = _response_event(
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_123",
                "type": "function_call",
                "status": "in_progress",
                "arguments": "",
                "call_id": "call_123",
                "name": "lookup",
                "caller": {"type": "program", "caller_id": "program_1"},
            },
            "output_index": 0,
            "sequence_number": 1,
        }
    )

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield native_event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        with pytest.raises(ValueError, match="Only direct"):
            _ = [event async for event in client.events(NormalizedRequest(model="m"))]


async def test_chat_completions_events_preserve_multiple_tool_call_indices(
    monkeypatch: pytest.MonkeyPatch,
):
    native_chunks = [
        _chat_chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}}]),
        _chat_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "first", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "second", "arguments": ""},
                            },
                        ]
                    },
                }
            ]
        ),
        _chat_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "second", "arguments": '{"b":2}'},
                            },
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "first", "arguments": '{"a":1}'},
                            },
                        ]
                    },
                }
            ]
        ),
        _chat_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
    ]

    async def stub_native_events(
        native_request: OpenAIChatCompletionsCreateRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        for native_chunk in native_chunks:
            yield native_chunk

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIChatCompletionsClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [event async for event in client.events(NormalizedRequest(model="m"))]

    assert events == [
        ResponseStartEvent(
            id="chatcmpl_123", created_at=2, model="reasoning-model", service_tier="default"
        ),
        PartStartEvent(
            index=0,
            part=ToolCall(tool_call_id="call_a", tool_name="first", args=""),
        ),
        PartStartEvent(
            index=1,
            part=ToolCall(tool_call_id="call_b", tool_name="second", args=""),
        ),
        PartDeltaEvent(index=1, delta=ToolCallDelta(args_delta='{"b":2}')),
        PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta='{"a":1}')),
        PartEndEvent(
            index=0,
            part=ToolCall(tool_call_id="call_a", tool_name="first", args='{"a":1}'),
        ),
        PartEndEvent(
            index=1,
            part=ToolCall(tool_call_id="call_b", tool_name="second", args='{"b":2}'),
        ),
        ResponseEndEvent(
            id="chatcmpl_123",
            created_at=2,
            model="reasoning-model",
            service_tier="default",
            termination_reason="tool_call",
        ),
    ]


@pytest.mark.parametrize(
    ("capabilities", "thinking_delta", "thinking"),
    [
        (
            VLLM_CAPABILITIES,
            ThinkingDelta(content_delta="Think"),
            Thinking(content="Think", summary=""),
        ),
        (
            OPENAI_CAPABILITIES,
            ThinkingDelta(summary_delta="Think"),
            Thinking(content="", summary="Think"),
        ),
    ],
)
async def test_chat_completions_events_translate_native_sdk_chunks(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: ProviderCapabilities,
    thinking_delta: ThinkingDelta,
    thinking: Thinking,
):
    native_chunks = [
        _chat_chunk(choices=[{"index": 0, "delta": {"role": "assistant"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"reasoning_content": "Think"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"content": "Answer"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]),
        _chat_chunk(
            choices=[],
            usage={
                "prompt_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 18,
            },
        ),
    ]
    captured_requests: list[OpenAIChatCompletionsCreateRequest] = []

    async def stub_native_events(
        native_request: OpenAIChatCompletionsCreateRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        captured_requests.append(native_request)
        for chunk in native_chunks:
            yield chunk

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIChatCompletionsClient(http_client, capabilities=capabilities)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        events = [
            event
            async for event in client.events(NormalizedRequest(input="Hi", model="m", stream=True))
        ]

    assert len(captured_requests) == 1
    assert isinstance(captured_requests[0], OpenAIChatCompletionsCreateRequest)
    assert captured_requests[0].model_dump() == {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
            }
        ],
        "model": "m",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert events == [
        ResponseStartEvent(
            id="chatcmpl_123", created_at=2, model="reasoning-model", service_tier="default"
        ),
        PartStartEvent(index=0, part=Thinking(content="", summary="")),
        PartDeltaEvent(index=0, delta=thinking_delta),
        PartEndEvent(index=0, part=thinking),
        PartStartEvent(index=1, part=TextOutput(content="")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="Answer")),
        PartEndEvent(index=1, part=TextOutput(content="Answer")),
        ResponseEndEvent(
            id="chatcmpl_123",
            created_at=2,
            model="reasoning-model",
            service_tier="default",
            termination_reason="stop",
            usage={
                "input_tokens": 11,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 18,
            },
        ),
    ]


async def test_chat_completions_events_reject_empty_native_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    async def stub_native_events(
        native_request: OpenAIChatCompletionsCreateRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        if False:
            yield

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIChatCompletionsClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        with pytest.raises(
            ValueError,
            match="Chat Completions stream did not contain a response chunk",
        ):
            [event async for event in client.events(NormalizedRequest(model="m"))]


async def test_chat_completions_events_retain_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    native_chunks = [
        _chat_chunk(choices=[{"index": 0, "delta": {"role": "assistant"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "length"}]),
    ]

    async def stub_native_events(
        native_request: OpenAIChatCompletionsCreateRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        for chunk in native_chunks:
            yield chunk

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIChatCompletionsClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [event async for event in client.events(NormalizedRequest(model="m"))]

    response_end = next(event for event in events if isinstance(event, ResponseEndEvent))
    assert response_end.termination_reason == "max_output_tokens"


async def test_vllm_uses_explicit_url_and_dummy_credential(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-api-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "environment-organization")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "environment-project")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.invalid/v1")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _sse_response("[DONE]")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = Molto(http_client=http_client)
    stream = await client.responses(
        body={"input": "Hi", "model": "m", "stream": True},
        provider="vllm",
        base_url="https://vllm.example/v1",
    )

    assert [event async for event in stream] == []
    assert len(requests) == 1
    assert str(requests[0].url) == "https://vllm.example/v1/responses"
    assert requests[0].headers["authorization"] == f"Bearer {VLLM_DUMMY_API_KEY}"
    assert "openai-organization" not in requests[0].headers
    assert "openai-project" not in requests[0].headers
    await client.close()


async def test_chat_completions_sdk_consumes_terminal_marker():
    valid_chunk = (
        '{"id":"chatcmpl_123","choices":[],"created":1,"model":"m",'
        '"object":"chat.completion.chunk"}'
    )
    malformed_after_done = (
        '{"choices":[],"created":1,"model":"m","object":"chat.completion.chunk"}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(valid_chunk, "[DONE]", malformed_after_done)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIChatCompletionsClient(
            http_client,
            capabilities=VLLM_CAPABILITIES,
            api_key="test-api-key",
        )
        request = OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
        )
        events = [event async for event in client.native_events(request)]

    assert len(events) == 1
    assert type(events[0]) is OpenAIChatCompletionsStreamChunk
    assert events[0].id == "chatcmpl_123"


class _TrackingStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content

    async def aclose(self) -> None:
        self.closed = True


async def test_early_stop_closes_stream_but_keeps_shared_client_usable():
    first_chunk = (
        '{"id":"first","choices":[],"created":1,"model":"m","object":"chat.completion.chunk"}'
    )
    second_chunk = (
        '{"id":"second","choices":[],"created":2,"model":"m","object":"chat.completion.chunk"}'
    )
    response_streams = [
        _TrackingStream(_sse_response(first_chunk, second_chunk, "[DONE]").content),
        _TrackingStream(_sse_response(second_chunk, "[DONE]").content),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=response_streams.pop(0),
            headers={"content-type": "text/event-stream"},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = Molto(http_client=http_client)
    body = {"messages": [{"role": "user", "content": "Hi"}], "model": "m", "stream": True}

    first_response_stream = response_streams[0]
    first_events = await client.chat_completions(body=body, api_key="test-api-key")
    assert (await anext(first_events)).id == "first"
    await first_events.aclose()  # type: ignore[attr-defined]
    assert first_response_stream.closed
    assert not http_client.is_closed

    second_events = await client.chat_completions(body=body, api_key="test-api-key")
    assert [event.id async for event in second_events] == ["second"]
    assert not http_client.is_closed

    await client.close()
    assert http_client.is_closed


async def test_native_events_use_openai_sdk_errors():
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            401,
            json={
                "error": {
                    "code": "invalid_api_key",
                    "message": "Invalid key",
                    "type": "invalid_request_error",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIResponsesClient(
            http_client,
            capabilities=OPENAI_CAPABILITIES,
            api_key="bad-key",
        )
        request = OpenAIResponsesCreateRequest(input="Hi", model="m", stream=True)
        with pytest.raises(AuthenticationError) as exc_info:
            [event async for event in client.native_events(request)]

    assert exc_info.value.status_code == 401
    assert exc_info.value.code == "invalid_api_key"
    assert request_count == 1
