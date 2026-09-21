from collections.abc import AsyncIterator
from json import dumps

import httpx
import pytest
from openai import APIError, APIStatusError, OpenAIError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import NotFoundError as OpenAINotFoundError

from molto import Molto
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.core import VLLM_DUMMY_API_KEY
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ResourceNotFoundError,
    UnsupportedInput,
)
from molto.types.normalized import (
    CacheBreakpoint,
    FunctionTool,
    FunctionToolChoice,
    ImageUrlInput,
    ModelResponseStreamEvent,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    PromptCache,
    Reasoning,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
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
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesResponse,
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
    adapter = OpenAIResponsesAdapter(VLLM_CAPABILITIES)
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


def test_responses_client_preserves_history_item_ids():
    body = OpenAIResponsesClient._request_body(
        NormalizedRequest(
            input=[
                Thinking(content="Prior thought", summary="", id="rs_123"),
                TextOutput(content="Answer", id="msg_123"),
            ],
            model="m",
        )
    )

    assert body["input"][0]["id"] == "rs_123"
    assert body["input"][1]["id"] == "msg_123"


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


@pytest.mark.parametrize("summary", [None, "auto", "detailed"])
def test_chat_completions_client_maps_supported_reasoning_configuration(summary):
    body = OpenAIChatCompletionsClient._request_body(
        NormalizedRequest(model="m", reasoning=Reasoning(effort="high", summary=summary))
    )

    assert body["reasoning_effort"] == "high"


def test_chat_completions_client_remaps_parallel_image_result_after_tool_messages():
    request = NormalizedRequest(
        input=[
            UserInput(content=[TextInput(content="Call all tools.")]),
            Thinking(content="", summary="I should call all tools in parallel."),
            ToolCall(
                tool_call_id="call_a",
                tool_name="read_image",
                args='{"file_path": "a.png"}',
            ),
            ToolCall(
                tool_call_id="call_b",
                tool_name="read_image",
                args='{"file_path": "b.png"}',
            ),
            ToolCall(
                tool_call_id="call_c",
                tool_name="read_text",
                args='{"file_path": "c.txt"}',
            ),
            ToolCall(
                tool_call_id="call_d",
                tool_name="read_image",
                args='{"file_path": "d.png"}',
            ),
            ToolReturn(
                tool_call_id="call_a",
                content=[ImageUrlInput(url="data:image/png;base64,AA==")],
            ),
            ToolReturn(
                tool_call_id="call_b",
                content=[ImageUrlInput(url="data:image/png;base64,AA==")],
            ),
            ToolReturn(
                tool_call_id="call_c",
                content=[TextInput(content="Text from c.txt")],
            ),
            ToolReturn(
                tool_call_id="call_d",
                content=[ImageUrlInput(url="data:image/png;base64,AA==")],
            ),
            UserInput(content=[TextInput(content="Summarize all results.")]),
        ],
        model="m",
    )
    messages = OpenAIChatCompletionsClient._request_body(request)["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
        "tool",
        "user",
        "user",
        "user",
        "user",
    ]
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Call all tools."}],
        },
        {
            "role": "assistant",
            "reasoning_content": "I should call all tools in parallel.",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "read_image",
                        "arguments": '{"file_path": "a.png"}',
                    },
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "read_image",
                        "arguments": '{"file_path": "b.png"}',
                    },
                },
                {
                    "id": "call_c",
                    "type": "function",
                    "function": {
                        "name": "read_text",
                        "arguments": '{"file_path": "c.txt"}',
                    },
                },
                {
                    "id": "call_d",
                    "type": "function",
                    "function": {
                        "name": "read_image",
                        "arguments": '{"file_path": "d.png"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The image result of tool call 'call_a' will be attached as user prompt "
                        "after all tool responses."
                    ),
                }
            ],
            "tool_call_id": "call_a",
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The image result of tool call 'call_b' will be attached as user prompt "
                        "after all tool responses."
                    ),
                }
            ],
            "tool_call_id": "call_b",
        },
        {
            "role": "tool",
            "content": [{"type": "text", "text": "Text from c.txt"}],
            "tool_call_id": "call_c",
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The image result of tool call 'call_d' will be attached as user prompt "
                        "after all tool responses."
                    ),
                }
            ],
            "tool_call_id": "call_d",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Image returned by tool call 'call_a':",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Image returned by tool call 'call_b':",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Image returned by tool call 'call_d':",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize all results."}],
        },
    ]


def test_chat_completions_client_flushes_image_results_before_later_assistant_history():
    request = NormalizedRequest(
        input=[
            UserInput(content=[TextInput(content="Call both tools.")]),
            Thinking(content="", summary="I should call both tools in parallel."),
            ToolCall(
                tool_call_id="call_a",
                tool_name="read_image",
                args='{"file_path": "a.png"}',
            ),
            ToolCall(
                tool_call_id="call_b",
                tool_name="read_text",
                args='{"file_path": "b.txt"}',
            ),
            ToolReturn(
                tool_call_id="call_a",
                content=[ImageUrlInput(url="data:image/png;base64,AA==")],
            ),
            ToolReturn(
                tool_call_id="call_b",
                content=[TextInput(content="Text from b.txt")],
            ),
            Thinking(content="", summary="Let me try again."),
            ToolCall(
                tool_call_id="call_c",
                tool_name="read_image",
                args='{"file_path": "c.png"}',
            ),
            ToolReturn(
                tool_call_id="call_c",
                content=[ImageUrlInput(url="data:image/png;base64,AA==")],
            ),
            TextOutput(content="Assistant reply."),
            UserInput(content=[TextInput(content="Summarize both results.")]),
        ],
        model="m",
    )
    messages = OpenAIChatCompletionsClient._request_body(request)["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "user",
    ]
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Call both tools."}],
        },
        {
            "role": "assistant",
            "reasoning_content": "I should call both tools in parallel.",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "read_image",
                        "arguments": '{"file_path": "a.png"}',
                    },
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "read_text",
                        "arguments": '{"file_path": "b.txt"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The image result of tool call 'call_a' will be attached as user prompt "
                        "after all tool responses."
                    ),
                }
            ],
            "tool_call_id": "call_a",
        },
        {
            "role": "tool",
            "content": [{"type": "text", "text": "Text from b.txt"}],
            "tool_call_id": "call_b",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Image returned by tool call 'call_a':",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
        {
            "role": "assistant",
            "reasoning_content": "Let me try again.",
            "tool_calls": [
                {
                    "id": "call_c",
                    "type": "function",
                    "function": {
                        "name": "read_image",
                        "arguments": '{"file_path": "c.png"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The image result of tool call 'call_c' will be attached as user prompt "
                        "after all tool responses."
                    ),
                }
            ],
            "tool_call_id": "call_c",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Image returned by tool call 'call_c':",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "Assistant reply.",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize both results."}],
        },
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
        ToolReturn(
            tool_call_id="call_123",
            tool_name=None,
            content=[TextInput(content="result")],
        ),
    ]
    assert chat.input == [
        TextOutput(content="Calling lookup."),
        ToolCall(tool_call_id="call_123", tool_name="lookup", args="{}"),
        ToolReturn(
            tool_call_id="call_123",
            tool_name=None,
            content=[TextInput(content="result")],
        ),
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

    with pytest.raises(UnsupportedInput, match=message):
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

    with pytest.raises(UnsupportedInput, match=message):
        OpenAIResponsesAdapter(OPENAI_CAPABILITIES).normalize_request(
            OpenAIResponsesCreateRequest.model_construct(input=[item], model="m")
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
            ToolReturn(
                tool_call_id="call_123",
                tool_name=None,
                content=[TextInput(content="result")],
            ),
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
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": [{"type": "input_text", "text": "result"}],
        },
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
        {
            "role": "tool",
            "content": [{"type": "text", "text": "result"}],
            "tool_call_id": "call_123",
        },
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
        input=[
            ToolReturn(
                content=[TextInput(content="failed")],
                tool_call_id="call_123",
                outcome="failed",
            )
        ],
        model="m",
    )

    with pytest.raises(UnsupportedInput, match="cannot preserve non-successful"):
        client._request_body(request)


@pytest.mark.parametrize(
    ("prompt_cache", "expected"),
    [
        (
            PromptCache(key="turn-1", mode="implicit", ttl="30m", retention="24h"),
            {
                "prompt_cache_key": "turn-1",
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                "prompt_cache_retention": "24h",
            },
        ),
        (PromptCache(mode="explicit"), {"prompt_cache_options": {"mode": "explicit"}}),
        (PromptCache(key="turn-1"), {"prompt_cache_key": "turn-1"}),
        (PromptCache(retention="in_memory"), {"prompt_cache_retention": "in_memory"}),
    ],
)
def test_openai_clients_project_prompt_cache_fields(prompt_cache, expected):
    """Ensures both OpenAI endpoints receive cache intent without invented settings."""
    request = NormalizedRequest(model="m", stream=True, prompt_cache=prompt_cache)

    for body in (
        OpenAIResponsesClient._request_body(request),
        OpenAIChatCompletionsClient._request_body(request),
    ):
        assert {
            field: body[field] for field in body if field.startswith("prompt_cache")
        } == expected


def test_openai_clients_omit_unset_prompt_cache():
    """Ensures absent cache intent leaves provider defaults untouched."""
    request = NormalizedRequest(model="m", stream=True)

    for body in (
        OpenAIResponsesClient._request_body(request),
        OpenAIChatCompletionsClient._request_body(request),
    ):
        assert not [field for field in body if field.startswith("prompt_cache")]


@pytest.mark.parametrize("ttl", ["5m", "30m", "1h"])
def test_openai_clients_map_prompt_cache_ttl(ttl):
    """Ensures TTL mapping uses 30m without implying stronger guarantees through retention."""
    request = NormalizedRequest(model="m", stream=True, prompt_cache=PromptCache(ttl=ttl))

    for client in (OpenAIResponsesClient, OpenAIChatCompletionsClient):
        body = client._request_body(request)
        assert body["prompt_cache_options"] == {"ttl": "30m"}
        assert "prompt_cache_retention" not in body


@pytest.mark.parametrize(
    ("client", "input_field", "text_type"),
    [
        (OpenAIResponsesClient, "input", "input_text"),
        (OpenAIChatCompletionsClient, "messages", "text"),
    ],
)
def test_openai_clients_project_unmarked_instructions_as_text_parts(
    client, input_field, text_type
):
    request = NormalizedRequest(
        model="m",
        input=[
            SystemPrompt(content="First instruction"),
            SystemPrompt(content="Second instruction"),
        ],
    )

    body = client._request_body(request)

    assert body[input_field] == [
        {"role": "developer", "content": [{"type": text_type, "text": text}]}
        for text in ("First instruction", "Second instruction")
    ]
    assert "prompt_cache_options" not in body


@pytest.mark.parametrize(
    ("client", "input_field", "text_type", "image"),
    [
        (
            OpenAIResponsesClient,
            "input",
            "input_text",
            {
                "type": "input_image",
                "image_url": "https://example.com/image.png",
                "detail": "high",
            },
        ),
        (
            OpenAIChatCompletionsClient,
            "messages",
            "text",
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "detail": "high"},
            },
        ),
    ],
)
def test_openai_clients_preserve_content_cache_breakpoints(client, input_field, text_type, image):
    """Ensures both OpenAI endpoints retain marker positions and valid assistant shapes."""
    marker = CacheBreakpoint()
    request = NormalizedRequest(
        model="m",
        input=[
            SystemPrompt(content="Instructions", cache_breakpoint=marker),
            SystemPrompt(content="Uncached instructions"),
            UserInput(
                content=[
                    TextInput(content="Uncached part"),
                    TextInput(content="Cached prefix", cache_breakpoint=marker),
                    ImageUrlInput(
                        url="https://example.com/image.png", detail="high", cache_breakpoint=marker
                    ),
                ]
            ),
            TextOutput(content="Previous answer", cache_breakpoint=marker),
        ],
    )

    body = client._request_body(request)

    assert body[input_field] == [
        {
            "role": "developer",
            "content": [
                {
                    "type": text_type,
                    "text": "Instructions",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "role": "developer",
            "content": [{"type": text_type, "text": "Uncached instructions"}],
        },
        {
            "role": "user",
            "content": [
                {"type": text_type, "text": "Uncached part"},
                {
                    "type": text_type,
                    "text": "Cached prefix",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {**image, "prompt_cache_breakpoint": {"mode": "explicit"}},
            ],
        },
        {
            "role": "assistant",
            **({"type": "message", "status": "completed"} if input_field == "input" else {}),
            "content": [
                {
                    "type": "output_text" if input_field == "input" else text_type,
                    "text": "Previous answer",
                    **({"annotations": [], "logprobs": []} if input_field == "input" else {}),
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
    ]
    assert "prompt_cache_options" not in body


@pytest.mark.parametrize("client", [OpenAIResponsesClient, OpenAIChatCompletionsClient])
@pytest.mark.parametrize("ttl", ["30m", "5m", "1h"])
def test_openai_clients_map_breakpoint_lifetimes(client, ttl):
    """Ensures shared-TTL mapping retains mode and key without inventing retention."""
    request = NormalizedRequest(
        model="m",
        prompt_cache=PromptCache(mode="explicit", key="turn-1"),
        input=[
            UserInput(
                content=[TextInput(content="Cached", cache_breakpoint=CacheBreakpoint(ttl=ttl))]
            )
        ],
    )

    body = client._request_body(request)
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert body["prompt_cache_key"] == "turn-1"
    assert "prompt_cache_retention" not in body


def test_openai_clients_preserve_tool_text_cache_breakpoint():
    """Ensures tool text parts stay intact so markers delimit the same result prefix."""
    request = NormalizedRequest(
        model="m",
        input=[
            ToolReturn(
                tool_call_id="call_123",
                content=[
                    TextInput(content="First"),
                    TextInput(content="Last", cache_breakpoint=CacheBreakpoint()),
                ],
            )
        ],
    )

    assert OpenAIResponsesClient._request_body(request)["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": [
                {"type": "input_text", "text": "First"},
                {
                    "type": "input_text",
                    "text": "Last",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
            ],
        }
    ]
    assert OpenAIChatCompletionsClient._request_body(request)["messages"] == [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Last", "prompt_cache_breakpoint": {"mode": "explicit"}},
            ],
        }
    ]


def test_responses_client_preserves_tool_image_cache_breakpoint():
    """Ensures marked tool images remain in their original result to preserve cache prefixes."""
    request = NormalizedRequest(
        model="m",
        input=[
            ToolReturn(
                tool_call_id="call_123",
                content=[
                    ImageUrlInput(
                        url="https://example.com/image.png", cache_breakpoint=CacheBreakpoint()
                    ),
                ],
            )
        ],
    )

    assert OpenAIResponsesClient._request_body(request)["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": [
                {
                    "type": "input_image",
                    "image_url": "https://example.com/image.png",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }
    ]


@pytest.mark.parametrize("item_id", [None, "msg_123"])
def test_responses_client_preserves_assistant_marker_and_history_id(item_id):
    """Ensures marked assistant history retains output_text and its optional message ID."""
    request = NormalizedRequest(
        model="m",
        input=[TextOutput(content="Answer", id=item_id, cache_breakpoint=CacheBreakpoint())],
    )

    assert OpenAIResponsesClient._request_body(request)["input"] == [
        {
            **({"id": item_id} if item_id is not None else {}),
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Answer",
                    "annotations": [],
                    "logprobs": [],
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }
    ]


@pytest.mark.parametrize("marker_position", ["image", "same_result_text", "next_result_text"])
def test_chat_client_rejects_cache_breakpoint_after_relocated_tool_image(marker_position):
    """Asserts that tool-image relocation cannot silently change marked cache prefixes."""
    marker = CacheBreakpoint()
    image_result = ToolReturn(
        tool_call_id="call_a",
        content=[
            ImageUrlInput(
                url="https://example.com/image.png",
                cache_breakpoint=marker if marker_position == "image" else None,
            )
        ],
    )
    text = TextInput(content="Result", cache_breakpoint=marker)
    request = NormalizedRequest(model="m", input=[image_result])
    if marker_position == "same_result_text":
        image_result.content.append(text)
    elif marker_position == "next_result_text":
        request.input.append(ToolReturn(tool_call_id="call_b", content=[text]))

    with pytest.raises(
        UnsupportedInput, match="tool-result images must move past the marked part"
    ):
        OpenAIChatCompletionsClient._request_body(request)


@pytest.mark.parametrize(
    ("normalized_request", "error"),
    [
        (NormalizedRequest(model="m", max_tool_calls=1), "max_tool_calls"),
        (
            NormalizedRequest(model="m", reasoning=Reasoning(summary="concise")),
            "concise reasoning summaries",
        ),
        (
            NormalizedRequest(
                input=[
                    Thinking(
                        content="",
                        summary="summary",
                        signature="opaque-state",
                    )
                ],
                model="m",
            ),
            "thinking signatures",
        ),
        (
            NormalizedRequest(
                input=[
                    ToolReturn(
                        tool_name="lookup",
                        content=[TextInput(content="result")],
                        tool_call_id="call_123",
                    )
                ],
                model="m",
            ),
            "tool return names",
        ),
    ],
)
def test_chat_completions_client_rejects_unrepresentable_request_fields(
    normalized_request: NormalizedRequest,
    error: str,
):
    with pytest.raises(UnsupportedInput, match=error):
        OpenAIChatCompletionsClient._request_body(normalized_request)


def test_chat_completions_client_ignores_responses_history_item_ids():
    def request(item_id: str | None) -> NormalizedRequest:
        return NormalizedRequest(
            input=[
                Thinking(content="thought", summary="", id=item_id),
                TextOutput(content="answer", id=item_id),
                ToolCall(
                    id=item_id,
                    tool_call_id="call_123",
                    tool_name="lookup",
                    args="{}",
                ),
                ToolReturn(
                    id=item_id,
                    content=[TextInput(content="result")],
                    tool_call_id="call_123",
                ),
            ],
            model="m",
        )

    assert OpenAIChatCompletionsClient._request_body(
        request("responses_item_id")
    ) == OpenAIChatCompletionsClient._request_body(request(None))


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
                "content": [],
                "encrypted_content": "request-state",
                "summary": [{"type": "summary_text", "text": "Prior thought"}],
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
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="New thought"))
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="response-state"))
        yield PartEndEvent(
            index=0,
            part=Thinking(content="", summary="New thought", signature="response-state"),
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
                "type": "response.reasoning_summary_text.delta",
                "delta": "Think",
                "item_id": "rs_123",
                "output_index": 0,
                "sequence_number": 2,
                "summary_index": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Think"}],
                    "content": [],
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

        events = [
            event
            async for event in client.events(
                NormalizedRequest(input=[UserInput(content=[TextInput(content="Hi")])], model="m")
            )
        ]

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
        PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Think")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="response-state")),
        PartEndEvent(
            index=0,
            part=Thinking(
                content="",
                summary="Think",
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
                    "encrypted_content": "response-state",
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
        PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="response-state")),
        PartEndEvent(
            index=0,
            part=Thinking(
                content="",
                summary="Summary",
                id="rs_123",
                signature="response-state",
            ),
        ),
    ]


def _message_item_events(
    item_id: str, text: str, output_index: int, sequence_number: int
) -> list[OpenAIResponsesStreamEvent]:
    return [
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
                "output_index": output_index,
                "sequence_number": sequence_number,
            }
        ),
        _response_event(
            {
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": text,
                "item_id": item_id,
                "logprobs": [],
                "output_index": output_index,
                "sequence_number": sequence_number + 1,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                "output_index": output_index,
                "sequence_number": sequence_number + 2,
            }
        ),
    ]


async def test_responses_events_omit_completely_empty_reasoning_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A reasoning item with no content, summary, signature, or deltas carries
    # nothing to translate, so its part lifecycle is omitted from the IR and the
    # surviving parts are reindexed to stay dense.
    empty_reasoning_item = {
        "id": "rs_123",
        "type": "reasoning",
        "content": [],
        "summary": [],
    }
    native_events = [
        *_message_item_events("msg_a", "Hi", 0, 0),
        _response_event(
            {
                "type": "response.output_item.added",
                "item": empty_reasoning_item,
                "output_index": 1,
                "sequence_number": 3,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": empty_reasoning_item,
                "output_index": 1,
                "sequence_number": 4,
            }
        ),
        *_message_item_events("msg_b", "Bye", 2, 5),
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
        PartStartEvent(index=0, part=TextOutput(content="", id="msg_a")),
        PartDeltaEvent(index=0, delta=TextDelta(content_delta="Hi")),
        PartEndEvent(index=0, part=TextOutput(content="Hi", id="msg_a")),
        PartStartEvent(index=1, part=TextOutput(content="", id="msg_b")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="Bye")),
        PartEndEvent(index=1, part=TextOutput(content="Bye", id="msg_b")),
    ]


async def test_responses_events_preserve_encrypted_only_reasoning_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An encrypted reasoning item without text deltas still carries opaque state.
    native_events = [
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [],
                    "summary": [],
                    "encrypted_content": "initial-state",
                },
                "output_index": 0,
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [],
                    "summary": [],
                    "encrypted_content": "final-state",
                },
                "output_index": 0,
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

    assert events == [
        PartStartEvent(
            index=0,
            part=Thinking(content="", summary="", id="rs_123", signature="initial-state"),
        ),
        PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="final-state")),
        PartEndEvent(
            index=0,
            part=Thinking(content="", summary="", id="rs_123", signature="final-state"),
        ),
    ]


async def test_responses_events_start_signed_reasoning_part_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The signed start carries meaningful opaque state; it must not wait for
    # later events that may never arrive if the stream stalls or fails.
    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [],
                    "summary": [],
                    "encrypted_content": "initial-state",
                },
                "output_index": 0,
                "sequence_number": 0,
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        stream = client.events(NormalizedRequest(model="m"))
        first = await anext(stream)
        await stream.aclose()

    assert first == PartStartEvent(
        index=0,
        part=Thinking(content="", summary="", id="rs_123", signature="initial-state"),
    )


@pytest.mark.parametrize(
    ("capabilities", "added_content", "added_summary", "expected_part"),
    [
        (
            VLLM_CAPABILITIES,
            [{"type": "reasoning_text", "text": "Initial"}],
            [],
            Thinking(content="Initial", summary="", id="rs_123"),
        ),
        (
            OPENAI_CAPABILITIES,
            [],
            [{"type": "summary_text", "text": "Summary"}],
            Thinking(content="", summary="Summary", id="rs_123"),
        ),
    ],
    ids=["content", "summary"],
)
async def test_responses_events_preserve_reasoning_added_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: ProviderCapabilities,
    added_content: list[dict[str, object]],
    added_summary: list[dict[str, object]],
    expected_part: Thinking,
) -> None:
    # The `added` snapshot may already carry reasoning text; the part starts
    # immediately with that snapshot instead of waiting for later events.
    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": added_content,
                    "summary": added_summary,
                },
                "output_index": 0,
                "sequence_number": 0,
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=capabilities)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        stream = client.events(NormalizedRequest(model="m"))
        first = await anext(stream)
        await stream.aclose()

    assert first == PartStartEvent(index=0, part=expected_part)


async def test_responses_events_reject_reasoning_outside_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "raw"}],
                    "summary": [],
                },
                "output_index": 0,
                "sequence_number": 0,
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        with pytest.raises(ValueError, match="cannot contain raw text"):
            await anext(client.events(NormalizedRequest(model="m")))


def test_responses_client_rejects_unsigned_summary_history() -> None:
    request = NormalizedRequest(
        input=[Thinking(content="", summary="Summary")],
        model="m",
    )

    with pytest.raises(ValueError, match="summary requires a signature"):
        OpenAIResponsesClient._request_body(request)


async def test_responses_events_preserve_reasoning_added_snapshot_and_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The start snapshot is the initial state, not a delta: the completed part
    # carries the snapshot plus subsequent deltas exactly once.
    native_events = [
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Initial"}],
                    "summary": [],
                },
                "output_index": 0,
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_text.delta",
                "content_index": 0,
                "delta": " more",
                "item_id": "rs_123",
                "output_index": 0,
                "sequence_number": 1,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Initial more"}],
                    "summary": [],
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
        client = OpenAIResponsesClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [event async for event in client.events(NormalizedRequest(model="m"))]

    assert events == [
        PartStartEvent(index=0, part=Thinking(content="Initial", summary="", id="rs_123")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta=" more")),
        PartEndEvent(index=0, part=Thinking(content="Initial more", summary="", id="rs_123")),
    ]


async def test_responses_events_defer_completely_empty_reasoning_part_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only a completely empty reasoning item is deferred; with no further events
    # it produces nothing at all.
    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield _response_event(
            {
                "type": "response.output_item.added",
                "item": {"id": "rs_123", "type": "reasoning", "content": [], "summary": []},
                "output_index": 0,
                "sequence_number": 0,
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        stream = client.events(NormalizedRequest(model="m"))
        with pytest.raises(StopAsyncIteration):
            await anext(stream)


async def test_responses_events_complete_reasoning_part_without_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A reasoning item whose text only appears in the completed item still gets a
    # balanced part lifecycle: the deferred start is flushed at completion.
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
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Done-only thought"}],
                    "summary": [],
                },
                "output_index": 0,
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
        client = OpenAIResponsesClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [event async for event in client.events(NormalizedRequest(model="m"))]

    assert events == [
        PartStartEvent(index=0, part=Thinking(content="", summary="", id="rs_123")),
        PartEndEvent(
            index=0,
            part=Thinking(content="Done-only thought", summary="", id="rs_123"),
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


async def test_responses_adapter_projects_encrypted_only_reasoning_lifecycle() -> None:
    adapter = OpenAIResponsesAdapter(OPENAI_CAPABILITIES)
    request = adapter.normalize_request(
        OpenAIResponsesCreateRequest(input="Question", model="m", stream=True)
    )

    async def normalized_events() -> AsyncIterator[ModelResponseStreamEvent]:
        yield ResponseStartEvent(id="resp_123", created_at=1, model="m")
        yield PartStartEvent(
            index=0,
            part=Thinking(content="", summary="", signature="initial-state"),
        )
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="final-state"))
        yield PartEndEvent(
            index=0,
            part=Thinking(content="", summary="", signature="final-state"),
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
        "response.output_item.done",
        "response.completed",
    ]
    added = next(event for event in events if event.type == "response.output_item.added")
    done = next(event for event in events if event.type == "response.output_item.done")
    assert added.item.encrypted_content == "initial-state"
    assert done.item.content == []
    assert done.item.summary == []
    assert done.item.encrypted_content == "final-state"


@pytest.mark.parametrize("capabilities", [VLLM_CAPABILITIES, OPENAI_CAPABILITIES])
async def test_responses_adapter_omits_empty_reasoning_part_lifecycle(
    capabilities: ProviderCapabilities,
) -> None:
    # A reasoning part with no content, summary, signature, or deltas carries no
    # projectable state, so only its item lifecycle is emitted. Native Responses
    # reasoning items look the same unless summary or encrypted content is opted in.
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

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]
    done = next(event for event in events if event.type == "response.output_item.done")
    assert done.item.type == "reasoning"
    assert done.item.content == []
    assert done.item.summary == []
    assert done.item.encrypted_content is None


async def test_responses_events_translate_non_empty_part_added_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Part-added snapshots can carry initial text; translate them as deltas so
    # the streamed text stays consistent with the completed snapshots. Raw
    # reasoning and summaries use separate reasoning items, as required by the
    # Responses lifecycle. Empty part-added echoes remain ignored (see
    # test_responses_events_ignore_...).
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
                "type": "response.content_part.added",
                "content_index": 0,
                "item_id": "rs_123",
                "output_index": 0,
                "part": {"type": "reasoning_text", "text": "Initial"},
                "sequence_number": 1,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Initial"}],
                    "summary": [],
                },
                "output_index": 0,
                "sequence_number": 2,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {"id": "rs_456", "type": "reasoning", "content": [], "summary": []},
                "output_index": 1,
                "sequence_number": 3,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_summary_part.added",
                "item_id": "rs_456",
                "output_index": 1,
                "part": {"type": "summary_text", "text": "Sum"},
                "sequence_number": 4,
                "summary_index": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_456",
                    "type": "reasoning",
                    "content": [],
                    "summary": [{"type": "summary_text", "text": "Sum"}],
                    "encrypted_content": "response-state",
                },
                "output_index": 1,
                "sequence_number": 5,
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
                "output_index": 2,
                "sequence_number": 6,
            }
        ),
        _response_event(
            {
                "type": "response.content_part.added",
                "content_index": 0,
                "item_id": "msg_123",
                "output_index": 2,
                "part": {
                    "type": "output_text",
                    "text": "Answer",
                    "annotations": [],
                    "logprobs": [],
                },
                "sequence_number": 7,
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
                "output_index": 2,
                "sequence_number": 8,
            }
        ),
    ]

    async def collect(
        selected_events: list[OpenAIResponsesStreamEvent],
        capabilities: ProviderCapabilities,
    ) -> list[ModelResponseStreamEvent]:
        async def stub_native_events(
            native_request: OpenAIResponsesCreateRequest,
        ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
            for event in selected_events:
                yield event

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_unexpected_http)
        ) as http_client:
            client = OpenAIResponsesClient(http_client, capabilities=capabilities)
            monkeypatch.setattr(client, "native_events", stub_native_events)
            return [event async for event in client.events(NormalizedRequest(model="m"))]

    events = [
        *(await collect(native_events[:3], VLLM_CAPABILITIES)),
        *(await collect(native_events[3:], OPENAI_CAPABILITIES)),
    ]

    assert events == [
        PartStartEvent(index=0, part=Thinking(content="", summary="", id="rs_123")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="Initial")),
        PartEndEvent(
            index=0,
            part=Thinking(content="Initial", summary="", id="rs_123"),
        ),
        PartStartEvent(index=0, part=Thinking(content="", summary="", id="rs_456")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Sum")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="response-state")),
        PartEndEvent(
            index=0,
            part=Thinking(
                content="",
                summary="Sum",
                id="rs_456",
                signature="response-state",
            ),
        ),
        PartStartEvent(index=1, part=TextOutput(content="", id="msg_123")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="Answer")),
        PartEndEvent(index=1, part=TextOutput(content="Answer", id="msg_123")),
    ]


async def test_responses_events_ignore_empty_deltas_on_deferred_reasoning_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Empty deltas are lifecycle echoes: they must not materialize a deferred
    # reasoning part, so the part is still omitted at completion.
    empty_item = {"id": "rs_123", "type": "reasoning", "content": [], "summary": []}
    native_events = [
        _response_event(
            {
                "type": "response.output_item.added",
                "item": empty_item,
                "output_index": 0,
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_text.delta",
                "content_index": 0,
                "delta": "",
                "item_id": "rs_123",
                "output_index": 0,
                "sequence_number": 1,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "",
                "item_id": "rs_123",
                "output_index": 0,
                "sequence_number": 2,
                "summary_index": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": empty_item,
                "output_index": 0,
                "sequence_number": 3,
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

    assert events == []


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
                            content="",
                            summary="Prior thought",
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
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "first", "arguments": '{"a":1}'},
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
        PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta='{"a":1}')),
        PartEndEvent(
            index=0,
            part=ToolCall(tool_call_id="call_a", tool_name="first", args='{"a":1}'),
        ),
        PartStartEvent(
            index=1,
            part=ToolCall(tool_call_id="call_b", tool_name="second", args=""),
        ),
        PartDeltaEvent(index=1, delta=ToolCallDelta(args_delta='{"b":2}')),
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


async def test_chat_completions_events_reject_noncontiguous_tool_call_deltas(
    monkeypatch: pytest.MonkeyPatch,
):
    native_chunks = [
        _chat_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": call_index,
                                "id": f"call_{call_index}",
                                "type": "function",
                                "function": {"name": "read", "arguments": ""},
                            }
                        ]
                    },
                }
            ]
        )
        for call_index in (0, 1, 0)
    ]

    async def stub_native_events(
        native_request: OpenAIChatCompletionsCreateRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        for native_chunk in native_chunks:
            yield native_chunk

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIChatCompletionsClient(http_client, capabilities=OPENAI_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        with pytest.raises(ValueError, match="must be contiguous by index"):
            _ = [event async for event in client.events(NormalizedRequest(model="m"))]


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
            async for event in client.events(
                NormalizedRequest(
                    input=[UserInput(content=[TextInput(content="Hi")])], model="m", stream=True
                )
            )
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


async def test_chat_completions_events_omit_empty_reasoning_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Empty `reasoning_content` chunks carry nothing: no thinking part starts.
    native_chunks = [
        _chat_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": None, "reasoning_content": ""},
                }
            ]
        ),
        _chat_chunk(choices=[{"index": 0, "delta": {"reasoning_content": ""}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
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

    assert events == [
        ResponseStartEvent(
            id="chatcmpl_123", created_at=2, model="reasoning-model", service_tier="default"
        ),
        ResponseEndEvent(
            id="chatcmpl_123",
            created_at=2,
            model="reasoning-model",
            service_tier="default",
            termination_reason="stop",
        ),
    ]


async def test_chat_completions_events_interleave_reasoning_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Models may emit text in between reasoning. Each kind switch ends the
    # active part and starts a new one; empty reasoning chunks are no-ops and
    # never split a text part.
    native_chunks = [
        _chat_chunk(choices=[{"index": 0, "delta": {"role": "assistant"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"reasoning_content": "R1"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"content": "T1"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"reasoning_content": ""}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"content": "T2"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"reasoning_content": "R2"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
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

    assert events == [
        ResponseStartEvent(
            id="chatcmpl_123", created_at=2, model="reasoning-model", service_tier="default"
        ),
        PartStartEvent(index=0, part=Thinking(content="", summary="")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="R1")),
        PartEndEvent(index=0, part=Thinking(content="R1", summary="")),
        PartStartEvent(index=1, part=TextOutput(content="")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="T1")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="T2")),
        PartEndEvent(index=1, part=TextOutput(content="T1T2")),
        PartStartEvent(index=2, part=Thinking(content="", summary="")),
        PartDeltaEvent(index=2, delta=ThinkingDelta(content_delta="R2")),
        PartEndEvent(index=2, part=Thinking(content="R2", summary="")),
        ResponseEndEvent(
            id="chatcmpl_123",
            created_at=2,
            model="reasoning-model",
            service_tier="default",
            termination_reason="stop",
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
    await client.aclose()


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


async def test_responses_native_response_uses_openai_sdk_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIResponsesClient(
            http_client,
            capabilities=OPENAI_CAPABILITIES,
            api_key="test-api-key",
        )
        request = OpenAIResponsesCreateRequest(input="Hi", model="m", stream=False)
        response = await client.native_response(request)

    assert type(response) is OpenAIResponsesResponse
    assert response.id == "resp_123"


async def test_chat_completions_native_response_uses_openai_sdk_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_123",
                "choices": [],
                "created": 1,
                "model": "m",
                "object": "chat.completion",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIChatCompletionsClient(
            http_client,
            capabilities=OPENAI_CAPABILITIES,
            api_key="test-api-key",
        )
        request = OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": "Hi"}], model="m", stream=False
        )
        response = await client.native_response(request)

    assert type(response) is OpenAIChatCompletionsResponse
    assert response.id == "chatcmpl_123"


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
    async with await client.chat_completions(body=body, api_key="test-api-key") as first_events:
        assert (await anext(first_events)).id == "first"
    assert first_response_stream.closed
    assert not http_client.is_closed

    second_events = await client.chat_completions(body=body, api_key="test-api-key")
    assert [event.id async for event in second_events] == ["second"]
    assert not http_client.is_closed

    await client.aclose()
    assert http_client.is_closed


async def test_native_events_normalize_authentication_error():
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

    error = exc_info.value
    assert error.response.status_code == 401
    assert error.body["code"] == "invalid_api_key"
    assert type(error.__cause__) is OpenAIAuthenticationError
    assert request_count == 1


@pytest.mark.parametrize("client_type", [OpenAIResponsesClient, OpenAIChatCompletionsClient])
@pytest.mark.parametrize(
    "error",
    [
        OpenAIError("The `api_key` and `workload_identity` arguments are mutually exclusive"),
        TypeError("Missing credentials."),
        APIError(
            "Missing credentials.",
            request=httpx.Request("POST", "https://example.test"),
            body=None,
        ),
    ],
)
async def test_missing_credentials_recognition_requires_sdk_type_and_message(client_type, error):
    async with httpx.AsyncClient() as http_client:
        client = client_type(http_client, capabilities=OPENAI_CAPABILITIES)
        with pytest.raises(type(error)) as exc_info:
            client._handle_error(error)
    assert exc_info.value is error


@pytest.mark.parametrize("client_type", [OpenAIResponsesClient, OpenAIChatCompletionsClient])
async def test_missing_credentials_recognizes_sdk_error_subclasses(client_type):
    class CredentialsError(OpenAIError):
        pass

    error = CredentialsError("Missing credentials.")
    async with httpx.AsyncClient() as http_client:
        client = client_type(http_client, capabilities=OPENAI_CAPABILITIES)
        with pytest.raises(AuthenticationError) as exc_info:
            client._handle_error(error)
    assert exc_info.value.__cause__ is error
    assert exc_info.value.response is None


@pytest.mark.parametrize(
    ("status", "code"),
    [(400, "invalid_request_error"), (400, None), (401, None), (404, None)],
)
@pytest.mark.parametrize("client_type", [OpenAIResponsesClient, OpenAIChatCompletionsClient])
async def test_error_recognition_does_not_use_status_alone(
    status: int, code: str | None, client_type
):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    error = APIStatusError(
        "Your input exceeds the context window of this model.",
        response=response,
        body={"code": code},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = client_type(http_client, capabilities=OPENAI_CAPABILITIES)
        with pytest.raises(APIStatusError) as exc_info:
            client._handle_error(error)
    assert exc_info.value is error


@pytest.mark.parametrize("client_type", [OpenAIResponsesClient, OpenAIChatCompletionsClient])
@pytest.mark.parametrize("status", [400, 401, 404, 422])
@pytest.mark.parametrize(
    ("sdk_error_type", "code", "normalized_type"),
    [
        (APIStatusError, "context_length_exceeded", ContextLengthExceededError),
        (APIStatusError, "model_not_found", ModelNotFoundError),
        (OpenAIAuthenticationError, "model_not_found", ModelNotFoundError),
        (OpenAINotFoundError, "model_not_found", ModelNotFoundError),
        (OpenAIAuthenticationError, None, AuthenticationError),
        (OpenAINotFoundError, None, ResourceNotFoundError),
    ],
)
async def test_error_recognition_uses_code_and_sdk_class(
    client_type, status, sdk_error_type, code, normalized_type
):
    class ProviderError(sdk_error_type):
        pass

    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    error = ProviderError("Provider failure", response=response, body={"code": code})
    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = client_type(http_client, capabilities=OPENAI_CAPABILITIES)
        with pytest.raises(normalized_type) as exc_info:
            client._handle_error(error)
    assert exc_info.value.__cause__ is error
    assert exc_info.value.response is response
    assert exc_info.value.response.status_code == status
    assert exc_info.value.body is error.body


@pytest.mark.parametrize("status_error", [False, True])
async def test_context_stream_error_distinguishes_api_error_subclasses(status_error):
    class StreamAPIError(APIError):
        pass

    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(200, request=request)
    body = {"code": "context_length_exceeded"}
    error = (
        APIStatusError(
            "Context exceeded", response=httpx.Response(422, request=request), body=body
        )
        if status_error
        else StreamAPIError("Context exceeded", request=request, body=body)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        with pytest.raises(ContextLengthExceededError) as exc_info:
            client._handle_error(error, response=response)
    assert exc_info.value.__cause__ is error
    assert exc_info.value.response is (error.response if status_error else response)
    assert exc_info.value.response.status_code == (422 if status_error else 200)
    assert exc_info.value.body is body


async def test_context_stream_error_requires_response():
    request = httpx.Request("POST", "https://example.test")
    error = APIError("Context exceeded", request=request, body={"code": "context_length_exceeded"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        with pytest.raises(APIError) as exc_info:
            client._handle_error(error)
    assert exc_info.value is error


@pytest.mark.parametrize("status", [200, 201])
async def test_context_stream_error_recognition_ignores_status(status):
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status, request=request)
    error = APIError("Context exceeded", request=request, body={"code": "context_length_exceeded"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client, capabilities=OPENAI_CAPABILITIES)
        with pytest.raises(ContextLengthExceededError) as exc_info:
            client._handle_error(error, response=response)
    assert exc_info.value.__cause__ is error
    assert exc_info.value.response is response
    assert exc_info.value.response.status_code == status


@pytest.mark.parametrize("downstream", ["responses", "chat/completions", "messages"])
@pytest.mark.parametrize(("status", "code"), [(200, "unknown_error"), (201, "unknown_error")])
async def test_unknown_responses_stream_error_passes_through(downstream, status, code):
    error_body = {"code": code, "message": "Provider-specific failure"}

    def handler(request):
        response = _sse_response(dumps({"type": "error", "error": error_body}))
        response.status_code = status
        return response

    body = {"model": "m", "stream": True}
    if downstream == "responses":
        body["input"] = "Hi"
    else:
        body["messages"] = [{"role": "user", "content": "Hi"}]
        if downstream == "messages":
            body["max_tokens"] = 16
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        with pytest.raises(APIError) as exc_info:
            stream = await coordinator(body=body, endpoint="responses", api_key="test-key")
            async with stream:
                [event async for event in stream]
    assert type(exc_info.value) is APIError
    assert exc_info.value.body == error_body
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("endpoint", ["responses", "chat/completions"])
async def test_native_events_normalize_known_sdk_errors(endpoint: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": "model_not_found",
                    "message": "The requested model 'm' does not exist.",
                    "param": "model",
                    "type": "invalid_request_error",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        if endpoint == "responses":
            client = OpenAIResponsesClient(
                http_client,
                capabilities=OPENAI_CAPABILITIES,
                api_key="test-api-key",
            )
            request = OpenAIResponsesCreateRequest(input="Hi", model="m", stream=True)
        else:
            client = OpenAIChatCompletionsClient(
                http_client,
                capabilities=OPENAI_CAPABILITIES,
                api_key="test-api-key",
            )
            request = OpenAIChatCompletionsCreateRequest(
                messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
            )
        with pytest.raises(ModelNotFoundError) as exc_info:
            [event async for event in client.native_events(request)]

    error = exc_info.value
    assert error.response.status_code == 404
    assert isinstance(error.__cause__, APIStatusError)
