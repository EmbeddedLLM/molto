from collections.abc import AsyncIterator

import httpx
import pytest
from openai import NotFoundError

from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.exceptions import ModelNotFoundError, UnsupportedInput
from molto.types.normalized import (
    CacheBreakpoint,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    PromptCache,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UserInput,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesResponse,
)
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient

_CAPABILITIES = PROVIDER_CAPABILITIES["openai"]


@pytest.mark.parametrize(
    "content",
    [
        "a",
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": [{"type": "input_text", "text": "a"}]}],
    ],
    ids=["string", "easy-message", "typed-message"],
)
def test_responses_text_input(content):
    request = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(input=content, model="m")
    )
    assert request.input == [UserInput(content=[TextInput(content="a")])]


@pytest.mark.parametrize(
    "content",
    [
        [{"role": "developer", "content": "a"}],
        [{"role": "system", "content": [{"type": "input_text", "text": "a"}]}],
    ],
    ids=["easy-message", "typed-message"],
)
def test_responses_developer_role_normalization(content):
    request = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(input=content, model="m")
    )
    assert request.input == [SystemPrompt(content="a")]


def test_adapters_preserve_multimodal_input_boundaries_and_order():
    responses = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "before"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                        {"type": "input_text", "text": "after"},
                    ],
                }
            ],
            model="m",
        )
    )
    chat = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(
        OpenAIChatCompletionsCreateRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA=="},
                        },
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
            model="m",
        )
    )

    assert responses.input == chat.input
    assert OpenAIResponsesClient._process_input(chat) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "before"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                {"type": "input_text", "text": "after"},
            ],
        }
    ]
    assert OpenAIChatCompletionsClient._messages(responses) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
                {"type": "text", "text": "after"},
            ],
        }
    ]


def test_responses_adapter_rejects_image_file_id():
    with pytest.raises(UnsupportedInput, match="requires image_url"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file_123"}],
            }
        )


def test_responses_adapter_rejects_input_file():
    with pytest.raises(
        UnsupportedInput, match="Unsupported Responses user content type: input_file"
    ):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": "file_123"}],
            }
        )


def test_responses_adapter_rejects_untranslated_request_fields():
    request = OpenAIResponsesCreateRequest(input="Hello", model="m", metadata={"key": "value"})

    with pytest.raises(UnsupportedInput, match=r"Responses request fields.*metadata"):
        OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(request)


def test_responses_adapter_ignores_encrypted_reasoning_include():
    request_without_include = OpenAIResponsesCreateRequest(input="Hello", model="m")
    request_with_include = OpenAIResponsesCreateRequest(
        input="Hello", model="m", include=["reasoning.encrypted_content"]
    )

    normalized_without_include = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        request_without_include
    )
    normalized_with_include = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        request_with_include
    )

    assert normalized_with_include == normalized_without_include


def test_responses_adapter_ignores_disabled_store():
    request_without_store = OpenAIResponsesCreateRequest(input="Hello", model="m")
    request_with_store_disabled = OpenAIResponsesCreateRequest(
        input="Hello", model="m", store=False
    )

    normalized_without_store = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        request_without_store
    )
    normalized_with_store_disabled = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        request_with_store_disabled
    )

    assert normalized_with_store_disabled == normalized_without_store


def test_responses_adapter_rejects_enabled_store():
    request = OpenAIResponsesCreateRequest(input="Hello", model="m", store=True)

    with pytest.raises(UnsupportedInput, match=r"store=true"):
        OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(request)


def test_responses_adapter_rejects_untranslated_include_values():
    request = OpenAIResponsesCreateRequest(
        input="Hello",
        model="m",
        include=["message.output_text.logprobs"],
    )

    with pytest.raises(UnsupportedInput, match=r"include values.*message.output_text.logprobs"):
        OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(request)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_text", "text": "hello", "prompt_cache_breakpoint": {"mode": "explicit"}},
        {
            "type": "input_image",
            "image_url": "https://example.com/a.png",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
    ],
    ids=["text", "image"],
)
def test_responses_adapter_normalizes_prompt_cache_breakpoint(part):
    """Ensures text and image markers retain the OpenAI defaults needed for translation."""
    request = OpenAIResponsesCreateRequest(model="m", input=[{"role": "user", "content": [part]}])
    normalized = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(request)
    assert list(normalized.iter_cache_breakpoints()) == [CacheBreakpoint()]
    assert normalized.prompt_cache == PromptCache(mode="implicit", ttl="30m")


@pytest.mark.parametrize(
    "message",
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello", "prompt_cache_breakpoint": {"mode": "explicit"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {"type": "text", "text": "result", "prompt_cache_breakpoint": {"mode": "explicit"}}
            ],
        },
    ],
    ids=["user", "tool"],
)
def test_chat_adapter_normalizes_prompt_cache_breakpoint(message):
    """Ensures normalization preserves user and tool-result cache boundaries."""
    normalized = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(
        OpenAIChatCompletionsCreateRequest(model="m", messages=[message], stream=True)
    )
    assert list(normalized.iter_cache_breakpoints()) == [CacheBreakpoint()]
    assert normalized.prompt_cache == PromptCache(mode="implicit", ttl="30m")


def test_openai_adapters_normalize_prompt_cache_fields():
    """Ensures both OpenAI protocols normalize cache intent without losing settings."""
    cache = {
        "prompt_cache_key": "turn-1",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_retention": "24h",
    }
    responses = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(model="m", input="hi", stream=True, **cache)
    )
    chat = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(
        OpenAIChatCompletionsCreateRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            **cache,
        )
    )
    expected = PromptCache(key="turn-1", mode="explicit", ttl="30m", retention="24h")

    assert responses.prompt_cache == expected
    assert chat.prompt_cache == expected


def test_openai_adapters_omit_unconfigured_prompt_cache():
    """Ensures empty or omitted options do not introduce cache intent."""
    empty_options = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(model="m", input="hi", prompt_cache_options={})
    )
    unset = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(
        OpenAIChatCompletionsCreateRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    )

    assert empty_options.prompt_cache is None
    assert unset.prompt_cache is None


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}],
        [{"type": "file", "file": {"file_id": "file_123"}}],
    ],
    ids=["audio", "file"],
)
def test_chat_adapter_rejects_unsupported_user_content(content):
    adapter = OpenAIChatCompletionsAdapter(_CAPABILITIES)
    with pytest.raises(UnsupportedInput, match="Unsupported Chat user content type"):
        adapter._normalize_message({"role": "user", "content": content})


def test_chat_input_validation_rejects_image_tool_results():
    request = OpenAIChatCompletionsCreateRequest.model_construct(
        messages=[
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/result.png"},
                    }
                ],
            }
        ],
        model="m",
    )

    with pytest.raises(UnsupportedInput, match="cannot represent image tool result"):
        OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(request)


def test_chat_adapter_rejects_untranslated_request_fields():
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="m",
        stop=["END"],
    )

    with pytest.raises(UnsupportedInput, match=r"Chat Completions request fields.*stop"):
        OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(request)


def test_chat_adapter_normalizes_deprecated_max_tokens():
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="m",
        max_tokens=10,
    )

    normalized = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(request)

    assert normalized.max_output_tokens == 10


def test_chat_adapter_rejects_conflicting_max_token_fields():
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="m",
        max_completion_tokens=10,
        max_tokens=20,
    )

    with pytest.raises(UnsupportedInput, match="cannot set both"):
        OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(request)


def test_chat_adapter_rejects_disabling_usage():
    request = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="m",
        stream=True,
        stream_options={"include_usage": False},
    )

    with pytest.raises(UnsupportedInput, match="always includes usage"):
        OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(request)


def test_chat_adapter_accepts_explicit_usage():
    normalized = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(
        OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="m",
            stream=True,
            stream_options={"include_usage": True},
        )
    )

    assert normalized.stream is True


@pytest.mark.parametrize(
    ("role", "part_type", "expected_type"),
    [
        ("developer", "input_text", "system-prompt"),
        ("assistant", "output_text", "text-output"),
    ],
)
def test_responses_adapter_validates_content_parts_by_message_role(role, part_type, expected_type):
    [item] = OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
        {"type": "message", "role": role, "content": [{"type": part_type, "text": "x"}]}
    )

    assert item.content_type == expected_type


def test_responses_adapter_rejects_wrong_content_part_for_message_role():
    with pytest.raises(UnsupportedInput, match="assistant message content cannot be translated"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}],
            }
        )


async def _stream(*events: ModelResponseStreamEvent) -> AsyncIterator[ModelResponseStreamEvent]:
    for event in events:
        yield event


def _responses_adapter(capabilities: ProviderCapabilities):
    adapter = OpenAIResponsesAdapter(capabilities)
    request = adapter.normalize_request(
        OpenAIResponsesCreateRequest(input="Question", model="m", stream=True)
    )
    return adapter, request


def _chat_adapter(capabilities: ProviderCapabilities):
    adapter = OpenAIChatCompletionsAdapter(capabilities)
    request = adapter.normalize_request(
        OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
        )
    )
    return adapter, request


_RESPONSE_START = ResponseStartEvent(id="resp_123", created_at=1, model="m")
_RESPONSE_END = ResponseEndEvent(id="resp_123", created_at=1, model="m", termination_reason="stop")
_CHAT_START = ResponseStartEvent(id="chatcmpl_123", created_at=1, model="m")


async def test_responses_response_projection_returns_terminal_response() -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    response = await adapter.project_response(
        adapter.project_events(
            _stream(
                _RESPONSE_START,
                PartStartEvent(index=0, part=TextOutput(content="")),
                PartDeltaEvent(index=0, delta=TextDelta(content_delta="Answer")),
                PartEndEvent(index=0, part=TextOutput(content="Answer")),
                ResponseEndEvent(
                    id="resp_123",
                    created_at=1,
                    model="m",
                    termination_reason="stop",
                    usage=Usage(
                        input_tokens=3,
                        input_tokens_details={"cached_tokens": 1},
                        output_tokens=2,
                        output_tokens_details={"reasoning_tokens": 0},
                        total_tokens=5,
                    ),
                ),
            ),
            request,
        ),
        request,
    )

    assert type(response) is OpenAIResponsesResponse
    assert response.id == "resp_123"
    assert response.status == "completed"
    assert response.output[0].content[0].text == "Answer"
    assert response.usage is not None
    assert response.usage.input_tokens_details.cached_tokens == 1


async def test_responses_response_projection_drains_projected_stream() -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)
    exhausted = False

    async def events() -> AsyncIterator[ModelResponseStreamEvent]:
        nonlocal exhausted
        yield _RESPONSE_START
        yield _RESPONSE_END
        exhausted = True

    await adapter.project_response(adapter.project_events(events(), request), request)

    assert exhausted


async def test_responses_response_projection_returns_incomplete_response() -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    response = await adapter.project_response(
        adapter.project_events(
            _stream(
                _RESPONSE_START,
                ResponseEndEvent(
                    id="resp_123",
                    created_at=1,
                    model="m",
                    termination_reason="max_output_tokens",
                ),
            ),
            request,
        ),
        request,
    )

    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"


async def test_responses_response_projection_rejects_missing_terminal_event() -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match="without a terminal response"):
        await adapter.project_response(
            adapter.project_events(_stream(_RESPONSE_START), request),
            request,
        )


async def test_chat_response_projection_accumulates_projected_chunks() -> None:
    adapter, request = _chat_adapter(PROVIDER_CAPABILITIES["vllm"])

    response = await adapter.project_response(
        adapter.project_events(
            _stream(
                _CHAT_START,
                PartStartEvent(index=0, part=Thinking(content="", summary="")),
                PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="Think")),
                PartEndEvent(index=0, part=Thinking(content="Think", summary="")),
                PartStartEvent(index=1, part=TextOutput(content="")),
                PartDeltaEvent(index=1, delta=TextDelta(content_delta="Answer")),
                PartEndEvent(index=1, part=TextOutput(content="Answer")),
                PartStartEvent(
                    index=2,
                    part=ToolCall(
                        tool_name="weather",
                        args="",
                        tool_call_id="call_weather",
                    ),
                ),
                PartDeltaEvent(index=2, delta=ToolCallDelta(args_delta='{"city":')),
                PartDeltaEvent(index=2, delta=ToolCallDelta(args_delta='"Paris"}')),
                PartEndEvent(
                    index=2,
                    part=ToolCall(
                        tool_name="weather",
                        args='{"city":"Paris"}',
                        tool_call_id="call_weather",
                    ),
                ),
                ResponseEndEvent(
                    id="chatcmpl_123",
                    created_at=1,
                    model="m",
                    termination_reason="tool_call",
                    usage=Usage(
                        input_tokens=8,
                        input_tokens_details={
                            "cached_tokens": 3,
                            "cache_write_tokens": 2,
                        },
                        output_tokens=5,
                        output_tokens_details={"reasoning_tokens": 1},
                        total_tokens=13,
                    ),
                ),
            ),
            request,
        ),
        request,
    )

    assert type(response) is OpenAIChatCompletionsResponse
    assert response.id == "chatcmpl_123"
    assert response.object == "chat.completion"
    assert response.service_tier == "default"
    [choice] = response.choices
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "Answer"
    assert choice.message.reasoning_content == "Think"
    assert choice.message.role == "assistant"
    assert choice.message.tool_calls is not None
    [tool_call] = choice.message.tool_calls
    assert tool_call.model_dump() == {
        "id": "call_weather",
        "function": {
            "arguments": '{"city":"Paris"}',
            "name": "weather",
            "parsed_arguments": None,
        },
        "type": "function",
        "index": 0,
    }
    assert response.usage is not None
    assert response.usage.prompt_tokens == 8
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.cached_tokens == 3
    assert response.usage.prompt_tokens_details.cache_write_tokens == 2
    assert response.usage.completion_tokens_details is not None
    assert response.usage.completion_tokens_details.reasoning_tokens == 1


async def test_chat_response_projection_handles_completion_without_content() -> None:
    adapter, request = _chat_adapter(_CAPABILITIES)

    response = await adapter.project_response(
        adapter.project_events(
            _stream(
                _CHAT_START,
                ResponseEndEvent(
                    id="chatcmpl_123",
                    created_at=1,
                    model="m",
                    termination_reason="content_filter",
                ),
            ),
            request,
        ),
        request,
    )

    [choice] = response.choices
    assert choice.finish_reason == "content_filter"
    assert choice.message.role == "assistant"
    assert choice.message.content is None


async def test_chat_response_projection_rejects_missing_finish_reason() -> None:
    adapter, request = _chat_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match="without a finish reason"):
        await adapter.project_response(
            adapter.project_events(
                _stream(
                    _CHAT_START,
                    PartStartEvent(index=0, part=TextOutput(content="unfinished")),
                ),
                request,
            ),
            request,
        )


@pytest.mark.parametrize(
    ("capabilities", "part", "added_type", "done_types", "field"),
    [
        (
            PROVIDER_CAPABILITIES["vllm"],
            Thinking(content="Raw thought", summary=""),
            "response.content_part.added",
            ("response.reasoning_text.done", "response.content_part.done"),
            "content",
        ),
        (
            PROVIDER_CAPABILITIES["openai"],
            Thinking(content="", summary="Summary", signature="opaque-state"),
            "response.reasoning_summary_part.added",
            ("response.reasoning_summary_text.done", "response.reasoning_summary_part.done"),
            "summary",
        ),
    ],
    ids=["raw", "summary"],
)
async def test_responses_projection_completes_thinking_text_arriving_without_deltas(
    capabilities: ProviderCapabilities,
    part: Thinking,
    added_type: str,
    done_types: tuple[str, str],
    field: str,
) -> None:
    adapter, request = _responses_adapter(capabilities)

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _RESPONSE_START,
                PartStartEvent(index=0, part=Thinking(content="", summary="")),
                PartEndEvent(index=0, part=part),
                _RESPONSE_END,
            ),
            request,
        )
    ]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        added_type,
        *done_types,
        "response.output_item.done",
        "response.completed",
    ]
    done = next(event for event in events if event.type == "response.output_item.done")
    text = part.content or part.summary
    assert [part.text for part in getattr(done.item, field)] == [text]
    assert done.item.encrypted_content == part.signature


@pytest.mark.parametrize(
    (
        "capabilities",
        "start",
        "deltas",
        "completed",
        "part_added_type",
        "text_delta_type",
        "completed_field",
    ),
    [
        (
            PROVIDER_CAPABILITIES["openai"],
            Thinking(content="", summary="", signature=None),
            (
                ThinkingDelta(summary_delta="I am"),
                ThinkingDelta(summary_delta=" reasoning"),
            ),
            Thinking(
                content="",
                summary="I am reasoning",
                signature="opaque-signature",
            ),
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "summary",
        ),
        (
            PROVIDER_CAPABILITIES["vllm"],
            Thinking(content="", summary="", signature=None),
            (
                ThinkingDelta(content_delta="I am"),
                ThinkingDelta(content_delta=" reasoning"),
            ),
            Thinking(content="I am reasoning", summary="", signature=None),
            "response.content_part.added",
            "response.reasoning_text.delta",
            "content",
        ),
        (
            PROVIDER_CAPABILITIES["vllm"],
            Thinking(content="I", summary="", signature=None),
            (
                ThinkingDelta(content_delta=" am"),
                ThinkingDelta(content_delta=" reasoning"),
            ),
            Thinking(content="I am reasoning", summary="", signature=None),
            "response.content_part.added",
            "response.reasoning_text.delta",
            "content",
        ),
    ],
    ids=["summary-empty-start", "raw-empty-start", "raw-non-empty-start"],
)
async def test_responses_projection_preserves_streamed_reasoning(
    capabilities: ProviderCapabilities,
    start: Thinking,
    deltas: tuple[ThinkingDelta, ...],
    completed: Thinking,
    part_added_type: str,
    text_delta_type: str,
    completed_field: str,
) -> None:
    adapter, request = _responses_adapter(capabilities)

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _RESPONSE_START,
                PartStartEvent(index=0, part=start),
                *(PartDeltaEvent(index=0, delta=delta) for delta in deltas),
                PartEndEvent(index=0, part=completed),
                _RESPONSE_END,
            ),
            request,
        )
    ]

    part_added = next(event for event in events if event.type == part_added_type)
    text_deltas = [event.delta for event in events if event.type == text_delta_type]
    done = next(event for event in events if event.type == "response.output_item.done")

    assert part_added.part.text == (start.content or start.summary)
    assert text_deltas == [delta.content_delta or delta.summary_delta for delta in deltas]
    assert [part.text for part in getattr(done.item, completed_field)] == ["I am reasoning"]
    assert done.item.encrypted_content == completed.signature


async def test_responses_projection_preserves_signature_only_reasoning_snapshots() -> None:
    adapter, request = _responses_adapter(PROVIDER_CAPABILITIES["openai"])

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _RESPONSE_START,
                PartStartEvent(
                    index=0,
                    part=Thinking(content="", summary="", signature="initial-signature"),
                ),
                PartEndEvent(
                    index=0,
                    part=Thinking(content="", summary="", signature="final-signature"),
                ),
                _RESPONSE_END,
            ),
            request,
        )
    ]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]
    added = next(event for event in events if event.type == "response.output_item.added")
    done = next(event for event in events if event.type == "response.output_item.done")
    assert added.item.content == added.item.summary == []
    assert added.item.encrypted_content == "initial-signature"
    assert done.item.content == done.item.summary == []
    assert done.item.encrypted_content == "final-signature"


@pytest.mark.parametrize(
    "deltas",
    [
        (
            ThinkingDelta(content_delta="raw"),
            ThinkingDelta(summary_delta="summary"),
        ),
    ],
)
async def test_responses_projection_rejects_reasoning_outside_capability(
    deltas: tuple[ThinkingDelta, ...],
) -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match="cannot contain raw text"):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _RESPONSE_START,
                    PartStartEvent(index=0, part=Thinking(content="", summary="")),
                    *(PartDeltaEvent(index=0, delta=delta) for delta in deltas),
                ),
                request,
            )
        ]


def test_responses_normalization_rejects_unsigned_summary_history() -> None:
    request = OpenAIResponsesCreateRequest(
        input=[
            {
                "type": "reasoning",
                "content": [],
                "summary": [{"type": "summary_text", "text": "Summary"}],
            }
        ],
        model="m",
    )

    with pytest.raises(ValueError, match="summary requires a signature"):
        OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(request)


async def test_responses_projection_rejects_unsigned_summary_completion() -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match="summary requires a signature"):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _RESPONSE_START,
                    PartStartEvent(index=0, part=Thinking(content="", summary="")),
                    PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Summary")),
                    PartEndEvent(index=0, part=Thinking(content="", summary="Summary")),
                ),
                request,
            )
        ]


async def test_responses_projection_rejects_thinking_part_with_content_and_summary() -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match="cannot contain both raw reasoning"):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _RESPONSE_START,
                    PartStartEvent(index=0, part=Thinking(content="", summary="")),
                    PartEndEvent(index=0, part=Thinking(content="raw", summary="summary")),
                ),
                request,
            )
        ]


async def test_responses_projection_signature_only_delta_defers_to_completed_item() -> None:
    # Responses has no mid-stream signature event, so a signature-only delta emits
    # nothing and does not register a text representation; the completed item
    # carries the final snapshot.
    adapter, request = _responses_adapter(_CAPABILITIES)

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _RESPONSE_START,
                PartStartEvent(
                    index=0, part=Thinking(content="", summary="", signature="initial")
                ),
                PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="final")),
                PartEndEvent(
                    index=0, part=Thinking(content="", summary="Summary", signature="final")
                ),
                _RESPONSE_END,
            ),
            request,
        )
    ]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = next(event for event in events if event.type == "response.output_item.added")
    done = next(event for event in events if event.type == "response.output_item.done")
    assert added.item.encrypted_content == "initial"
    assert [part.text for part in done.item.summary] == ["Summary"]
    assert done.item.encrypted_content == "final"


@pytest.mark.parametrize(
    ("delta", "message"),
    [
        (ToolCallDelta(tool_name_delta="Extra"), "cannot update a function name"),
        (ToolCallDelta(tool_call_id="call_new"), "cannot update a call ID"),
    ],
)
async def test_responses_projection_rejects_tool_identity_updates(
    delta: ToolCallDelta, message: str
) -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match=message):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _RESPONSE_START,
                    PartStartEvent(
                        index=0,
                        part=ToolCall(tool_name="First", args="", tool_call_id="call_one"),
                    ),
                    PartDeltaEvent(index=0, delta=delta),
                ),
                request,
            )
        ]


async def test_responses_projection_rejects_delta_before_part_start() -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match="delta before its output item started"):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _RESPONSE_START,
                    PartDeltaEvent(index=0, delta=TextDelta(content_delta="text")),
                ),
                request,
            )
        ]


@pytest.mark.parametrize(
    ("part", "expected_id"),
    [
        (Thinking(content="", summary="", id="rs_custom"), "rs_custom"),
        (TextOutput(content="", id="msg_custom"), "msg_custom"),
        (
            ToolCall(tool_name="First", args="", tool_call_id="call_one", id="fc_custom"),
            "fc_custom",
        ),
    ],
)
async def test_responses_projection_preserves_native_part_ids(
    part: Thinking | TextOutput | ToolCall, expected_id: str
) -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    events = [
        event
        async for event in adapter.project_events(
            _stream(_RESPONSE_START, PartStartEvent(index=0, part=part)),
            request,
        )
    ]

    added = next(event for event in events if event.type == "response.output_item.added")
    assert added.item.id == expected_id


@pytest.mark.parametrize(
    ("part", "prefix"),
    [
        (Thinking(content="", summary=""), ""),
        (TextOutput(content=""), "msg_"),
        (ToolCall(tool_name="First", args="", tool_call_id="call_one"), "fc_"),
    ],
)
async def test_responses_projection_generates_prefixed_item_ids(
    part: Thinking | TextOutput | ToolCall, prefix: str
) -> None:
    adapter, request = _responses_adapter(_CAPABILITIES)

    events = [
        event
        async for event in adapter.project_events(
            _stream(_RESPONSE_START, PartStartEvent(index=0, part=part)),
            request,
        )
    ]

    added = next(event for event in events if event.type == "response.output_item.added")
    assert added.item.id is not None
    assert added.item.id.startswith(prefix)
    assert len(added.item.id) == len(prefix) + 32


@pytest.mark.parametrize(
    ("capabilities", "part", "expected_delta"),
    [
        (
            _CAPABILITIES,
            Thinking(content="", summary=""),
            {"role": "assistant", "content": None, "reasoning_content": ""},
        ),
        (
            PROVIDER_CAPABILITIES["vllm"],
            Thinking(content="Raw thought", summary=""),
            {"role": "assistant", "content": None, "reasoning_content": "Raw thought"},
        ),
        (
            _CAPABILITIES,
            Thinking(content="", summary="Summarized thought"),
            {"role": "assistant", "content": None, "reasoning_content": "Summarized thought"},
        ),
        (
            _CAPABILITIES,
            TextOutput(content=""),
            {"role": "assistant", "content": "", "refusal": None},
        ),
        (
            _CAPABILITIES,
            TextOutput(content="Ready answer"),
            {"role": "assistant", "content": "Ready answer", "refusal": None},
        ),
        (
            _CAPABILITIES,
            ToolCall(tool_name="First", args="{}", tool_call_id="call_one"),
            {"role": "assistant", "content": None, "refusal": None},
        ),
    ],
)
async def test_chat_projection_initial_chunk_matches_first_part(
    capabilities: ProviderCapabilities,
    part: Thinking | TextOutput | ToolCall,
    expected_delta: dict,
) -> None:
    adapter, request = _chat_adapter(capabilities)

    events = [
        event
        async for event in adapter.project_events(
            _stream(_CHAT_START, PartStartEvent(index=0, part=part)),
            request,
        )
    ]

    assert events[0].choices[0].delta.model_dump(exclude_unset=True) == expected_delta


async def test_chat_projection_preserves_non_empty_later_part_starts() -> None:
    # A non-initial part start may already carry text; preserve it as its own
    # chunk instead of waiting for deltas that may not exist.
    adapter, request = _chat_adapter(_CAPABILITIES)

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _CHAT_START,
                PartStartEvent(index=0, part=TextOutput(content="First")),
                PartEndEvent(index=0, part=TextOutput(content="First")),
                PartStartEvent(index=1, part=Thinking(content="", summary="Second thought")),
                PartEndEvent(index=1, part=Thinking(content="", summary="Second thought")),
                PartStartEvent(index=2, part=TextOutput(content="Second")),
                PartEndEvent(index=2, part=TextOutput(content="Second")),
            ),
            request,
        )
    ]

    assert [event.choices[0].delta.model_dump(exclude_unset=True) for event in events] == [
        {"role": "assistant", "content": "First", "refusal": None},
        {"reasoning_content": "Second thought"},
        {"content": "Second"},
    ]


async def test_responses_projection_seeds_lifecycle_from_non_empty_part_starts() -> None:
    # A non-empty start snapshot opens the Responses text lifecycle with that
    # text, keeping the stream consistent with the completed item.
    adapter, request = _responses_adapter(PROVIDER_CAPABILITIES["vllm"])

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _RESPONSE_START,
                PartStartEvent(index=0, part=Thinking(content="Initial", summary="", id="rs_1")),
                PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta=" more")),
                PartEndEvent(
                    index=0, part=Thinking(content="Initial more", summary="", id="rs_1")
                ),
                PartStartEvent(index=1, part=TextOutput(content="Answer", id="msg_1")),
                PartEndEvent(index=1, part=TextOutput(content="Answer", id="msg_1")),
                _RESPONSE_END,
            ),
            request,
        )
    ]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    part_adds = [event for event in events if event.type == "response.content_part.added"]
    assert [part_adds[0].part.text, part_adds[1].part.text] == ["Initial", "Answer"]
    reasoning_delta = next(
        event for event in events if event.type == "response.reasoning_text.delta"
    )
    assert reasoning_delta.delta == " more"
    done_items = [event for event in events if event.type == "response.output_item.done"]
    assert [part.text for part in done_items[0].item.content] == ["Initial more"]
    assert [part.text for part in done_items[1].item.content] == ["Answer"]


async def test_chat_projection_skips_empty_thinking_delta() -> None:
    adapter, request = _chat_adapter(_CAPABILITIES)

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _CHAT_START,
                PartStartEvent(index=0, part=Thinking(content="", summary="")),
                PartDeltaEvent(index=0, delta=ThinkingDelta()),
            ),
            request,
        )
    ]

    assert [event.choices[0].delta.model_dump(exclude_unset=True) for event in events] == [
        {"role": "assistant", "content": None, "reasoning_content": ""}
    ]


async def test_chat_projection_applies_tool_identity_deltas() -> None:
    adapter, request = _chat_adapter(_CAPABILITIES)

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _CHAT_START,
                PartStartEvent(
                    index=0,
                    part=ToolCall(tool_name="First", args="", tool_call_id="call_one"),
                ),
                PartDeltaEvent(
                    index=0,
                    delta=ToolCallDelta(
                        tool_name_delta="Extra", tool_call_id="call_two", args_delta="{}"
                    ),
                ),
            ),
            request,
        )
    ]

    tool_call = events[-1].choices[0].delta.tool_calls[0]
    assert tool_call.model_dump(exclude_unset=True) == {
        "index": 0,
        "id": "call_two",
        "function": {"name": "Extra", "arguments": "{}"},
    }


async def test_chat_projection_rejects_tool_delta_before_call_start() -> None:
    adapter, request = _chat_adapter(_CAPABILITIES)

    with pytest.raises(ValueError, match="tool delta before its call started"):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _CHAT_START,
                    PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta="{}")),
                ),
                request,
            )
        ]


def _model_not_found_error() -> ModelNotFoundError:
    body = {
        "error": {
            "code": "model_not_found",
            "message": "Unknown model",
            "param": "model",
            "type": "invalid_request_error",
        }
    }
    response = httpx.Response(
        404,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    try:
        raise NotFoundError("Unknown model", response=response, body=body)
    except NotFoundError as error:
        try:
            raise ModelNotFoundError(
                error.message, response=error.response, body=error.body
            ) from error
        except ModelNotFoundError as normalized_error:
            return normalized_error


async def _failing_events() -> AsyncIterator[ModelResponseStreamEvent]:
    raise _model_not_found_error()
    yield


@pytest.mark.parametrize("native", [True, False], ids=["native", "translated"])
async def test_error_projection_selects_subclass_from_status(native: bool) -> None:
    if native:
        # Native passthrough never normalizes a request.
        adapter = OpenAIResponsesAdapter(_CAPABILITIES)
        stream = adapter.project_native_events(_failing_events())
    else:
        adapter, request = _responses_adapter(_CAPABILITIES)
        stream = adapter.project_events(_failing_events(), request)

    with pytest.raises(NotFoundError) as exc_info:
        [event async for event in stream]

    error = exc_info.value
    assert type(error) is NotFoundError
    assert isinstance(error.__cause__, ModelNotFoundError)
    assert type(error.__cause__.__cause__) is NotFoundError
    assert error is not error.__cause__.__cause__


@pytest.mark.parametrize("role", ["system", "developer", "assistant"])
def test_chat_normalizes_marked_text_arrays_without_merging_parts(role):
    """Ensures text parts stay separate so instruction and assistant cache prefixes survive."""
    request = OpenAIChatCompletionsCreateRequest(
        model="m",
        messages=[
            {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": "Cached",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    },
                    {"type": "text", "text": "Uncached"},
                ],
            }
        ],
    )
    normalized = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(request)
    item_type = TextOutput if role == "assistant" else SystemPrompt
    assert normalized.input == [
        item_type(content="Cached", cache_breakpoint=CacheBreakpoint()),
        item_type(content="Uncached"),
    ]
    assert normalized.prompt_cache == PromptCache(mode="implicit", ttl="30m")
