from collections.abc import AsyncIterator

import pytest

from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.normalized import (
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
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
    UserInput,
)
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
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
    with pytest.raises(ValueError, match="requires image_url"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file_123"}],
            }
        )


def test_responses_adapter_rejects_input_file():
    with pytest.raises(ValueError, match="Unsupported Responses user content type: input_file"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": "file_123"}],
            }
        )


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
def test_responses_adapter_rejects_prompt_cache_breakpoint(part):
    with pytest.raises(ValueError, match="not supported for translation"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {"role": "user", "content": [part]}
        )


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
    with pytest.raises(ValueError, match="Unsupported Chat user content type"):
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

    with pytest.raises(ValueError, match="cannot represent image tool result"):
        OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(request)


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
    with pytest.raises(ValueError, match="assistant message content cannot be translated"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "x"}],
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
