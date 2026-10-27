"""Cross-adapter projection of normalized terminal and usage state.

One normalized stream must project into three downstream protocols, so the
behaviour for a reason the target cannot express belongs to a single table here
rather than to each adapter's own tests.
"""

import pytest

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.normalized import (
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TerminationReason,
    TextOutput,
)
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest

CAPABILITIES = PROVIDER_CAPABILITIES["openai"]


async def _events(reason: TerminationReason):
    yield ResponseStartEvent(id="r", created_at=1, model="m")
    part = TextOutput(content="answer")
    yield PartStartEvent(index=0, part=part)
    yield PartEndEvent(index=0, part=part)
    yield ResponseEndEvent(id="r", created_at=1, model="m", termination_reason=reason)


async def _chat(reason: TerminationReason):
    adapter = OpenAIChatCompletionsAdapter(CAPABILITIES)
    request = adapter.normalize_request(
        OpenAIChatCompletionsCreateRequest(
            model="m", messages=[{"role": "user", "content": "q"}], stream=True
        )
    )
    events = [event async for event in adapter.project_events(_events(reason), request)]
    return next(
        choice.finish_reason
        for event in events
        for choice in event.choices
        if choice.finish_reason
    )


async def _messages(reason: TerminationReason):
    adapter = AnthropicMessagesAdapter(CAPABILITIES)
    request = adapter.normalize_request(
        AnthropicMessagesCreateRequest(
            model="m",
            max_tokens=10,
            messages=[{"role": "user", "content": "q"}],
            stream=True,
        )
    )
    events = [event async for event in adapter.project_events(_events(reason), request)]
    return next(event.delta.stop_reason for event in events if event.type == "message_delta")


async def _responses(reason: TerminationReason):
    adapter = OpenAIResponsesAdapter(CAPABILITIES)
    request = adapter.normalize_request(OpenAIResponsesCreateRequest(model="m", input="q"))
    events = [event async for event in adapter.project_events(_events(reason), request)]
    terminal = events[-1]
    assert terminal.type in {"response.completed", "response.incomplete"}
    return terminal.response.status, terminal.response.incomplete_details


@pytest.mark.parametrize(
    ("reason", "chat", "messages"),
    [
        ("stop", "stop", "end_turn"),
        ("max_output_tokens", "length", "max_tokens"),
        ("content_filter", "content_filter", "end_turn"),
        ("pause", "length", "pause_turn"),
        # Unknown is not a pause: a target without an incomplete state reports
        # normal completion rather than the output limit or a continuation hint.
        ("unknown", "stop", "end_turn"),
        ("refusal", "stop", "refusal"),
    ],
)
async def test_terminal_reason_projects_into_chat_and_messages(
    reason: TerminationReason, chat: str, messages: str
):
    """Ensures each normalized terminal reason maps to the documented target value."""
    assert await _chat(reason) == chat
    assert await _messages(reason) == messages


@pytest.mark.parametrize(
    ("reason", "status", "details"),
    [
        ("stop", "completed", None),
        ("max_output_tokens", "incomplete", {"reason": "max_output_tokens"}),
        ("content_filter", "incomplete", {"reason": "content_filter"}),
        ("pause", "incomplete", None),
        ("unknown", "incomplete", None),
    ],
)
async def test_terminal_reason_projects_into_responses(reason, status, details):
    """Ensures Responses keeps incomplete states and leaves a reason-less cause null."""
    projected_status, incomplete_details = await _responses(reason)
    assert projected_status == status
    actual_details = None if incomplete_details is None else incomplete_details.model_dump()
    assert actual_details == details
