"""Shared termination semantics across downstream protocol adapters."""

import pytest

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.normalized import ResponseEndEvent, ResponseStartEvent, TerminationReason
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest


@pytest.mark.parametrize(
    ("termination_reason", "chat_reason", "messages_reason", "responses_reason"),
    [
        ("max_output_tokens", "length", "max_tokens", "max_output_tokens"),
        ("content_filter", "content_filter", "end_turn", "content_filter"),
        ("unknown", "stop", "end_turn", None),
        ("pause", "length", "pause_turn", None),
    ],
)
async def test_incomplete_termination_projections(
    termination_reason: TerminationReason,
    chat_reason: str,
    messages_reason: str,
    responses_reason: str | None,
) -> None:
    """Ensures all adapters preserve explicit reasons and distinguish unknown from pause."""

    async def normalized_events():
        yield ResponseStartEvent(id="resp_123", created_at=1, model="m")
        yield ResponseEndEvent(
            id="resp_123", created_at=1, model="m", termination_reason=termination_reason
        )

    chat = OpenAIChatCompletionsAdapter(PROVIDER_CAPABILITIES["openai"])
    request = chat.normalize_request(
        OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
        )
    )
    chat_events = [event async for event in chat.project_events(normalized_events(), request)]
    assert chat_events[-1].choices[0].finish_reason == chat_reason

    messages = AnthropicMessagesAdapter(PROVIDER_CAPABILITIES["anthropic"])
    request = messages.normalize_request(
        AnthropicMessagesCreateRequest(
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            max_tokens=100,
            stream=True,
        )
    )
    message_events = [
        event async for event in messages.project_events(normalized_events(), request)
    ]
    terminal_delta = next(event for event in message_events if event.type == "message_delta")
    assert terminal_delta.delta.stop_reason == messages_reason

    responses = OpenAIResponsesAdapter(PROVIDER_CAPABILITIES["openai"])
    request = responses.normalize_request(
        OpenAIResponsesCreateRequest(input="Hi", model="m", stream=True)
    )
    response_events = [
        event async for event in responses.project_events(normalized_events(), request)
    ]
    terminal = response_events[-1]
    assert terminal.type == "response.incomplete"
    assert terminal.response.status == "incomplete"
    assert terminal.response.model_dump()["incomplete_details"] == (
        {"reason": responses_reason} if responses_reason is not None else None
    )
