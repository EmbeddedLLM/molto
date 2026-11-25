"""Gemini's OpenAI-compatible Chat transport."""

from collections.abc import AsyncIterator
from contextlib import aclosing

from molto.types.normalized import TerminationReason
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.upstream.openai import ChatCompletionsClient


class GeminiChatCompletionsClient(ChatCompletionsClient):
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    REQUIRED_API_KEY = "GEMINI_API_KEY"
    OMIT_NULL_MESSAGE_FIELDS = True
    OMIT_INCLUDE_OBFUSCATION = True
    SIGNATURE_NAMESPACE = "google"

    @staticmethod
    def _termination_reason(finish_reason: str, *, has_tool_calls: bool) -> TerminationReason:
        # Gemini ends completed tool calls with "stop" on its native Chat API.
        if finish_reason == "stop" and has_tool_calls:
            return "tool_call"
        return ChatCompletionsClient._termination_reason(
            finish_reason, has_tool_calls=has_tool_calls
        )

    async def native_events(
        self, request: OpenAIChatCompletionsCreateRequest
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        # Repair only missing indices. Association uses call IDs, never a chunk's
        # array positions; already indexed provider chunks retain native fidelity.
        identities: dict[int, dict[str, int]] = {}
        async with aclosing(super().native_events(request)) as source:
            async for chunk in source:
                # Only an unindexed call is rewritten, so other chunks pass
                # through without the per-chunk cost of a copy.
                if any(
                    tool.index is None
                    for choice in chunk.choices
                    for tool in choice.delta.tool_calls or []
                ):
                    chunk = chunk.model_copy(deep=True)
                for choice in chunk.choices:
                    calls = identities.setdefault(choice.index, {})
                    for tool in choice.delta.tool_calls or []:
                        if tool.index is not None:
                            continue
                        if tool.id:
                            if tool.id in calls:
                                tool.index = calls[tool.id]
                                # Repeated IDs identify fragments; SDK accumulation
                                # would otherwise concatenate these snapshots.
                                tool.id = None
                            else:
                                tool.index = calls[tool.id] = len(calls)
                        elif len(calls) == 1:
                            tool.index = next(iter(calls.values()))
                        else:
                            raise ValueError(
                                "Gemini tool fragment has no unambiguous index or ID."
                            )
                yield chunk
