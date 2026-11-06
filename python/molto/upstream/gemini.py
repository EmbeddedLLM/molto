"""Gemini's OpenAI-compatible Chat transport."""

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import TextOutput, Thinking, ToolCall
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.upstream.openai import OpenAIChatCompletionsClient


def _register_call(calls: dict[int, str | None], index: int, call_id: str | None) -> None:
    previous = calls.get(index)
    if previous and call_id and previous != call_id:
        raise ValueError("Gemini tool index changed call identity.")
    if call_id and any(i != index and known_id == call_id for i, known_id in calls.items()):
        raise ValueError("Gemini tool identity changed index.")
    calls[index] = call_id or previous


class GeminiChatCompletionsClient(OpenAIChatCompletionsClient):
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    REQUIRED_API_KEY = "GEMINI_API_KEY"

    @staticmethod
    def _thought_signature(extra_content: dict[str, Any] | None) -> str | None:
        if not extra_content:
            return None
        google = extra_content.get("google") if isinstance(extra_content, dict) else None
        if (
            not isinstance(extra_content, dict)
            or set(extra_content) != {"google"}
            or not isinstance(google, dict)
            or set(google) != {"thought_signature"}
            or not isinstance(google["thought_signature"], str)
            or not google["thought_signature"]
        ):
            raise ValueError(
                "Gemini extra_content must contain only a nonempty thought_signature."
            )
        return google["thought_signature"]

    @staticmethod
    def _starts_assistant_message(part: Thinking) -> bool:
        # A signature-only part follows the text or call it authenticates.
        return not part.signature or bool(part.content or part.summary)

    @staticmethod
    def _assistant_message(parts: list[Thinking | TextOutput | ToolCall]) -> dict[str, Any]:
        message = OpenAIChatCompletionsClient._assistant_message(
            [part for part in parts if not isinstance(part, Thinking) or not part.signature]
        )
        target: dict[str, Any] | None = None
        tool_index = 0
        for part in parts:
            if isinstance(part, TextOutput):
                target = message
            elif isinstance(part, ToolCall):
                target = message["tool_calls"][tool_index]
                tool_index += 1
            elif part.signature:
                if part.content or part.summary or target is None or "extra_content" in target:
                    raise UnsupportedInput(
                        "Gemini requires each signature-only thinking part immediately after "
                        "its text or tool call."
                    )
                target["extra_content"] = {"google": {"thought_signature": part.signature}}
                target = None
            else:
                target = None
        return message

    async def native_events(
        self, request: OpenAIChatCompletionsCreateRequest
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        # Repair only missing indices. Association uses call IDs, never a chunk's
        # array positions; already indexed provider chunks retain native fidelity.
        identities: dict[int, dict[int, str | None]] = {}
        announced_ids: dict[int, set[str]] = {}
        async with aclosing(super().native_events(request)) as source:
            async for original in source:
                chunk = original.model_copy(deep=True)
                for choice in chunk.choices:
                    calls = identities.setdefault(choice.index, {})
                    announced = announced_ids.setdefault(choice.index, set())
                    tools = choice.delta.tool_calls or []
                    # Reserve explicit indices first, including those appearing
                    # after an unindexed call in this chunk.
                    for tool in tools:
                        if tool.index is not None:
                            if type(tool.index) is not int or tool.index < 0:
                                raise ValueError(
                                    "Gemini tool indices must be nonnegative integers."
                                )
                            _register_call(calls, tool.index, tool.id)
                    for tool in tools:
                        index = tool.index
                        if index is None:
                            matches = [
                                i for i, call_id in calls.items() if tool.id and call_id == tool.id
                            ]
                            if matches:
                                index = matches[0]
                            elif tool.id:
                                index = 0
                                while index in calls:
                                    index += 1
                            elif (
                                len(calls) == 1
                                and len(tools) == 1
                                and not (tool.function and tool.function.name)
                            ):
                                index = next(iter(calls))
                            else:
                                raise ValueError(
                                    "Gemini tool fragment has no unambiguous index or ID."
                                )
                            tool.index = index
                            _register_call(calls, index, tool.id)
                            if tool.id in announced:
                                # IDs used for inference are complete identity
                                # snapshots; SDK accumulation concatenates them
                                # unless subsequent snapshots are omitted.
                                tool.id = None
                        if tool.id:
                            announced.add(tool.id)
                yield chunk
