"""Gemini's OpenAI-compatible Chat transport."""

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import TerminationReason, TextOutput, Thinking, ToolCall
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.upstream.openai import ChatCompletionsClient


def _register_call(calls: dict[int, str | None], index: int, call_id: str | None) -> None:
    previous = calls.get(index)
    if previous and call_id and previous != call_id:
        raise ValueError("Gemini tool index changed call identity.")
    if call_id and any(i != index and known_id == call_id for i, known_id in calls.items()):
        raise ValueError("Gemini tool identity changed index.")
    calls[index] = call_id or previous


class GeminiChatCompletionsClient(ChatCompletionsClient):
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    REQUIRED_API_KEY = "GEMINI_API_KEY"
    OMIT_NULL_MESSAGE_FIELDS = True
    OMIT_INCLUDE_OBFUSCATION = True

    @staticmethod
    def _termination_reason(finish_reason: str, *, has_tool_calls: bool) -> TerminationReason:
        # Gemini ends completed tool calls with "stop" on its native Chat API.
        if finish_reason == "stop" and has_tool_calls:
            return "tool_call"
        return ChatCompletionsClient._termination_reason(
            finish_reason, has_tool_calls=has_tool_calls
        )

    @staticmethod
    def _thought_signature(extra_content: dict[str, Any] | None) -> str | None:
        if not extra_content:
            return None
        google = extra_content.get("google", {}) if isinstance(extra_content, dict) else None
        if not isinstance(google, dict):
            raise ValueError("Gemini extra_content.google must be an object.")
        # Other keys are extensions this client does not model; only the
        # signature is continuation state that the next request must carry.
        if "thought_signature" not in google:
            return None
        signature = google["thought_signature"]
        if not isinstance(signature, str) or not signature:
            raise ValueError(
                "Gemini extra_content.google.thought_signature must be a nonempty string."
            )
        return signature

    @staticmethod
    def _starts_assistant_message(part: Thinking) -> bool:
        # A signature-only part follows the text or call it authenticates.
        return not part.signature or bool(part.content or part.summary)

    @staticmethod
    def _assistant_message(parts: list[Thinking | TextOutput | ToolCall]) -> dict[str, Any]:
        message = ChatCompletionsClient._assistant_message(
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
                # A signature that follows no text or call was attached to the
                # message itself, which Gemini sends when a turn has no text.
                target = message if target is None else target
                if part.content or part.summary or "extra_content" in target:
                    raise UnsupportedInput(
                        "Gemini requires signature-only thinking parts, at most one for the "
                        "message and one for each tool call."
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
