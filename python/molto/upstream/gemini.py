"""Gemini's OpenAI-compatible Chat transport."""

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    ModelResponseStreamEvent,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TerminationReason,
    TextOutput,
    Thinking,
    ToolCall,
)
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
        # Signatures prefix their content and do not split parallel tool calls.
        return not part.signature or bool(part.content or part.summary)

    @classmethod
    def _messages(cls, request: NormalizedRequest) -> list[dict[str, Any]]:
        def is_signature(part: Any) -> bool:
            return isinstance(part, Thinking) and not cls._starts_assistant_message(part)

        items = []
        for item in request.input or []:
            if isinstance(item, AssistantInput):
                signature_start = len(items)
                while signature_start and is_signature(items[signature_start - 1]):
                    signature_start -= 1
                prefix_start = signature_start
                while prefix_start and (
                    isinstance(items[prefix_start - 1], ToolCall)
                    or is_signature(items[prefix_start - 1])
                ):
                    prefix_start -= 1
                if prefix_start and isinstance(items[prefix_start - 1], AssistantInput):
                    # Calls after an explicit text message stay with that message.
                    prefix_start = signature_start
                if prefix_start < len(items):
                    # A standalone Responses signature belongs to the following
                    # text message. Unattached calls can join it, but two explicit
                    # assistant messages must keep their separate boundaries.
                    item = item.model_copy(
                        update={"content": [*items[prefix_start:], *item.content]}
                    )
                    del items[prefix_start:]
            items.append(item)
        return super()._messages(request.model_copy(update={"input": items}))

    @staticmethod
    def _assistant_message(parts: list[Thinking | TextOutput | ToolCall]) -> dict[str, Any]:
        content = []
        for part in parts:
            if isinstance(part, Thinking) and part.signature:
                if part.content or part.summary:
                    raise UnsupportedInput("Gemini requires signature-only thinking parts.")
                continue
            content.append(part)
        message = ChatCompletionsClient._assistant_message(content)
        signature: str | None = None
        tool_index = 0

        def attach(target: dict[str, Any]) -> None:
            nonlocal signature
            if signature is None:
                return
            if "extra_content" in target:
                raise UnsupportedInput(
                    "Gemini requires at most one signature for the message and each tool call."
                )
            target["extra_content"] = {"google": {"thought_signature": signature}}
            signature = None

        for part in parts:
            if isinstance(part, Thinking) and part.signature:
                if signature is not None:
                    raise UnsupportedInput(
                        "Gemini requires at most one signature before each text or tool call."
                    )
                signature = part.signature
            elif isinstance(part, TextOutput):
                attach(message)
            elif isinstance(part, ToolCall):
                attach(message["tool_calls"][tool_index])
                tool_index += 1
        # Without following content, the signature remains message-level state.
        attach(message)
        return message

    async def _events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        message_signature: str | None = None
        tool_signatures: dict[int, str] = {}
        tool_ids: dict[int, str] = {}

        async def capture_signatures() -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
            nonlocal message_signature
            async with aclosing(self.native_events(self._native_request(request))) as source:
                async for chunk in source:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        signature = self._thought_signature(getattr(delta, "extra_content", None))
                        if signature is not None:
                            message_signature = signature
                        for tool in delta.tool_calls or []:
                            if tool.id is not None:
                                tool_ids[tool.index] = tool.id
                            signature = self._thought_signature(
                                getattr(tool, "extra_content", None)
                            )
                            if signature is not None:
                                tool_signatures[tool.index] = signature
                    yield chunk

        buffered: list[ModelResponseStreamEvent] = []
        # A final message signature can authenticate text emitted before calls.
        # Keep this buffering specific to Gemini translation; native chunks and
        # other providers retain their ordinary streaming behavior.
        async with aclosing(capture_signatures()) as source:
            async with aclosing(self._normalize_stream(source)) as events:
                async for event in events:
                    if isinstance(event, ResponseStartEvent):
                        yield event
                    else:
                        buffered.append(event)

        signatures_by_id = {
            tool_ids[index]: signature for index, signature in tool_signatures.items()
        }
        signatures: dict[int, str] = {}
        first_text: int | None = None
        first_call: int | None = None
        for event in buffered:
            if not isinstance(event, PartStartEvent):
                continue
            if isinstance(event.part, TextOutput) and first_text is None:
                first_text = event.index
            elif isinstance(event.part, ToolCall):
                if first_call is None:
                    first_call = event.index
                if event.part.tool_call_id in signatures_by_id:
                    signatures[event.index] = signatures_by_id[event.part.tool_call_id]
        if message_signature is not None:
            if first_text is not None:
                signatures[first_text] = message_signature
                message_signature = None
            elif first_call is not None and first_call not in signatures:
                # Without text, message-level state prefixes the first call.
                # A separately signed call keeps its own state; preserve any
                # additional message signature as a standalone trailing part.
                signatures[first_call] = message_signature
                message_signature = None

        next_index = 0
        indices: dict[int, int] = {}

        def signature_events(signature: str) -> tuple[PartStartEvent, PartEndEvent]:
            nonlocal next_index
            part = Thinking(content="", summary="", signature=signature)
            index = next_index
            next_index += 1
            return PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)

        for event in buffered:
            if isinstance(event, PartStartEvent):
                if event.index in signatures:
                    for signature_event in signature_events(signatures[event.index]):
                        yield signature_event
                indices[event.index] = next_index
                next_index += 1
            elif isinstance(event, ResponseEndEvent) and message_signature is not None:
                for signature_event in signature_events(message_signature):
                    yield signature_event
            if isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent)):
                event = event.model_copy(update={"index": indices[event.index]})
            yield event

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
