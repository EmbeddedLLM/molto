"""Compose OpenAI Chat Completions models from portable output IR."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator

from molto.providers.openai.usage import usage_to_openai
from molto.types.ir.events import (
    OutputEvent,
    OutputEventSequence,
    PreservedOpenAIStreamEvent,
    ResponseIncompleteEvent,
)
from molto.types.ir.output_items import AssistantMessageItem, FunctionCallItem
from molto.types.ir.portable import OutputIR, OutputTextPart, RefusalPart
from molto.types.openai.chat_completions.models_response import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionResponse,
)


def _finish_reason(output: OutputIR, has_tool_calls: bool) -> str:
    if isinstance(output.event_sequence.terminal, ResponseIncompleteEvent):
        return {"max_output_tokens": "length", "content_filter": "content_filter"}.get(
            output.incomplete_reason or "", "length"
        )
    return "tool_calls" if has_tool_calls else "stop"


def _response_values(output: OutputIR) -> dict:
    message = next(
        (item for item in output.output.values if isinstance(item, AssistantMessageItem)),
        None,
    )
    content = None
    refusal = None
    annotations = None
    if message is not None:
        if message.annotations is not None:
            annotations = [
                {
                    "type": annotation.type,
                    "url_citation": annotation.model_dump(mode="python", exclude={"type"}),
                }
                for annotation in message.annotations
            ]
        for part in message.content:
            if isinstance(part, OutputTextPart):
                content = part.text
                if annotations is None and part.annotations is not None:
                    annotations = [
                        {
                            "type": annotation.type,
                            "url_citation": annotation.model_dump(mode="python", exclude={"type"}),
                        }
                        for annotation in part.annotations
                    ]
            elif isinstance(part, RefusalPart):
                refusal = part.refusal

    function_calls = [item for item in output.output.values if isinstance(item, FunctionCallItem)]
    tool_calls = [
        {
            "id": item.call_id,
            "type": "function",
            "function": {"name": item.name, "arguments": item.arguments},
        }
        for item in function_calls
    ]
    message_values = {
        "role": "assistant",
        "content": content,
        "refusal": refusal,
    }
    if message is not None and message.audio is not None:
        message_values["audio"] = message.audio.model_dump(mode="python")
    if annotations is not None:
        message_values["annotations"] = annotations
    if tool_calls:
        message_values["tool_calls"] = tool_calls
    choice_values = {
        "index": output.metadata.choice_index,
        "message": message_values,
        "finish_reason": _finish_reason(output, bool(function_calls)),
    }
    if output.metadata.logprobs_present:
        choice_values["logprobs"] = None
    values = {
        "id": output.id,
        "choices": [choice_values],
        "created": int(output.metadata.created_at or 0),
        "model": output.metadata.model,
        "object": "chat.completion",
        "system_fingerprint": output.metadata.system_fingerprint,
        "usage": usage_to_openai(output.usage, "chat/completions"),
    }
    if output.metadata.service_tier_present:
        values["service_tier"] = output.metadata.service_tier
    return values


def from_output_ir(output: OutputIR) -> OpenAIChatCompletionResponse:
    """Compose one Chat response from semantic OpenAI-compatible output."""

    return OpenAIChatCompletionResponse.model_validate(_response_values(output))


async def from_output_event_stream(
    events: AsyncIterable[OutputEvent],
    *,
    include_usage: bool = True,
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Compose a Chat stream, forwarding matching native frames immediately.

    ``include_usage`` gates the usage on the synthesized terminal chunk; native
    frames are forwarded verbatim.
    """

    event_values: list[OutputEvent] = []
    matching_frames = False
    async for event in events:
        if isinstance(event, PreservedOpenAIStreamEvent) and event.endpoint == "chat/completions":
            matching_frames = True
            if event.event_type != "chat.completion.chunk":
                raise ValueError(f"Unexpected OpenAI Chat stream event {event.event_type!r}.")
            yield OpenAIChatCompletionChunk.model_validate(json.loads(event.payload_json))
            continue
        event_values.append(event)
    if matching_frames:
        return

    sequence = OutputEventSequence(events=tuple(event_values))
    output = OutputIR.from_output_events(sequence)
    values = _response_values(output)
    choice = values["choices"][0]
    message = choice["message"]
    delta = {
        "role": "assistant",
        "content": message["content"],
        "refusal": message["refusal"],
    }
    if "tool_calls" in message:
        delta["tool_calls"] = [
            {"index": index, **tool_call} for index, tool_call in enumerate(message["tool_calls"])
        ]
    if "audio" in message:
        delta["audio"] = message["audio"]
    yield OpenAIChatCompletionChunk.model_validate(
        {
            "id": output.id,
            "created": values["created"],
            "model": values["model"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": choice["finish_reason"],
                }
            ],
            "usage": values["usage"] if include_usage else None,
        }
    )
