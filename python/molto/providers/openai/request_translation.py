"""Build OpenAI SDK request dictionaries from the portable IR."""

from __future__ import annotations

import json
from typing import Any

from molto.types.ir.portable import (
    AllowedFunctionTools,
    CodeInterpreterCallRequestItem,
    FunctionCallRequestItem,
    FunctionCallResultItem,
    FunctionToolChoice,
    FunctionToolDefinition,
    ImageGenerationCallRequestItem,
    InputAudioPart,
    InputImagePart,
    InputTextPart,
    JsonObjectOutputFormat,
    OpenAIHostedToolChoice,
    OpenAIHostedToolDefinition,
    OutputTextPart,
    ReasoningRequestItem,
    RefusalPart,
    RequestMessageItem,
    TextOutputFormat,
    WebSearchCallRequestItem,
)
from molto.utils.exceptions import BadInputError

from .types import OpenAIEndpoint


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _content_part(part: InputTextPart | InputImagePart | InputAudioPart) -> dict[str, Any]:
    if isinstance(part, InputTextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, InputImagePart):
        return {
            "type": "image_url",
            "image_url": {"url": part.image_url, "detail": part.detail},
        }
    return {"type": "input_audio", "input_audio": {"data": part.data, "format": part.format}}


def _chat_content(item: RequestMessageItem) -> str | list[dict[str, Any]]:
    if item.input_form == "responses_output_item":
        raise BadInputError("Chat Completions cannot represent Responses output history.")
    if len(item.content) == 1 and isinstance(item.content[0], InputTextPart):
        return item.content[0].text
    return [_content_part(part) for part in item.content]


def _chat_messages(ir) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if ir.instructions is not None:
        messages.append({"role": "developer", "content": ir.instructions})
    for item in ir.items:
        if isinstance(item, RequestMessageItem):
            message = {"role": item.role, "content": _chat_content(item)}
            if item.role == "assistant" and not item.content:
                message["content"] = None
            if item.audio_id is not None:
                if item.role != "assistant":
                    raise BadInputError("Previous audio references require an assistant message.")
                message["audio"] = {"id": item.audio_id}
            messages.append(message)
        elif isinstance(item, FunctionCallRequestItem):
            if item.namespace is not None:
                raise BadInputError("Chat Completions cannot represent namespaced function calls.")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {"name": item.name, "arguments": item.arguments},
                        }
                    ],
                }
            )
        elif isinstance(item, FunctionCallResultItem):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output,
                }
            )
        else:
            raise BadInputError(f"Chat Completions cannot represent {type(item).__name__}.")
    return messages


def _responses_content(item: RequestMessageItem) -> str | list[dict[str, Any]]:
    if len(item.content) == 1 and isinstance(item.content[0], InputTextPart):
        return item.content[0].text
    result = []
    for part in item.content:
        if isinstance(part, InputTextPart):
            result.append({"type": "input_text", "text": part.text})
        elif isinstance(part, InputImagePart):
            result.append(
                {"type": "input_image", "image_url": part.image_url, "detail": part.detail}
            )
        elif isinstance(part, (OutputTextPart, RefusalPart)):
            raise BadInputError("Output content requires a Responses output-message item.")
        else:
            raise BadInputError("Responses cannot represent Chat input audio.")
    return result


def _responses_output_message(item: RequestMessageItem) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in item.content:
        if isinstance(part, OutputTextPart):
            content.append(
                {
                    "type": "output_text",
                    "text": part.text,
                    "annotations": [
                        annotation.model_dump(mode="python")
                        for annotation in part.annotations or ()
                    ],
                }
            )
        elif isinstance(part, RefusalPart):
            content.append({"type": "refusal", "refusal": part.refusal})
        else:
            raise BadInputError("Responses output-message history requires output content parts.")
    return _without_none(
        {
            "type": "message",
            "id": item.id,
            "status": item.status.value if item.status is not None else None,
            "role": "assistant",
            "content": content,
            "phase": item.phase,
        }
    )


def _responses_input(ir) -> str | list[dict[str, Any]]:
    if len(ir.items) == 1 and isinstance(ir.items[0], RequestMessageItem):
        item = ir.items[0]
        if item.role == "user" and item.input_form != "responses_item":
            return _responses_content(item)
    result: list[dict[str, Any]] = []
    for item in ir.items:
        if isinstance(item, RequestMessageItem):
            if item.input_form == "responses_output_item":
                result.append(_responses_output_message(item))
                continue
            if item.audio_id is not None:
                raise BadInputError("Responses cannot represent previous Chat audio references.")
            message = {"role": item.role, "content": _responses_content(item)}
            if item.type_present:
                message["type"] = "message"
            result.append(message)
        elif isinstance(item, FunctionCallRequestItem):
            result.append(
                _without_none(
                    {
                        "type": "function_call",
                        "id": item.id,
                        "status": item.status.value if item.status is not None else None,
                        "call_id": item.call_id,
                        "name": item.name,
                        "namespace": item.namespace,
                        "arguments": item.arguments,
                    }
                )
            )
        elif isinstance(item, FunctionCallResultItem):
            result.append(
                _without_none(
                    {
                        "type": "function_call_output",
                        "id": item.id,
                        "call_id": item.call_id,
                        "output": item.output,
                    }
                )
            )
        elif isinstance(item, ReasoningRequestItem):
            result.append(
                _without_none(
                    {
                        "type": "reasoning",
                        "id": item.id,
                        "summary": [
                            {"type": "summary_text", "text": text} for text in item.summary
                        ],
                        "content": [
                            {"type": "reasoning_text", "text": text} for text in item.content
                        ],
                        "encrypted_content": item.encrypted_content,
                        "status": item.status.value if item.status is not None else None,
                    }
                )
            )
        elif isinstance(item, WebSearchCallRequestItem):
            action = item.action.model_dump(mode="python", exclude_none=True)
            if "queries" in action:
                action["queries"] = list(action["queries"])
            if "sources" in action:
                action["sources"] = list(action["sources"])
            result.append(
                {
                    "type": "web_search_call",
                    "id": item.id,
                    "action": action,
                    "status": item.status,
                }
            )
        elif isinstance(item, CodeInterpreterCallRequestItem):
            result.append(
                {
                    "type": "code_interpreter_call",
                    "id": item.id,
                    "code": item.code,
                    "container_id": item.container_id,
                    "outputs": (
                        [output.model_dump(mode="python") for output in item.outputs]
                        if item.outputs is not None
                        else None
                    ),
                    "status": item.status,
                }
            )
        elif isinstance(item, ImageGenerationCallRequestItem):
            # The API requires `id` on image-generation input items and resolves
            # it against the persisted item; inline `result` is not an upload path.
            image_call: dict[str, Any] = {
                "type": "image_generation_call",
                "id": item.id,
            }
            for field in item.fields_present:
                image_call[field] = getattr(item, field)
            result.append(image_call)
        else:
            raise BadInputError(f"Responses cannot represent {type(item).__name__}.")
    return result


def _tool(
    tool: FunctionToolDefinition | OpenAIHostedToolDefinition,
    endpoint: OpenAIEndpoint,
) -> dict[str, Any]:
    if isinstance(tool, OpenAIHostedToolDefinition):
        if endpoint != "responses":
            raise BadInputError("Chat Completions cannot represent Responses hosted tools.")
        return json.loads(tool.payload_json)
    function = _without_none(
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
    )
    return (
        {"type": "function", "function": function}
        if endpoint == "chat_completions"
        else {"type": "function", **function}
    )


def _tool_choice(choice, endpoint: OpenAIEndpoint) -> Any:
    if choice is None or isinstance(choice, str):
        return choice
    if isinstance(choice, FunctionToolChoice):
        return (
            {"type": "function", "function": {"name": choice.name}}
            if endpoint == "chat_completions"
            else {"type": "function", "name": choice.name}
        )
    if isinstance(choice, AllowedFunctionTools):
        tools = [
            (
                {"type": "function", "function": {"name": tool.name}}
                if endpoint == "chat_completions"
                else {"type": "function", "name": tool.name}
            )
            for tool in choice.tools
        ]
        return {"type": "allowed_tools", "mode": choice.mode, "tools": tools}
    if isinstance(choice, OpenAIHostedToolChoice):
        if endpoint != "responses":
            raise BadInputError("Chat Completions cannot represent hosted tool choices.")
        return json.loads(choice.payload_json)
    raise BadInputError("OpenAI-compatible endpoint cannot represent this tool choice.")


def _output_format(ir, endpoint: OpenAIEndpoint) -> dict[str, Any] | None:
    if isinstance(ir.output_format, TextOutputFormat):
        return None
    if isinstance(ir.output_format, JsonObjectOutputFormat):
        return {"type": "json_object"}
    fields = _without_none(
        {
            "name": ir.output_format.name,
            "schema": ir.output_format.schema_,
            "description": ir.output_format.description,
            "strict": ir.output_format.strict,
        }
    )
    return (
        {"type": "json_schema", "json_schema": fields}
        if endpoint == "chat_completions"
        else {"type": "json_schema", **fields}
    )


def _state_values(ir) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if ir.state is None:
        return values
    for reference in ir.state.values:
        if reference.kind == "response":
            values["previous_response_id"] = reference.value
        elif reference.kind == "conversation":
            values["conversation"] = reference.value
    return values


def build_openai_request(ir, endpoint: OpenAIEndpoint) -> dict[str, Any]:
    """Build canonical OpenAI SDK arguments after capability validation."""

    generation = ir.generation
    request: dict[str, Any] = {"model": ir.model.split("/", 1)[-1]}
    if ir.stream is not None:
        request["stream"] = ir.stream
    if ir.stream_options is not None:
        request["stream_options"] = ir.stream_options.model_dump(mode="python", exclude_unset=True)
    request.update(
        _without_none(
            {
                "prompt_cache_key": ir.prompt_cache_key,
                "prompt_cache_retention": ir.prompt_cache_retention,
                "safety_identifier": ir.safety_identifier,
                "store": ir.store,
                "user": ir.user,
                "temperature": generation.temperature,
                "top_p": generation.top_p,
                "top_logprobs": generation.top_logprobs,
                "parallel_tool_calls": generation.parallel_tool_calls,
                "max_output_tokens": generation.max_output_tokens,
                "max_tool_calls": generation.max_tool_calls,
            }
        )
    )
    format_payload = _output_format(ir, endpoint)

    if endpoint != "chat_completions" and (
        ir.generation.audio_output is not None
        or ir.generation.modalities is not None
        and "audio" in ir.generation.modalities
        or ir.generation.web_search_options is not None
        or any(
            isinstance(item, RequestMessageItem) and item.audio_id is not None for item in ir.items
        )
    ):
        raise BadInputError("Responses cannot represent Chat audio.")

    if endpoint == "chat_completions":
        if ir.generation.audio_output is not None:
            request["audio"] = ir.generation.audio_output.model_dump(mode="python")
        if ir.generation.modalities is not None:
            request["modalities"] = list(ir.generation.modalities)
        if ir.generation.web_search_options is not None:
            request["web_search_options"] = ir.generation.web_search_options.model_dump(
                mode="python", exclude_unset=True
            )
        if generation.reasoning_effort is not None:
            request["reasoning_effort"] = generation.reasoning_effort
        request["messages"] = _chat_messages(ir)
        if format_payload is not None:
            request["response_format"] = (
                {"type": "json_object"}
                if format_payload["type"] == "json_object"
                else format_payload
            )
        if generation.max_output_tokens is not None:
            request["max_completion_tokens"] = request.pop("max_output_tokens")
    else:
        request["input"] = _responses_input(ir)
        if ir.include is not None:
            request["include"] = list(ir.include)
        if ir.instructions is not None:
            request["instructions"] = ir.instructions
        reasoning = _without_none(
            {
                "effort": generation.reasoning_effort,
                "summary": generation.reasoning_summary,
            }
        )
        if reasoning:
            request["reasoning"] = reasoning
        request.update(_state_values(ir))
        if format_payload is not None:
            request["text"] = {"format": format_payload}

    if ir.tools:
        request["tools"] = [_tool(tool, endpoint) for tool in ir.tools]
    if ir.tool_choice is not None:
        request["tool_choice"] = _tool_choice(ir.tool_choice, endpoint)
    return request
