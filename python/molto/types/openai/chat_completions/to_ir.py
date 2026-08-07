"""Convert an OpenAI Chat Completions request into the portable IR."""

from __future__ import annotations

from typing import Any, Literal

from molto.types.ir.portable import (
    AllowedFunctionTools,
    AudioOutputConfig,
    CustomVoice,
    FunctionCallRequestItem,
    FunctionCallResultItem,
    FunctionToolChoice,
    FunctionToolDefinition,
    GenerationConfig,
    InputAudioPart,
    InputImagePart,
    InputIR,
    InputTextPart,
    JsonObjectOutputFormat,
    JsonSchemaOutputFormat,
    RequestMessageItem,
    StreamOptions,
    TextOutputFormat,
    WebSearchOptions,
)
from molto.types.openai.chat_completions.models_request import (
    ChatCompletionAllowedToolsChoice,
    ChatCompletionAssistantMessage,
    ChatCompletionContentPartImage,
    ChatCompletionContentPartInputAudio,
    ChatCompletionContentPartText,
    ChatCompletionMessage,
    ChatCompletionNamedToolChoice,
    ChatCompletionResponseFormatJsonObject,
    ChatCompletionResponseFormatJsonSchema,
    ChatCompletionToolMessage,
    OpenAIChatCompletionRequest,
)
from molto.utils.exceptions import BadInputError


def _chat_content_to_ir(content):
    if content is None:
        return ()
    if isinstance(content, str):
        return (InputTextPart(text=content),)
    parts = []
    for part in content:
        if isinstance(part, ChatCompletionContentPartText):
            parts.append(InputTextPart(text=part.text))
        elif isinstance(part, ChatCompletionContentPartImage):
            parts.append(
                InputImagePart(image_url=part.image_url.url, detail=part.image_url.detail)
            )
        elif isinstance(part, ChatCompletionContentPartInputAudio):
            parts.append(
                InputAudioPart(data=part.input_audio.data, format=part.input_audio.format)
            )
        else:
            raise BadInputError(f"Unsupported Chat content part: {type(part).__name__}.")
    return tuple(parts)


def _chat_messages_to_ir(messages: list[ChatCompletionMessage]):
    items = []
    for message in messages:
        if message.name is not None:
            raise BadInputError("Chat message names have no IR representation.")
        if isinstance(message, ChatCompletionAssistantMessage) and message.refusal is not None:
            raise BadInputError("Chat assistant refusal history has no IR representation.")
        if isinstance(message, ChatCompletionToolMessage):
            items.append(
                FunctionCallResultItem(
                    call_id=message.tool_call_id,
                    output=message.content or "",
                )
            )
            continue
        items.append(
            RequestMessageItem(
                role=message.role,
                content=_chat_content_to_ir(message.content),
                audio_id=(
                    message.audio.id
                    if isinstance(message, ChatCompletionAssistantMessage)
                    and message.audio is not None
                    else None
                ),
            )
        )
        if isinstance(message, ChatCompletionAssistantMessage):
            for tool_call in message.tool_calls or ():
                items.append(
                    FunctionCallRequestItem(
                        call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                )
    return tuple(items)


def request_to_ir(request: OpenAIChatCompletionRequest) -> InputIR:
    """Convert a public Chat request into the portable OpenAI contract."""

    # InputIR represents one completion. n=1 is the OpenAI default and therefore
    # has no independent IR field; larger choice sets cannot be reconstructed.
    if request.n not in (None, 1):
        raise BadInputError("Chat n values other than 1 are unsupported.")

    unsupported_fields = (
        "logprobs",
        "seed",
        "service_tier",
        "stop",
    )
    supplied = [
        field
        for field in unsupported_fields
        if field in request.model_fields_set and getattr(request, field) is not None
    ]
    if supplied:
        raise BadInputError(
            "Chat request fields without an IR representation are unsupported: "
            + ", ".join(supplied)
            + "."
        )

    extra_body = request.extra_body or {}
    unsupported_extra_body = set(extra_body) - {"web_search_options"}
    if unsupported_extra_body:
        fields = ", ".join(sorted(unsupported_extra_body))
        raise BadInputError(f"Unsupported Chat extra_body fields: {fields}.")
    web_search_options = (
        WebSearchOptions.model_validate(extra_body["web_search_options"])
        if "web_search_options" in extra_body
        else None
    )

    tools = tuple(
        FunctionToolDefinition(
            name=tool.function.name,
            parameters=tool.function.parameters,
            description=tool.function.description,
            strict=tool.function.strict,
        )
        for tool in request.tools or ()
    )
    tool_choice: Any = request.tool_choice
    if isinstance(tool_choice, ChatCompletionNamedToolChoice):
        tool_choice = FunctionToolChoice(name=tool_choice.function.name)
    elif isinstance(tool_choice, ChatCompletionAllowedToolsChoice):
        tool_choice = AllowedFunctionTools(
            mode=tool_choice.mode,
            tools=tuple(FunctionToolChoice(name=tool.function.name) for tool in tool_choice.tools),
        )

    if isinstance(request.response_format, ChatCompletionResponseFormatJsonObject):
        output_format = JsonObjectOutputFormat()
    elif isinstance(request.response_format, ChatCompletionResponseFormatJsonSchema):
        output_format = JsonSchemaOutputFormat(
            name=request.response_format.name or "",
            schema=request.response_format.json_schema,
            description=request.response_format.description,
            strict=request.response_format.strict,
        )
    else:
        output_format = TextOutputFormat()

    max_output_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    stream_options = (
        StreamOptions.model_validate(
            request.stream_options.model_dump(mode="python", exclude_unset=True)
        )
        if request.stream_options is not None
        else None
    )
    return InputIR(
        model=request.model,
        stream=request.stream,
        stream_options=stream_options,
        include_usage=(stream_options.include_usage if stream_options is not None else None),
        store=request.store,
        prompt_cache_key=request.prompt_cache_key,
        prompt_cache_retention=request.prompt_cache_retention,
        safety_identifier=request.safety_identifier,
        user=request.user,
        items=_chat_messages_to_ir(request.messages),
        output_format=output_format,
        generation=GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            top_logprobs=request.top_logprobs,
            reasoning_effort=request.reasoning_effort,
            parallel_tool_calls=request.parallel_tool_calls,
            modalities=tuple(request.modalities) if request.modalities is not None else None,
            audio_output=(
                AudioOutputConfig(
                    format=request.audio.format,
                    voice=(
                        CustomVoice(id=request.audio.voice.id)
                        if hasattr(request.audio.voice, "id")
                        else request.audio.voice
                    ),
                )
                if request.audio is not None
                else None
            ),
            web_search_options=web_search_options,
        ),
        tools=tools,
        tool_choice=tool_choice,
    )


def prepare_chat_request(
    request: OpenAIChatCompletionRequest,
    endpoint: Literal["chat/completions", "responses"],
) -> InputIR:
    """Build provider input for the selected Chat endpoint."""

    provider_ir = request_to_ir(request)
    if endpoint == "responses":
        provider_ir = provider_ir.model_copy(update={"stream_options": None})
    return provider_ir
