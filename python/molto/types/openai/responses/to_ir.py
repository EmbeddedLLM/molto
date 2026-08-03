"""Convert OpenAI Responses requests into the portable IR."""

from __future__ import annotations

import json
from typing import Any

from molto.types.ir.portable import (
    AllowedFunctionTools,
    CodeInterpreterCallRequestItem,
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLog,
    FunctionCallRequestItem,
    FunctionCallResultItem,
    FunctionToolChoice,
    FunctionToolDefinition,
    GenerationConfig,
    ImageGenerationCallRequestItem,
    InputImagePart,
    InputIR,
    InputTextPart,
    ItemStatus,
    JsonObjectOutputFormat,
    JsonSchemaOutputFormat,
    OpenAIHostedToolChoice,
    OpenAIHostedToolDefinition,
    OutputTextPart,
    ReasoningRequestItem,
    RefusalPart,
    RequestMessageItem,
    StateReference,
    StateReferences,
    StreamOptions,
    TextOutputFormat,
    UrlCitation,
    WebSearchActionFindInPage,
    WebSearchActionOpenPage,
    WebSearchActionSearch,
    WebSearchCallRequestItem,
    WebSearchSource,
)
from molto.utils.exceptions import BadInputError

from .models_items import (
    OpenAIAllowedToolsChoice,
    OpenAICodeOutputImage,
    OpenAICodeOutputLog,
    OpenAICodeToolCall,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolChoice,
    OpenAIFunctionToolOutput,
    OpenAIImageContent,
    OpenAIImageGenerationCallInput,
    OpenAIInput,
    OpenAIInputMessage,
    OpenAIJsonObjectFormat,
    OpenAIJsonSchemaFormat,
    OpenAIOutputItem,
    OpenAIOutputRefusal,
    OpenAIOutputTextContent,
    OpenAIReasoningItem,
    OpenAIResponsesFunctionTool,
    OpenAITextContent,
    OpenAIWebSearchActionFindInPage,
    OpenAIWebSearchActionOpenPage,
    OpenAIWebSearchActionSearch,
    OpenAIWebSearchToolCall,
)


def _content_to_ir(content: str | list[OpenAITextContent | OpenAIImageContent]):
    if isinstance(content, str):
        return (InputTextPart(text=content),)
    result = []
    for part in content:
        if isinstance(part, OpenAITextContent):
            result.append(InputTextPart(text=part.text))
        elif isinstance(part, OpenAIImageContent):
            result.append(InputImagePart(image_url=part.image_url, detail=part.detail))
        else:
            raise BadInputError(f"Unsupported Responses content part: {type(part).__name__}.")
    return tuple(result)


def _output_content_to_ir(
    content: list[OpenAIOutputTextContent | OpenAIOutputRefusal],
):
    result = []
    for part in content:
        if isinstance(part, OpenAIOutputTextContent):
            if part.logprobs:
                raise BadInputError(
                    "Responses output-message history with logprobs is unsupported."
                )
            try:
                annotations = tuple(
                    UrlCitation.model_validate(annotation) for annotation in part.annotations
                )
            except ValueError as exc:
                raise BadInputError(
                    "Responses output-message history contains unsupported annotations."
                ) from exc
            result.append(OutputTextPart(text=part.text, annotations=annotations))
        else:
            result.append(RefusalPart(refusal=part.refusal))
    return tuple(result)


def _web_search_call_to_ir(item: OpenAIWebSearchToolCall) -> WebSearchCallRequestItem:
    action = item.action
    if isinstance(action, OpenAIWebSearchActionSearch):
        sources = None
        if action.sources is not None:
            source_values = []
            for source in action.sources:
                if source.type != "url" or source.url is None:
                    raise BadInputError("Web-search history supports only URL sources with a URL.")
                source_values.append(WebSearchSource(url=source.url))
            sources = tuple(source_values)
        ir_action = WebSearchActionSearch(
            query=action.query,
            queries=tuple(action.queries) if action.queries is not None else None,
            sources=sources,
        )
    elif isinstance(action, OpenAIWebSearchActionOpenPage):
        ir_action = WebSearchActionOpenPage(url=action.url)
    elif isinstance(action, OpenAIWebSearchActionFindInPage):
        if action.url is None:
            raise BadInputError("Find-in-page history requires a URL.")
        ir_action = WebSearchActionFindInPage(pattern=action.pattern, url=action.url)
    else:
        raise BadInputError("Web-search history requires a completed action.")
    return WebSearchCallRequestItem(
        id=item.id,
        action=ir_action,
        status=item.status,
    )


def _code_interpreter_call_to_ir(
    item: OpenAICodeToolCall,
) -> CodeInterpreterCallRequestItem:
    outputs = None
    if item.outputs is not None:
        output_values = []
        for output in item.outputs:
            if isinstance(output, OpenAICodeOutputLog):
                if output.logs is None:
                    raise BadInputError("Code-interpreter log history requires log text.")
                output_values.append(CodeInterpreterOutputLog(logs=output.logs))
            elif isinstance(output, OpenAICodeOutputImage):
                output_values.append(CodeInterpreterOutputImage(url=output.url))
        outputs = tuple(output_values)
    return CodeInterpreterCallRequestItem(
        id=item.id,
        code=item.code,
        container_id=item.container_id,
        outputs=outputs,
        status=item.status,
    )


def _input_items_to_ir(input_value: str | list[OpenAIInput]):
    if isinstance(input_value, str):
        return (
            RequestMessageItem(
                role="user",
                content=(InputTextPart(text=input_value),),
                input_form="responses_string",
            ),
        )

    items = []
    for item in input_value:
        if isinstance(item, OpenAIInputMessage):
            items.append(
                RequestMessageItem(
                    role=item.role,
                    content=_content_to_ir(item.content),
                    input_form="responses_item",
                    type_present="type" in item.model_fields_set,
                )
            )
        elif isinstance(item, OpenAIOutputItem):
            items.append(
                RequestMessageItem(
                    id=item.id,
                    role="assistant",
                    content=_output_content_to_ir(item.content),
                    status=ItemStatus(item.status),
                    phase=item.phase,
                    input_form="responses_output_item",
                    type_present=True,
                )
            )
        elif isinstance(item, OpenAIReasoningItem):
            items.append(
                ReasoningRequestItem(
                    id=item.id,
                    summary=tuple(part.text for part in item.summary),
                    content=tuple(part.text for part in item.content),
                    encrypted_content=item.encrypted_content,
                    status=ItemStatus(item.status) if item.status is not None else None,
                )
            )
        elif isinstance(item, OpenAIFunctionToolCall):
            items.append(
                FunctionCallRequestItem(
                    id=item.id,
                    status=ItemStatus(item.status) if item.status is not None else None,
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                )
            )
        elif isinstance(item, OpenAIFunctionToolOutput):
            if not isinstance(item.output, str):
                raise BadInputError("Responses function-call output must be text.")
            if item.status is not None:
                raise BadInputError(
                    "Responses function-call output status has no IR representation."
                )
            items.append(
                FunctionCallResultItem(id=item.id, call_id=item.call_id, output=item.output)
            )
        elif isinstance(item, OpenAIWebSearchToolCall):
            items.append(_web_search_call_to_ir(item))
        elif isinstance(item, OpenAICodeToolCall):
            items.append(_code_interpreter_call_to_ir(item))
        elif isinstance(item, OpenAIImageGenerationCallInput):
            input_fields = {"result", "status"}
            items.append(
                ImageGenerationCallRequestItem(
                    id=item.id,
                    result=item.result,
                    status=item.status,
                    fields_present=frozenset(item.model_fields_set & input_fields),
                )
            )
        else:
            raise BadInputError(
                f"Responses input item {type(item).__name__} is not OpenAI-compatible IR."
            )
    return tuple(items)


def _output_format(request) -> object:
    text = getattr(request, "text", None)
    if getattr(text, "requested_verbosity", None) is not None:
        raise BadInputError("Responses text verbosity has no IR representation.")
    text_format = getattr(text, "format", None)
    if isinstance(text_format, OpenAIJsonObjectFormat):
        return JsonObjectOutputFormat()
    if isinstance(text_format, OpenAIJsonSchemaFormat):
        return JsonSchemaOutputFormat(
            name=text_format.name,
            schema=text_format.schema_,
            description=text_format.description,
            strict=text_format.strict,
        )
    return TextOutputFormat()


def _tool_choice(choice: Any):
    if choice is None or isinstance(choice, str):
        return choice
    if isinstance(choice, OpenAIFunctionToolChoice):
        return FunctionToolChoice(name=choice.name)
    if isinstance(choice, OpenAIAllowedToolsChoice):
        if all(isinstance(tool, OpenAIFunctionToolChoice) for tool in choice.tools):
            return AllowedFunctionTools(
                mode=choice.mode,
                tools=tuple(FunctionToolChoice(name=tool.name) for tool in choice.tools),
            )
    payload = choice.model_dump(mode="python", exclude_unset=True)
    return OpenAIHostedToolChoice(
        choice_type=str(payload.get("type") or "unknown"),
        payload_json=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )


def _state(request, model: str) -> StateReferences | None:
    provider = model.split("/", 1)[0]
    values = []
    previous_response_id = getattr(request, "previous_response_id", None)
    if previous_response_id is not None:
        values.append(
            StateReference(
                id=previous_response_id,
                provider=provider,
                kind="response",
                value=previous_response_id,
            )
        )
    conversation = getattr(request, "conversation", None)
    if conversation is not None:
        conversation_id = conversation if isinstance(conversation, str) else conversation.id
        values.append(
            StateReference(
                id=conversation_id,
                provider=provider,
                kind="conversation",
                value=conversation_id,
            )
        )
    return StateReferences(provider=provider, values=tuple(values)) if values else None


def _responses_request_to_ir(request) -> InputIR:
    unsupported_fields = (
        "background",
        "extra_body",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "service_tier",
        "truncation",
    )
    supplied = [
        field
        for field in unsupported_fields
        if field in request.model_fields_set and getattr(request, field, None) is not None
    ]
    if supplied:
        raise BadInputError(
            "Responses request fields without an IR representation are unsupported: "
            + ", ".join(supplied)
            + "."
        )
    tools = []
    for tool in request.tools or ():
        if isinstance(tool, OpenAIResponsesFunctionTool):
            tools.append(
                FunctionToolDefinition(
                    name=tool.name,
                    parameters=tool.parameters,
                    description=tool.description,
                    strict=tool.strict,
                )
            )
        else:
            payload = tool.model_dump(mode="python", exclude_unset=True)
            tools.append(
                OpenAIHostedToolDefinition(
                    tool_type=str(payload["type"]),
                    payload_json=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                )
            )

    reasoning = getattr(request, "reasoning", None)
    return InputIR(
        model=request.model,
        stream=request.stream,
        stream_options=(
            StreamOptions.model_validate(
                request.stream_options.model_dump(mode="python", exclude_unset=True)
            )
            if request.stream_options is not None
            else None
        ),
        store=request.store,
        instructions=request.instructions,
        include=tuple(request.include) if request.include is not None else None,
        items=_input_items_to_ir(request.input),
        output_format=_output_format(request),
        generation=GenerationConfig(
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            top_logprobs=request.top_logprobs,
            reasoning_effort=reasoning.effort if reasoning is not None else None,
            reasoning_summary=reasoning.summary if reasoning is not None else None,
            parallel_tool_calls=request.parallel_tool_calls,
            max_tool_calls=request.max_tool_calls,
        ),
        tools=tuple(tools),
        tool_choice=_tool_choice(request.tool_choice),
        state=_state(request, request.model),
    )
