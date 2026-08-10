# adapter.normalize_request(...)
# adapter.project_event(...)
# adapter.project_response(...)

# client.events(...)
from openai.types.responses.response_input_param import (
    EasyInputMessageParam,
    FunctionCallOutput,
    Message,
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)

from molto.types.normalized import NormalizedRequest, SystemPrompt, TextInput, TextOutput
from molto.types.openai_responses import OpenAIResponsesCreateRequest


class OpenAIResponsesAdapter:
    def __init__(self):
        pass

    def normalize_request(self, request: OpenAIResponsesCreateRequest):
        norm_fields = NormalizedRequest.model_fields.keys()
        extra_fields = {k: v for k, v in request.model_dump().items() if k not in norm_fields}
        # Process inputs
        from typing import Union

        __ = Union[
            EasyInputMessageParam,
            Message,
            ResponseOutputMessageParam,
            # ResponseFileSearchToolCallParam,
            # ResponseComputerToolCallParam,
            # ComputerCallOutput,
            # ResponseFunctionWebSearchParam,
            ResponseFunctionToolCallParam,
            FunctionCallOutput,
            # ToolSearchCall,
            # ResponseToolSearchOutputItemParamParam,
            # AdditionalTools,
            ResponseReasoningItemParam,
            # ResponseCompactionItemParamParam,
            # ImageGenerationCall,
            # ResponseCodeInterpreterToolCallParam,
            # LocalShellCall,
            # LocalShellCallOutput,
            # ShellCall,
            # ShellCallOutput,
            # ApplyPatchCall,
            # ApplyPatchCallOutput,
            # McpListTools,
            # McpApprovalRequest,
            # McpApprovalResponse,
            # McpCall,
            # ResponseCustomToolCallOutputParam,
            # ResponseCustomToolCallParam,
            # CompactionTrigger,
            # ItemReference,
            # Program,
            # ProgramOutput,
        ]
        normed_input = None
        if isinstance(request.input, str):
            normed_input = request.input
        elif isinstance(request.input, list):
            normed_input = []
            for i in request.input:
                if i.type == "message":
                    if i.role == "assistant":
                        normed_input.append(TextOutput(content=i.content))
                    elif i.role == "user":
                        if isinstance(i.content, str):
                            normed_input.append(TextInput(content=i.content))
                        else:
                            for j in i.content:
                                if j.type == "input_text":
                                    normed_input.append(TextInput(content=j.text))
                                else:
                                    raise ValueError(f"Unsupported message type: {j.type}")
                    else:
                        normed_input.append(SystemPrompt(content=i.content))
        return NormalizedRequest(
            input=normed_input,
            max_output_tokens=request.max_output_tokens,
            max_tool_calls=request.max_tool_calls,
            model=request.model,
            parallel_tool_calls=request.parallel_tool_calls,
            reasoning=request.reasoning,
            stream=request.stream,
            stream_options=request.stream_options,
            temperature=request.temperature,
            tool_choice=request.tool_choice,
            tools=request.tools,
            top_p=request.top_p,
            extra_fields=extra_fields,
        )
