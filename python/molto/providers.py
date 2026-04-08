# Providers will map the intermediate representation to Pydantic-AI's various settings
import json
from collections.abc import AsyncIterator
from typing import Any

from httpx import AsyncClient
from pydantic_ai import (
    DeferredToolRequests,
    ExternalToolset,
    ModelMessage,
    ModelSettings,
    RunContext,
    ToolDefinition,
    ToolReturnPart,
    UsageLimits,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets.abstract import AbstractToolset

from molto.configs import ENV_CONFIG
from molto.runner import AgentSettings, RunSettings
from molto.types.ir import (
    AgentError,
    AgentMessageDelta,
    AgentMessageEnd,
    AgentMessageStart,
    AgentRunResult,
    AgentThoughtDelta,
    AgentThoughtEnd,
    AgentThoughtStart,
    AgentToolDelta,
    AgentToolEnd,
    AgentToolStart,
    AllowedToolsChoice,
    BuiltinToolChoice,
    CodeInterpreterTool,
    FunctionTool,
    FunctionToolChoice,
    InputIR,
    JsonSchemaFormat,
    McpTool,
    McpToolChoice,
    OutputIR,
    WebSearchTool,
)
from molto.types.openai.responses import (
    OpenAIFunctionToolCall,
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIReasoningContent,
    OpenAIReasoningItem,
    OpenAIResponsesResponse,
    OpenAIResponsesResponseError,
    OpenAIResponsesStream,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamPart,
    OpenAIResponsesStreamText,
)
from molto.utils import uuid7_str
from molto.utils.exceptions import BadInputError

# if TYPE_CHECKING:
#     from molto.mcp.runtime_client import BuiltinMcpRuntimeClient


class OpenAIResponsesRequiredToolChoiceCapability(AbstractCapability[None]):
    def get_model_settings(self):
        def check_messages_for_tool_calls(messages: list[ModelMessage]):
            is_tool_called = False
            for message in messages:
                for part in message.parts:
                    if isinstance(part, ToolReturnPart) and part.outcome == "success":
                        is_tool_called = True
            return is_tool_called

        def check_tool_calls(ctx: RunContext[None]):
            if not ctx.model_settings:
                ctx.model_settings = ModelSettings()
            if ctx.usage.tool_calls > 0 or check_messages_for_tool_calls(ctx.messages):
                # If tool calls exist, remove tool_choice if extra_body exists
                if ctx.model_settings.get("extra_body"):
                    ctx.model_settings["extra_body"].pop("tool_choice", None)
                return ctx.model_settings

            # If no tool calls, ensure tool_choice is set to "required"
            if ctx.model_settings.get("extra_body") is None:
                ctx.model_settings["extra_body"] = {}

            ctx.model_settings["extra_body"]["tool_choice"] = "required"
            return ctx.model_settings

        return check_tool_calls


class _Provider:
    async def parse_agent_settings(self, ir) -> AgentSettings:
        """Parse input IR into pydantic-ai agent settings."""
        raise NotImplementedError

    async def parse_run_settings(self, ir) -> RunSettings:
        """Parse input IR into pydantic-ai run settings."""
        raise NotImplementedError

    async def compose_outputs(
        self,
        output_stream: AsyncIterator[OutputIR],
    ):
        """Compose vendor-specific streaming response from Output IR stream."""
        raise NotImplementedError

    async def get_output(
        self,
        output_stream: AsyncIterator[OutputIR],
    ):
        """Compose vendor-specific response from Output IR stream."""
        raise NotImplementedError


class OpenAIResponsesProvider(_Provider):
    def __init__(self, ir: InputIR | None = None) -> None:
        self._ir = ir

    async def parse_agent_settings(
        self, ir: InputIR, http_client: AsyncClient | None = None
    ) -> AgentSettings:
        """Parse input IR into pydantic-ai agent settings."""
        model = OpenAIResponsesModel(
            model_name=ir.model.split("/")[-1],
            provider=OpenAIProvider(
                api_key=ENV_CONFIG.openai_api_key_plain,
                http_client=http_client,
            ),
        )
        model_settings = OpenAIResponsesModelSettings(
            temperature=ir.temperature,
            top_p=ir.top_p,
            max_tokens=ir.max_output_tokens,
            openai_reasoning_effort=ir.reasoning_effort,
            openai_reasoning_summary=ir.reasoning_summary,
            openai_logprobs=True if ir.top_logprobs else False,
            openai_top_logprobs=ir.top_logprobs,
            parallel_tool_calls=ir.parallel_tool_calls,
            openai_store=ir.store,
            openai_previous_response_id=ir.previous_response_id,
        )
        return AgentSettings(
            model=model,
            model_settings=model_settings,
            output_type=dict[str, Any]
            if isinstance(ir.text_format, JsonSchemaFormat)
            else [str, DeferredToolRequests],
            capabilities=[OpenAIResponsesRequiredToolChoiceCapability()]
            if ir.tool_choice == "required"
            else None,
        )

    async def parse_run_settings(self, ir: InputIR) -> RunSettings:
        # TODO: (Amer) improve this
        """Parse input IR into pydantic-ai run settings."""
        # Build function tool definitions, keyed by name for filtering
        function_tool_defs: dict[str, ToolDefinition] = {}
        if ir.tools:
            for t in ir.tools:
                if isinstance(t, FunctionTool):
                    function_tool_defs[t.name] = ToolDefinition(
                        name=t.name,
                        description=t.description or "",
                        parameters_json_schema=t.parameters,
                        strict=t.strict,
                    )
                elif isinstance(t, CodeInterpreterTool):
                    pass  # TODO: wire builtin code interpreter
                elif isinstance(t, WebSearchTool):
                    pass  # TODO: wire builtin web search
                elif isinstance(t, McpTool):
                    pass  # TODO: wire MCP toolset

        # Apply tool_choice: filter tools and build an instruction override when forcing a call
        tc = ir.tool_choice
        instruction_override: str | None = None

        if tc == "none":
            function_tool_defs.clear()
        elif tc == "auto":
            pass
        elif tc == "required":
            # (Amer) I added a capability that will handle this.
            pass
        elif isinstance(tc, BuiltinToolChoice):
            function_tool_defs.clear()
            # TODO: activate the specific builtin
            instruction_override = (
                f"You must call the `{tc.name}` tool before producing the final answer. "
                "Do not answer directly without calling it."
            )
        elif isinstance(tc, FunctionToolChoice):
            selected = function_tool_defs.get(tc.name)
            if selected is None:
                raise BadInputError(
                    f"`tool_choice.name` {tc.name!r} is not present in effective tools."
                )
            function_tool_defs = {tc.name: selected}
            instruction_override = (
                f"You must call the function `{tc.name}` before producing the final answer. "
                "Do not answer directly without calling it."
            )
        elif isinstance(tc, McpToolChoice):
            function_tool_defs.clear()
            # TODO: activate the specific MCP tool
            if tc.name is None:
                instruction_override = (
                    f"You must call a tool from the MCP server `{tc.server_label}` before "
                    "producing the final answer. Do not answer directly without a tool call."
                )
            else:
                instruction_override = (
                    f"You must call the MCP tool `{tc.server_label}:{tc.name}` before producing "
                    "the final answer. Do not answer directly without calling it."
                )
        elif isinstance(tc, AllowedToolsChoice):
            allowed_function_names = {
                e.name for e in tc.tools if isinstance(e, FunctionToolChoice)
            }
            function_tool_defs = {
                k: v for k, v in function_tool_defs.items() if k in allowed_function_names
            }
            # TODO: filter builtins and MCP tools to allowed set
            if tc.mode == "required":
                instruction_override = (
                    "You must call at least one of the allowed tools before producing the final "
                    "answer. Do not answer directly without a tool call."
                )

        # Assemble toolsets
        toolsets: list[AbstractToolset[Any]] = []
        if function_tool_defs:
            toolsets.append(ExternalToolset(tool_defs=list(function_tool_defs.values())))

        # Merge user instructions with any tool-choice instruction override
        instructions: str | None
        if not instruction_override:
            instructions = ir.instructions
        elif not ir.instructions:
            instructions = instruction_override
        else:
            instructions = f"{ir.instructions.rstrip()}\n\n{instruction_override}"

        return RunSettings(
            message_history=ir.message_history,
            instructions=instructions,
            toolsets=toolsets or None,
            usage_limits=UsageLimits(tool_calls_limit=None),
        )

    async def compose_outputs(
        self,
        output_stream: AsyncIterator[OutputIR],
    ):
        """Yield OpenAI Responses streaming SSE events from the Output IR stream."""
        assert self._ir is not None, "compose_outputs requires ir set on provider"
        ir = self._ir
        seq = 0
        output_index = 0
        output_items: list = []
        error: OpenAIResponsesResponseError | None = None

        # Per-tool-call streaming state keyed by tool_call_id
        active_tools: dict[str, dict] = {}

        # Text streaming state
        current_text_id: str | None = None
        current_text = ""

        # Reasoning streaming state
        current_reasoning_id: str | None = None
        current_reasoning = ""

        initial_response = OpenAIResponsesResponse(
            model=ir.model.split("/")[-1],
            instructions=ir.instructions,
            status="in_progress",
        )
        yield OpenAIResponsesStream(
            type="response.created",
            response=initial_response,
            sequence_number=seq,
        )
        seq += 1
        yield OpenAIResponsesStream(
            type="response.in_progress",
            response=initial_response,
            sequence_number=seq,
        )
        seq += 1

        async for event in output_stream:
            if isinstance(event, AgentMessageStart):
                current_text = ""
                current_text_id = uuid7_str("msg_")
                yield OpenAIResponsesStreamOutput(
                    type="response.output_item.added",
                    item=OpenAIOutputItem(id=current_text_id, content=[], status="in_progress"),
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1
                yield OpenAIResponsesStreamPart(
                    type="response.content_part.added",
                    part=OpenAIOutputTextContent(text=""),
                    item_id=current_text_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1

            elif isinstance(event, AgentMessageDelta):
                current_text += event.content
                yield OpenAIResponsesStreamText(
                    type="response.output_text.delta",
                    delta=event.content,
                    item_id=current_text_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1

            elif isinstance(event, AgentMessageEnd):
                yield OpenAIResponsesStreamText(
                    type="response.output_text.done",
                    text=current_text,
                    item_id=current_text_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1
                yield OpenAIResponsesStreamPart(
                    type="response.content_part.done",
                    part=OpenAIOutputTextContent(text=current_text),
                    item_id=current_text_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1
                completed_item = OpenAIOutputItem(
                    id=current_text_id,
                    content=[OpenAIOutputTextContent(text=current_text)],
                    status="completed",
                )
                output_items.append(completed_item)
                yield OpenAIResponsesStreamOutput(
                    type="response.output_item.done",
                    item=completed_item,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1
                output_index += 1

            elif isinstance(event, AgentThoughtStart):
                current_reasoning = ""
                current_reasoning_id = uuid7_str("rs_")
                yield OpenAIResponsesStreamOutput(
                    type="response.output_item.added",
                    item=OpenAIReasoningItem(id=current_reasoning_id, status="in_progress"),
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1

            elif isinstance(event, AgentThoughtDelta):
                current_reasoning += event.content
                yield OpenAIResponsesStreamText(
                    type="response.reasoning.delta",
                    delta=event.content,
                    item_id=current_reasoning_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1

            elif isinstance(event, AgentThoughtEnd):
                yield OpenAIResponsesStreamText(
                    type="response.reasoning.done",
                    text=current_reasoning,
                    item_id=current_reasoning_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1
                completed_item = OpenAIReasoningItem(
                    id=current_reasoning_id,
                    content=(
                        [OpenAIReasoningContent(text=current_reasoning)]
                        if current_reasoning
                        else []
                    ),
                    status="completed",
                )
                output_items.append(completed_item)
                yield OpenAIResponsesStreamOutput(
                    type="response.output_item.done",
                    item=completed_item,
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1
                output_index += 1

            elif isinstance(event, AgentToolStart):
                item_id = uuid7_str("fc_")
                active_tools[event.tool_call_id] = {
                    "id": item_id,
                    "tool_name": event.tool_name,
                    "args_buf": "",
                    "output_index": output_index,
                }
                yield OpenAIResponsesStreamOutput(
                    type="response.output_item.added",
                    item=OpenAIFunctionToolCall(
                        id=item_id,
                        call_id=event.tool_call_id,
                        name=event.tool_name,
                        arguments="",
                        status="in_progress",
                    ),
                    output_index=output_index,
                    sequence_number=seq,
                )
                seq += 1
                output_index += 1

            elif isinstance(event, AgentToolDelta):
                tc = active_tools.get(event.tool_call_id)
                if tc:
                    tc["args_buf"] += event.args
                    yield OpenAIResponsesStreamText(
                        type="response.function_call_arguments.delta",
                        delta=event.args,
                        item_id=tc["id"],
                        output_index=tc["output_index"],
                        sequence_number=seq,
                    )
                    seq += 1

            elif isinstance(event, AgentToolEnd):
                tc = active_tools.get(event.tool_call_id)
                if tc:
                    yield OpenAIResponsesStreamText(
                        type="response.function_call_arguments.done",
                        arguments=tc["args_buf"],
                        item_id=tc["id"],
                        output_index=tc["output_index"],
                        sequence_number=seq,
                    )
                    seq += 1

            elif isinstance(event, AgentRunResult):
                for tool_call in event.tool_calls or []:
                    args_str = json.dumps(
                        tool_call.args, ensure_ascii=False, separators=(",", ":")
                    )
                    tc = active_tools.pop(tool_call.tool_call_id, None)
                    item_id = tc["id"] if tc else uuid7_str("fc_")
                    item_output_index = tc["output_index"] if tc else output_index
                    completed_item = OpenAIFunctionToolCall(
                        id=item_id,
                        call_id=tool_call.tool_call_id,
                        name=tool_call.tool_name,
                        arguments=args_str,
                        status="completed",
                    )
                    output_items.append(completed_item)
                    yield OpenAIResponsesStreamOutput(
                        type="response.output_item.done",
                        item=completed_item,
                        output_index=item_output_index,
                        sequence_number=seq,
                    )
                    seq += 1

            elif isinstance(event, AgentError):
                error = OpenAIResponsesResponseError(
                    code="server_error",
                    message=event.error,
                )

        final_response = OpenAIResponsesResponse(
            model=ir.model.split("/")[-1],
            instructions=ir.instructions,
            status="completed" if error is None else "failed",
            output=output_items,
            error=error,
        )
        yield OpenAIResponsesStream(
            type="response.completed" if error is None else "response.failed",
            response=final_response,
            sequence_number=seq,
        )

    async def get_output(
        self,
        output_stream: AsyncIterator[OutputIR],
    ) -> OpenAIResponsesResponse:
        """Return a single OpenAI Responses response by parsing the terminal AgentRunResult event."""
        ir = self._ir
        response = None
        async for event in output_stream:
            if isinstance(event, AgentRunResult):
                output_items = []
                for tool_call in event.tool_calls or []:
                    output_items.append(
                        OpenAIFunctionToolCall(
                            id=uuid7_str("fc_"),
                            call_id=tool_call.tool_call_id,
                            name=tool_call.tool_name,
                            arguments=json.dumps(
                                tool_call.args, ensure_ascii=False, separators=(",", ":")
                            ),
                            status="completed",
                        )
                    )
                if event.message:
                    output_items.append(
                        OpenAIOutputItem(
                            id=uuid7_str("msg_"),
                            content=[OpenAIOutputTextContent(text=event.message)],
                            status="completed",
                        )
                    )
                error = (
                    OpenAIResponsesResponseError(code="server_error", message=event.error)
                    if event.error
                    else None
                )
                response = OpenAIResponsesResponse(
                    model=ir.model.split("/")[-1],
                    instructions=ir.instructions,
                    status="completed" if error is None else "failed",
                    output=output_items,
                    error=error,
                )
        # This is done to properly exit the async generator
        if response is not None:
            return response
        return OpenAIResponsesResponse(
            model=ir.model.split("/")[-1],
            instructions=ir.instructions,
            status="completed",
            output=[],
        )


class OpenAIChatCompletionsProvider(_Provider):
    pass


#     def parse_agent_settings(self) -> AgentSettings:
#         """
#         These parameters are not supported by Responses API:
#         - id
#         - n
#         - presence_penalty
#         - frequency_penalty
#         - logit_bias
#         - stop / stop_sequences
#         - seed

#         These parameters are not supported by vLLM:
#         - user / openai_user
#         - service_tier / openai_service_tier
#         - prediction / openai_prediction
#         - safety_identifier
#         - prompt_cache_key / openai_prompt_cache_key
#         - prompt_cache_retention / openai_prompt_cache_retention

#         Unused OpenAIChatModelSettings fields:
#         - timeout: float | Timeout
#         - extra_headers: dict[str, str]
#         - extra_body: object

#         vLLMResponses fields that are mapped in `as_run_settings`:
#         - input
#         - instructions
#         - tools
#         - tool_choice
#         - previous_response_id

#         vLLMResponses fields that are used when creating Agent:
#         - model

#         vLLMResponses fields that are not used:
#         - metadata
#         """
#         kwargs: dict[str, Any] = self.model_dump(exclude_unset=True, exclude_none=True)

#         max_output_tokens = kwargs.pop("max_output_tokens", None)
#         if max_output_tokens:
#             kwargs["max_tokens"] = max_output_tokens
#         reasoning_effort = kwargs.pop("reasoning", {}).get("effort")
#         if reasoning_effort:
#             kwargs["openai_reasoning_effort"] = reasoning_effort
#         top_logprobs = kwargs.pop("top_logprobs", None)
#         if top_logprobs:
#             kwargs["openai_logprobs"] = True
#             kwargs["openai_top_logprobs"] = top_logprobs
#         else:
#             kwargs["openai_logprobs"] = False

#         return OpenAIChatModelSettings(**kwargs)

#     async def parse_run_settings(self, ir) -> RunSettings:
#         """
#         Converts the request into a dictionary of run settings for Pydantic AI.

#         Notes:

#         If system prompt is "A" and instructions is "B", then pydantic ai will form this history:
#         ```yaml
#         messages:
#         - content: A
#         role: system
#         - content: B
#         role: system
#         - content: Hi.
#         role: user
#         ```
#         """
#         # mcp_tool_name_map: dict[str, McpToolRef] = {}
#         # mcp_internal_name_by_ref: dict[McpToolRef, str] = {}

#         # def _register_internal_mcp_tool(ref: McpToolRef) -> str:
#         #     existing_name = mcp_internal_name_by_ref.get(ref)
#         #     if existing_name is not None:
#         #         return existing_name
#         #     try:
#         #         internal_tool_name = build_internal_mcp_tool_name(
#         #             server_label=ref.server_label,
#         #             tool_name=ref.tool_name,
#         #             existing_map=mcp_tool_name_map,
#         #         )
#         #     except ValueError as exc:
#         #         raise BadInputError(str(exc)) from exc
#         #     mcp_tool_name_map[internal_tool_name] = ref
#         #     mcp_internal_name_by_ref[ref] = internal_tool_name
#         #     return internal_tool_name

#         # if not isinstance(self.input, str):
#         #     historical_mcp_refs = sorted(
#         #         {
#         #             McpToolRef(server_label=msg.server_label, tool_name=msg.name)
#         #             for msg in self.input
#         #             if isinstance(msg, OpenAIMcpToolCall)
#         #         },
#         #         key=lambda ref: (ref.server_label, ref.tool_name),
#         #     )
#         #     for ref in historical_mcp_refs:
#         #         _register_internal_mcp_tool(ref)

#         # Process chat history
#         # `previous_response_id` hydration is implemented in Layer 4 (ResponseStore). This method expects
#         # `self.input` to already include any prior context/tool outputs that should be part of the upstream prompt.
#         if isinstance(self.input, str):
#             message_history = [
#                 ModelRequest(parts=[UserPromptPart(content=self.input)]),
#             ]
#         else:
#             message_history = []
#             for msg in self.input:
#                 if isinstance(msg, (OpenAIInputMessage, OpenAIInputItem)):
#                     if msg.role in ("system", "developer"):
#                         if not isinstance(msg.content, str):
#                             # NOTE: Can system prompt be non-string?
#                             raise BadInputError("System prompt must be a string.")
#                         message_history.append(
#                             ModelRequest(parts=[SystemPromptPart(content=msg.content)])
#                         )
#                         continue

#                     if msg.role == "user":
#                         if isinstance(msg.content, str):
#                             message_history.append(
#                                 ModelRequest(parts=[UserPromptPart(content=msg.content)])
#                             )
#                         else:
#                             message_history.append(
#                                 ModelRequest(
#                                     parts=[
#                                         UserPromptPart(
#                                             content=[
#                                                 c.text
#                                                 if isinstance(c, OpenAITextContent)
#                                                 else ImageUrl(
#                                                     url=c.image_url,
#                                                     vendor_metadata=dict(detail=c.detail),
#                                                 )
#                                                 for c in msg.content
#                                             ]
#                                         )
#                                     ]
#                                 )
#                             )
#                         continue

#                     if msg.role == "assistant":
#                         if not isinstance(msg.content, str):
#                             raise BadInputError(
#                                 "Assistant message input must be a string. "
#                                 "To replay assistant outputs, use `OpenAIOutputItem`."
#                             )
#                         message_history.append(
#                             ModelResponse(
#                                 parts=[TextPart(content=msg.content, id=None)],
#                                 # NOTE: Do we need to provide the provider name?
#                                 # provider_name="openai",
#                             )
#                         )
#                         continue

#                     raise BadInputError(f"Invalid message role: {msg.role}")

#                 if isinstance(msg, OpenAIOutputItem):
#                     msg_id = msg.id
#                     for content_item in msg.content:
#                         if isinstance(content_item, OpenAIOutputTextContent):
#                             message_history.append(
#                                 ModelResponse(
#                                     parts=[TextPart(content=content_item.text, id=msg_id)],
#                                     # provider_name="openai",
#                                 )
#                             )
#                         elif isinstance(content_item, OpenAIOutputRefusal):
#                             message_history.append(
#                                 ModelResponse(
#                                     parts=[TextPart(content=content_item.refusal, id=msg_id)],
#                                     # provider_name="openai",
#                                 )
#                             )
#                         else:
#                             raise BadInputError(
#                                 f"Invalid assistant content type: {type(content_item)}"
#                             )
#                     continue

#                 if isinstance(msg, OpenAIReasoningItem):
#                     msg_id = msg.id
#                     content = ""
#                     if msg.content:
#                         content = "".join(c.text for c in msg.content)
#                     if (not msg.encrypted_content) and msg.summary:
#                         content = "".join(c.text for c in msg.summary)
#                     message_history.append(
#                         ModelResponse(
#                             parts=[
#                                 ThinkingPart(
#                                     content=content,
#                                     signature=msg.encrypted_content,
#                                     id=msg_id,
#                                     # provider_name="openai",
#                                 )
#                             ],
#                             # provider_name="openai",
#                         )
#                     )
#                     continue

#                 if isinstance(msg, OpenAIFunctionToolCall):
#                     msg_id = msg.id
#                     message_history.append(
#                         ModelResponse(
#                             parts=[
#                                 ToolCallPart(
#                                     tool_name=msg.name,
#                                     args=msg.arguments,
#                                     tool_call_id=msg.call_id,
#                                     id=msg_id,
#                                 )
#                             ],
#                             # provider_name="openai",
#                         )
#                     )
#                     continue

#                 if isinstance(msg, OpenAIFunctionToolOutput):
#                     tool_name: str = next(
#                         (
#                             i.name
#                             for i in self.input
#                             if isinstance(i, OpenAIFunctionToolCall) and i.call_id == msg.call_id
#                         ),
#                         "",
#                     )
#                     tool_output: str
#                     if isinstance(msg.output, str):
#                         tool_output = msg.output
#                     else:
#                         tool_output = json.dumps(
#                             [c.model_dump(mode="python") for c in msg.output],
#                             ensure_ascii=False,
#                             separators=(",", ":"),
#                         )
#                     message_history.append(
#                         ModelRequest(
#                             parts=[
#                                 ToolReturnPart(
#                                     tool_name=tool_name,
#                                     # TODO: Handle list[OpenAITextContent | OpenAIImageContent]
#                                     content=tool_output,
#                                     tool_call_id=msg.call_id,
#                                 ),
#                             ]
#                         )
#                     )
#                     continue

#                 if isinstance(msg, OpenAICodeToolCall):
#                     msg_id = msg.id
#                     message_history.append(
#                         ModelResponse(
#                             parts=[
#                                 BuiltinToolCallPart(
#                                     tool_name="code_execution",
#                                     args={
#                                         "container_id": msg.container_id,
#                                         "code": msg.code,
#                                     },
#                                     tool_call_id=msg_id,
#                                     # provider_name="openai",
#                                 ),
#                                 BuiltinToolReturnPart(
#                                     tool_name="code_execution",
#                                     content={"status": "completed"},
#                                     tool_call_id=msg_id,
#                                     # provider_name="openai",
#                                 ),
#                             ],
#                             # provider_name="openai",
#                         )
#                     )
#                     continue

#                 if isinstance(msg, OpenAIWebSearchToolCall):
#                     msg_id = msg.id
#                     action_payload = (
#                         msg.action.model_dump(mode="python", exclude_none=True)
#                         if msg.action is not None
#                         else {}
#                     )
#                     message_history.append(
#                         ModelResponse(
#                             parts=[
#                                 BuiltinToolCallPart(
#                                     tool_name=WEB_SEARCH_TOOL,
#                                     args=action_payload,
#                                     tool_call_id=msg_id,
#                                 ),
#                                 BuiltinToolReturnPart(
#                                     tool_name=WEB_SEARCH_TOOL,
#                                     content={"status": msg.status, "action": action_payload},
#                                     tool_call_id=msg_id,
#                                 ),
#                             ]
#                         )
#                     )
#                     continue

#                 # if isinstance(msg, OpenAIMcpToolCall):
#                 #     msg_id = msg.id
#                 #     internal_tool_name = _register_internal_mcp_tool(
#                 #         McpToolRef(server_label=msg.server_label, tool_name=msg.name)
#                 #     )
#                 #     mcp_args: dict[str, Any] = {}
#                 #     try:
#                 #         parsed_args = json.loads(msg.arguments)
#                 #         if isinstance(parsed_args, dict):
#                 #             mcp_args = parsed_args
#                 #     except Exception:
#                 #         mcp_args = {}
#                 #     tool_return_content: dict[str, Any] = {"status": msg.status}
#                 #     if msg.output is not None:
#                 #         tool_return_content["output"] = msg.output
#                 #     if msg.error is not None:
#                 #         tool_return_content["error"] = msg.error
#                 #     message_history.append(
#                 #         ModelResponse(
#                 #             parts=[
#                 #                 BuiltinToolCallPart(
#                 #                     tool_name=internal_tool_name,
#                 #                     args=mcp_args,
#                 #                     tool_call_id=msg_id,
#                 #                 ),
#                 #                 BuiltinToolReturnPart(
#                 #                     tool_name=internal_tool_name,
#                 #                     content=tool_return_content,
#                 #                     tool_call_id=msg_id,
#                 #                 ),
#                 #             ]
#                 #         )
#                 #     )
#                 #     continue

#                 raise BadInputError(f"Invalid message type: {type(msg)}")

#         # TODO: Re-enable this
#         # Process tool list
#         # builtin_tools: list[Tool] = []
#         # deferred_tools: list[ToolDefinition] = []
#         # mcp_declarations: dict[str, OpenAIResponsesMcpTool] = {}
#         # mcp_servers: dict[str, ResolvedMcpServerTools] = {}
#         # selected_mcp_tool_infos_by_server: dict[str, dict[str, McpToolInfo]] = {}
#         # if self.tools:
#         #     for t in self.tools:
#         #         match t:
#         #             case OpenAIResponsesFunctionTool():
#         #                 if t.name.startswith(HOSTED_MCP_INTERNAL_PREFIX):
#         #                     raise BadInputError(
#         #                         f"Function tool names starting with {HOSTED_MCP_INTERNAL_PREFIX!r} are reserved."
#         #                     )
#         #                 deferred_tools.append(
#         #                     ToolDefinition(
#         #                         name=t.name,
#         #                         parameters_json_schema=t.parameters,
#         #                         strict=t.strict,
#         #                         description=t.description,
#         #                     )
#         #                 )
#         #             case vLLMResponsesCodeTool():
#         #                 builtin_tools.append(TOOLS[CODE_INTERPRETER_TOOL])
#         #             case OpenAIResponsesWebSearchTool():
#         #                 builtin_tools.append(TOOLS[WEB_SEARCH_TOOL])
#         #             case OpenAIResponsesMcpTool():
#         #                 if t.server_label in mcp_declarations:
#         #                     raise BadInputError(
#         #                         "Duplicate MCP declarations are not allowed for the same "
#         #                         f"`server_label`: {t.server_label!r}."
#         #                     )
#         #                 mcp_declarations[t.server_label] = t
#         #             case _:
#         #                 raise BadInputError(f"Invalid tool type: {type(t)}")

#         # if mcp_declarations:
#         #     mcp_servers = await resolve_mcp_declarations(
#         #         declarations=mcp_declarations,
#         #         builtin_mcp_runtime_client=builtin_mcp_runtime_client,
#         #         request_remote_enabled=request_remote_enabled,
#         #         request_remote_url_checks_enabled=request_remote_url_checks_enabled,
#         #         request_remote_toolset_builder=build_request_remote_toolset,
#         #     )

#         internal_tool_choice_instruction: str | None = None

#         if self.tool_choice == "auto":
#             pass
#         elif self.tool_choice == "none":
#             builtin_tools = []
#             deferred_tools = []
#             mcp_servers = {}
#             selected_mcp_tool_infos_by_server = {}
#         elif self.tool_choice == "required":
#             if not builtin_tools and not deferred_tools and not mcp_servers:
#                 raise BadInputError(
#                     '`tool_choice="required"` requires at least one effective tool.'
#                 )
#             internal_tool_choice_instruction = _build_required_tool_choice_instruction(kind="any")
#         elif isinstance(self.tool_choice, OpenAIAllowedToolsChoice):
#             # TODO: Re-enable
#             # (
#             #     builtin_tools,
#             #     deferred_tools,
#             #     mcp_servers,
#             #     selected_mcp_tool_infos_by_server,
#             # ) = _apply_allowed_tools_choice(
#             #     choice=self.tool_choice,
#             #     builtin_tools=builtin_tools,
#             #     deferred_tools=deferred_tools,
#             #     mcp_servers=mcp_servers,
#             # )
#             if self.tool_choice.mode == "auto":
#                 pass
#             else:
#                 if not builtin_tools and not deferred_tools and not mcp_servers:
#                     raise BadInputError(
#                         '`tool_choice.allowed_tools` with `mode="required"` '
#                         "requires at least one effective tool."
#                     )
#                 internal_tool_choice_instruction = _build_required_tool_choice_instruction(
#                     kind="allowed"
#                 )
#         elif isinstance(self.tool_choice, OpenAIHostedToolChoice):
#             requested_builtin = _normalize_hosted_tool_type(self.tool_choice.type)
#             builtin_tools = [t for t in builtin_tools if t.name == requested_builtin]
#             if not builtin_tools:
#                 raise BadInputError(
#                     f"`tool_choice.type={self.tool_choice.type!r}` references a built-in tool "
#                     "not present in effective tools."
#                 )
#             deferred_tools = []
#             mcp_servers = {}
#             selected_mcp_tool_infos_by_server = {}
#             internal_tool_choice_instruction = _build_required_tool_choice_instruction(
#                 kind="builtin",
#                 builtin_name=requested_builtin,
#             )
#         elif isinstance(self.tool_choice, OpenAIFunctionToolChoice):
#             builtin_tools = []
#             deferred_tools = [t for t in deferred_tools if t.name == self.tool_choice.name]
#             if not deferred_tools:
#                 raise BadInputError(
#                     f"`tool_choice.name` {self.tool_choice.name!r} is not present in effective tools."
#                 )
#             mcp_servers = {}
#             selected_mcp_tool_infos_by_server = {}
#             internal_tool_choice_instruction = _build_required_tool_choice_instruction(
#                 kind="function",
#                 function_name=self.tool_choice.name,
#             )
#         elif isinstance(self.tool_choice, OpenAIMcpToolChoice):
#             if not mcp_servers:
#                 raise BadInputError(
#                     '`tool_choice.type="mcp"` requires at least one MCP tool declaration.'
#                 )
#             selected_server = mcp_servers.get(self.tool_choice.server_label)
#             if selected_server is None:
#                 raise BadInputError(
#                     '`tool_choice.type="mcp"` references a `server_label` '
#                     "not present in effective tools."
#                 )
#             selected_tools = selected_server.allowed_tool_infos
#             if self.tool_choice.name is not None:
#                 if self.tool_choice.name not in selected_tools:
#                     raise BadInputError(
#                         f"`tool_choice.name` {self.tool_choice.name!r} is not allowed for "
#                         f"server {self.tool_choice.server_label!r}."
#                     )
#                 selected_tools = {self.tool_choice.name: selected_tools[self.tool_choice.name]}
#             mcp_servers = {self.tool_choice.server_label: selected_server}
#             selected_mcp_tool_infos_by_server = {self.tool_choice.server_label: selected_tools}  # noqa: F841
#             builtin_tools = []
#             deferred_tools = []
#             internal_tool_choice_instruction = _build_required_tool_choice_instruction(
#                 kind="mcp",
#                 server_label=self.tool_choice.server_label,
#                 mcp_tool_name=self.tool_choice.name,
#             )
#         else:
#             raise BadInputError(f"Invalid `tool_choice` provided: {self.tool_choice}.")

#         # TODO: Re-enable this
#         # mcp_resolved_tools: list[ResolvedMcpTool] = []
#         # for server_label, server_runtime in sorted(mcp_servers.items()):
#         #     tool_infos = selected_mcp_tool_infos_by_server.get(
#         #         server_label, server_runtime.allowed_tool_infos
#         #     )
#         #     for tool_name, tool_info in tool_infos.items():
#         #         normalized_schema = _normalize_mcp_input_schema(
#         #             server_label=server_label,
#         #             tool_name=tool_name,
#         #             input_schema=tool_info.input_schema,
#         #         )
#         #         ref = McpToolRef(
#         #             server_label=server_label,
#         #             tool_name=tool_name,
#         #             mode=server_runtime.mode,
#         #         )
#         #         internal_tool_name = _register_internal_mcp_tool(ref)

#         #         description = (
#         #             tool_info.description.strip()
#         #             if isinstance(tool_info.description, str) and tool_info.description.strip()
#         #             else f"MCP tool {server_label}:{tool_name}"
#         #         )
#         #         mcp_resolved_tools.append(
#         #             ResolvedMcpTool(
#         #                 internal_name=internal_tool_name,
#         #                 ref=ref,
#         #                 mcp_toolset=server_runtime.mcp_toolset,
#         #                 mcp_tool_name=tool_name,
#         #                 description=description,
#         #                 input_schema=normalized_schema,
#         #                 schema_validator=Draft202012Validator(normalized_schema),
#         #                 secret_values=server_runtime.secret_values,
#         #                 mcp_tool=server_runtime.allowed_mcp_tools_by_name.get(tool_name),
#         #             )
#         #         )
#         toolsets: list[AbstractToolset[Any]] = []
#         if builtin_tools:
#             toolsets.append(FunctionToolset(tools=builtin_tools))
#         # TODO: Re-enable this
#         # if mcp_resolved_tools:
#         #     toolsets.append(
#         #         McpGatewayToolset(
#         #             tools=mcp_resolved_tools,
#         #             id="vllm_responses_mcp",
#         #         )
#         #     )
#         if deferred_tools:
#             toolsets.append(ExternalToolset(tool_defs=deferred_tools))
#         run_settings: AgentRunSettings = {
#             "message_history": message_history,
#             "instructions": _merge_internal_instructions(
#                 user_instructions=self.instructions,
#                 internal_instruction=internal_tool_choice_instruction,
#             ),
#             "toolsets": toolsets or None,
#             # Parity note: do not enforce request `max_tool_calls` in pydantic_ai runtime limits.
#             # Keep `max_tool_calls` as request/response metadata only, matching observed OpenAI behavior.
#             # Preserve pydantic_ai default request_limit safety guard for non-terminating loops.
#             "usage_limits": UsageLimits(tool_calls_limit=None),
#         }
#         # return run_settings, builtin_tools, mcp_tool_name_map
#         return run_settings, builtin_tools


class AnthropicMessagesProvider(_Provider):
    pass
