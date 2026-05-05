# Providers will map the intermediate representation to Pydantic-AI's various settings
import json
from collections.abc import AsyncIterator
from typing import Any

from httpx import AsyncClient
from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DeferredToolRequests,
    ExternalToolset,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StructuredDict,
    SystemPromptPart,
    ThinkingPart,
    ToolCallPart,
    ToolDefinition,
    ToolReturnPart,
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai import (
    TextPart as PydanticAITextPart,
)
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.builtin_tools import CodeExecutionTool as OpenAICodeExecutionTool
from pydantic_ai.builtin_tools import WebSearchTool as OpenAIWebSearchTool
from pydantic_ai.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets.abstract import AbstractToolset

from molto.configs import ENV_CONFIG
from molto.runner import AgentSettings, RunSettings
from molto.types.ir import (
    AgentMessage,
    AgentThought,
    AgentToolCalls,
    AgentToolResult,
    AllowedToolsChoice,
    BuiltinToolChoice,
    CodeInterpreterTool,
    FunctionTool,
    FunctionToolChoice,
    Input,
    InputCodeToolCall,
    InputHistoryItem,
    InputMcpToolCall,
    InputWebSearchToolCall,
    JsonSchemaFormat,
    McpTool,
    McpToolChoice,
    OutputIR,
    SystemMessage,
    TextPart,
    UserMessage,
    WebSearchTool,
)
from molto.types.openai.responses import (
    WEB_SEARCH_TOOL,
    from_ir_stream,
)
from molto.utils.exceptions import BadInputError


class _Provider:
    async def parse_agent_settings(
        self, ir, http_client: AsyncClient | None = None
    ) -> AgentSettings:
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
    def __init__(self, ir: Input | None = None) -> None:
        self._ir = ir

    async def parse_agent_settings(
        self, ir: Input, http_client: AsyncClient | None = None
    ) -> AgentSettings:
        """Parse input IR into pydantic-ai agent settings."""
        model = OpenAIResponsesModel(
            model_name=ir.model.split("/")[-1],
            provider=OpenAIProvider(
                api_key=ENV_CONFIG.openai_api_key_plain,
                http_client=http_client,
            ),
        )

        extra_body = ir.extra_body or {}
        tool_choice = ir.tool_choice
        if tool_choice == "auto" or tool_choice == "required":
            extra_body["tool_choice"] = tool_choice
        if ir.safety_identifier is not None:
            extra_body["safety_identifier"] = ir.safety_identifier
        if ir.max_tool_calls is not None:
            extra_body["max_tool_calls"] = ir.max_tool_calls
        if ir.conversation_id is not None:
            extra_body["conversation"] = ir.conversation_id
        include = ir.include or []
        ms_kwargs: dict[str, Any] = {}
        if ir.temperature is not None:
            ms_kwargs["temperature"] = ir.temperature
        if ir.top_p is not None:
            ms_kwargs["top_p"] = ir.top_p
        if ir.max_output_tokens is not None:
            ms_kwargs["max_tokens"] = ir.max_output_tokens
        if ir.reasoning_effort is not None:
            ms_kwargs["openai_reasoning_effort"] = ir.reasoning_effort
        if ir.reasoning_summary is not None:
            ms_kwargs["openai_reasoning_summary"] = ir.reasoning_summary
        if ir.top_logprobs is not None:
            ms_kwargs["openai_logprobs"] = True
            ms_kwargs["openai_top_logprobs"] = ir.top_logprobs
        if ir.parallel_tool_calls is not None:
            ms_kwargs["parallel_tool_calls"] = ir.parallel_tool_calls
        if ir.store is not None:
            ms_kwargs["openai_store"] = ir.store
        if ir.previous_response_id is not None:
            ms_kwargs["openai_previous_response_id"] = ir.previous_response_id
        if ir.service_tier is not None:
            ms_kwargs["openai_service_tier"] = ir.service_tier
        if ir.prompt_cache_key is not None:
            ms_kwargs["openai_prompt_cache_key"] = ir.prompt_cache_key
        if ir.prompt_cache_retention is not None:
            ms_kwargs["openai_prompt_cache_retention"] = ir.prompt_cache_retention
        if ir.truncation is not None:
            ms_kwargs["openai_truncation"] = ir.truncation
        if ir.verbosity is not None:
            ms_kwargs["openai_text_verbosity"] = ir.verbosity
        if include:
            if "web_search_call.action.sources" in include:
                ms_kwargs["openai_include_web_search_sources"] = True
            if "code_interpreter_call.outputs" in include:
                ms_kwargs["openai_include_code_execution_outputs"] = True
            if "file_search_call.results" in include:
                ms_kwargs["openai_include_file_search_results"] = True
        if extra_body:
            ms_kwargs["extra_body"] = extra_body
        model_settings = OpenAIResponsesModelSettings(**ms_kwargs)
        return AgentSettings(
            model=model,
            model_settings=model_settings,
            output_type=StructuredDict(
                ir.text_format.json_schema,
                name=ir.text_format.name,
                description=ir.text_format.description,
            )
            if isinstance(ir.text_format, JsonSchemaFormat)
            else [str, DeferredToolRequests],
        )

    @staticmethod
    def convert_input_to_messages(input: str | list[InputHistoryItem]) -> list[ModelMessage]:
        """Convert IR input history into pydantic-ai ModelMessage objects."""
        if isinstance(input, str):
            return [ModelRequest(parts=[UserPromptPart(content=input)])]

        message_history: list[ModelMessage] = []
        for msg in input:
            if isinstance(msg, SystemMessage):
                message_history.append(ModelRequest(parts=[SystemPromptPart(content=msg.content)]))

            elif isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    message_history.append(
                        ModelRequest(parts=[UserPromptPart(content=msg.content)])
                    )
                else:
                    message_history.append(
                        ModelRequest(
                            parts=[
                                UserPromptPart(
                                    content=[
                                        c.text
                                        if isinstance(c, TextPart)
                                        else ImageUrl(
                                            url=c.url, vendor_metadata={"detail": c.detail}
                                        )
                                        for c in msg.content
                                    ]
                                )
                            ]
                        )
                    )

            elif isinstance(msg, AgentMessage):
                message_history.append(
                    ModelResponse(parts=[PydanticAITextPart(content=msg.content, id=msg.id)])
                )

            elif isinstance(msg, AgentThought):
                message_history.append(
                    ModelResponse(
                        parts=[
                            ThinkingPart(content=msg.content, signature=msg.signature, id=msg.id)
                        ]
                    )
                )

            elif isinstance(msg, AgentToolCalls):
                for tc in msg.tool_calls:
                    message_history.append(
                        ModelResponse(
                            parts=[
                                ToolCallPart(
                                    tool_name=tc.tool_name,
                                    args=json.dumps(tc.args),
                                    tool_call_id=tc.tool_call_id,
                                    id=tc.id,
                                )
                            ]
                        )
                    )

            elif isinstance(msg, AgentToolResult):
                content = msg.result if isinstance(msg.result, str) else json.dumps(msg.result)
                message_history.append(
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name=msg.tool_name,
                                content=content,
                                tool_call_id=msg.tool_call_id,
                            )
                        ]
                    )
                )

            elif isinstance(msg, InputCodeToolCall):
                message_history.append(
                    ModelResponse(
                        parts=[
                            BuiltinToolCallPart(
                                tool_name="code_execution",
                                args={"container_id": msg.container_id, "code": msg.code},
                                tool_call_id=msg.id,
                            ),
                            BuiltinToolReturnPart(
                                tool_name="code_execution",
                                content={"status": "completed"},
                                tool_call_id=msg.id,
                            ),
                        ]
                    )
                )

            elif isinstance(msg, InputWebSearchToolCall):
                action = msg.action or {}
                message_history.append(
                    ModelResponse(
                        parts=[
                            BuiltinToolCallPart(
                                tool_name=WEB_SEARCH_TOOL,
                                args=action,
                                tool_call_id=msg.id,
                            ),
                            BuiltinToolReturnPart(
                                tool_name=WEB_SEARCH_TOOL,
                                content={"status": msg.status, "action": action},
                                tool_call_id=msg.id,
                            ),
                        ]
                    )
                )

            elif isinstance(msg, InputMcpToolCall):
                args = json.loads(msg.arguments) if msg.arguments else {}
                message_history.append(
                    ModelResponse(
                        parts=[
                            BuiltinToolCallPart(
                                tool_name=f"mcp_{msg.server_label}_{msg.name}",
                                args=args,
                                tool_call_id=msg.id,
                            ),
                            BuiltinToolReturnPart(
                                tool_name=f"mcp_{msg.server_label}_{msg.name}",
                                content={
                                    "status": msg.status,
                                    "output": msg.output,
                                    "error": msg.error,
                                },
                                tool_call_id=msg.id,
                            ),
                        ]
                    )
                )

            else:
                raise BadInputError(f"Invalid message type: {type(msg)}")

        return message_history

    async def parse_run_settings(self, ir: Input) -> RunSettings:
        """Parse input IR into pydantic-ai run settings."""
        # Collect function tools keyed by name
        function_tool_defs: dict[str, ToolDefinition] = {}
        # Collect builtin tools keyed by canonical name ("code_interpreter", "web_search")
        builtin_tool_map: dict[str, AbstractToolset[Any]] = {}
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
                    builtin_tool_map["code_interpreter"] = OpenAICodeExecutionTool()
                elif isinstance(t, WebSearchTool):
                    ws_tool_kwargs: dict[str, Any] = {
                        "search_context_size": t.search_context_size,
                    }
                    if t.allowed_domains:
                        ws_tool_kwargs["allowed_domains"] = t.allowed_domains
                    if t.user_location is not None:
                        ws_tool_kwargs["user_location"] = {
                            k: v
                            for k, v in {
                                "city": t.user_location.city,
                                "country": t.user_location.country,
                                "region": t.user_location.region,
                                "timezone": t.user_location.timezone,
                            }.items()
                            if v is not None
                        }
                    builtin_tool_map["web_search"] = OpenAIWebSearchTool(**ws_tool_kwargs)
                elif isinstance(t, McpTool):
                    pass  # TODO: wire MCP toolset

        # Apply tool_choice: filter both sets down to what the user selected
        tc = ir.tool_choice

        if tc == "none":
            function_tool_defs.clear()
            builtin_tool_map.clear()
        elif tc == "auto":
            pass
        elif tc == "required":
            if len(ir.tools) == 0:
                raise BadInputError(
                    "tool_choice cannot be 'auto' or 'required' if no tools are provided."
                )
        elif isinstance(tc, BuiltinToolChoice):
            function_tool_defs.clear()
            builtin_tool_map = {k: v for k, v in builtin_tool_map.items() if k == tc.name}
        elif isinstance(tc, FunctionToolChoice):
            selected = function_tool_defs.get(tc.name)
            if selected is None:
                raise BadInputError(
                    f"`tool_choice.name` {tc.name!r} is not present in effective tools."
                )
            function_tool_defs = {tc.name: selected}
            builtin_tool_map.clear()
        elif isinstance(tc, McpToolChoice):
            function_tool_defs.clear()
            builtin_tool_map.clear()
            # TODO: activate the specific MCP tool
        elif isinstance(tc, AllowedToolsChoice):
            allowed_function_names = {
                e.name for e in tc.tools if isinstance(e, FunctionToolChoice)
            }
            allowed_builtin_names = {e.name for e in tc.tools if isinstance(e, BuiltinToolChoice)}
            function_tool_defs = {
                k: v for k, v in function_tool_defs.items() if k in allowed_function_names
            }
            builtin_tool_map = {
                k: v for k, v in builtin_tool_map.items() if k in allowed_builtin_names
            }

        # Assemble toolsets from whatever survived filtering
        toolsets: list[AbstractToolset[Any]] = []
        builtin_tools: list[AbstractBuiltinTool] = []
        if function_tool_defs:
            toolsets.append(ExternalToolset(tool_defs=list(function_tool_defs.values())))
        if builtin_tool_map:
            builtin_tools.extend(builtin_tool_map.values())
        return RunSettings(
            message_history=self.convert_input_to_messages(ir.input)
            if ir.input is not None
            else [],
            instructions=ir.instructions,
            toolsets=toolsets or None,
            usage_limits=UsageLimits(tool_calls_limit=ir.max_tool_calls),
            builtin_tools=builtin_tools,
        )

    async def compose_outputs(
        self,
        output_stream: AsyncIterator[OutputIR],
    ):
        assert self._ir is not None
        async for event in from_ir_stream(self._ir, output_stream):
            yield event

    async def get_output(
        self,
        output_stream: AsyncIterator[OutputIR],
    ):
        assert self._ir is not None
        last_event = None
        async for event in self.compose_outputs(output_stream):
            last_event = event
        assert last_event is not None
        return last_event.response


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
