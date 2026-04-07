# Providers will map the intermediate representation to Pydantic-AI's various settings
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from time import time
from typing import Annotated, Any, Literal, TypedDict, Union

from httpx import AsyncClient
from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from pydantic import (
    AliasChoices,
    BaseModel,
    Discriminator,
    Field,
    WrapValidator,
    field_validator,
)
from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ExternalToolset,
    FunctionToolset,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    Tool,
    ToolCallPart,
    # ToolDefinition,
    ToolReturnPart,
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_extra_types.timezone_name import TimeZoneName

from molto.runner import AgentSettings, Runner, RunSettings

# from vllm_responses.mcp.gateway_toolset import McpGatewayToolset, ResolvedMcpTool
# from vllm_responses.mcp.resolver import (
#     ResolvedMcpServerTools,
#     build_request_remote_toolset,
#     resolve_mcp_declarations,
# )
# from vllm_responses.mcp.types import McpToolInfo, McpToolRef
# from vllm_responses.mcp.utils import (
#     HOSTED_MCP_INTERNAL_PREFIX,
#     build_internal_mcp_tool_name,
#     normalize_mcp_input_schema,
# )
# from vllm_responses.tools import TOOLS
from molto.utils import uuid7_str
from molto.utils.exceptions import BadInputError

# if TYPE_CHECKING:
#     from molto.mcp.runtime_client import BuiltinMcpRuntimeClient


class _Provider:
    _run_settings: RunSettings
    _agent_settings: ModelSettings

    def __init__(
        self,
        body: Any,
        http_client: AsyncClient,
    ):
        self._body = body
        self._http_client = http_client

    async def close(self) -> None:
        await self._http_client.aclose()

    async def run(self):
        """Non-streaming execution."""
        self._parse_run_settings()
        self._parse_agent_settings()
        runner = Runner(self._run_settings, self._agent_settings)
        output_ir = await runner.run()
        return self._compose_output(output_ir)

    async def run_stream(self) -> AsyncIterator:
        """Streaming execution."""
        self._parse_run_settings()
        self._parse_agent_settings()
        runner = Runner(self._run_settings, self._agent_settings)
        async for event in runner.run_stream():
            yield self._compose_output(event)

    async def _parse_run_settings(self) -> None:
        raise NotImplementedError

    async def _parse_agent_settings(self) -> None:
        raise NotImplementedError

    async def _compose_outputs(self) -> None:
        raise NotImplementedError


class OpenAIResponsesProvider(_Provider):
    async def _parse_run_settings(self) -> None:
        raise NotImplementedError

    async def _parse_agent_settings(self) -> None:
        raise NotImplementedError

    async def _compose_output(self) -> None:
        raise NotImplementedError


# class OpenAIChatCompletionsProvider(_Provider):
#     async def _parse_run_settings(self) -> None:
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

#     def _parse_agent_settings(self) -> None:
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
