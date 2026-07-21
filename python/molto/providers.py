# Providers will map the intermediate representation to Pydantic-AI's various settings
import json
from collections.abc import AsyncIterator
from typing import Any

from httpx import AsyncClient
from pydantic_ai import (
    DeferredToolRequests,
    ExternalToolset,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeOutput,
    NativeToolCallPart,
    NativeToolReturnPart,
    PromptedOutput,
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
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.openai import (
    OpenAIChatModelSettings,
    OpenAIModel,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.native_tools import CodeExecutionTool as OpenAICodeExecutionTool
from pydantic_ai.native_tools import WebSearchTool as OpenAIWebSearchTool
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
    CodeInterpreterTool,
    FunctionTool,
    FunctionToolChoice,
    Input,
    InputCodeToolCall,
    InputHistoryItem,
    InputMcpToolCall,
    InputWebSearchToolCall,
    JsonObjectFormat,
    JsonSchemaFormat,
    McpTool,
    McpToolChoice,
    NativeToolChoice,
    OutputIR,
    SystemMessage,
    TextPart,
    UserMessage,
    WebSearchTool,
)
from molto.types.openai.chat_completions import (
    from_ir as chat_from_ir,
)
from molto.types.openai.chat_completions import (
    from_ir_stream as chat_from_ir_stream,
)
from molto.types.openai.responses import (
    WEB_SEARCH_TOOL,
    from_ir_stream,
)
from molto.utils.exceptions import BadInputError


class _Provider:
    async def parse_agent_settings(
        self, ir, http_client: AsyncClient | None = None, *, api_key: str | None = None
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


class _OpenAIProviderBase(_Provider):
    def __init__(self, ir: Input | None = None) -> None:
        self._ir = ir

    @staticmethod
    def _validate_sampling_params(ir: Input) -> None:
        if ir.temperature is not None and not (0 <= ir.temperature <= 2):
            raise BadInputError(f"temperature must be between 0 and 2, got {ir.temperature}")
        if ir.max_output_tokens is not None and ir.max_output_tokens < 1:
            raise BadInputError(
                f"max_output_tokens must be at least 1, got {ir.max_output_tokens}"
            )

    @staticmethod
    def _build_openai_provider(
        http_client: AsyncClient | None,
        api_key: str | None,
    ) -> OpenAIProvider:
        return OpenAIProvider(
            api_key=api_key or ENV_CONFIG.openai_api_key_plain,
            http_client=http_client,
        )

    @staticmethod
    def _build_common_model_settings(ir: Input) -> dict[str, Any]:
        ms_kwargs: dict[str, Any] = {}
        if ir.temperature is not None:
            ms_kwargs["temperature"] = ir.temperature
        if ir.top_p is not None:
            ms_kwargs["top_p"] = ir.top_p
        if ir.max_output_tokens is not None:
            ms_kwargs["max_tokens"] = ir.max_output_tokens
        if ir.top_logprobs is not None:
            ms_kwargs["openai_logprobs"] = True
            ms_kwargs["openai_top_logprobs"] = ir.top_logprobs
        if ir.parallel_tool_calls is not None:
            ms_kwargs["parallel_tool_calls"] = ir.parallel_tool_calls
        if ir.frequency_penalty is not None:
            ms_kwargs["frequency_penalty"] = ir.frequency_penalty
        if ir.presence_penalty is not None:
            ms_kwargs["presence_penalty"] = ir.presence_penalty
        if ir.extra_body:
            ms_kwargs["extra_body"] = dict(ir.extra_body)
        return ms_kwargs

    @staticmethod
    def _build_output_type(ir: Input) -> Any:
        if isinstance(ir.text_format, JsonSchemaFormat):
            return NativeOutput(
                StructuredDict(ir.text_format.json_schema),
                name=ir.text_format.name,
                description=ir.text_format.description,
                strict=ir.text_format.strict,
            )
        if isinstance(ir.text_format, JsonObjectFormat):
            return PromptedOutput(
                StructuredDict({"type": "object", "additionalProperties": True}), template=False
            )
        return [str, DeferredToolRequests]

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
                    args = tc.raw_args if tc.raw_args is not None else json.dumps(tc.args)
                    message_history.append(
                        ModelResponse(
                            parts=[
                                ToolCallPart(
                                    tool_name=tc.tool_name,
                                    args=args,
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
                            NativeToolCallPart(
                                tool_name="code_execution",
                                args={"container_id": msg.container_id, "code": msg.code},
                                tool_call_id=msg.id,
                            ),
                            NativeToolReturnPart(
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
                            NativeToolCallPart(
                                tool_name=WEB_SEARCH_TOOL,
                                args=action,
                                tool_call_id=msg.id,
                            ),
                            NativeToolReturnPart(
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
                            NativeToolCallPart(
                                tool_name=f"mcp_{msg.server_label}_{msg.name}",
                                args=args,
                                tool_call_id=msg.id,
                            ),
                            NativeToolReturnPart(
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

    @staticmethod
    def _collect_tools(
        ir: Input,
    ) -> tuple[dict[str, ToolDefinition], dict[str, AbstractNativeTool]]:
        function_tool_defs: dict[str, ToolDefinition] = {}
        native_tool_map: dict[str, AbstractNativeTool] = {}
        if not ir.tools:
            return function_tool_defs, native_tool_map
        for t in ir.tools:
            if isinstance(t, FunctionTool):
                function_tool_defs[t.name] = ToolDefinition(
                    name=t.name,
                    description=t.description or "",
                    parameters_json_schema=t.parameters,
                    strict=t.strict,
                )
            elif isinstance(t, CodeInterpreterTool):
                native_tool_map["code_interpreter"] = OpenAICodeExecutionTool()
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
                native_tool_map["web_search"] = OpenAIWebSearchTool(**ws_tool_kwargs)
            elif isinstance(t, McpTool):
                pass
        return function_tool_defs, native_tool_map

    @staticmethod
    def _apply_tool_choice(
        ir: Input,
        function_tool_defs: dict[str, ToolDefinition],
        native_tool_map: dict[str, AbstractNativeTool],
    ) -> tuple[dict[str, ToolDefinition], dict[str, AbstractNativeTool]]:
        tc = ir.tool_choice
        if tc == "none":
            return {}, {}
        if tc == "auto":
            return function_tool_defs, native_tool_map
        if tc == "required":
            if not ir.tools:
                raise BadInputError(
                    "tool_choice cannot be 'auto' or 'required' if no tools are provided."
                )
            return function_tool_defs, native_tool_map
        if isinstance(tc, NativeToolChoice):
            native_tool_map = {k: v for k, v in native_tool_map.items() if k == tc.name}
            return {}, native_tool_map
        if isinstance(tc, FunctionToolChoice):
            selected = function_tool_defs.get(tc.name)
            if selected is None:
                raise BadInputError(
                    f"`tool_choice.name` {tc.name!r} is not present in effective tools."
                )
            return {tc.name: selected}, {}
        if isinstance(tc, McpToolChoice):
            return {}, {}
        if isinstance(tc, AllowedToolsChoice):
            allowed_function_names = {
                e.name for e in tc.tools if isinstance(e, FunctionToolChoice)
            }
            allowed_native_names = {e.name for e in tc.tools if isinstance(e, NativeToolChoice)}
            return (
                {k: v for k, v in function_tool_defs.items() if k in allowed_function_names},
                {k: v for k, v in native_tool_map.items() if k in allowed_native_names},
            )
        return function_tool_defs, native_tool_map

    @classmethod
    def _apply_chat_tool_choice(
        cls,
        ir: Input,
        function_tool_defs: dict[str, ToolDefinition],
    ) -> dict[str, ToolDefinition]:
        tc = ir.tool_choice
        if tc == "none":
            return {}
        if tc in (None, "auto"):
            return function_tool_defs
        if tc == "required":
            if not function_tool_defs:
                raise BadInputError(
                    "tool_choice cannot be 'required' if no function tools are provided."
                )
            return function_tool_defs
        if isinstance(tc, FunctionToolChoice):
            selected = function_tool_defs.get(tc.name)
            if selected is None:
                raise BadInputError(
                    f"`tool_choice.name` {tc.name!r} is not present in effective tools."
                )
            return {tc.name: selected}
        if isinstance(tc, AllowedToolsChoice):
            allowed_function_names = {
                e.name for e in tc.tools if isinstance(e, FunctionToolChoice)
            }
            selected = {k: v for k, v in function_tool_defs.items() if k in allowed_function_names}
            if tc.mode == "required" and not selected:
                raise BadInputError("allowed tool_choice requires at least one function tool.")
            return selected
        return function_tool_defs

    @classmethod
    def _build_run_settings(
        cls,
        ir: Input,
        function_tool_defs: dict[str, ToolDefinition],
        native_tool_map: dict[str, AbstractNativeTool] | None = None,
    ) -> RunSettings:
        toolsets: list[AbstractToolset[Any]] = []
        native_tools: list[AbstractNativeTool] = []
        if function_tool_defs:
            toolsets.append(ExternalToolset(tool_defs=list(function_tool_defs.values())))
        if native_tool_map:
            native_tools.extend(native_tool_map.values())
        return RunSettings(
            message_history=(
                cls.convert_input_to_messages(ir.input) if ir.input is not None else []
            ),
            instructions=ir.instructions,
            toolsets=toolsets or None,
            usage_limits=UsageLimits(tool_calls_limit=ir.max_tool_calls),
            capabilities=[NativeTool(tool=t) for t in native_tools] or None,
        )


class OpenAIResponsesProvider(_OpenAIProviderBase):
    @staticmethod
    def _build_model_settings(ir: Input) -> OpenAIResponsesModelSettings:
        openai = ir.openai
        responses = ir.responses
        extra_body = dict(ir.extra_body or {})
        tool_choice = ir.tool_choice
        if tool_choice == "auto" or tool_choice == "required":
            extra_body["tool_choice"] = tool_choice
        if responses.safety_identifier is not None:
            extra_body["safety_identifier"] = responses.safety_identifier
        if ir.max_tool_calls is not None:
            extra_body["max_tool_calls"] = ir.max_tool_calls
        if responses.conversation_id is not None:
            extra_body["conversation"] = responses.conversation_id

        include = responses.include or []
        ms_kwargs = OpenAIResponsesProvider._build_common_model_settings(ir)
        if ir.reasoning_effort is not None:
            ms_kwargs["openai_reasoning_effort"] = ir.reasoning_effort
        if ir.reasoning_summary is not None:
            ms_kwargs["openai_reasoning_summary"] = ir.reasoning_summary
        if openai.store is not None:
            ms_kwargs["openai_store"] = openai.store
        if responses.previous_response_id is not None:
            ms_kwargs["openai_previous_response_id"] = responses.previous_response_id
        if openai.service_tier is not None:
            ms_kwargs["openai_service_tier"] = openai.service_tier
        if openai.prompt_cache_key is not None:
            ms_kwargs["openai_prompt_cache_key"] = openai.prompt_cache_key
        if openai.prompt_cache_retention is not None:
            ms_kwargs["openai_prompt_cache_retention"] = openai.prompt_cache_retention
        if responses.truncation is not None:
            ms_kwargs["openai_truncation"] = responses.truncation
        if responses.verbosity is not None:
            ms_kwargs["openai_text_verbosity"] = responses.verbosity
        if include:
            if "web_search_call.action.sources" in include:
                ms_kwargs["openai_include_web_search_sources"] = True
            if "code_interpreter_call.outputs" in include:
                ms_kwargs["openai_include_code_execution_outputs"] = True
            if "file_search_call.results" in include:
                ms_kwargs["openai_include_file_search_results"] = True
        if extra_body:
            ms_kwargs["extra_body"] = extra_body
        return OpenAIResponsesModelSettings(**ms_kwargs)

    async def parse_agent_settings(
        self,
        ir: Input,
        http_client: AsyncClient | None = None,
        *,
        api_key: str | None = None,
    ) -> AgentSettings:
        self._validate_sampling_params(ir)
        model = OpenAIResponsesModel(
            model_name=ir.model.split("/")[-1],
            provider=self._build_openai_provider(http_client, api_key),
        )
        model_settings = self._build_model_settings(ir)
        return AgentSettings(
            model=model,
            model_settings=model_settings,
            output_type=self._build_output_type(ir),
        )

    async def parse_run_settings(self, ir: Input) -> RunSettings:
        function_tool_defs, native_tool_map = self._collect_tools(ir)
        function_tool_defs, native_tool_map = self._apply_tool_choice(
            ir, function_tool_defs, native_tool_map
        )
        return self._build_run_settings(ir, function_tool_defs, native_tool_map)

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


class OpenAIChatCompletionsProvider(_OpenAIProviderBase):
    @staticmethod
    def _build_model_settings(ir: Input) -> OpenAIChatModelSettings:
        openai = ir.openai
        ms_kwargs = OpenAIChatCompletionsProvider._build_common_model_settings(ir)
        if ir.tool_choice in {"none", "required"}:
            ms_kwargs["tool_choice"] = ir.tool_choice
        elif isinstance(ir.tool_choice, FunctionToolChoice):
            ms_kwargs["tool_choice"] = [ir.tool_choice.name]
        elif isinstance(ir.tool_choice, AllowedToolsChoice):
            ms_kwargs["tool_choice"] = (
                [
                    choice.name
                    for choice in ir.tool_choice.tools
                    if isinstance(choice, FunctionToolChoice)
                ]
                if ir.tool_choice.mode == "required"
                else "auto"
            )
        if ir.reasoning_effort is not None:
            ms_kwargs["openai_reasoning_effort"] = ir.reasoning_effort
        if openai.store is not None:
            ms_kwargs["openai_store"] = openai.store
        if openai.service_tier is not None:
            ms_kwargs["openai_service_tier"] = openai.service_tier
        if openai.prompt_cache_key is not None:
            ms_kwargs["openai_prompt_cache_key"] = openai.prompt_cache_key
        if openai.prompt_cache_retention is not None:
            ms_kwargs["openai_prompt_cache_retention"] = openai.prompt_cache_retention
        return OpenAIChatModelSettings(**ms_kwargs)

    @staticmethod
    def _build_output_type(ir: Input) -> Any:
        if isinstance(ir.text_format, JsonSchemaFormat):
            return NativeOutput(
                StructuredDict(ir.text_format.json_schema),
                name=ir.text_format.name,
                description=ir.text_format.description,
                strict=ir.text_format.strict,
            )
        if isinstance(ir.text_format, JsonObjectFormat):
            return PromptedOutput(
                StructuredDict({"type": "object", "additionalProperties": True}), template=False
            )
        return [str, DeferredToolRequests]

    async def parse_agent_settings(
        self,
        ir: Input,
        http_client: AsyncClient | None = None,
        *,
        api_key: str | None = None,
    ) -> AgentSettings:
        self._validate_sampling_params(ir)
        model = OpenAIModel(
            model_name=ir.model.split("/")[-1],
            provider=self._build_openai_provider(http_client, api_key),
        )
        model_settings = self._build_model_settings(ir)
        if model_settings.get("tool_choice") == "required" or isinstance(
            model_settings.get("tool_choice"), list
        ):

            def agent_model_settings(_: Any) -> OpenAIChatModelSettings:
                return model_settings

        else:
            agent_model_settings = model_settings
        return AgentSettings(
            model=model,
            model_settings=agent_model_settings,
            output_type=self._build_output_type(ir),
        )

    async def parse_run_settings(self, ir: Input) -> RunSettings:
        function_tool_defs, _native_tool_map = self._collect_tools(ir)
        function_tool_defs = self._apply_chat_tool_choice(ir, function_tool_defs)
        return self._build_run_settings(ir, function_tool_defs)

    async def compose_outputs(
        self,
        output_stream: AsyncIterator[OutputIR],
    ):
        assert self._ir is not None
        async for event in chat_from_ir_stream(self._ir, output_stream):
            yield event

    async def get_output(
        self,
        output_stream: AsyncIterator[OutputIR],
    ):
        assert self._ir is not None
        return await chat_from_ir(self._ir, output_stream)


class AnthropicMessagesProvider(_Provider):
    pass
