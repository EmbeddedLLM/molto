from time import perf_counter
from typing import Any, AsyncIterator, TypedDict

from httpx import AsyncClient
from loguru import logger
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DeferredToolRequests,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    UnexpectedModelBehavior,
    UsageLimits,
    capture_run_messages,
)
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.toolsets.abstract import AbstractToolset

from molto.types.ir import (
    AgentBuiltinToolCall,
    AgentError,
    AgentEvent,
    AgentMessageDelta,
    AgentMessageEnd,
    AgentMessageStart,
    AgentRunResult,
    AgentThoughtDelta,
    AgentThoughtEnd,
    AgentThoughtStart,
    AgentToolCall,
    AgentToolDelta,
    AgentToolEnd,
    AgentToolResult,
    AgentToolStart,
    InputTokensDetails,
    OutputTokensDetails,
    Usage,
)


class AgentSettings(TypedDict):
    """
    Kwargs passed to `Agent(...)`.

    Keep this aligned with the public `pydantic_ai` signature to avoid untyped
    `dict[str, Any]` plumbing.
    """

    model: Model
    model_settings: ModelSettings | None
    http_client: AsyncClient
    output_type: OutputSpec = [str, DeferredToolRequests]
    # Unused for now but may be useful in the future for things like auto-adding capabilities based on IR features (e.g. tool choice)
    capabilities: list[AbstractCapability[None]] | None = None


class RunSettings(TypedDict):
    """
    Kwargs passed to `Agent.run_stream_events(...)`.

    Keep this aligned with the public `pydantic_ai` run signature to avoid untyped
    `dict[str, Any]` plumbing.
    """

    message_history: list[ModelMessage]
    instructions: str | None
    toolsets: list[AbstractToolset[Any]] | None
    usage_limits: UsageLimits
    builtin_tools: list[AbstractBuiltinTool] | None


class Runner:
    def __init__(
        self,
        agent_settings: AgentSettings,
        run_settings: RunSettings,
    ):
        self._agent_id = "Temp"
        self._agent_settings = agent_settings
        self._run_settings = run_settings
        self._tool_id2name: dict[str, str] = {}
        self._builtin_tool_calls: dict[str, dict] = {}
        self._agent = Agent(
            **self._agent_settings,
        )

    async def run_stream(self) -> AsyncIterator[AgentEvent]:
        with capture_run_messages() as messages:
            event_count = 0
            t0 = perf_counter()
            try:
                # logger.debug(f"{self._run_settings=}")
                async for event in self._agent.run_stream_events(
                    **self._run_settings,
                ):
                    event_count += 1
                    if isinstance(event, PartStartEvent):
                        # New event / thought bubble
                        if isinstance(event.part, TextPart):
                            # Signals start and contains the first delta
                            yield AgentMessageStart()
                            if event.part.content:
                                yield AgentMessageDelta(content=event.part.content)
                        elif isinstance(event.part, ThinkingPart):
                            # Signals start and contains the first delta
                            yield AgentThoughtStart()
                            if event.part.content or event.part.signature:
                                yield AgentThoughtDelta(
                                    content=event.part.content,
                                    signature=event.part.signature,
                                )
                        elif isinstance(event.part, ToolCallPart):
                            # Signals start
                            _tool_name = event.part.tool_name
                            _tool_call_id = event.part.tool_call_id
                            self._tool_id2name[_tool_call_id] = _tool_name
                            yield AgentToolStart(tool_call_id=_tool_call_id, tool_name=_tool_name)
                            if event.part.has_content():
                                yield AgentToolDelta(
                                    tool_call_id=_tool_call_id,
                                    tool_name=_tool_name,
                                    args=event.part.args_as_json_str(),
                                )
                        elif isinstance(event.part, BuiltinToolCallPart):
                            self._builtin_tool_calls[event.part.tool_call_id] = {
                                "tool_name": event.part.tool_name,
                                "tool_call_id": event.part.tool_call_id,
                                "args": event.part.args
                                if isinstance(event.part.args, dict)
                                else None,
                            }
                        elif isinstance(event.part, BuiltinToolReturnPart):
                            tc = self._builtin_tool_calls.get(event.part.tool_call_id)
                            if tc:
                                yield AgentBuiltinToolCall(
                                    tool_name=tc["tool_name"],
                                    tool_call_id=tc["tool_call_id"],
                                    args=tc.get("args"),
                                    result=event.part.content,
                                )
                    elif isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            if event.delta.content_delta:
                                yield AgentMessageDelta(content=event.delta.content_delta)
                        elif isinstance(event.delta, ThinkingPartDelta):
                            if event.delta.content_delta or event.part.signature_delta:
                                yield AgentThoughtDelta(
                                    content=event.part.content_delta,
                                    signature=event.part.signature_delta,
                                )
                        elif isinstance(event.delta, ToolCallPartDelta):
                            if event.delta.args_delta:
                                _delta_tool_call_id = event.delta.tool_call_id
                                _tool_name = self._tool_id2name.get(_delta_tool_call_id, "")
                                yield AgentToolDelta(
                                    tool_call_id=_delta_tool_call_id,
                                    tool_name=_tool_name,
                                    args=str(event.delta.args_delta),
                                )
                    elif isinstance(event, PartEndEvent):
                        if isinstance(event.part, TextPart):
                            yield AgentMessageEnd()
                        elif isinstance(event.part, ThinkingPart):
                            yield AgentThoughtEnd()
                        elif isinstance(event.part, ToolCallPart):
                            yield AgentToolEnd(tool_call_id=event.part.tool_call_id)
                        elif isinstance(event.part, BuiltinToolCallPart):
                            tc = self._builtin_tool_calls.get(event.part.tool_call_id)
                            if tc:
                                tc["args"] = (
                                    event.part.args if isinstance(event.part.args, dict) else None
                                )
                    elif isinstance(event, FunctionToolCallEvent):
                        yield AgentToolCall(
                            tool_call_id=event.tool_call_id,
                            tool_name=event.part.tool_name,
                            args=event.part.args_as_dict(),
                        )
                    elif isinstance(event, FunctionToolResultEvent):
                        # Display status bubble showing tool result summary
                        yield AgentToolResult(
                            tool_call_id=event.result.tool_call_id,
                            tool_name=event.result.tool_name,
                            result=event.result.content,
                        )
                    elif isinstance(event, AgentRunResultEvent):
                        _output = event.result.output
                        _raw_usage = event.result.usage()
                        _usage = Usage(
                            input_tokens=_raw_usage.input_tokens,
                            output_tokens=_raw_usage.output_tokens,
                            total_tokens=_raw_usage.input_tokens + _raw_usage.output_tokens,
                            input_tokens_details=InputTokensDetails(
                                cached_tokens=_raw_usage.cache_read_tokens,
                            ),
                            output_tokens_details=OutputTokensDetails(
                                reasoning_tokens=_raw_usage.details.get("reasoning_tokens", 0),
                            ),
                        )
                        _completed_at = int(event.result.timestamp().timestamp())
                        _logprobs = None
                        _incomplete_details = None
                        for msg in reversed(event.result.new_messages()):
                            if isinstance(msg, ModelResponse):
                                if msg.provider_details and "logprobs" in msg.provider_details:
                                    _logprobs = msg.provider_details["logprobs"]
                                if msg.finish_reason == "length":
                                    _incomplete_details = "max_output_tokens"
                                elif (
                                    msg.finish_reason is None
                                    and msg.provider_details
                                    and msg.provider_details.get("finish_reason")
                                    == "max_output_tokens"
                                ):
                                    _incomplete_details = "max_output_tokens"
                                break
                        if isinstance(_output, DeferredToolRequests):
                            yield AgentRunResult(
                                tool_calls=[
                                    AgentToolCall(
                                        tool_call_id=_call.tool_call_id,
                                        tool_name=_call.tool_name,
                                        args=_call.args_as_dict(),
                                    )
                                    for _call in _output.calls
                                ],
                                usage=_usage,
                                completed_at=_completed_at,
                                logprobs=_logprobs,
                                incomplete_details=_incomplete_details,
                            )
                        else:
                            yield AgentRunResult(
                                message=_output if isinstance(_output, str) else None,
                                usage=_usage,
                                completed_at=_completed_at,
                                logprobs=_logprobs,
                                incomplete_details=_incomplete_details,
                            )
                    else:
                        # Other events we don't handle yet (e.g. tool call end, thought bubble end)
                        continue
            except UnexpectedModelBehavior as e:
                yield AgentError(error=str(e), cause=repr(e.__cause__))
            # print(messages)
            # for m in messages:
            #     print("--------")
            #     if isinstance(m, ModelRequest):
            #         print(m)
            #     else:
            #         for mm in m.parts:
            #             print("> ", mm)
        t = perf_counter() - t0
        logger.info(
            (
                f'Agent "{self._agent_id}" run completed in {t:.1f} seconds '
                f"({event_count:,d} events, {event_count / t:.2f} event/s, {len(messages):,d} messages)"
            )
        )
