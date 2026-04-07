from time import perf_counter
from typing import Any, AsyncIterator, TypedDict

from loguru import logger
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    DeferredToolRequests,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
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
from pydantic_ai.toolsets.abstract import AbstractToolset

from molto.types.ir import (
    AgentError,
    AgentEvent,
    AgentMessageDelta,
    AgentMessageEnd,
    AgentMessageStart,
    AgentThoughtDelta,
    AgentThoughtEnd,
    AgentThoughtStart,
    AgentToolCall,
    AgentToolCalls,
    AgentToolDelta,
    AgentToolEnd,
    AgentToolResult,
    AgentToolStart,
)


class AgentSettings(TypedDict):
    """
    Kwargs passed to `Agent(...)`.

    Keep this aligned with the public `pydantic_ai` signature to avoid untyped
    `dict[str, Any]` plumbing.
    """


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


class Runner:
    def __init__(
        self,
        run_settings: AgentSettings,
        agent_settings: RunSettings,
    ):
        self._run_settings = run_settings
        self._agent_settings = agent_settings
        self._tool_id2name: dict[str, str] = {}
        self._agent = Agent(
            **self._agent_settings,
        )

    async def _run_stream(self) -> AsyncIterator[AgentEvent]:
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
                            self._tool_id2name[event.part.tool_call_id] = _tool_name
                            yield AgentToolStart()
                            if event.part.has_content():
                                yield AgentToolDelta(
                                    tool_name=_tool_name,
                                    args=event.part.args_as_json_str(),
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
                                _tool_name = self._tool_id2name.get(event.delta.tool_call_id, None)
                                yield AgentToolDelta(
                                    tool_name=_tool_name,
                                    args=str(event.delta.args_delta),
                                )
                    elif isinstance(event, PartEndEvent):
                        if isinstance(event.part, TextPart):
                            yield AgentMessageEnd()
                        elif isinstance(event.part, ThinkingPart):
                            yield AgentThoughtEnd()
                        elif isinstance(event.part, ToolCallPart):
                            yield AgentToolEnd()
                    elif isinstance(event, FunctionToolCallEvent):
                        raise NotImplementedError
                    elif isinstance(event, FunctionToolResultEvent):
                        # Display status bubble showing tool result summary
                        yield AgentToolResult(
                            tool_call_id=event.result.tool_call_id,
                            tool_name=event.result.tool_name,
                            result=event.result.content,
                        )
                    elif isinstance(event, AgentRunResultEvent):
                        _output = event.result.output
                        if isinstance(_output, DeferredToolRequests):
                            yield AgentToolCalls(
                                tool_calls=[
                                    AgentToolCall(
                                        tool_call_id=_call.tool_call_id,
                                        tool_name=_call.tool_name,
                                        args=_call.args_as_dict(),
                                    )
                                    for _call in _output.approvals
                                ],
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
