import json
from time import perf_counter
from typing import Any, AsyncIterator, TypedDict

from httpx import AsyncClient
from loguru import logger
from pydantic_ai import (
    Agent,
    AgentRunError,
    AgentRunResultEvent,
    DeferredToolRequests,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelHTTPError,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    NativeToolCallPart,
    NativeToolReturnPart,
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
    UserError,
    capture_run_messages,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.toolsets.abstract import AbstractToolset

from molto.types.ir import (
    RESPONSE_TOOL_NAME,
    AgentError,
    AgentEvent,
    AgentMessageDelta,
    AgentMessageEnd,
    AgentMessageStart,
    AgentNativeToolCall,
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
from molto.utils.exceptions import (
    BadInputError,
    ContextOverflowError,
    ExternalAuthError,
    ModelOverloadError,
    MoltoException,
    RateLimitExceedError,
    ServerBusyError,
    UnexpectedError,
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
    capabilities: list[AbstractCapability[Any]] | None = None


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
    # Native tools is constructed this way in Pydantic AI.
    capabilities: list[AbstractCapability[Any]] | None


def _map_pydantic_ai_error(exc: Exception) -> MoltoException:
    if isinstance(exc, UnexpectedModelBehavior):
        return UnexpectedError(str(exc))
    if isinstance(exc, UserError):
        return BadInputError(str(exc))
    if isinstance(exc, ModelHTTPError):
        status = exc.status_code
        body = str(exc.body) if exc.body else str(exc)
        if status == 400:
            msg = str(body)
            if "context" in msg.lower() and ("length" in msg.lower() or "token" in msg.lower()):
                return ContextOverflowError(msg)
            return BadInputError(msg)
        if status == 401:
            return ExternalAuthError(body)
        if status == 403:
            return BadInputError(body)
        if status == 404:
            return BadInputError(body)
        if status == 429:
            return RateLimitExceedError(body, limit=0, remaining=0, reset_at=0)
        if status == 500:
            return ServerBusyError(body)
        if status in (503, 529):
            return ModelOverloadError(body)
        return UnexpectedError(f"HTTP {status}: {body}")
    if isinstance(exc, AgentRunError):
        return UnexpectedError(str(exc))
    return UnexpectedError(f"{type(exc).__name__}: {exc}")


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
        self._native_tool_calls: dict[str, dict] = {}
        self._current_text_id: str | None = None
        self._current_thought_id: str | None = None
        # When `output_type` is a `StructuredDict(...)`, pydantic-ai routes
        # structured output through an internal `final_result` tool call. Track
        # that tool call so the runner can surface it as a text message rather
        # than a tool-call event in the IR.
        _output_type = self._agent_settings["output_type"]
        self._structured_output_active = isinstance(_output_type, type) and getattr(
            _output_type, "__is_model_like__", False
        )
        self._structured_output_tool_id: str | None = None
        self._agent = Agent(
            **self._agent_settings,
        )

    def _event_to_ir(self, event: Any) -> AsyncIterator[AgentEvent]:
        print("event from runner:\n", event)
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, TextPart):
                self._current_text_id = event.part.id
                yield AgentMessageStart(id=event.part.id)
                if event.part.content:
                    yield AgentMessageDelta(content=event.part.content, id=event.part.id)
            elif isinstance(event.part, ThinkingPart):
                self._current_thought_id = event.part.id
                yield AgentThoughtStart(id=event.part.id)
                if event.part.content or event.part.signature:
                    yield AgentThoughtDelta(
                        content=event.part.content,
                        signature=event.part.signature,
                        id=event.part.id,
                    )
            elif isinstance(event.part, ToolCallPart):
                _tool_name = event.part.tool_name
                _tool_call_id = event.part.tool_call_id
                if self._structured_output_active and _tool_name == RESPONSE_TOOL_NAME:
                    self._structured_output_tool_id = _tool_call_id
                    self._current_text_id = _tool_call_id
                    yield AgentMessageStart(id=_tool_call_id)
                    if event.part.has_content():
                        yield AgentMessageDelta(
                            content=event.part.args_as_json_str(),
                            id=_tool_call_id,
                        )
                else:
                    self._tool_id2name[_tool_call_id] = _tool_name
                    yield AgentToolStart(
                        tool_call_id=_tool_call_id, tool_name=_tool_name, id=event.part.id
                    )
                    if event.part.has_content():
                        yield AgentToolDelta(
                            tool_call_id=_tool_call_id,
                            tool_name=_tool_name,
                            args=event.part.args_as_json_str(),
                        )
            elif isinstance(event.part, NativeToolCallPart):
                self._native_tool_calls[event.part.tool_call_id] = {
                    "tool_name": event.part.tool_name,
                    "tool_call_id": event.part.tool_call_id,
                    "args": event.part.args if isinstance(event.part.args, dict) else None,
                    "id": event.part.id,
                }
            elif isinstance(event.part, NativeToolReturnPart):
                tc = self._native_tool_calls.get(event.part.tool_call_id)
                if tc:
                    yield AgentNativeToolCall(
                        tool_name=tc["tool_name"],
                        tool_call_id=tc["tool_call_id"],
                        args=tc.get("args"),
                        result=event.part.content,
                        id=tc.get("id"),
                    )

        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                if event.delta.content_delta:
                    yield AgentMessageDelta(
                        content=event.delta.content_delta,
                        id=self._current_text_id,
                    )
            elif isinstance(event.delta, ThinkingPartDelta):
                if event.delta.content_delta or event.delta.signature_delta:
                    yield AgentThoughtDelta(
                        content=event.delta.content_delta,
                        signature=event.delta.signature_delta,
                        id=self._current_thought_id,
                    )
            elif isinstance(event.delta, ToolCallPartDelta):
                if event.delta.args_delta:
                    _delta_tool_call_id = event.delta.tool_call_id
                    if (
                        self._structured_output_tool_id is not None
                        and _delta_tool_call_id == self._structured_output_tool_id
                    ):
                        yield AgentMessageDelta(
                            content=str(event.delta.args_delta),
                            id=_delta_tool_call_id,
                        )
                    else:
                        _tool_name = self._tool_id2name.get(_delta_tool_call_id, "")
                        yield AgentToolDelta(
                            tool_call_id=_delta_tool_call_id,
                            tool_name=_tool_name,
                            args=str(event.delta.args_delta),
                        )

        elif isinstance(event, PartEndEvent):
            if isinstance(event.part, TextPart):
                _part_logprobs = None
                if event.part.provider_details and "logprobs" in event.part.provider_details:
                    _part_logprobs = event.part.provider_details["logprobs"]
                yield AgentMessageEnd(id=event.part.id, logprobs=_part_logprobs)
                self._current_text_id = None
            elif isinstance(event.part, ThinkingPart):
                yield AgentThoughtEnd(id=event.part.id)
                self._current_thought_id = None
            elif isinstance(event.part, ToolCallPart):
                if (
                    self._structured_output_tool_id is not None
                    and event.part.tool_call_id == self._structured_output_tool_id
                ):
                    yield AgentMessageEnd(id=event.part.tool_call_id)
                    self._current_text_id = None
                    self._structured_output_tool_id = None
                else:
                    yield AgentToolEnd(tool_call_id=event.part.tool_call_id)
            elif isinstance(event.part, NativeToolCallPart):
                tc = self._native_tool_calls.get(event.part.tool_call_id)
                if tc:
                    tc["args"] = event.part.args if isinstance(event.part.args, dict) else None
                    if event.part.id is not None:
                        tc["id"] = event.part.id

        elif isinstance(event, FunctionToolCallEvent):
            yield AgentToolCall(
                tool_call_id=event.tool_call_id,
                tool_name=event.part.tool_name,
                args=event.part.args_as_dict(),
                id=event.part.id,
            )

        elif isinstance(event, FunctionToolResultEvent):
            yield AgentToolResult(
                tool_call_id=event.result.tool_call_id,
                tool_name=event.result.tool_name,
                result=event.result.content,
            )

        elif isinstance(event, AgentRunResultEvent):
            yield from self._result_to_ir(event)

    def _result_to_ir(self, event: AgentRunResultEvent) -> AsyncIterator[AgentEvent]:
        _output = event.result.output
        _raw_usage = event.result.usage
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
        _completed_at = int(event.result.timestamp.timestamp())
        _logprobs = None
        _incomplete_details = None
        _provider_response_id = None
        for msg in reversed(event.result.new_messages()):
            if isinstance(msg, ModelResponse):
                if msg.provider_response_id is not None:
                    _provider_response_id = msg.provider_response_id
                if msg.provider_details and "logprobs" in msg.provider_details:
                    _logprobs = msg.provider_details["logprobs"]
                else:
                    for _part in msg.parts:
                        if (
                            isinstance(_part, TextPart)
                            and _part.provider_details
                            and "logprobs" in _part.provider_details
                        ):
                            _logprobs = (_logprobs or []) + _part.provider_details["logprobs"]
                if msg.finish_reason == "length":
                    _incomplete_details = "max_output_tokens"
                elif (
                    msg.finish_reason is None
                    and msg.provider_details
                    and msg.provider_details.get("finish_reason") == "max_output_tokens"
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
                provider_response_id=_provider_response_id,
            )
        else:
            if isinstance(_output, str):
                _message = _output
            elif isinstance(_output, dict):
                _message = json.dumps(_output, ensure_ascii=False, separators=(",", ":"))
            else:
                _message = None
            yield AgentRunResult(
                message=_message,
                usage=_usage,
                completed_at=_completed_at,
                logprobs=_logprobs,
                incomplete_details=_incomplete_details,
                provider_response_id=_provider_response_id,
            )

    async def run_stream(self) -> AsyncIterator[AgentEvent]:
        with capture_run_messages() as messages:
            event_count = 0
            t0 = perf_counter()
            try:
                async for event in self._agent.run_stream_events(
                    **self._run_settings,
                ):
                    event_count += 1
                    for ir_event in self._event_to_ir(event):
                        yield ir_event
            except (UnexpectedModelBehavior, UserError, ModelHTTPError, AgentRunError) as e:
                yield AgentError(error=str(e), cause=repr(e.__cause__))
                raise _map_pydantic_ai_error(e) from e
            except Exception as e:
                yield AgentError(error=str(e), cause=repr(e.__cause__))
                raise UnexpectedError(str(e)) from e
        t = perf_counter() - t0
        logger.debug(
            (
                f'Agent "{self._agent_id}" run completed in {t:.1f} seconds '
                f"({event_count:,d} events, {event_count / t:.2f} event/s, {len(messages):,d} messages)"
            )
        )
