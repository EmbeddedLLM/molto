from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from molto.types.ir import AgentRunResult, AgentToolCall, AgentToolCalls, OutputIR
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.chat_completions import from_ir as chat_completions_from_ir
from molto.types.openai.responses import OpenAIFunctionToolCall, OpenAIResponsesRequest
from molto.types.openai.responses import from_ir as responses_from_ir


def _single_history_tool_call(items: object) -> AgentToolCall:
    assert isinstance(items, list)
    tool_call_items = [item for item in items if isinstance(item, AgentToolCalls)]
    assert len(tool_call_items) == 1
    assert len(tool_call_items[0].tool_calls) == 1
    return tool_call_items[0].tool_calls[0]


async def _output_stream(*events: OutputIR) -> AsyncIterator[OutputIR]:
    for event in events:
        yield event


@pytest.mark.parametrize(
    ("arguments", "expected_args", "expected_raw_args"),
    [
        ('{"city":"Paris"}', {"city": "Paris"}, None),
        ("{bad json", {}, "{bad json"),
        ('["not","an","object"]', {}, '["not","an","object"]'),
    ],
)
def test_chat_completion_history_tool_args_are_safe(
    arguments: str,
    expected_args: dict[str, Any],
    expected_raw_args: str | None,
) -> None:
    request = OpenAIChatCompletionRequest(
        model="gpt-test",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            }
        ],
    )

    tool_call = _single_history_tool_call(request.to_ir().input)

    assert tool_call.args == expected_args
    assert tool_call.raw_args == expected_raw_args


@pytest.mark.parametrize(
    ("arguments", "expected_args", "expected_raw_args"),
    [
        ('{"city":"Paris"}', {"city": "Paris"}, None),
        ("{bad json", {}, "{bad json"),
        ('["not","an","object"]', {}, '["not","an","object"]'),
    ],
)
def test_responses_history_tool_args_are_safe(
    arguments: str,
    expected_args: dict[str, Any],
    expected_raw_args: str | None,
) -> None:
    request = OpenAIResponsesRequest(
        model="gpt-test",
        input=[
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": arguments,
            }
        ],
    )

    tool_call = _single_history_tool_call(request.to_ir().input)

    assert tool_call.args == expected_args
    assert tool_call.raw_args == expected_raw_args


async def test_chat_completion_response_preserves_raw_tool_args() -> None:
    ir = OpenAIChatCompletionRequest(
        model="gpt-test",
        messages=[{"role": "user", "content": "hello"}],
    ).to_ir()

    response = await chat_completions_from_ir(
        ir,
        _output_stream(
            AgentRunResult(
                tool_calls=[
                    AgentToolCall(
                        tool_call_id="call_1",
                        tool_name="lookup",
                        args={},
                        raw_args='["not","an","object"]',
                    )
                ]
            )
        ),
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.arguments == '["not","an","object"]'


async def test_responses_response_preserves_raw_tool_args() -> None:
    ir = OpenAIResponsesRequest(model="gpt-test", input="hello").to_ir()

    response = await responses_from_ir(
        ir,
        _output_stream(
            AgentRunResult(
                tool_calls=[
                    AgentToolCall(
                        id="fc_1",
                        tool_call_id="call_1",
                        tool_name="lookup",
                        args={},
                        raw_args='["not","an","object"]',
                    )
                ]
            )
        ),
    )

    assert len(response.output) == 1
    tool_call = response.output[0]
    assert isinstance(tool_call, OpenAIFunctionToolCall)
    assert tool_call.arguments == '["not","an","object"]'
