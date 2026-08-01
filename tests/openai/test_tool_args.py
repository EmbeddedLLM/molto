"""Tool-argument preservation across both OpenAI public formats.

Consolidates the near-verbatim `tests/responses/test_tool_args.py` and
`tests/chat_completions/test_tool_args.py` pair: history tool-call arguments
must survive request-to-IR verbatim for both formats, and Responses output
tool-call arguments must survive SDK-payload-to-public-response verbatim.
"""

from __future__ import annotations

import pytest

from molto.providers.openai.output_translation import responses_response_to_output_events
from molto.types.ir.portable import FunctionCallRequestItem, collect_output_ir
from molto.types.openai import OpenAIRequest
from molto.types.openai.responses import (
    OpenAIFunctionToolCall,
    from_output_ir,
)

_ARGUMENTS = [
    '{"city":"Paris"}',
    "{bad json",
    '["not","an","object"]',
]


def _single_history_tool_call(items: object) -> FunctionCallRequestItem:
    assert isinstance(items, tuple)
    tool_call_items = [item for item in items if isinstance(item, FunctionCallRequestItem)]
    assert len(tool_call_items) == 1
    return tool_call_items[0]


@pytest.mark.parametrize("arguments", _ARGUMENTS)
def test_responses_history_tool_arguments_are_preserved_verbatim(arguments: str) -> None:
    request = OpenAIRequest.responses(
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

    tool_call = _single_history_tool_call(request.to_ir().items)

    assert tool_call.arguments == arguments


@pytest.mark.parametrize("arguments", _ARGUMENTS)
def test_chat_history_tool_arguments_are_preserved_verbatim(arguments: str) -> None:
    request = OpenAIRequest.chat(
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

    tool_call = _single_history_tool_call(request.to_ir().items)

    assert tool_call.arguments == arguments


@pytest.mark.parametrize("arguments", _ARGUMENTS)
async def test_responses_output_tool_arguments_are_preserved_verbatim(arguments: str) -> None:
    payload = {
        "id": "resp_1",
        "object": "response",
        "created_at": 123,
        "model": "gpt-test",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "status": "completed",
                "arguments": arguments,
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    output = await collect_output_ir(responses_response_to_output_events(payload))
    response = from_output_ir(output)

    tool_call = response.output[0]
    assert isinstance(tool_call, OpenAIFunctionToolCall)
    assert tool_call.arguments == arguments
