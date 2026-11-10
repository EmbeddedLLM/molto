"""Native history transport tests using explicit synthetic Google extension payloads."""

from copy import deepcopy

import httpx
import pytest
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

from molto import Molto
from molto.utils.io import dumps_sse, json_loads
from molto.utils.test import collect_sdk_response


def _history(message):
    """Remove only SDK parser and stream bookkeeping from actual collected output."""
    history = message.model_dump(
        exclude_unset=True,
        exclude={
            "parsed": True,
            "tool_calls": {"__all__": {"index": True, "function": {"parsed_arguments": True}}},
        },
    )
    for field in ("annotations", "audio", "function_call", "refusal"):
        if history.get(field) is None:
            history.pop(field, None)
    return history


async def _consume(result, stream):
    if not stream:
        return result.choices[0].message, [result.model_dump()]
    state = ChatCompletionStreamState()
    events = []
    async for event in result:
        events.append(event.model_dump())
        state.handle_chunk(event)
    return state.get_final_completion().choices[0].message, events


@pytest.mark.parametrize("stream", [False, True])
async def test_google_native_sdk_history_round_trip(stream):
    """Ensures all Google attachments survive native SDK parity and exact next-turn transport."""

    def google(state):
        return {"google": {"thought_signature": state}}

    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "lookup", "arguments": f'{{"n":{i}}}'},
            "extra_content": google(f"call-{i}"),
        }
        for i in range(2)
    ]
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": calls,
        "extra_content": google(" message\n"),
    }
    chunks = []
    for i, call in enumerate(calls):
        first = deepcopy(call)
        first["index"] = i
        first["function"]["arguments"] = '{"n":'
        delta = {"tool_calls": [first]}
        if i == 0:
            delta.update(role="assistant", content=None, extra_content=assistant["extra_content"])
        chunks.append(delta)
        chunks.append({"tool_calls": [{"index": i, "function": {"arguments": f"{i}}}"}}]})
    chunk_base = {
        "id": "chatcmpl_1",
        "created": 1,
        "model": "model",
        "object": "chat.completion.chunk",
    }
    sse = "".join(
        dumps_sse({**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        for delta in chunks
    )
    sse += dumps_sse(
        {**chunk_base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    )
    sse += "data: [DONE]\n\n"
    response = {
        "id": "chatcmpl_1",
        "created": 1,
        "model": "model",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": assistant, "finish_reason": "tool_calls"}],
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "First", "extra_content": google("part-1")},
            {"type": "text", "text": "Second", "extra_content": google("part-2")},
        ],
        "extra_content": google("user"),
    }
    results = [
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": f"file-{i}",
            "extra_content": {
                "google": {
                    "parts": [
                        {
                            "file_data": {
                                "mime_type": "application/pdf",
                                "display_name": f"file-{i}",
                                "file_uri": f"gs://example/file-{i}.pdf",
                            }
                        }
                    ]
                }
            },
        }
        for i, call in enumerate(calls)
    ]
    extra_body = {
        "ordinary_extra": True,
        "extra_body": {"google": {"thinking_config": {"include_thoughts": True}}},
    }
    requests = []

    def handler(request):
        requests.append(json_loads(request.content))
        assert str(request.url) == "https://example.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        if stream:
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as sdk_http:
        sdk = AsyncOpenAI(
            api_key="test-key",
            base_url="https://example.test/v1",
            http_client=sdk_http,
            max_retries=0,
        )
        sdk_message, sdk_events = await _consume(
            await sdk.chat.completions.create(
                model="model", messages=[user], stream=stream, extra_body=extra_body
            ),
            stream,
        )
    assert _history(sdk_message) == assistant
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        actual_message, actual_events = await _consume(
            await molto.chat_completions(
                {"model": "model", "messages": [user], "stream": stream},
                api_key="test-key",
                base_url="https://example.test/v1",
                extra_body=extra_body,
            ),
            stream,
        )
        assert actual_events == sdk_events
        assert _history(actual_message) == assistant
        await _consume(
            await molto.chat_completions(
                {
                    "model": "model",
                    "messages": [user, _history(actual_message), *results],
                    "stream": stream,
                },
                api_key="test-key",
                base_url="https://example.test/v1",
                extra_body=extra_body,
            ),
            stream,
        )
    expected = {"model": "model", "messages": [user], "stream": stream, **extra_body}
    assert requests[:2] == [expected, expected]
    assert requests[2] == {**expected, "messages": [user, assistant, *results]}


@pytest.mark.parametrize("turn", [1, 2])
async def test_recorded_native_tool_history_matches_sdk(turn):
    """Ensures eagerly validated call history retains recorded request bytes and SDK events."""
    from pathlib import Path

    from molto.utils.io import read_yaml

    cassette = read_yaml(
        Path(__file__).parents[1]
        / "cassettes"
        / "upstream_ir"
        / "tool_use"
        / f"openai-chat_completions-stream-{turn:02d}.yaml"
    )
    body = cassette["request"]["body"]
    requests = []

    def handler(request):
        requests.append(json_loads(request.content))
        assert request.url.path == cassette["request"]["path"]
        return httpx.Response(
            cassette["response"]["status_code"],
            text="".join(cassette["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as sdk_http:
        sdk = AsyncOpenAI(api_key="test-key", http_client=sdk_http, max_retries=0)
        expected_message, expected_events = await _consume(
            await sdk.chat.completions.create(**body), True
        )
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        actual_message, actual_events = await _consume(
            await molto.chat_completions(body, api_key="test-key"), True
        )
    assert requests == [body, body]
    assert actual_events == expected_events
    assert _history(actual_message) == _history(expected_message)


@pytest.mark.parametrize("stream", [False, True])
async def test_openrouter_native_reasoning_format_history(stream):
    """Ensures native Responses output carries exact format into the following SDK request."""
    item = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": " Think\n"}],
        "status": "completed",
        "format": "unknown",
    }
    response = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "model",
        "status": "completed",
        "output": [item],
    }
    event = {"type": "response.completed", "sequence_number": 0, "response": response}
    requests = []

    def handler(request):
        requests.append(json_loads(request.content))
        if stream:
            return httpx.Response(
                200, text=dumps_sse(event), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=response)

    async def collect(result):
        if stream:
            events = [event async for event in result]
            return events[-1].response, [event.model_dump() for event in events]
        return result, [result.model_dump()]

    body = {"model": "model", "input": "Think", "stream": stream}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as sdk_http:
        sdk = AsyncOpenAI(api_key="test-key", http_client=sdk_http, max_retries=0)
        expected, expected_events = await collect(await sdk.responses.create(**body))
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        actual, actual_events = await collect(await molto.responses(body, api_key="test-key"))
        assert actual_events == expected_events
        history = [item.model_dump(exclude_unset=True) for item in actual.output]
        assert history == [item]
        await collect(await molto.responses({**body, "input": history}, api_key="test-key"))
    assert requests == [body, body, {**body, "input": [item]}]


@pytest.mark.parametrize("endpoint", ["chat/completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_format_cannot_be_silently_translated(endpoint, stream):
    """Ensures opaque provider reasoning format is rejected before translated transport."""
    from molto.types.exceptions import UnsupportedInput

    def handler(request):
        raise AssertionError("Unsupported history must fail before transport")

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        with pytest.raises(UnsupportedInput, match="format"):
            await molto.responses(
                {
                    "model": "model",
                    "stream": stream,
                    "input": [
                        {
                            "type": "reasoning",
                            "summary": [],
                            "content": [{"type": "reasoning_text", "text": "Think"}],
                            "format": "unknown",
                        }
                    ],
                },
                endpoint=endpoint,
                provider="vllm",
                api_key="test-key",
            )


@pytest.mark.parametrize("coordinator", ["chat_completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_format_output_rejects_unsupported_translation(coordinator, stream):
    """Ensures source reasoning format cannot disappear when projecting provider output."""
    event = {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Think"}],
            "format": "unknown",
        },
    }
    created = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_1",
            "created_at": 1,
            "model": "model",
            "status": "in_progress",
            "output": [],
        },
    }

    def handler(request):
        return httpx.Response(
            200,
            text=dumps_sse(created) + dumps_sse(event),
            headers={"content-type": "text/event-stream"},
        )

    body = {"model": "model", "messages": [{"role": "user", "content": "Think"}], "stream": stream}
    if coordinator == "messages":
        body["max_tokens"] = 32
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        with pytest.raises(ValueError, match="format"):
            result = await getattr(molto, coordinator)(
                body, endpoint="responses", provider="vllm", api_key="test-key"
            )
            if stream:
                [event async for event in result]


def _google_stream(deltas):
    """Build synthetic Google chunks, retaining late attachment snapshots."""
    chunk = {
        "id": "chatcmpl_google",
        "created": 1,
        "model": "model",
        "object": "chat.completion.chunk",
    }
    sse = "".join(
        dumps_sse({**chunk, "choices": [{"index": 0, "delta": delta}]}) for delta in deltas
    )
    sse += dumps_sse(
        {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )
    return httpx.Response(
        200,
        stream=httpx.ByteStream((sse + "data: [DONE]\n\n").encode()),
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize("coordinator", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_google_translated_parallel_signature_history(coordinator, stream):
    """Ensures late signature snapshots retain exact text and parallel-call associations."""
    text_state = {"google": {"thought_signature": " message\n"}}
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "lookup", "arguments": f'{{"n":{i}}}'},
            "extra_content": {"google": {"thought_signature": f"call-{i}"}},
        }
        for i in range(2)
    ]
    deltas = [{"content": "Checking both."}, {"extra_content": text_state}]
    for i, call in enumerate(calls):
        first = deepcopy(call)
        first["index"] = i
        first["function"]["arguments"] = '{"n":'
        first["extra_content"]["google"]["thought_signature"] = f"earlier-{i}"
        deltas.extend(
            [
                {"tool_calls": [first]},
                {"tool_calls": [{"index": i, "function": {"arguments": f"{i}}}"}}]},
                {"tool_calls": [{"index": i, "extra_content": call["extra_content"]}]},
            ]
        )
    requests = []

    def handler(request):
        requests.append(json_loads(request.content))
        return _google_stream(deltas)

    body = {"model": "model", "stream": stream}
    if coordinator == "responses":
        body["input"] = [{"role": "user", "content": "Check both."}]
    else:
        body.update(messages=[{"role": "user", "content": "Check both."}], max_tokens=32)

    async def collect(result):
        if not stream:
            return result
        async with result:
            return collect_sdk_response(coordinator, [event async for event in result])

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        create = getattr(molto, coordinator)
        result = await collect(
            await create(body, endpoint="chat/completions", provider="gemini", api_key="test-key")
        )
        if coordinator == "responses":
            assert [item.type for item in result.output] == [
                "message",
                "reasoning",
                "function_call",
                "reasoning",
                "function_call",
                "reasoning",
            ]
            assert [
                item.encrypted_content for item in result.output if item.type == "reasoning"
            ] == [" message\n", "call-0", "call-1"]
            history = [item.model_dump(exclude_unset=True) for item in result.output]
            body["input"].extend(history)
            body["input"].extend(
                {"type": "function_call_output", "call_id": call["id"], "output": "ok"}
                for call in calls
            )
        else:
            assert [item.type for item in result.content] == [
                "text",
                "thinking",
                "tool_use",
                "thinking",
                "tool_use",
                "thinking",
            ]
            assert [item.signature for item in result.content if item.type == "thinking"] == [
                " message\n",
                "call-0",
                "call-1",
            ]
            body["messages"].extend(
                [
                    {
                        "role": "assistant",
                        "content": [item.model_dump() for item in result.content],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": call["id"], "content": "ok"}
                            for call in calls
                        ],
                    },
                ]
            )
        await collect(
            await create(body, endpoint="chat/completions", provider="gemini", api_key="test-key")
        )
    assert len(requests) == 2
    assert requests[1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Check both."}]},
        {
            "role": "assistant",
            "content": "Checking both.",
            "extra_content": text_state,
            "tool_calls": calls,
        },
        *[
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": [{"type": "text", "text": "ok"}],
            }
            for call in calls
        ],
    ]


@pytest.mark.parametrize(
    "extra_content",
    [
        {"google": {"thought_signature": ""}},
        {"google": {"thought_signature": None}},
        {"google": {"thought_signature": 7}},
        {"google": {"thought_signature": "state", "unknown": True}},
        {"google": "invalid"},
        "invalid",
    ],
)
@pytest.mark.parametrize("placement", ["message", "call"])
async def test_google_malformed_signature_closes_stream(extra_content, placement):
    """Ensures malformed Google state fails explicitly and releases the upstream response."""
    delta = {"content": "Text"}
    if placement == "message":
        delta["extra_content"] = extra_content
    else:
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
                "extra_content": extra_content,
            }
        ]
    responses = []

    def handler(request):
        response = _google_stream([delta])
        responses.append(response)
        return response

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        with pytest.raises(ValueError, match="extra_content"):
            result = await molto.responses(
                {"model": "model", "input": "Hello", "stream": True},
                endpoint="chat/completions",
                provider="gemini",
                api_key="test-key",
            )
            async with result:
                [event async for event in result]
        assert len(responses) == 1 and responses[0].is_closed
