"""Native history transport tests using explicit synthetic Google extension payloads."""

from copy import deepcopy

import httpx
import pytest
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

from molto import Molto
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    NormalizedRequest,
    PartEndEvent,
    TextOutput,
    Thinking,
    ToolCall,
)
from molto.upstream.gemini import GeminiChatCompletionsClient
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


@pytest.mark.parametrize(
    "item",
    [
        {
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Think"}],
            "format": "named-v1",
        },
        {
            "summary": [{"type": "summary_text", "text": "Think"}],
            "format": "unknown",
        },
        {"summary": [], "encrypted_content": "opaque", "format": "unknown"},
    ],
    ids=["named", "formatted-summary", "formatted-state"],
)
@pytest.mark.parametrize("endpoint", ["chat/completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_format_cannot_be_silently_translated(item, endpoint, stream):
    """Ensures a reasoning format describing opaque state is rejected before translated transport."""

    def handler(request):
        raise AssertionError("Unsupported history must fail before transport")

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        with pytest.raises(UnsupportedInput, match="format cannot be translated"):
            await molto.responses(
                {"model": "model", "stream": stream, "input": [{"type": "reasoning", **item}]},
                endpoint=endpoint,
                provider="vllm",
                api_key="test-key",
            )


def test_unknown_reasoning_format_on_raw_history_is_omitted():
    """Ensures OpenRouter's raw reasoning history can leave its native route without the label."""
    from molto.adapters.openai import OpenAIResponsesAdapter
    from molto.types.openai_responses import OpenAIResponsesCreateRequest

    request = OpenAIResponsesCreateRequest(
        model="model",
        input=[
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": " Think\n"}],
                "format": "unknown",
            }
        ],
    )
    normalized = OpenAIResponsesAdapter(PROVIDER_CAPABILITIES["vllm"]).normalize_request(request)
    assert normalized.input == [Thinking(id="rs_1", content=" Think\n", summary="")]


@pytest.mark.parametrize("coordinator", ["chat_completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_format_output_rejects_unsupported_translation(coordinator, stream):
    """Ensures OpenAI output cannot lose a reasoning format its own API never defines."""
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
                body, endpoint="responses", provider="openai", api_key="test-key"
            )
            if stream:
                [event async for event in result]


@pytest.mark.parametrize("coordinator", ["chat_completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_shared_responses_client_ignores_reasoning_format(coordinator, stream):
    """Ensures a shared protocol client completes output carrying a format it does not model."""
    item = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Think"}],
        "format": "named-v1",
    }
    response = {"id": "resp_1", "created_at": 1, "model": "model", "output": []}
    events = [
        {"type": "response.created", "response": {**response, "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {
            "type": "response.completed",
            "response": {**response, "status": "completed", "output": [item]},
        },
    ]
    sse = "".join(dumps_sse({**event, "sequence_number": i}) for i, event in enumerate(events))
    body = {"model": "model", "messages": [{"role": "user", "content": "Think"}], "stream": stream}
    if coordinator == "messages":
        body["max_tokens"] = 32
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, text=sse, headers={"content-type": "text/event-stream"}
                )
            )
        )
    ) as molto:
        result = await getattr(molto, coordinator)(
            body, endpoint="responses", provider="vllm", api_key="test-key"
        )
        if stream:
            async with result:
                result = collect_sdk_response(
                    "chat/completions" if coordinator == "chat_completions" else coordinator,
                    [event async for event in result],
                )
    if coordinator == "chat_completions":
        assert result.choices[0].message.reasoning_content == "Think"
    else:
        assert [(block.type, block.thinking) for block in result.content] == [
            ("thinking", "Think")
        ]


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


@pytest.mark.parametrize(
    ("extra_content", "signatures"),
    [
        ({"google": {"thought_signature": "state", "unknown": True}, "other": {}}, ["state"]),
        ({"google": {"unknown": True}}, []),
        ({"other": {}}, []),
    ],
    ids=["beside-signature", "without-signature", "without-google"],
)
async def test_google_signature_ignores_unmodelled_keys(extra_content, signatures):
    """Ensures extra Google keys neither fail the response nor hide its signature."""
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: _google_stream([{"content": "Text", "extra_content": extra_content}])
            )
        )
    ) as molto:
        result = await molto.responses(
            {"model": "model", "input": "Hello"},
            endpoint="chat/completions",
            provider="gemini",
            api_key="test-key",
        )
    assert result.output_text == "Text"
    assert [
        item.encrypted_content for item in result.output if item.type == "reasoning"
    ] == signatures


def _google_call(i):
    return {
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": "lookup", "arguments": f'{{"n":{i}}}'},
        "extra_content": {"google": {"thought_signature": f"call-{i}"}},
    }


@pytest.mark.parametrize("text", [True, False])
async def test_google_signatures_normalize_to_ordered_parts(text):
    """Ensures each signature in one response becomes its own part and restores its attachment."""
    state = {"google": {"thought_signature": " message\n"}}
    calls = [_google_call(i) for i in range(2)]
    deltas = [{"content": "Checking."}] if text else []
    deltas.append({"role": "assistant", "extra_content": state})
    deltas.extend({"tool_calls": [{**call, "index": i}]} for i, call in enumerate(calls))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _google_stream(deltas))
    ) as transport:
        client = GeminiChatCompletionsClient(
            transport, api_key="test-key", capabilities=PROVIDER_CAPABILITIES["gemini"]
        )
        events = [
            event async for event in client.events(NormalizedRequest(model="model", stream=True))
        ]
    ends = [event for event in events if isinstance(event, PartEndEvent)]
    parts = [event.part for event in ends]
    assert [event.index for event in ends] == list(range(len(ends)))
    assert parts == [
        *([TextOutput(content="Checking.")] if text else []),
        Thinking(content="", summary="", signature=" message\n"),
        ToolCall(tool_call_id="call_0", tool_name="lookup", args='{"n":0}'),
        Thinking(content="", summary="", signature="call-0"),
        ToolCall(tool_call_id="call_1", tool_name="lookup", args='{"n":1}'),
        Thinking(content="", summary="", signature="call-1"),
    ]
    history = NormalizedRequest(model="model", input=[AssistantInput(content=parts)])
    assert GeminiChatCompletionsClient._native_request(history).model_dump()["messages"] == [
        {
            "role": "assistant",
            **({"content": "Checking."} if text else {}),
            "extra_content": state,
            "tool_calls": calls,
        }
    ]


def test_google_history_rejects_a_second_signature_for_one_attachment():
    """Ensures a signature Gemini has no place for is rejected instead of dropped."""
    parts = [
        TextOutput(content="Text"),
        Thinking(content="", summary="", signature="first"),
        Thinking(content="", summary="", signature="second"),
    ]
    with pytest.raises(UnsupportedInput, match="at most one"):
        GeminiChatCompletionsClient._native_request(
            NormalizedRequest(model="model", input=[AssistantInput(content=parts)])
        )


@pytest.mark.parametrize("coordinator", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_google_message_signature_without_text_round_trips(coordinator, stream):
    """Ensures a message signature that arrives without text is retained and restored."""
    state = {"google": {"thought_signature": " message\n"}}
    call = _google_call(0)
    deltas = [
        {"tool_calls": [{**call, "index": 0}]},
        {"role": "assistant", "extra_content": state},
    ]
    requests = []

    def handler(request):
        requests.append(json_loads(request.content))
        return _google_stream(deltas)

    body = {"model": "model", "stream": stream}
    if coordinator == "responses":
        body["input"] = [{"role": "user", "content": "Check."}]
    else:
        body.update(messages=[{"role": "user", "content": "Check."}], max_tokens=32)

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
            assert [
                (item.type, getattr(item, "encrypted_content", None)) for item in result.output
            ] == [("function_call", None), ("reasoning", "call-0"), ("reasoning", " message\n")]
            body["input"].extend(item.model_dump(exclude_unset=True) for item in result.output)
            body["input"].append(
                {"type": "function_call_output", "call_id": call["id"], "output": "ok"}
            )
        else:
            assert [(item.type, getattr(item, "signature", None)) for item in result.content] == [
                ("tool_use", None),
                ("thinking", "call-0"),
                ("thinking", " message\n"),
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
                        ],
                    },
                ]
            )
        await collect(
            await create(body, endpoint="chat/completions", provider="gemini", api_key="test-key")
        )
    assert requests[1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Check."}]},
        {"role": "assistant", "extra_content": state, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": [{"type": "text", "text": "ok"}]},
    ]


@pytest.mark.parametrize(
    ("provider", "omits_nulls"),
    [
        ("openai", False),
        ("vllm", False),
        ("togetherai", False),
        ("bedrock", False),
        ("fireworks", True),
        ("gemini", True),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_native_history_nulls_reach_only_providers_that_accept_them(
    provider, omits_nulls, stream
):
    """Ensures echoed null message fields are forwarded faithfully except where a provider rejects them."""
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": "state", "nested": None}},
    }
    assistant = {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "audio": None,
        "function_call": None,
        "tool_calls": [call],
    }
    body = {
        "model": "model",
        "stream": stream,
        "messages": [
            {"role": "user", "content": "Look it up."},
            assistant,
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }
    requests = []

    def handler(request):
        requests.append(json_loads(request.content))
        if stream:
            return _google_stream([{"content": "Done."}])
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "created": 1,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        result = await molto.chat_completions(
            deepcopy(body),
            provider=provider,
            api_key="test-key",
            base_url="https://example.test/v1",
        )
        if stream:
            async with result:
                [event async for event in result]
    assert requests[0]["messages"][1] == (
        {"role": "assistant", "tool_calls": [call]} if omits_nulls else assistant
    )


@pytest.mark.parametrize("coordinator", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_google_message_signature_after_unsigned_call_is_rejected(coordinator, stream):
    """Ensures message state cannot silently become tool state on translated continuation."""
    call = _google_call(0)
    call.pop("extra_content")
    deltas = [
        {"tool_calls": [{**call, "index": 0}]},
        {"extra_content": {"google": {"thought_signature": "message-state"}}},
    ]
    responses = []

    def handler(_):
        response = _google_stream(deltas)
        responses.append(response)
        return response

    body = {"model": "model", "stream": stream}
    if coordinator == "responses":
        body["input"] = "Check."
    else:
        body.update(messages=[{"role": "user", "content": "Check."}], max_tokens=32)
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        with pytest.raises(ValueError, match="cannot preserve its attachment"):
            result = await getattr(molto, coordinator)(
                body, endpoint="chat/completions", provider="gemini", api_key="test-key"
            )
            if stream:
                async with result:
                    [event async for event in result]
    assert len(responses) == 1 and responses[0].is_closed


@pytest.mark.parametrize("reasoning", [False, True])
async def test_google_message_signature_without_text_or_calls(reasoning):
    """Ensures standalone message state survives an empty or reasoning-only response."""
    deltas = [{"reasoning_content": "Think"}] if reasoning else []
    deltas.append({"extra_content": {"google": {"thought_signature": "message-state"}}})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _google_stream(deltas))
    ) as transport:
        client = GeminiChatCompletionsClient(
            transport, api_key="test-key", capabilities=PROVIDER_CAPABILITIES["gemini"]
        )
        parts = [
            event.part
            async for event in client.events(NormalizedRequest(model="model", stream=True))
            if isinstance(event, PartEndEvent)
        ]
    assert parts == [
        *([Thinking(content="", summary="Think")] if reasoning else []),
        Thinking(content="", summary="", signature="message-state"),
    ]
    history = GeminiChatCompletionsClient._native_request(
        NormalizedRequest(model="model", input=[AssistantInput(content=parts)])
    )
    assert history.model_dump()["messages"] == [
        {
            "role": "assistant",
            **({"reasoning_content": "Think"} if reasoning else {"content": []}),
            "extra_content": {"google": {"thought_signature": "message-state"}},
        }
    ]


@pytest.mark.parametrize("endpoint", ["chat/completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_raw_unknown_format_history_translates_with_recorded_sdk_output(endpoint, stream):
    """Ensures raw OpenRouter history reaches both protocols and preserves recorded SDK text."""
    from pathlib import Path

    from anthropic import AsyncAnthropic

    from molto.utils.io import read_yaml

    provider = "openai" if endpoint == "chat/completions" else "anthropic"
    cassette = read_yaml(
        Path(__file__).parents[1]
        / "cassettes"
        / "upstream_ir"
        / "text_chat"
        / f"{provider}-{endpoint.replace('/', '_')}-stream-00.yaml"
    )
    requests = []

    def handler(request):
        requests.append(json_loads(request.content))
        return httpx.Response(
            200,
            text="".join(cassette["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        sdk = (AsyncOpenAI if endpoint == "chat/completions" else AsyncAnthropic)(
            api_key="test-key", http_client=transport, max_retries=0
        )
        create = (
            sdk.chat.completions.create if endpoint == "chat/completions" else sdk.messages.create
        )
        async with await create(**cassette["request"]["body"]) as result:
            expected = collect_sdk_response(endpoint, [event async for event in result])
    body = {
        "model": cassette["request"]["body"]["model"],
        "stream": stream,
        "max_output_tokens": 32,
        "input": [
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": " Think\n"}],
                "format": "unknown",
            },
            {"role": "user", "content": "Continue."},
        ],
    }
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        actual = await molto.responses(
            body, endpoint=endpoint, provider="vllm", api_key="test-key"
        )
        if stream:
            async with actual:
                actual = collect_sdk_response("responses", [event async for event in actual])
    assert requests[1] == {
        "model": body["model"],
        "stream": True,
        "max_completion_tokens" if endpoint == "chat/completions" else "max_tokens": 32,
        "messages": [
            {
                "role": "assistant",
                **(
                    {"reasoning_content": " Think\n"}
                    if endpoint == "chat/completions"
                    else {"content": [{"type": "thinking", "thinking": " Think\n"}]}
                ),
            },
            {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
        ],
        **(
            {"stream_options": {"include_usage": True}}
            if endpoint == "chat/completions"
            else {
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "high"},
            }
        ),
    }
    assert actual.output_text == (
        expected.choices[0].message.content
        if endpoint == "chat/completions"
        else "".join(block.text for block in expected.content if block.type == "text")
    )
