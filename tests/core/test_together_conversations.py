"""Historical PR #76 matched conversations; these are not fresh live provider claims."""

from copy import deepcopy
from itertools import product
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from molto import Molto
from molto.utils.io import dumps_sse, json_loads, read_yaml
from molto.utils.test import (
    assert_chat_completions_chunks_match,
    reconstructable_anthropic_event,
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
)
from tests.core.history_helpers import (
    _assert_history_matches,
    _collect_response,
    _continue_history,
    _response_history,
)
from tests.core.test_interface import (
    _assert_responses_stream_invariants,
    _reconstructable_tool_responses_event,
    _without_irrelevant_parallel_tool_calls,
)

ROOT = Path(__file__).parents[1] / "cassettes" / "reference_pr76"
SCENARIOS = (
    "simple_text",
    "reasoning_raw",
    "input_image_base64",
    "input_image_remote_url",
    "tool_call",
    "tool_call_choice",
    "tool_call_parallel",
)


def _endpoint(path):
    provider, _, suffix = path.name.partition("-")
    protocol = suffix.split("-", 1)[0].replace("chat_completions", "chat/completions")
    return protocol, provider


def _paths(scenario, endpoint, mode):
    protocol, provider = endpoint
    return sorted((ROOT / scenario).glob(f"{provider}-{protocol.replace('/', '_')}-{mode}-*.yaml"))


def _cases():
    cases = []
    for scenario in SCENARIOS:
        endpoints = [
            _endpoint(path) for path in sorted((ROOT / scenario).glob("*-stream-00.yaml"))
        ]
        for downstream, upstream in product(endpoints, repeat=2):
            if "togetherai" not in (downstream[1], upstream[1]):
                continue
            for stream in (True, False):
                translated = downstream[0] != upstream[0]
                mode = "stream" if stream or translated else "non_stream"
                if not _paths(scenario, downstream, mode) or not _paths(scenario, upstream, mode):
                    assert mode == "non_stream", "Missing streaming counterpart"
                    continue
                cases.append(
                    pytest.param(
                        scenario,
                        downstream,
                        upstream,
                        stream,
                        id=f"{scenario}-{downstream}-{upstream}-{stream}",
                    )
                )
    return cases


def _sdk_body(body):
    body = deepcopy(body)
    extras = (
        {"reasoning": body.pop("reasoning")}
        if "messages" in body and "reasoning" in body
        else None
    )
    return body, extras


def _response(cassette):
    response = cassette["response"]
    if "sse" in response:
        return httpx.Response(
            response["status_code"],
            text="".join(response["sse"]),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(response["status_code"], json=response["body"])


async def _sdk_result(path, protocol):
    cassette = read_yaml(path)
    body, extras = _sdk_body(cassette["request"]["body"])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _response(cassette))
    ) as transport:
        sdk = (
            AsyncAnthropic(api_key="test", http_client=transport)
            if protocol == "messages"
            else AsyncOpenAI(api_key="test", http_client=transport)
        )
        create = (
            sdk.messages.create
            if protocol == "messages"
            else sdk.responses.create
            if protocol == "responses"
            else sdk.chat.completions.create
        )
        result = await create(**body, extra_body=extras)
        if not body.get("stream", False):
            return result, None
        async with result:
            events = [event async for event in result]
        return _collect_response(events, protocol), events


def _semantics(response, protocol):
    """Extract independently observable content, calls, usage, and termination for comparison."""
    if protocol == "responses":
        text = response.output_text
        calls = [
            (item.call_id, item.name, json_loads(item.arguments))
            for item in response.output
            if item.type == "function_call"
        ]
        reasoning = [
            (
                "".join(part.text for part in item.content or []),
                "".join(part.text for part in item.summary),
                item.encrypted_content,
            )
            for item in response.output
            if item.type == "reasoning"
        ]
        usage = (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens,
        )
        assert response.status == "completed"
    elif protocol == "chat/completions":
        message = response.choices[0].message
        text = message.content or ""
        calls = [
            (call.id, call.function.name, json_loads(call.function.arguments))
            for call in message.tool_calls or []
        ]
        raw = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        reasoning = [(raw, "", None)] if raw else []
        usage = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )
        assert response.choices[0].finish_reason == ("tool_calls" if calls else "stop")
    else:
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            (block.id, block.name, block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        reasoning = [
            (block.thinking, "", block.signature or None)
            for block in response.content
            if block.type == "thinking"
        ]
        input_tokens = (
            response.usage.input_tokens
            + (response.usage.cache_creation_input_tokens or 0)
            + (response.usage.cache_read_input_tokens or 0)
        )
        usage = (
            input_tokens,
            response.usage.output_tokens,
            input_tokens + response.usage.output_tokens,
        )
        assert response.stop_reason == ("tool_use" if calls else "end_turn")
    return response.id, response.model, text, calls, reasoning, usage


def _assert_wire_body(actual, expected, protocol, translated):
    """Compare entire request copies without rewriting any actual continuation history."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    key = "input" if protocol == "responses" else "messages"
    _assert_history_matches(actual.pop(key), expected.pop(key), protocol, translated=translated)
    if translated:
        actual = _without_irrelevant_parallel_tool_calls(actual)
        expected = _without_irrelevant_parallel_tool_calls(expected)
    assert actual == expected


def _assert_translated_events(actual, expected, body, protocol, upstream, scenario):
    """Assert independently parsed observable deltas and lifecycle, including chunk boundaries."""
    actual_sse = [dumps_sse(event) for event in actual]
    expected_sse = [dumps_sse(event) for event in expected]
    if protocol == "responses":
        _assert_responses_stream_invariants(actual_sse)
        actual = [reconstructable_responses_event(event, body) for event in actual_sse]
        expected = [reconstructable_responses_event(event, body) for event in expected_sse]
        for event in actual + expected:
            if upstream == "messages" and isinstance(event.get("response"), dict):
                event["response"].pop("created_at", None)
            if scenario.startswith("tool_call"):
                _reconstructable_tool_responses_event(event)
        assert actual == expected
    elif protocol == "chat/completions":
        actual = [
            chunk
            for event in actual_sse
            for chunk in reconstructable_chat_completions_chunks(event, body)
        ]
        expected = [
            chunk
            for event in expected_sse
            for chunk in reconstructable_chat_completions_chunks(event, body)
        ]
        if upstream == "messages":
            for chunk in actual + expected:
                chunk.pop("created", None)
        assert_chat_completions_chunks_match(actual, expected)
    else:
        assert [reconstructable_anthropic_event(event) for event in actual_sse] == [
            reconstructable_anthropic_event(event) for event in expected_sse
        ]


@pytest.mark.parametrize(("scenario", "downstream", "upstream", "stream"), _cases())
async def test_together_matched_conversations(scenario, downstream, upstream, stream):
    """Ensures actual SDK-compatible outputs become exact next-turn requests across all routes."""
    protocol, _ = downstream
    upstream_protocol, provider = upstream
    translated = protocol != upstream_protocol
    mode = "stream" if stream or translated else "non_stream"
    downstream_paths = _paths(scenario, downstream, mode)
    upstream_paths = _paths(scenario, upstream, mode)
    canonical_paths = sorted((ROOT / scenario).glob(f"*-responses-{mode}-*.yaml"))
    assert len(downstream_paths) == len(upstream_paths) == len(canonical_paths)
    assert len(downstream_paths) >= 2
    key = "input" if protocol == "responses" else "messages"
    history = deepcopy(read_yaml(downstream_paths[0])["request"]["body"][key])
    requests, responses = [], []
    expected_request = None

    def handler(request):
        turn = len(requests)
        requests.append(request)
        assert request.method == "POST"
        host = (
            "api.together.ai"
            if provider == "togetherai"
            else "api.anthropic.com"
            if upstream_protocol == "messages"
            else "api.openai.com"
        )
        assert str(request.url) == f"https://{host}/v1/{upstream_protocol}"
        assert request.headers[
            "x-api-key" if upstream_protocol == "messages" else "authorization"
        ] == ("test" if upstream_protocol == "messages" else "Bearer test")
        _assert_wire_body(
            json_loads(request.content), expected_request, upstream_protocol, translated
        )
        response = _response(read_yaml(upstream_paths[turn]))
        responses.append(response)
        return response

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        for turn, path in enumerate(downstream_paths):
            body, extras = _sdk_body(read_yaml(path)["request"]["body"])
            body[key] = deepcopy(history)
            body["stream"] = stream
            if translated:
                expected_request = deepcopy(read_yaml(upstream_paths[turn])["request"]["body"])
                _, extras = _sdk_body(expected_request)
            else:
                expected_request = {**body, **(extras or {})}
            result = await getattr(client, protocol.replace("/", "_"))(
                body=body,
                endpoint=upstream_protocol,
                provider=provider,
                api_key="test",
                extra_body=extras,
            )
            if stream:
                async with result:
                    events = [event async for event in result]
                result = _collect_response(events, protocol)
            sdk_result, sdk_events = await _sdk_result(
                path if translated else upstream_paths[turn], protocol
            )
            if translated:
                if stream:
                    reference_body, _ = _sdk_body(read_yaml(path)["request"]["body"])
                    _assert_translated_events(
                        events, sdk_events, reference_body, protocol, upstream_protocol, scenario
                    )
                _assert_history_matches(
                    _response_history(result, protocol),
                    _response_history(sdk_result, protocol),
                    protocol,
                    translated=True,
                )
            elif stream:
                assert [event.model_dump() for event in events] == [
                    event.model_dump() for event in sdk_events
                ]
            else:
                assert result.model_dump() == sdk_result.model_dump()
            canonical, _ = await _sdk_result(canonical_paths[turn], "responses")
            assert _semantics(result, protocol) == _semantics(canonical, "responses")
            assert responses[-1].is_closed
            assert len(requests) == turn + 1
            if turn + 1 < len(downstream_paths):
                next_history = read_yaml(downstream_paths[turn + 1])["request"]["body"][key]
                history = _continue_history(
                    history,
                    _response_history(result, protocol),
                    next_history,
                    protocol,
                    translated=translated,
                )
