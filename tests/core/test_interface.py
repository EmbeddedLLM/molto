from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Literal

import pytest
from anthropic import AsyncAnthropic
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import AsyncOpenAI
from openai import BadRequestError as OpenAIBadRequestError
from pydantic import BaseModel

from molto import Molto
from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, PROVIDER_CAPABILITIES, Provider
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
)
from molto.types.exceptions import UnsupportedInput
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
)
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.upstream.together import TogetherAIChatCompletionsClient
from molto.utils.io import dumps_sse, json_loads, load_sse
from molto.utils.test import (
    assert_chat_completions_chunks_match,
    assert_contains_expected,
    reconstructable_anthropic_event,
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
)
from tests.core.cassette_helpers import (
    ENCRYPTED_REASONING_SCENARIOS,
    MATCHED_CASSETTE_DIR,
    PROMPT_CACHE_CASES,
    PROMPT_CACHE_SCENARIOS,
    PROTOCOLS,
    RESPONSES_MESSAGES_ROUTES,
    ROUTES,
    TOOL_CALL_SCENARIOS,
    Protocol,
    _assert_matched_response,
    _assert_reconstructable_tool_responses_request,
    _assert_responses_stream_invariants,
    _coordinator,
    _get_http_client,
    _matched_cassette_path,
    _protocol_name,
    _provider_from_cassette_path,
    _read_cassette,
    _sdk_stream_events,
    _split_sdk_request_body,
    _sse_records,
)
from tests.core.history_helpers import (
    _assert_history_matches,
    _collect_response,
    _continue_history,
    _response_history,
)

Endpoint = tuple[Protocol, Provider]
CONVERSATION_SCENARIOS = (
    "simple_text",
    "reasoning_raw",
    "reasoning_encrypted",
    "input_image_base64",
    "input_image_remote_url",
    "input_image_with_reasoning_encrypted",
    *TOOL_CALL_SCENARIOS,
)


def _matched_routes(
    scenario: str, mode: Literal["stream", "non_stream"] = "stream"
) -> tuple[tuple[Endpoint, Endpoint], ...]:
    endpoints = [
        (protocol, _provider_from_cassette_path(path))
        for protocol in PROTOCOLS
        for path in sorted(
            (MATCHED_CASSETTE_DIR / scenario).glob(f"*-{_protocol_name(protocol)}-{mode}-00.yaml")
        )
    ]
    assert endpoints, f"No matched {mode} cassettes found in {scenario!r}"
    if mode == "non_stream":
        return tuple(
            (downstream, upstream)
            for downstream, upstream in _matched_routes(scenario)
            if downstream[0] != upstream[0]
        ) + tuple(
            (downstream, upstream)
            for downstream, upstream in product(endpoints, repeat=2)
            if downstream[0] == upstream[0]
        )
    return tuple(product(endpoints, repeat=2))


def _matched_cassette_paths(
    scenario: str,
    endpoint: Endpoint,
    mode: Literal["stream", "non_stream"] = "stream",
) -> dict[str, Path]:
    protocol, provider = endpoint
    protocol_name = _protocol_name(protocol)
    filename_marker = f"-{protocol_name}-"
    cassette_paths: dict[str, Path] = {}
    for path in sorted(
        (MATCHED_CASSETTE_DIR / scenario).glob(f"{provider}{filename_marker}{mode}-*.yaml")
    ):
        variant = path.name.partition(filename_marker)[2]
        cassette_paths[variant] = path
    return cassette_paths


def _sdk_request(path: Path, protocol: Protocol) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read a recorded request as SDK body arguments and HTTP body extensions."""
    return _split_sdk_request_body(
        deepcopy(_read_cassette(path)["request"]["body"]),
        protocol,
        _provider_from_cassette_path(path),
    )


def _reconstructable_encrypted_responses_event(event: dict[str, Any]) -> dict[str, Any]:
    event.pop("obfuscation", None)

    item = event.get("item")
    if isinstance(item, dict):
        item.pop("status", None)

    response = event.get("response")
    if isinstance(response, dict):
        response.pop("max_output_tokens", None)
        response.pop("reasoning", None)
        for output_item in response.get("output") or []:
            output_item.pop("status", None)
            # output_item.pop("encrypted_content", None)

    return event


def _reconstructable_tool_responses_event(event: dict[str, Any]) -> dict[str, Any]:
    response = event.get("response")
    if not isinstance(response, dict):
        return event

    # These response-envelope values are server defaults or adjustments and are
    # not present in Chat Completions or Messages stream events.
    for field_name in ("max_output_tokens", "reasoning", "service_tier"):
        response.pop(field_name, None)
    for tool in response.get("tools") or []:
        if isinstance(tool, dict):
            tool.pop("output_schema", None)
    return event


def _assert_sse_events_contain(
    actual_sse_events: list[str], expected_sse_events: list[str]
) -> None:
    assert len(actual_sse_events) == len(expected_sse_events)
    for index, (actual_sse, expected_sse) in enumerate(
        zip(actual_sse_events, expected_sse_events, strict=True)
    ):
        assert_contains_expected(
            load_sse(actual_sse),
            load_sse(expected_sse),
            path=f"event[{index}]",
        )


def _completed_responses_body(cassette_path: Path) -> dict[str, Any]:
    cassette = _read_cassette(cassette_path)
    response = cassette["response"]
    if "body" in response:
        return response["body"]
    for record in reversed(_sse_records(cassette, "responses")):
        event = load_sse(record)
        if event.get("type") in {"response.completed", "response.incomplete"}:
            return event["response"]
    raise AssertionError("Responses cassette has no terminal response event.")


async def _assert_matched_non_stream(
    scenario: str,
    downstream_endpoint: Endpoint,
    upstream_endpoint: Endpoint,
) -> None:
    downstream, _ = downstream_endpoint
    upstream, provider = upstream_endpoint
    translated = downstream != upstream
    mode: Literal["stream", "non_stream"] = "stream" if translated else "non_stream"
    downstream_paths = _matched_cassette_paths(scenario, downstream_endpoint, mode)
    upstream_paths = _matched_cassette_paths(scenario, upstream_endpoint, mode)
    responses_endpoint = (
        "responses",
        _provider_from_cassette_path(
            _matched_cassette_path(scenario, "responses", f"{mode}-00.yaml")
        ),
    )
    responses_paths = _matched_cassette_paths(scenario, responses_endpoint, mode)
    assert downstream_paths, f"No matched {downstream!r} {mode} cassettes in {scenario!r}"
    assert downstream_paths.keys() == upstream_paths.keys() == responses_paths.keys(), (
        f"Mismatched cassette variants for {scenario!r}: "
        f"{downstream!r} has {sorted(downstream_paths)}, "
        f"{upstream!r} has {sorted(upstream_paths)}, "
        f"'responses' has {sorted(responses_paths)}"
    )
    variants = sorted(downstream_paths)
    history_key = "input" if downstream == "responses" else "messages"
    history = deepcopy(
        _read_cassette(downstream_paths[variants[0]])["request"]["body"][history_key]
    )
    requests = []

    async with _get_http_client(
        [upstream_paths[variant] for variant in variants],
        upstream,
        translated=translated,
        request_cassette_paths=(
            None if translated else [downstream_paths[variant] for variant in variants]
        ),
        compare_history=True,
        requests=requests,
    ) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        for turn, variant in enumerate(variants):
            request_body, extra_body = _sdk_request(downstream_paths[variant], downstream)
            request_body[history_key] = deepcopy(history)
            if translated:
                _, extra_body = _sdk_request(upstream_paths[variant], upstream)
                request_body["stream"] = False
            response = await coordinator(
                body=request_body,
                endpoint=upstream,
                provider=provider,
                api_key="test-api-key",
                extra_body=extra_body,
            )
            _assert_matched_response(
                response,
                downstream,
                _completed_responses_body(responses_paths[variant]),
            )
            if translated:
                sdk_body, sdk_extra_body = _sdk_request(downstream_paths[variant], downstream)
                sdk_events = await _sdk_stream_events(
                    downstream_paths[variant], downstream, sdk_body, extra_body=sdk_extra_body
                )
                _assert_history_matches(
                    _response_history(response, downstream),
                    _response_history(_collect_response(sdk_events, downstream), downstream),
                    downstream,
                    translated=True,
                )
            assert len(requests) == turn + 1
            if not translated:
                assert json_loads(requests[-1].content) == {**request_body, **(extra_body or {})}
            if turn + 1 < len(variants):
                next_body = _read_cassette(downstream_paths[variants[turn + 1]])["request"]["body"]
                history = _continue_history(
                    history,
                    _response_history(response, downstream),
                    next_body[history_key],
                    downstream,
                    translated=translated,
                )


@pytest.mark.parametrize(
    ("scenario", "variant", "downstream", "upstream"),
    [
        (scenario, variant, downstream, upstream)
        for scenario, variant in [
            ("simple_text", "stream-00.yaml"),
            ("reasoning_raw", "stream-00.yaml"),
            ("reasoning_raw", "stream-01.yaml"),
            ("tool_call", "stream-00.yaml"),
            ("tool_call", "stream-01.yaml"),
            ("tool_call", "stream-02.yaml"),
            *PROMPT_CACHE_CASES,
        ]
        for downstream, upstream in _matched_routes(scenario)
    ],
    ids=str,
)
def test_matched_request_body_matches_across_protocols(
    downstream: Endpoint,
    upstream: Endpoint,
    scenario: str,
    variant: str,
):
    """Ensures matched provider requests preserve equivalent semantics across endpoints."""
    downstream, downstream_provider = downstream
    upstream, upstream_provider = upstream
    downstream_path = _matched_cassette_path(scenario, downstream, variant, downstream_provider)
    upstream_path = _matched_cassette_path(scenario, upstream, variant, upstream_provider)
    downstream_body, extra_body = _sdk_request(downstream_path, downstream)
    request_type = {
        "responses": OpenAIResponsesCreateRequest,
        "chat/completions": OpenAIChatCompletionsCreateRequest,
        "messages": AnthropicMessagesCreateRequest,
    }[downstream]
    request = request_type.model_validate(downstream_body)

    if downstream == upstream:
        actual_body = request.model_dump()
        expected_body = _read_cassette(downstream_path)["request"]["body"]
    else:
        _, extra_body = _sdk_request(upstream_path, upstream)
        expected_body = _read_cassette(upstream_path)["request"]["body"]
        capabilities = PROVIDER_CAPABILITIES[_provider_from_cassette_path(upstream_path)]
        adapter = {
            "responses": OpenAIResponsesAdapter(capabilities),
            "chat/completions": OpenAIChatCompletionsAdapter(capabilities),
            "messages": AnthropicMessagesAdapter(capabilities),
        }[downstream]
        normalized_request = adapter.normalize_request(request)
        project_request = {
            "responses": OpenAIResponsesClient._request_body,
            "chat/completions": (
                TogetherAIChatCompletionsClient._request_body
                if upstream_provider == "togetherai"
                else OpenAIChatCompletionsClient._request_body
            ),
            "messages": AnthropicMessagesClient._request_body,
        }[upstream]
        actual_body = project_request(normalized_request)

    actual_body.update(extra_body or {})

    if scenario == "tool_call" and upstream == "responses" and downstream != upstream:
        _assert_reconstructable_tool_responses_request(actual_body, expected_body)
    else:
        assert actual_body == expected_body


async def _assert_stream_events(
    cassette_path_pairs: list[tuple[Path, Path]],
    downstream_endpoint: Endpoint,
    upstream_endpoint: Endpoint,
) -> None:
    downstream, _ = downstream_endpoint
    upstream, provider = upstream_endpoint
    first_downstream_path, _ = cassette_path_pairs[0]
    scenario = first_downstream_path.parent.name
    conversation = scenario in CONVERSATION_SCENARIOS
    translated = downstream != upstream
    history_key = "input" if downstream == "responses" else "messages"
    history = deepcopy(_read_cassette(first_downstream_path)["request"]["body"][history_key])
    requests = []
    async with _get_http_client(
        [upstream_path for _, upstream_path in cassette_path_pairs],
        upstream,
        translated=downstream != upstream,
        request_cassette_paths=(
            [downstream_path for downstream_path, _ in cassette_path_pairs]
            if downstream == upstream
            else None
        ),
        compare_history=conversation,
        requests=requests,
    ) as http_client:
        client = Molto(http_client=http_client)
        for turn, (downstream_path, upstream_path) in enumerate(cassette_path_pairs):
            cassette = _read_cassette(downstream_path)
            request_body, extra_body = _sdk_request(downstream_path, downstream)
            if conversation:
                request_body[history_key] = deepcopy(history)
            if downstream != upstream:
                _, extra_body = _sdk_request(upstream_path, upstream)
            coordinator = {
                "responses": client.responses,
                "chat/completions": client.chat_completions,
                "messages": client.messages,
            }[downstream]
            response = await coordinator(
                body=request_body,
                endpoint=upstream,
                provider=provider,
                api_key="test-api-key",
                extra_body=extra_body,
            )
            async with response:
                events = [event async for event in response]
            assert len(requests) == turn + 1
            if not translated:
                assert json_loads(requests[-1].content) == {**request_body, **(extra_body or {})}
            expected_sse_events = _sse_records(cassette, downstream)
            actual_sse_events = []
            for event in events:
                assert isinstance(event, BaseModel)
                actual_sse_events.append(dumps_sse(event))
            if downstream == "responses":
                assert len(events) == len(expected_sse_events)
                _assert_responses_stream_invariants(actual_sse_events)
                if scenario in PROMPT_CACHE_SCENARIOS:
                    usage = events[-1].response.usage
                    assert usage.input_tokens_details.cached_tokens > 0
                    assert usage.input_tokens_details.cache_write_tokens == 0
                if upstream == "responses":
                    _assert_sse_events_contain(actual_sse_events, expected_sse_events)
                elif upstream == "chat/completions":
                    # Chat does not expose Responses output-item IDs or the exact terminal
                    # timestamp or phase. Every other translated field must still match exactly.
                    actual_events = [
                        reconstructable_responses_event(sse, body=deepcopy(request_body))
                        for sse in actual_sse_events
                    ]
                    expected_events = [
                        reconstructable_responses_event(sse, body=deepcopy(request_body))
                        for sse in expected_sse_events
                    ]
                    if scenario in TOOL_CALL_SCENARIOS:
                        for event in actual_events + expected_events:
                            _reconstructable_tool_responses_event(event)
                    assert actual_events == expected_events
                elif upstream == "messages":
                    actual_events = [
                        reconstructable_responses_event(sse, body=deepcopy(request_body))
                        for sse in actual_sse_events
                    ]
                    expected_events = [
                        reconstructable_responses_event(sse, body=deepcopy(request_body))
                        for sse in expected_sse_events
                    ]
                    for event in actual_events + expected_events:
                        response_data = event.get("response")
                        if isinstance(response_data, dict):
                            response_data.pop("created_at", None)
                        if scenario in ENCRYPTED_REASONING_SCENARIOS:
                            _reconstructable_encrypted_responses_event(event)
                        if scenario in TOOL_CALL_SCENARIOS:
                            _reconstructable_tool_responses_event(event)
                    assert actual_events == expected_events
            elif downstream == "chat/completions":
                if upstream == "chat/completions":
                    sdk_events = await _sdk_stream_events(
                        upstream_path, upstream, request_body, extra_body=extra_body
                    )
                    assert [type(event) for event in events] == [
                        type(event) for event in sdk_events
                    ]
                    assert [
                        event.model_dump(mode="json", exclude_unset=True) for event in events
                    ] == [
                        event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
                    ]
                elif upstream in {"responses", "messages"}:
                    actual_events = [
                        chunk
                        for sse in actual_sse_events
                        for chunk in reconstructable_chat_completions_chunks(
                            sse, body=deepcopy(request_body)
                        )
                    ]
                    expected_events = [
                        chunk
                        for sse in expected_sse_events
                        for chunk in reconstructable_chat_completions_chunks(
                            sse, body=deepcopy(request_body)
                        )
                    ]
                    if upstream == "messages":
                        for chunk in actual_events + expected_events:
                            chunk.pop("created", None)
                    assert_chat_completions_chunks_match(actual_events, expected_events)
            elif downstream == "messages":
                if upstream == "messages":
                    assert len(events) == len(expected_sse_events)
                    _assert_sse_events_contain(actual_sse_events, expected_sse_events)
                else:
                    actual_events = [
                        reconstructable_anthropic_event(sse) for sse in actual_sse_events
                    ]
                    expected_events = [
                        reconstructable_anthropic_event(sse) for sse in expected_sse_events
                    ]
                    assert actual_events == expected_events
            else:
                raise ValueError(f"Unsupported downstream protocol: {downstream}")
            if conversation and turn + 1 < len(cassette_path_pairs):
                next_path, _ = cassette_path_pairs[turn + 1]
                next_body = _read_cassette(next_path)["request"]["body"]
                history = _continue_history(
                    history,
                    _response_history(_collect_response(events, downstream), downstream),
                    next_body[history_key],
                    downstream,
                    translated=translated,
                )


async def _assert_matched_stream_events(
    scenario: str,
    downstream: Endpoint,
    upstream: Endpoint,
) -> None:
    downstream_paths = _matched_cassette_paths(scenario, downstream)
    upstream_paths = _matched_cassette_paths(scenario, upstream)
    assert downstream_paths, f"No matched {downstream!r} cassettes found in {scenario!r}"
    assert downstream_paths.keys() == upstream_paths.keys(), (
        f"Mismatched cassette variants for {scenario!r}: "
        f"{downstream!r} has {sorted(downstream_paths)}, "
        f"{upstream!r} has {sorted(upstream_paths)}"
    )
    cassette_path_pairs = [
        (downstream_paths[variant], upstream_paths[variant])
        for variant in sorted(downstream_paths)
    ]
    await _assert_stream_events(cassette_path_pairs, downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("simple_text", "non_stream"), ids=str
)
async def test_simple_text_non_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched simple text cassettes agree across provider endpoints."""
    await _assert_matched_non_stream("simple_text", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), _matched_routes("simple_text"), ids=str)
async def test_simple_text_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched simple text cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("simple_text", downstream, upstream)


@pytest.mark.parametrize(
    ("scenario", "downstream", "upstream"),
    [
        (scenario, downstream, upstream)
        for scenario in PROMPT_CACHE_SCENARIOS
        for downstream, upstream in _matched_routes(scenario)
    ],
    ids=str,
)
async def test_prompt_cache_stream(scenario: str, downstream: Endpoint, upstream: Endpoint):
    """Ensures matched prompt cache cassettes agree across provider endpoints."""
    await _assert_matched_stream_events(scenario, downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), _matched_routes("reasoning_raw"), ids=str)
async def test_reasoning_raw_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched reasoning raw cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("reasoning_raw", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("reasoning_encrypted"), ids=str
)
async def test_reasoning_encrypted_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched reasoning encrypted cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("reasoning_encrypted", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("reasoning_encrypted"), ids=str
)
async def test_reasoning_encrypted_non_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched reasoning encrypted cassettes agree across provider endpoints."""
    await _assert_matched_non_stream("reasoning_encrypted", downstream, upstream)


@pytest.mark.parametrize(
    ("scenario", "downstream", "upstream"),
    [
        (scenario, downstream, upstream)
        for scenario in CONVERSATION_SCENARIOS
        if scenario not in {"simple_text", "reasoning_encrypted"}
        for downstream, upstream in _matched_routes(scenario)
        if downstream[0] != upstream[0]
    ],
    ids=str,
)
async def test_translated_conversation_non_stream(
    scenario: str, downstream: Endpoint, upstream: Endpoint
):
    """Ensures collected translated responses survive continuation and match next-turn YAML."""
    await _assert_matched_non_stream(scenario, downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("input_image_base64"), ids=str
)
async def test_input_image_base64_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched input image base64 cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("input_image_base64", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("input_image_remote_url"), ids=str
)
async def test_input_image_remote_url_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched input image remote url cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("input_image_remote_url", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("input_image_with_reasoning_encrypted"), ids=str
)
async def test_input_image_with_reasoning_encrypted_stream(
    downstream: Endpoint, upstream: Endpoint
):
    """Ensures matched input image with reasoning encrypted cassettes agree across provider endpoints."""
    await _assert_matched_stream_events(
        "input_image_with_reasoning_encrypted", downstream, upstream
    )


@pytest.mark.parametrize(("downstream", "upstream"), _matched_routes("tool_call"), ids=str)
async def test_tool_call_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched tool call cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("tool_call", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), _matched_routes("tool_call_choice"), ids=str)
async def test_tool_call_choice_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched tool call choice cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("tool_call_choice", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("tool_call_image_result_parallel"), ids=str
)
async def test_tool_call_image_result_parallel_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched tool call image result parallel cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("tool_call_image_result_parallel", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("tool_call_parallel"), ids=str
)
async def test_tool_call_parallel_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched tool call parallel cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("tool_call_parallel", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("tool_call_with_reasoning_encrypted"), ids=str
)
async def test_tool_call_with_reasoning_encrypted_stream(downstream: Endpoint, upstream: Endpoint):
    """Ensures matched tool call with reasoning encrypted cassettes agree across provider endpoints."""
    await _assert_matched_stream_events("tool_call_with_reasoning_encrypted", downstream, upstream)


@pytest.mark.parametrize(
    ("downstream", "upstream"), _matched_routes("error-invalid_request_error"), ids=str
)
async def test_error_matched(downstream: Endpoint, upstream: Endpoint):
    """Ensures invalid-request errors preserve upstream SDK behavior across provider routes."""
    scenario = "error-invalid_request_error"
    variant = "stream-00.yaml"
    downstream, downstream_provider = downstream
    upstream, upstream_provider = upstream
    downstream_path = _matched_cassette_path(scenario, downstream, variant, downstream_provider)
    upstream_path = _matched_cassette_path(scenario, upstream, variant, upstream_provider)
    downstream_cassette = _read_cassette(downstream_path)
    upstream_cassette = _read_cassette(upstream_path)
    expected_response = upstream_cassette["response"]
    error_type = AnthropicBadRequestError if upstream == "messages" else OpenAIBadRequestError
    translated = downstream != upstream
    body, extra_body = _sdk_request(downstream_path, downstream)
    expected_body = deepcopy(downstream_cassette["request"]["body"])
    if translated:
        expected_body = deepcopy(upstream_cassette["request"]["body"])
        _, extra_body = _sdk_request(upstream_path, upstream)
        if upstream == "chat/completions" and "max_tokens" in expected_body:
            # Translated Chat requests serialize the limit as max_completion_tokens;
            # native requests retain the caller's exact field, including across providers.
            expected_body["max_completion_tokens"] = expected_body.pop("max_tokens")

    requests = []
    async with _get_http_client(
        [upstream_path],
        upstream,
        translated=translated,
        expected_body=expected_body,
        requests=requests,
    ) as http_client:
        with pytest.raises(error_type) as exc_info:
            await _coordinator(Molto(http_client=http_client), downstream)(
                body=body,
                endpoint=upstream,
                provider=upstream_provider,
                api_key="test-api-key",
                extra_body=extra_body,
            )
        assert exc_info.value.response.is_closed

    assert len(requests) == 1
    assert json_loads(requests[0].content) == expected_body
    sdk_body, sdk_extra_body = _split_sdk_request_body(
        deepcopy(expected_body),
        upstream,
        upstream_provider if translated else downstream_provider,
    )
    sdk_requests = []
    async with _get_http_client(
        [upstream_path], upstream, expected_body=expected_body, requests=sdk_requests
    ) as http_client:
        if upstream == "messages":
            sdk = AsyncAnthropic(api_key="test-api-key", http_client=http_client, max_retries=0)
            create = sdk.messages.create
        else:
            sdk = AsyncOpenAI(
                api_key="test-api-key",
                http_client=http_client,
                max_retries=0,
                base_url=(
                    "https://api.together.ai/v1"
                    if upstream_provider == "togetherai"
                    else "https://api.openai.com/v1"
                ),
            )
            create = (
                sdk.responses.create if upstream == "responses" else sdk.chat.completions.create
            )
        with pytest.raises(error_type) as sdk_exc_info:
            await create(**sdk_body, extra_body=sdk_extra_body)
        assert sdk_exc_info.value.response.is_closed

    assert len(sdk_requests) == 1
    assert json_loads(sdk_requests[0].content) == expected_body
    error, sdk_error = exc_info.value, sdk_exc_info.value
    assert type(error) is type(sdk_error) is error_type
    assert error.__cause__ is sdk_error.__cause__ is None
    assert error.status_code == sdk_error.status_code == expected_response["status_code"]
    assert error.message == sdk_error.message
    assert error.body == sdk_error.body
    assert error.response.content == sdk_error.response.content
    assert error.response.headers == sdk_error.response.headers
    assert error.request is error.response.request is requests[0]
    response_body = expected_response["body"]
    if isinstance(response_body, str):
        assert error.response.content == response_body.encode()
        response_body = json_loads(response_body)
    assert error.response.json() == response_body


@pytest.mark.parametrize("upstream", ["chat/completions", "messages"])
async def test_responses_translation_ignores_disabled_store(upstream: Protocol):
    """Ensures translated Responses requests ignore behavior-neutral store=false."""
    variant = "stream-00.yaml"
    downstream_path = _matched_cassette_path("simple_text", "responses", variant)
    upstream_path = _matched_cassette_path("simple_text", upstream, variant)
    request_body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
    request_body["store"] = False

    async with _get_http_client([upstream_path], upstream, translated=True) as http_client:
        client = Molto(http_client=http_client)
        stream = await client.responses(
            body=request_body,
            endpoint=upstream,
            provider=_provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
        )

        assert [event async for event in stream]


_OPENAI_PROMPT_CACHE: dict[str, Any] = {
    "prompt_cache_key": "codex-turn-1",
    "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
    "prompt_cache_retention": "24h",
}

_MESSAGES_PROMPT_CACHE: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}


def _simple_text_cassette(protocol: Protocol) -> Path:
    return _matched_cassette_path(
        "simple_text", protocol, "stream-00.yaml", provider=DEFAULT_PROVIDER_BY_ENDPOINT[protocol]
    )


async def _assert_request_parity(
    downstream: Protocol,
    upstream: Protocol,
    additions: dict[str, Any],
    expected_additions: dict[str, Any],
    *,
    provider: Provider | None = None,
) -> None:
    """Assert the outgoing request and parity with parsed downstream SDK events."""
    downstream_path = _simple_text_cassette(downstream)
    upstream_path = _simple_text_cassette(upstream)
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"]) | additions
    expected_body = deepcopy(_read_cassette(upstream_path)["request"]["body"]) | expected_additions

    async with _get_http_client(
        [upstream_path],
        upstream,
        translated=downstream != upstream,
        expected_body=expected_body,
    ) as http_client:
        stream = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body,
            endpoint=upstream,
            provider=provider or _provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
        )
        events = [event async for event in stream]

    # Reuse the recorded response without rewriting fixtures: these tests exercise
    # request projection and stream fidelity, not the provider's acceptance of the modified input.
    async with _get_http_client([downstream_path], downstream, expected_body=body) as http_client:
        if downstream == "messages":
            sdk_stream = await AsyncAnthropic(
                api_key="test-api-key", http_client=http_client
            ).messages.create(**body)
        else:
            sdk_client = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
            if downstream == "responses":
                sdk_stream = await sdk_client.responses.create(**body)
            else:
                sdk_stream = await sdk_client.chat.completions.create(**body)
        async with sdk_stream:
            sdk_events = [event async for event in sdk_stream]

    if downstream == upstream:
        assert [type(event) for event in events] == [type(event) for event in sdk_events]
        assert [event.model_dump(mode="json", exclude_unset=True) for event in events] == [
            event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
        ]
        return

    actual_sse = [dumps_sse(event) for event in events]
    expected_sse = [dumps_sse(event) for event in sdk_events]
    # The recorded response predates these request overrides, so compare its original
    # reconstructible fields rather than treating its request metadata as a fresh result.
    recorded_body = _read_cassette(downstream_path)["request"]["body"]
    if downstream == "responses":
        _assert_responses_stream_invariants(actual_sse)
        actual = [reconstructable_responses_event(sse, body=recorded_body) for sse in actual_sse]
        expected = [
            reconstructable_responses_event(sse, body=recorded_body) for sse in expected_sse
        ]
        if upstream == "messages":
            for event in actual + expected:
                if isinstance(event.get("response"), dict):
                    event["response"].pop("created_at", None)
    elif downstream == "chat/completions":
        actual = [
            chunk
            for sse in actual_sse
            for chunk in reconstructable_chat_completions_chunks(sse, body=body)
        ]
        expected = [
            chunk
            for sse in expected_sse
            for chunk in reconstructable_chat_completions_chunks(sse, body=body)
        ]
        if upstream == "messages":
            for chunk in actual + expected:
                chunk.pop("created", None)
    else:
        actual = [reconstructable_anthropic_event(sse) for sse in actual_sse]
        expected = [reconstructable_anthropic_event(sse) for sse in expected_sse]
    if downstream == "chat/completions":
        assert_chat_completions_chunks_match(actual, expected)
    else:
        assert actual == expected


async def _assert_prompt_cache_rejected(
    downstream: Protocol,
    upstream: Protocol,
    additions: dict[str, Any],
    match: str,
) -> None:
    """Assert an unportable cache value fails before any upstream request is made."""
    downstream_path = _simple_text_cassette(downstream)
    upstream_path = _simple_text_cassette(upstream)
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"]) | additions

    async with _get_http_client([]) as http_client:
        with pytest.raises(UnsupportedInput, match=match):
            await _coordinator(Molto(http_client=http_client), downstream)(
                body=body,
                endpoint=upstream,
                provider=_provider_from_cassette_path(upstream_path),
                api_key="test-api-key",
            )


@pytest.mark.parametrize(
    ("downstream", "upstream", "representation"),
    [(downstream, upstream, "raw") for downstream, upstream in ROUTES]
    + [
        (downstream, upstream, representation)
        for downstream, upstream in RESPONSES_MESSAGES_ROUTES
        for representation in ("summary", "signature")
    ],
)
@pytest.mark.parametrize("followed_by_user", [False, True])
async def test_reasoning_only_history(
    downstream: Protocol,
    upstream: Protocol,
    representation: str,
    followed_by_user: bool,
) -> None:
    """Ensures representable reasoning-only history reaches every upstream protocol."""
    reasoning = "Prior thought" if representation != "signature" else ""
    signed = representation != "raw"
    responses_item = {
        "type": "reasoning",
        "status": "completed",
        "content": [] if signed else [{"type": "reasoning_text", "text": reasoning}],
        "summary": [{"type": "summary_text", "text": reasoning}] if reasoning and signed else [],
        **({"encrypted_content": "opaque-state"} if signed else {}),
    }
    thinking_block = {
        "type": "thinking",
        "thinking": reasoning,
        **({"signature": "opaque-state"} if signed else {}),
    }
    additions: dict[Protocol, dict[str, Any]] = {
        "responses": {"input": [responses_item], "reasoning": {"effort": "none"}},
        "chat/completions": {
            "messages": [{"role": "assistant", "reasoning_content": reasoning}],
            "reasoning_effort": "none",
        },
        "messages": {
            "messages": [{"role": "assistant", "content": [thinking_block]}],
            "thinking": {"type": "disabled"},
        },
    }
    if followed_by_user:
        for protocol, fields in additions.items():
            key = "input" if protocol == "responses" else "messages"
            part_type = "input_text" if protocol == "responses" else "text"
            fields[key].append(
                {"role": "user", "content": [{"type": part_type, "text": "Continue"}]}
            )
    await _assert_request_parity(
        downstream,
        upstream,
        additions[downstream],
        additions[upstream],
        provider="openai" if signed else "vllm",
    )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("marked", [False, True])
async def test_assistant_history_message_boundaries(
    downstream: Protocol, upstream: Protocol, marked: bool
) -> None:
    """Ensures multipart assistant history retains parts, markers, and adjacent messages."""

    def history(protocol: Protocol) -> dict[str, Any]:
        messages = []
        for texts in (("First part.", "Second part."), ("Separate message.",)):
            content = []
            for index, text in enumerate(texts):
                part: dict[str, Any] = {"type": "text", "text": text}
                if protocol == "responses":
                    part.update(type="output_text", annotations=[], logprobs=[])
                if marked and index == 0:
                    if protocol == "messages":
                        part["cache_control"] = {"type": "ephemeral"}
                    else:
                        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
                content.append(part)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if protocol == "responses":
                message.update(type="message", status="completed")
            elif protocol == "chat/completions" and len(content) == 1 and not marked:
                message["content"] = texts[0]
            messages.append(message)
        additions = {"input" if protocol == "responses" else "messages": messages}
        if marked and protocol != "messages":
            additions["prompt_cache_options"] = {"mode": "explicit"}
        return additions

    additions = history(downstream)
    expected = history(upstream)
    if marked and downstream != upstream and downstream != "messages":
        if upstream == "messages":
            for message in expected["messages"]:
                message["content"][0]["cache_control"]["ttl"] = "1h"
        else:
            expected["prompt_cache_options"]["ttl"] = "30m"
    await _assert_request_parity(downstream, upstream, additions, expected)


@pytest.mark.parametrize("downstream", ["chat/completions", "messages"])
@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_separate_reasoning_and_text_history_messages(
    downstream: Protocol, upstream: Protocol
) -> None:
    """Ensures an explicit reasoning-only assistant message never absorbs the next message."""
    history: dict[Protocol, dict[str, Any]] = {
        "chat/completions": {
            "messages": [
                {"role": "assistant", "reasoning_content": "Prior thought"},
                {"role": "assistant", "content": "Prior answer"},
            ],
            "reasoning_effort": "none",
        },
        "messages": {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "Prior thought"}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Prior answer"}]},
            ],
            "thinking": {"type": "disabled"},
        },
        "responses": {
            "input": [
                {
                    "type": "reasoning",
                    "status": "completed",
                    "content": [{"type": "reasoning_text", "text": "Prior thought"}],
                    "summary": [],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Prior answer",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
            ],
            "reasoning": {"effort": "none"},
        },
    }
    await _assert_request_parity(
        downstream, upstream, history[downstream], history[upstream], provider="vllm"
    )


@pytest.mark.parametrize("with_text_and_tools", [False, True])
@pytest.mark.parametrize(
    ("reasoning_parts", "expected_reasoning"),
    [(["First", "second"], "First\nsecond"), (["First", "", "second"], "First\n\nsecond")],
)
async def test_chat_groups_consecutive_assistant_parts(
    with_text_and_tools: bool, reasoning_parts: list[str], expected_reasoning: str
) -> None:
    """Ensures reasoning joins with newlines while parts stay within their source message."""
    parts: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": text} for text in reasoning_parts
    ]
    expected_message: dict[str, Any] = {
        "role": "assistant",
        "reasoning_content": expected_reasoning,
    }
    if with_text_and_tools:
        text_parts = [{"type": "text", "text": text} for text in ("Answer ", "continued")]
        calls = [
            {"type": "tool_use", "id": call_id, "name": "lookup", "input": {}}
            for call_id in ("c1", "c2")
        ]
        parts.extend([*text_parts, *calls])
        expected_message["content"] = text_parts
        expected_message["tool_calls"] = [
            {
                "type": "function",
                "id": call["id"],
                "function": {"name": "lookup", "arguments": "{}"},
            }
            for call in calls
        ]
    await _assert_request_parity(
        "messages",
        "chat/completions",
        {
            "messages": [
                {"role": "assistant", "content": parts},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "Separate"}]},
            ],
            "thinking": {"type": "disabled"},
        },
        {
            "messages": [
                expected_message,
                {"role": "assistant", "reasoning_content": "Separate"},
            ],
            "reasoning_effort": "none",
        },
        provider="vllm",
    )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (
            [{"type": "text", "text": "Answer"}, {"type": "thinking", "thinking": "Later"}],
            "cannot preserve this assistant reasoning order",
        ),
        (
            [
                {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
                {"type": "thinking", "thinking": "After tool"},
            ],
            "cannot preserve this assistant reasoning order",
        ),
        (
            [
                {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
                {"type": "text", "text": "After tool"},
            ],
            "cannot preserve assistant text after tool calls",
        ),
    ],
)
async def test_chat_rejects_unrepresentable_assistant_part_order(content, match) -> None:
    """Ensures Chat rejects assistant order it cannot preserve within one message."""
    body = deepcopy(_read_cassette(_simple_text_cassette("messages"))["request"]["body"])
    body["messages"] = [{"role": "assistant", "content": content}]
    async with _get_http_client([]) as http_client:
        with pytest.raises(UnsupportedInput, match=match):
            await Molto(http_client=http_client).messages(
                body=body,
                endpoint="chat/completions",
                provider="vllm",
                api_key="test-api-key",
            )


@pytest.mark.parametrize("upstream", ["responses", "messages"])
async def test_interleaved_assistant_history(upstream: Protocol) -> None:
    """Ensures mixed assistant parts retain text runs and order around reasoning and calls."""
    first_texts = [{"type": "text", "text": text} for text in ("First", "Second")]
    last_texts = [{"type": "text", "text": text} for text in ("Third", "Fourth")]
    additions = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    *first_texts,
                    {"type": "thinking", "thinking": "A thought"},
                    {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
                    *last_texts,
                ],
            }
        ],
        "thinking": {"type": "disabled"},
    }
    if upstream == "messages":
        expected = additions
    else:
        text_messages = [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": part["text"],
                        "annotations": [],
                        "logprobs": [],
                    }
                    for part in parts
                ],
            }
            for parts in (first_texts, last_texts)
        ]
        expected = {
            "input": [
                text_messages[0],
                {
                    "type": "reasoning",
                    "status": "completed",
                    "content": [{"type": "reasoning_text", "text": "A thought"}],
                    "summary": [],
                },
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{}",
                    "call_id": "c1",
                    "status": "completed",
                },
                text_messages[1],
            ],
            "reasoning": {"effort": "none"},
        }
    await _assert_request_parity("messages", upstream, additions, expected, provider="vllm")


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_prompt_cache_translates_across_protocols(
    downstream: Protocol,
    upstream: Protocol,
):
    """Ensures cache settings reach every upstream route while preserving SDK event parity."""
    if downstream == "messages":
        additions = deepcopy(_MESSAGES_PROMPT_CACHE)
        expected_additions = (
            additions if upstream == "messages" else {"prompt_cache_options": {"mode": "implicit"}}
        )
        await _assert_request_parity(downstream, upstream, additions, expected_additions)
        return

    additions = deepcopy(_OPENAI_PROMPT_CACHE)
    if upstream == "messages":
        additions.pop("prompt_cache_retention")
        expected_additions = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    else:
        expected_additions = deepcopy(additions)
    await _assert_request_parity(downstream, upstream, additions, expected_additions)


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
@pytest.mark.parametrize("key", [None, "codex-turn-1"])
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (None, {}),
        ({}, {}),
        ({"mode": "implicit"}, {"cache_control": {"type": "ephemeral"}}),
        ({"ttl": "30m"}, {"cache_control": {"type": "ephemeral", "ttl": "1h"}}),
        ({"mode": "explicit"}, {}),
        ({"mode": "explicit", "ttl": "30m"}, {}),
    ],
)
async def test_messages_prompt_cache_options(downstream, key, options, expected):
    """Ensures Messages cache mode and lifetime are independent of the dropped caller key."""
    additions = {}
    if key is not None:
        additions["prompt_cache_key"] = key
    if options is not None:
        additions["prompt_cache_options"] = options
    await _assert_request_parity(
        downstream,
        "messages",
        additions,
        expected,
    )


@pytest.mark.parametrize(
    ("downstream", "additions", "upstream", "match"),
    [
        ("responses", {"prompt_cache_retention": "24h"}, "messages", "prompt cache retention"),
        (
            "chat/completions",
            {"prompt_cache_key": "turn", "prompt_cache_retention": "in_memory"},
            "messages",
            "prompt cache retention",
        ),
    ],
)
async def test_prompt_cache_rejects_unportable_values(
    downstream: Protocol,
    additions: dict[str, Any],
    upstream: Protocol,
    match: str,
):
    """Asserts that Messages routes reject retention guarantees they cannot preserve."""
    await _assert_prompt_cache_rejected(downstream, upstream, additions, match)


def _breakpoint_input(protocol: Protocol, position: str, marker: dict[str, Any]) -> dict[str, Any]:
    text_type = "input_text" if protocol == "responses" else "text"
    field = "cache_control" if protocol == "messages" else "prompt_cache_breakpoint"
    part = {"type": text_type, "text": "Cached prefix.", field: marker}
    suffix = {"type": text_type, "text": "Uncached suffix."}
    if position in {"image", "tool_image"}:
        part = (
            {"type": "input_image", "image_url": "https://example.com/image.png", field: marker}
            if protocol == "responses"
            else {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
                field: marker,
            }
            if protocol == "chat/completions"
            else {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image.png"},
                field: marker,
            }
        )
    user = {"role": "user", "content": [suffix]}
    messages = [user]
    if position in {"user", "image"}:
        messages = [{"role": "user", "content": [part, suffix]}]
    elif position == "system":
        if protocol == "messages":
            return {"system": [part, suffix], "messages": [user]}
        messages = [
            {"role": "developer", "content": [part]},
            {"role": "developer", "content": [suffix]},
            user,
        ]
    elif position == "assistant":
        messages = [{"role": "assistant", "content": [part]}, user]
        if protocol == "responses":
            messages[0].update(type="message", status="completed")
            part.update(type="output_text", annotations=[], logprobs=[])
    elif position in {"tool", "tool_image"}:
        if protocol == "responses":
            messages = [
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{}",
                    "call_id": "c1",
                    "status": "completed",
                },
                {"type": "function_call_output", "call_id": "c1", "output": [part, suffix]},
                user,
            ]
        elif protocol == "chat/completions":
            messages = [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": [part, suffix]},
                user,
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "c1", "name": "lookup", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": [part, suffix]}
                    ],
                },
                user,
            ]
    return {"input" if protocol == "responses" else "messages": messages}


@pytest.mark.parametrize(
    ("position", "downstream", "upstream"),
    [
        (position, downstream, upstream)
        for position in ("user", "image", "system", "assistant", "tool", "tool_image")
        for downstream, upstream in ROUTES
        if position != "tool_image" or "chat/completions" not in (downstream, upstream)
    ],
)
@pytest.mark.parametrize("mode", ["implicit", "explicit"])
async def test_cache_breakpoint_translation(downstream, upstream, position, mode):
    """Ensures translation preserves marker positions, cache modes, and SDK event parity."""
    openai_marker = {"mode": "explicit"}
    messages_marker = {"type": "ephemeral"}
    additions = _breakpoint_input(
        downstream, position, messages_marker if downstream == "messages" else openai_marker
    )
    if downstream == "messages":
        if mode == "implicit":
            additions["cache_control"] = {"type": "ephemeral"}
    else:
        additions["prompt_cache_options"] = {"mode": mode, "ttl": "30m"}
    if downstream == upstream:
        expected = deepcopy(additions)
    else:
        marker = {"type": "ephemeral", "ttl": "1h"} if upstream == "messages" else openai_marker
        expected = _breakpoint_input(upstream, position, marker)
        if upstream == "messages":
            if mode == "implicit":
                expected["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        else:
            expected["prompt_cache_options"] = {"mode": mode}
            if downstream != "messages":
                expected["prompt_cache_options"]["ttl"] = "30m"
    await _assert_request_parity(downstream, upstream, additions, expected)


@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_responses_assistant_cache_marker_with_history_id(upstream):
    """Ensures marked assistant history remains valid and Responses routes retain its ID."""
    additions = _breakpoint_input("responses", "assistant", {"mode": "explicit"})
    additions["input"][0]["id"] = "msg_123"
    additions["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    if upstream == "responses":
        expected = deepcopy(additions)
    else:
        marker = (
            {"type": "ephemeral", "ttl": "1h"} if upstream == "messages" else {"mode": "explicit"}
        )
        expected = _breakpoint_input(upstream, "assistant", marker)
        if upstream == "chat/completions":
            expected["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    await _assert_request_parity("responses", upstream, additions, expected)


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
@pytest.mark.parametrize(("count", "mode"), [(5, "explicit"), (5, "implicit"), (4, "implicit")])
async def test_messages_cache_breakpoint_count_is_left_to_upstream(downstream, count, mode):
    """Ensures all markers reach Messages so the upstream owns changing count limits."""
    text_type = "input_text" if downstream == "responses" else "text"
    content = [
        {
            "type": text_type,
            "text": str(i),
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
        for i in range(count)
    ]
    content.append({"type": text_type, "text": "Uncached tail"})
    additions = {
        "input" if downstream == "responses" else "messages": [
            {"role": "user", "content": content}
        ],
        "prompt_cache_options": {"mode": mode, "ttl": "30m"},
    }
    expected_content = [
        {
            "type": "text",
            "text": str(i),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
        for i in range(count)
    ]
    expected_content.append({"type": "text", "text": "Uncached tail"})
    expected = {"messages": [{"role": "user", "content": expected_content}]}
    if mode == "implicit":
        expected["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    await _assert_request_parity(downstream, "messages", additions, expected)


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
async def test_breakpoint_without_options_inherits_openai_defaults(downstream):
    """Ensures marker-only requests carry OpenAI cache defaults to Messages."""
    additions = _breakpoint_input(downstream, "user", {"mode": "explicit"})
    expected = _breakpoint_input("messages", "user", {"type": "ephemeral", "ttl": "1h"})
    expected["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    await _assert_request_parity(downstream, "messages", additions, expected)


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("ttl", ["5m", "1h"])
async def test_messages_request_cache_lifetime(upstream, ttl):
    """Ensures native lifetimes survive and OpenAI routes use the documented approximation."""
    additions = {"cache_control": {"type": "ephemeral", "ttl": ttl}}
    expected = (
        additions
        if upstream == "messages"
        else {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}
    )
    await _assert_request_parity("messages", upstream, additions, expected)


@pytest.mark.parametrize(
    ("upstream", "position"),
    [
        (upstream, position)
        for upstream in PROTOCOLS
        for position in ("user", "image", "system", "assistant", "tool", "tool_image")
        if upstream != "chat/completions" or position != "tool_image"
    ],
)
@pytest.mark.parametrize("ttl", ["5m", "1h"])
@pytest.mark.parametrize("mode", ["implicit", "explicit"])
async def test_messages_block_cache_lifetime(upstream, position, ttl, mode):
    """Ensures lifetime mapping preserves marker positions and automatic-cache mode."""
    additions = _breakpoint_input("messages", position, {"type": "ephemeral", "ttl": ttl})
    if mode == "implicit":
        additions["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    if upstream == "messages":
        expected = additions
    else:
        expected = _breakpoint_input(upstream, position, {"mode": "explicit"})
        expected["prompt_cache_options"] = {"mode": mode, "ttl": "30m"}
    await _assert_request_parity("messages", upstream, additions, expected)


@pytest.mark.parametrize("upstream", ["responses", "chat/completions"])
@pytest.mark.parametrize("automatic", [False, True])
async def test_messages_mixed_cache_lifetimes_use_one_openai_lifetime(upstream, automatic):
    """Ensures mixed lifetimes share one OpenAI TTL without losing markers or cache mode."""
    additions = _breakpoint_input("messages", "user", {"type": "ephemeral", "ttl": "1h"})
    additions["messages"][0]["content"][1]["cache_control"] = {
        "type": "ephemeral",
        "ttl": "5m",
    }
    if automatic:
        additions["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
    expected = _breakpoint_input(upstream, "user", {"mode": "explicit"})
    expected["input" if upstream == "responses" else "messages"][0]["content"][1][
        "prompt_cache_breakpoint"
    ] = {"mode": "explicit"}
    expected["prompt_cache_options"] = {
        "mode": "implicit" if automatic else "explicit",
        "ttl": "30m",
    }
    await _assert_request_parity("messages", upstream, additions, expected)
