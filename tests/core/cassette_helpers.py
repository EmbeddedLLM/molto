from copy import deepcopy
from functools import lru_cache
from itertools import pairwise, product
from pathlib import Path
from typing import Any, Literal, cast

from anthropic import AsyncAnthropic
from anthropic import AsyncStream as AnthropicAsyncStream
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI, AsyncStream
from pydantic import BaseModel

from molto import Molto
from molto.adapters.openai import OpenAIResponsesAdapter
from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, PROVIDER_CAPABILITIES, Provider
from molto.types.anthropic_messages import (
    AnthropicMessagesResponse,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsResponse,
)
from molto.types.openai_responses import OpenAIResponsesCreateRequest, OpenAIResponsesResponse
from molto.utils.io import json_loads, load_sse, read_yaml
from tests.core.history_helpers import _assert_history_matches

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes"
MATCHED_CASSETTE_DIR = CASSETTE_DIR / "matched"
Protocol = Literal["responses", "chat/completions", "messages"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions", "messages")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))
RESPONSES_MESSAGES_PROTOCOLS: tuple[Protocol, ...] = ("responses", "messages")
RESPONSES_MESSAGES_ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(
    product(RESPONSES_MESSAGES_PROTOCOLS, repeat=2)
)
# Scenarios whose Responses streams carry encrypted reasoning. Messages has no
# equivalents for Responses include or stream options, so these routes compare
# reconstructible requests and events instead of exact wire values.
ENCRYPTED_REASONING_SCENARIOS: tuple[str, ...] = (
    "reasoning_encrypted",
    "input_image_with_reasoning_encrypted",
    "tool_call_with_reasoning_encrypted",
)
TOOL_CALL_SCENARIOS: tuple[str, ...] = (
    "tool_call",
    "tool_call_choice",
    "tool_call_image_result_parallel",
    "tool_call_parallel",
    "tool_call_with_reasoning_encrypted",
)
PROMPT_CACHE_SCENARIOS: tuple[str, ...] = (
    "prompt_cache_implicit",
    "prompt_cache_explicit",
)
PROMPT_CACHE_CASES: tuple[tuple[str, str], ...] = tuple(
    product(PROMPT_CACHE_SCENARIOS, ("stream-00.yaml", "stream-01.yaml"))
)


def _protocol_name(protocol: Protocol) -> str:
    return {
        "chat/completions": "chat_completions",
        "messages": "messages",
        "responses": "responses",
    }[protocol]


def _provider_from_cassette_path(path: Path) -> Provider:
    provider, separator, _ = path.name.partition("-")
    if not separator or provider not in PROVIDER_CAPABILITIES:
        raise ValueError(f"Cannot determine provider from matched cassette: {path}")
    return cast(Provider, provider)


def _recorded_cassette_path(protocol: Protocol, cassette_name: str) -> Path:
    return CASSETTE_DIR / "openai" / _protocol_name(protocol) / "test_interface" / cassette_name


def _matched_cassette_path(
    scenario: str, protocol: Protocol, variant: str, provider: Provider | None = None
) -> Path:
    provider = provider or (
        "vllm" if scenario == "reasoning_raw" else DEFAULT_PROVIDER_BY_ENDPOINT[protocol]
    )
    return MATCHED_CASSETTE_DIR / scenario / f"{provider}-{_protocol_name(protocol)}-{variant}"


def _split_sdk_request_body(
    body: dict[str, Any], protocol: Protocol, provider: Provider
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Keep Together's reasoning option on the wire through the SDK's `extra_body`."""
    if protocol == "chat/completions" and provider == "togetherai" and "reasoning" in body:
        # Chat's SDK signature has no top-level reasoning argument. The original
        # wire body remains intact; extra_body restores this field during serialization.
        return (
            {field: value for field, value in body.items() if field != "reasoning"},
            {"reasoning": body["reasoning"]},
        )
    return body, None


@lru_cache(maxsize=5)
def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(
    cassette_paths: list[Path],
    upstream: Protocol = "responses",
    *,
    translated: bool = False,
    request_cassette_paths: list[Path] | None = None,
    expected_body: dict[str, Any] | None = None,
    expected_extra_body: dict[str, Any] | None = None,
    requests: list[Request] | None = None,
    response_headers: dict[str, str] | None = None,
    base_url: str | None = None,
    compare_history: bool = False,
) -> AsyncClient:
    """Replay responses and assert bodies from their own or separate request cassettes."""
    request_cassette_paths = (
        cassette_paths if request_cassette_paths is None else request_cassette_paths
    ).copy()
    assert len(request_cassette_paths) == len(cassette_paths)
    cassette_paths = cassette_paths.copy()

    def cassette_handler(request: Request) -> Response:
        if requests is not None:
            requests.append(request)
        cassette_path = cassette_paths.pop(0)
        request_cassette_path = request_cassette_paths.pop(0)
        cassette = _read_cassette(cassette_path)
        scenario = cassette_path.parent.name
        request_body = json_loads(request.content)
        try:
            assert request.method == cassette["request"]["method"]
            origin = (
                "https://api.anthropic.com" if upstream == "messages" else "https://api.openai.com"
            )
            if (
                upstream == "chat/completions"
                and cassette_path.is_relative_to(MATCHED_CASSETTE_DIR)
                and _provider_from_cassette_path(cassette_path) == "togetherai"
            ):
                origin = "https://api.together.ai"
            expected_url = f"{origin}{cassette['request']['path']}"
            if base_url is not None:
                path = cassette["request"]["path"] if upstream == "messages" else f"/{upstream}"
                expected_url = f"{base_url.rstrip('/')}{path}"
            assert str(request.url) == expected_url
            assert request.url.path == cassette["request"]["path"]
            expected = expected_body
            if expected is None:
                expected = _read_cassette(request_cassette_path)["request"]["body"]
            expected = {**expected, **(expected_extra_body or {})}
            if compare_history:
                history_key = "input" if upstream == "responses" else "messages"
                _assert_history_matches(
                    request_body[history_key],
                    expected[history_key],
                    upstream,
                    translated=translated,
                )
                # Only the comparison copy uses fixture representations. The actual
                # submitted history remains available in the captured HTTP request.
                request_body = {**request_body, history_key: expected[history_key]}
            if (
                translated
                and upstream == "responses"
                and scenario in ENCRYPTED_REASONING_SCENARIOS
            ):
                _assert_reconstructable_encrypted_responses_request(
                    request_body,
                    expected,
                )
            elif translated and upstream == "responses" and scenario in TOOL_CALL_SCENARIOS:
                _assert_reconstructable_tool_responses_request(request_body, expected)
            elif translated and upstream == "chat/completions" and scenario in TOOL_CALL_SCENARIOS:
                _assert_reconstructable_tool_chat_request(request_body, expected)
            else:
                assert request_body == expected
        except AssertionError as e:
            return Response(
                400,
                content=f"Bad request: {repr(e)}".encode(),
                headers={"content-type": "text/plain; charset=utf-8"},
            )
        cassette_response = cassette["response"]
        if "sse" in cassette_response:
            return Response(
                cassette_response["status_code"],
                content="".join(cassette_response["sse"]).encode(),
                headers={
                    "content-type": "text/event-stream; charset=utf-8",
                    **(response_headers or {}),
                },
            )
        if isinstance(cassette_response["body"], str):
            return Response(
                cassette_response["status_code"],
                content=cassette_response["body"].encode(),
                headers={"content-type": "application/json", **(response_headers or {})},
            )
        return Response(
            cassette_response["status_code"],
            json=cassette_response["body"],
            headers={"content-type": "application/json", **(response_headers or {})},
        )

    return AsyncClient(transport=MockTransport(cassette_handler))


async def _sdk_stream_events(
    cassette_path: Path,
    protocol: Protocol,
    request_body: dict[str, Any],
    *,
    extra_body: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> list[BaseModel]:
    """Parse a recorded upstream stream through the SDK using the caller's exact body."""
    if base_url is None:
        base_url = {
            "messages": "https://api.anthropic.com",
            "responses": "https://api.openai.com/v1",
            "chat/completions": (
                "https://api.together.ai/v1"
                if _provider_from_cassette_path(cassette_path) == "togetherai"
                else "https://api.openai.com/v1"
            ),
        }[protocol]
    async with _get_http_client(
        [cassette_path],
        protocol,
        expected_body=request_body,
        expected_extra_body=extra_body,
        base_url=base_url,
    ) as http_client:
        if protocol == "messages":
            sdk_client = AsyncAnthropic(
                api_key="test-api-key", base_url=base_url, http_client=http_client
            )
            sdk_stream = await sdk_client.messages.create(**request_body, extra_body=extra_body)
            assert isinstance(sdk_stream, AnthropicAsyncStream)
        else:
            sdk_client = AsyncOpenAI(
                api_key="test-api-key", base_url=base_url, http_client=http_client
            )
            create = (
                sdk_client.responses.create
                if protocol == "responses"
                else sdk_client.chat.completions.create
            )
            sdk_stream = await create(**request_body, extra_body=extra_body)
            assert isinstance(sdk_stream, AsyncStream)
        async with sdk_stream:
            return [event async for event in sdk_stream]


def _assert_reconstructable_encrypted_responses_request(
    actual_body: dict[str, Any],
    expected_body: dict[str, Any],
) -> None:
    adapter = OpenAIResponsesAdapter(PROVIDER_CAPABILITIES["openai"])

    def normalize(body: dict[str, Any]) -> dict[str, Any]:
        request = OpenAIResponsesCreateRequest.model_validate(body)
        normalized = adapter.normalize_request(request).model_dump(
            exclude={"extra_fields", "stream_options"}
        )
        for item in normalized.get("input") or []:
            if isinstance(item, dict):
                item.pop("id", None)
        return normalized

    # Messages has no equivalents for Responses include/stream options or item IDs.
    assert normalize(actual_body) == normalize(expected_body)


def _assert_reconstructable_tool_responses_request(
    actual_body: dict[str, Any],
    expected_body: dict[str, Any],
) -> None:
    def without_history_item_ids(body: dict[str, Any]) -> dict[str, Any]:
        body = _without_irrelevant_parallel_tool_calls(body)
        for item in body.get("input") or []:
            if isinstance(item, dict):
                item.pop("id", None)
                item.pop("phase", None)
        return body

    # Chat Completions and Messages preserve call IDs but expose no separate
    # Responses output-item IDs or commentary/final phases for assistant history.
    assert without_history_item_ids(actual_body) == without_history_item_ids(expected_body)


def _assert_reconstructable_tool_chat_request(
    actual_body: dict[str, Any],
    expected_body: dict[str, Any],
) -> None:
    assert _without_irrelevant_parallel_tool_calls(
        actual_body
    ) == _without_irrelevant_parallel_tool_calls(expected_body)


def _without_irrelevant_parallel_tool_calls(body: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(body)
    if body.get("tool_choice") == "none":
        # Messages cannot encode parallel-call control when tool use is disabled,
        # and the setting cannot affect a request that forbids every tool call.
        body.pop("parallel_tool_calls", None)
    return body


def _sse_records(cassette: dict[str, Any], protocol: Protocol) -> list[str]:
    records = []
    for event in "".join(cassette["response"]["sse"]).split("\n\n"):
        event = event.strip()
        if (
            event
            and event != "data: [DONE]"
            and not (protocol == "messages" and load_sse(event).get("type") == "ping")
        ):
            records.append(f"{event}\n\n")
    return records


def _responses_output_text(response: dict[str, Any]) -> str:
    return "".join(
        part["text"]
        for item in response["output"]
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )


def _assert_responses_stream_invariants(sse_events: list[str]) -> None:
    events = [load_sse(sse) for sse in sse_events]
    sequence_numbers = [event["sequence_number"] for event in events]
    assert all(current == previous + 1 for previous, current in pairwise(sequence_numbers)), (
        f"Non-contiguous sequence numbers: {sequence_numbers}"
    )

    output_item_ids = [
        event["item"]["id"]
        for event in events
        if event.get("type") == "response.output_item.added"
    ]
    assert len(output_item_ids) == len(set(output_item_ids)), (
        f"Duplicate output item IDs: {output_item_ids}"
    )


def _assert_matched_response(
    response: BaseModel,
    downstream: Protocol,
    expected: dict[str, Any] | OpenAIChatCompletionsResponse,
) -> None:
    """Assert response semantics against a canonical Responses fixture or parsed Chat response."""
    if isinstance(expected, OpenAIChatCompletionsResponse):
        message = expected.choices[0].message
        expected_text = message.content or ""
        expected_calls = [
            (call.id, call.function.name, call.function.arguments)
            for call in message.tool_calls or []
        ]
        expected_id, expected_model, expected_status = expected.id, expected.model, "completed"
        expected_usage = {
            "input_tokens": expected.usage.prompt_tokens,
            "output_tokens": expected.usage.completion_tokens,
            "total_tokens": expected.usage.total_tokens,
            "input_tokens_details": expected.usage.prompt_tokens_details.model_dump(
                exclude_none=True
            )
            if expected.usage.prompt_tokens_details is not None
            else {},
        }
        expected_reasoning = None
    else:
        expected_text = _responses_output_text(expected)
        expected_usage = expected["usage"]
        expected_id, expected_model, expected_status = (
            expected["id"],
            expected["model"],
            expected["status"],
        )
        expected_calls = [
            (item["call_id"], item["name"], item["arguments"])
            for item in expected["output"]
            if item["type"] == "function_call"
        ]
        expected_reasoning = next(
            (item for item in expected["output"] if item["type"] == "reasoning"), None
        )
    response_type = {
        "responses": OpenAIResponsesResponse,
        "chat/completions": OpenAIChatCompletionsResponse,
        "messages": AnthropicMessagesResponse,
    }[downstream]
    assert type(response) is response_type
    assert response.id == expected_id
    assert response.model == expected_model

    if downstream == "responses":
        assert isinstance(response, OpenAIResponsesResponse)
        assert response.output_text == expected_text
        assert response.status == expected_status
        assert [
            (item.call_id, item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        ] == expected_calls
        assert response.usage is not None
        assert response.usage.input_tokens == expected_usage["input_tokens"]
        assert response.usage.output_tokens == expected_usage["output_tokens"]
        assert response.usage.total_tokens == expected_usage["total_tokens"]
    elif downstream == "chat/completions":
        assert isinstance(response, OpenAIChatCompletionsResponse)
        assert len(response.choices) == 1
        assert (response.choices[0].message.content or "") == expected_text
        assert response.choices[0].finish_reason == ("tool_calls" if expected_calls else "stop")
        assert [
            (call.id, call.function.name, call.function.arguments)
            for call in response.choices[0].message.tool_calls or []
        ] == expected_calls
        assert response.usage is not None
        assert response.usage.prompt_tokens == expected_usage["input_tokens"]
        assert response.usage.completion_tokens == expected_usage["output_tokens"]
        assert response.usage.total_tokens == expected_usage["total_tokens"]
    else:
        assert isinstance(response, AnthropicMessagesResponse)
        assert "".join(block.text for block in response.content if block.type == "text") == (
            expected_text
        )
        assert response.stop_reason == ("tool_use" if expected_calls else "end_turn")
        assert [
            (block.id, block.name, block.input)
            for block in response.content
            if block.type == "tool_use"
        ] == [
            (call_id, name, json_loads(arguments)) for call_id, name, arguments in expected_calls
        ]
        input_details = expected_usage["input_tokens_details"]
        cache_write_tokens = input_details.get("cache_write_tokens", 0)
        cached_tokens = input_details.get("cached_tokens", 0)
        assert response.usage.input_tokens == (
            expected_usage["input_tokens"] - cache_write_tokens - cached_tokens
        )
        assert response.usage.cache_creation_input_tokens == cache_write_tokens
        assert response.usage.cache_read_input_tokens == cached_tokens
        assert response.usage.output_tokens == expected_usage["output_tokens"]

    if expected_reasoning is None:
        return

    expected_signature = expected_reasoning.get("encrypted_content")
    raw_reasoning = expected_reasoning.get("content") or []
    expected_thinking = "".join(
        part["text"] for part in raw_reasoning or expected_reasoning["summary"]
    )
    if downstream == "responses":
        assert isinstance(response, OpenAIResponsesResponse)
        reasoning = next(item for item in response.output if item.type == "reasoning")
        assert (
            "".join(part.text for part in reasoning.content or reasoning.summary)
            == expected_thinking
        )
        assert reasoning.encrypted_content == expected_signature
    elif downstream == "messages":
        assert isinstance(response, AnthropicMessagesResponse)
        thinking = next(block for block in response.content if block.type == "thinking")
        assert thinking.thinking == expected_thinking
        assert (thinking.signature or None) == expected_signature
    else:
        assert expected_signature is None and raw_reasoning
        assert response.choices[0].message.reasoning_content == expected_thinking


def _coordinator(client: Molto, protocol: Protocol):
    return {
        "responses": client.responses,
        "chat/completions": client.chat_completions,
        "messages": client.messages,
    }[protocol]
