"""Request-boundary validation and preservation of provider-owned input fields."""

from copy import deepcopy
from itertools import permutations
from typing import Any

import pytest
from anthropic import NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from anthropic import AsyncAnthropic
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import omit as anthropic_omit
from anthropic.types import TextBlock
from httpx import AsyncClient, Request
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai import AsyncOpenAI
from openai import BadRequestError as OpenAIBadRequestError
from openai import omit as openai_omit
from openai.types.shared import Reasoning
from pydantic import BaseModel, ValidationError, create_model

from molto import Molto
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.utils.test import assert_stream_lifecycle, collect_sdk_response
from tests.core.test_interface import (
    PROTOCOLS,
    ROUTES,
    Protocol,
    _assert_matched_response,
    _coordinator,
    _get_http_client,
    _matched_cassette_path,
    _read_cassette,
)

_REQUEST_MODELS = {
    "responses": OpenAIResponsesCreateRequest,
    "chat/completions": OpenAIChatCompletionsCreateRequest,
    "messages": AnthropicMessagesCreateRequest,
}
_TRANSLATED_ROUTES = tuple(permutations(PROTOCOLS, 2))


def _first_message(body: dict[str, Any], protocol: Protocol) -> dict[str, Any]:
    return body["input" if protocol == "responses" else "messages"][0]


async def _sdk_result(
    http_client: AsyncClient,
    protocol: Protocol,
    body: dict[str, Any],
    *,
    extra_body: dict[str, Any] | None = None,
):
    if protocol == "messages":
        create = AsyncAnthropic(api_key="test-api-key", http_client=http_client).messages.create
    else:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        create = sdk.responses.create if protocol == "responses" else sdk.chat.completions.create
    result = await create(**body, extra_body=extra_body)
    if body.get("stream"):
        async with result:
            return [event async for event in result]
    return result


async def _assert_text_response_parity(result, protocol: Protocol, mode: str) -> None:
    reference_path = _matched_cassette_path("simple_text", "responses", f"{mode}-00.yaml")
    reference_body = _read_cassette(reference_path)["request"]["body"]
    async with _get_http_client([reference_path]) as http_client:
        reference = await _sdk_result(http_client, "responses", reference_body)
        if mode == "stream":
            reference = collect_sdk_response("responses", reference)
    _assert_matched_response(result, protocol, reference)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
async def test_native_requests_preserve_provider_fields_and_sdk_parity(
    protocol: Protocol, stream: bool
) -> None:
    """Ensures native input reaches the SDK unchanged while extensions stay body fields."""
    mode = "stream" if stream else "non_stream"
    path = _matched_cassette_path("simple_text", protocol, f"{mode}-00.yaml")
    sdk_body = deepcopy(_read_cassette(path)["request"]["body"])
    sdk_body["stream"] = stream
    sdk_body["temperature"] = None
    sdk_body["service_tier"] = "provider-specific-tier"
    message = _first_message(sdk_body, protocol)
    message["provider_message"] = {"value": None}
    message["content"][0]["provider_part"] = True
    message["content"].append({"type": "provider-specific-part", "payload": {"value": None}})
    extensions = {
        "provider_option": {"value": "body-value"},
        "provider_null": None,
        "extra_headers": {"x-body-only": "must-not-be-a-header"},
        "extra_query": {"body_only": "must-not-be-a-query"},
        "timeout": "body-only-timeout",
        "extra_body": {"body_only": "must-not-be-merged"},
    }
    body = {**sdk_body, **extensions}
    overrides = {"provider_option": {"value": "override"}, "top_p": 0.75}
    original_body = deepcopy(body)
    original_overrides = deepcopy(overrides)
    expected = {**body, **overrides}
    requests: list[Request] = []

    # Synthetic request extensions exercise forwarding, not live provider acceptance.
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        reference = await _sdk_result(
            http_client, protocol, sdk_body, extra_body={**extensions, **overrides}
        )
    async with _get_http_client(
        [path], protocol, expected_body=expected, requests=requests
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), protocol)(
            body=body,
            endpoint=protocol,
            api_key="test-api-key",
            extra_body=overrides,
        )
        if stream:
            result = [event async for event in result]
            assert [type(event) for event in result] == [type(event) for event in reference]
            assert [event.model_dump() for event in result] == [
                event.model_dump() for event in reference
            ]
        else:
            assert type(result) is type(reference)
            assert result.model_dump() == reference.model_dump()

    assert len(requests) == 1
    assert "x-body-only" not in requests[0].headers
    assert not requests[0].url.query
    assert body == original_body
    assert overrides == original_overrides


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [1, "false"], ids=["integer", "string"])
async def test_native_non_boolean_stream_preserves_provider_error(
    protocol: Protocol, stream: int | str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensures truthy non-booleans use the response path and retain SDK error fidelity."""
    path = _matched_cassette_path("error-invalid_request_error", protocol, "stream-00.yaml")
    body = {**deepcopy(_read_cassette(path)["request"]["body"]), "stream": stream}
    error_type = AnthropicBadRequestError if protocol == "messages" else OpenAIBadRequestError
    client_type = {
        "responses": OpenAIResponsesClient,
        "chat/completions": OpenAIChatCompletionsClient,
        "messages": AnthropicMessagesClient,
    }[protocol]

    def unexpected_stream(*args: Any) -> None:
        raise AssertionError("Only the boolean True should select native_events")

    monkeypatch.setattr(client_type, "native_events", unexpected_stream)
    # Replay an existing provider error to verify forwarding and error handling,
    # without claiming live acceptance of the modified stream value.
    async with _get_http_client([path], protocol, expected_body=body) as http_client:
        with pytest.raises(error_type) as reference:
            await _sdk_result(http_client, protocol, body)
    requests: list[Request] = []
    async with _get_http_client(
        [path], protocol, expected_body=body, requests=requests
    ) as http_client:
        with pytest.raises(error_type) as actual:
            await _coordinator(Molto(http_client=http_client), protocol)(
                body=body, api_key="test-api-key"
            )
    assert len(requests) == 1
    assert type(actual.value) is type(reference.value)
    assert actual.value.status_code == reference.value.status_code
    assert actual.value.body == reference.value.body


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
async def test_native_molto_models_omit_null_parameters(protocol: Protocol, stream: bool) -> None:
    """Ensures Molto model omission rules retain exact request and parsed SDK parity."""
    mode = "stream" if stream else "non_stream"
    path = _matched_cassette_path("simple_text", protocol, f"{mode}-00.yaml")
    expected = deepcopy(_read_cassette(path)["request"]["body"])
    request = _REQUEST_MODELS[protocol].model_validate({**expected, "temperature": None})
    original = request.model_copy(deep=True)
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        reference = await _sdk_result(http_client, protocol, expected)
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        result = await _coordinator(Molto(http_client=http_client), protocol)(
            body=request, endpoint=protocol, api_key="test-api-key"
        )
        if stream:
            async with result:
                events = [event async for event in result]
            assert_stream_lifecycle(protocol, events)
            assert [type(event) for event in events] == [type(event) for event in reference]
            assert [event.model_dump() for event in events] == [
                event.model_dump() for event in reference
            ]
        else:
            assert type(result) is type(reference)
            assert result.model_dump() == reference.model_dump()
    assert request == original


@pytest.mark.parametrize("protocol", ["responses", "messages"])
async def test_native_nested_sdk_models_preserve_sdk_serialization(protocol: Protocol) -> None:
    """Ensures SDK model defaults remain omitted when nested in a native request body."""
    path = _matched_cassette_path("simple_text", protocol, "stream-00.yaml")
    expected = deepcopy(_read_cassette(path)["request"]["body"])
    body = deepcopy(expected)
    if protocol == "responses":
        body["reasoning"] = Reasoning(effort="low")
        expected["reasoning"] = {"effort": "low"}
    else:
        message = _first_message(body, protocol)
        message["content"] = [TextBlock(**part) for part in message["content"]]
    original = deepcopy(body)
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        reference = await _sdk_result(http_client, protocol, body)
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        result = await _coordinator(Molto(http_client=http_client), protocol)(
            body=body, endpoint=protocol, api_key="test-api-key"
        )
        events = [event async for event in result]
    assert [type(event) for event in events] == [type(event) for event in reference]
    assert [event.model_dump() for event in events] == [event.model_dump() for event in reference]
    assert body == original


@pytest.mark.parametrize(("downstream", "upstream"), _TRANSLATED_ROUTES)
async def test_translated_requests_reject_unknown_top_level_fields(
    downstream: Protocol, upstream: Protocol
) -> None:
    """Ensures translation retains its existing top-level schema validation."""
    path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    body = deepcopy(_read_cassette(path)["request"]["body"])
    body["provider_unknown"] = True
    requests: list[Request] = []
    async with _get_http_client([], requests=requests) as http_client:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            await _coordinator(Molto(http_client=http_client), downstream)(
                body=body, endpoint=upstream, api_key="test-api-key"
            )
    assert not requests


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_coordinators_reject_caller_defined_models(
    downstream: Protocol, upstream: Protocol
) -> None:
    """Ensures unsupported caller models fail before transport on every route."""
    path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    body = _read_cassette(path)["request"]["body"]
    request = create_model("CallerRequest", **{key: (Any, value) for key, value in body.items()})()
    requests: list[Request] = []
    async with _get_http_client([], requests=requests) as http_client:
        with pytest.raises(TypeError, match=_REQUEST_MODELS[downstream].__name__):
            await _coordinator(Molto(http_client=http_client), downstream)(
                body=request, endpoint=upstream, api_key="test-api-key"
            )
    assert not requests


@pytest.mark.parametrize(("downstream", "upstream"), _TRANSLATED_ROUTES)
async def test_translated_molto_model_inputs_retain_sdk_parity(
    downstream: Protocol, upstream: Protocol
) -> None:
    """Ensures supported Molto model bodies retain translated request and response parity."""
    path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    upstream_path = _matched_cassette_path("simple_text", upstream, "stream-00.yaml")
    body = deepcopy(_read_cassette(path)["request"]["body"])
    request = _REQUEST_MODELS[downstream].model_validate({**body, "stream": False})
    async with _get_http_client([upstream_path], upstream, translated=True) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=request, endpoint=upstream, api_key="test-api-key"
        )
    await _assert_text_response_parity(result, downstream, "stream")


@pytest.mark.parametrize("max_tokens", ["missing", None])
async def test_native_messages_preserve_local_sdk_argument_errors(max_tokens):
    """Ensures SDK-local argument failures remain identical and make no HTTP request."""
    body = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    if max_tokens != "missing":
        body["max_tokens"] = max_tokens
    requests: list[Request] = []
    async with _get_http_client([], requests=requests) as http_client:
        with pytest.raises(TypeError) as reference:
            await _sdk_result(http_client, "messages", body)
        with pytest.raises(TypeError) as actual:
            await Molto(http_client=http_client).messages(body=body, api_key="test-api-key")
    assert str(actual.value) == str(reference.value)
    assert not requests


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("marker", ["omit", "not-given"])
async def test_native_requests_preserve_sdk_omission_markers(protocol: Protocol, marker: str):
    """Ensures SDK omission markers reach the SDK without premature JSON serialization."""
    path = _matched_cassette_path("simple_text", protocol, "stream-00.yaml")
    expected = deepcopy(_read_cassette(path)["request"]["body"])
    markers = (
        {"omit": anthropic_omit, "not-given": ANTHROPIC_NOT_GIVEN}
        if protocol == "messages"
        else {"omit": openai_omit, "not-given": OPENAI_NOT_GIVEN}
    )
    body = {**expected, "temperature": markers[marker]}
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        reference = await _sdk_result(http_client, protocol, body)
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        stream = await _coordinator(Molto(http_client=http_client), protocol)(
            body=body, api_key="test-api-key"
        )
        async with stream:
            events = [event async for event in stream]
    assert [type(event) for event in events] == [type(event) for event in reference]
    assert [event.model_dump() for event in events] == [event.model_dump() for event in reference]
    assert body["temperature"] is markers[marker]


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_native_requests_preserve_iterable_history(protocol: Protocol):
    """Ensures inspecting native history does not consume SDK-supported input iterables."""
    path = _matched_cassette_path("simple_text", protocol, "stream-00.yaml")
    expected = deepcopy(_read_cassette(path)["request"]["body"])
    body = deepcopy(expected)
    history_key = "input" if protocol == "responses" else "messages"
    for item in body[history_key]:
        item["content"] = iter(item["content"])
    body[history_key] = iter(body[history_key])
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        reference = await _sdk_result(http_client, protocol, expected)
    async with _get_http_client([path], protocol, expected_body=expected) as http_client:
        stream = await _coordinator(Molto(http_client=http_client), protocol)(
            body=body, api_key="test-api-key"
        )
        async with stream:
            events = [event async for event in stream]
    assert [type(event) for event in events] == [type(event) for event in reference]
    assert [event.model_dump() for event in events] == [event.model_dump() for event in reference]


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_native_requests_reject_nested_caller_models(protocol: Protocol) -> None:
    """Ensures unsupported nested models cannot become iterable field/value pairs."""

    class CallerMetadata(BaseModel):
        label: str = "example"

    path = _matched_cassette_path("simple_text", protocol, "stream-00.yaml")
    body = deepcopy(_read_cassette(path)["request"]["body"])
    body["metadata"] = CallerMetadata()
    requests: list[Request] = []
    async with _get_http_client([], requests=requests) as http_client:
        with pytest.raises(TypeError, match="Serialize caller-defined Pydantic models"):
            await _coordinator(Molto(http_client=http_client), protocol)(
                body=body, api_key="test-api-key"
            )
    assert not requests
