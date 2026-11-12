"""Request-boundary validation and preservation of provider-owned input fields."""

from copy import deepcopy
from itertools import permutations
from typing import Any

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from httpx import AsyncClient, Request
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import ValidationError, create_model

from molto import Molto
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.exceptions import UnsupportedInput
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.utils.test import assert_stream_lifecycle, collect_sdk_response
from tests.core.test_interface import (
    PROTOCOLS,
    ROUTES,
    Protocol,
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


def _add_unknown_fields(body: dict[str, Any], protocol: Protocol) -> None:
    body["provider_option"] = {"enabled": True, "value": None}
    message = _first_message(body, protocol)
    message["provider_message"] = "message-extension"
    message["content"][0]["provider_part"] = {"value": None}


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


async def _assert_text_response_parity(result, protocol: Protocol) -> None:
    reference_path = _matched_cassette_path("simple_text", "responses", "stream-00.yaml")
    reference_body = _read_cassette(reference_path)["request"]["body"]
    async with _get_http_client([reference_path]) as http_client:
        reference = collect_sdk_response(
            "responses", await _sdk_result(http_client, "responses", reference_body)
        )
    assert result.id == reference.id
    assert result.model == reference.model
    if protocol == "responses":
        text = result.output_text
        input_tokens = result.usage.input_tokens
        output_tokens = result.usage.output_tokens
        assert result.status == reference.status
    elif protocol == "chat/completions":
        text = result.choices[0].message.content
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens
        assert result.choices[0].finish_reason == "stop"
    else:
        text = "".join(block.text for block in result.content if block.type == "text")
        input_tokens = result.usage.input_tokens
        output_tokens = result.usage.output_tokens
        assert result.stop_reason == "end_turn"
    assert text == reference.output_text
    assert input_tokens == reference.usage.input_tokens
    assert output_tokens == reference.usage.output_tokens


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "stream", [False, True, 1], ids=["non-stream", "stream", "integer-stream"]
)
async def test_native_requests_preserve_provider_fields_and_sdk_parity(
    protocol: Protocol, stream: bool | int
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
            drop_unknown=bool(stream),
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
async def test_native_model_input_retains_allowed_nested_extras(protocol: Protocol) -> None:
    """Ensures model inputs retain per-call allowed extras despite stricter declared schemas."""
    path = _matched_cassette_path("simple_text", protocol, "stream-00.yaml")
    body = deepcopy(_read_cassette(path)["request"]["body"])
    _add_unknown_fields(body, protocol)
    body["temperature"] = None
    request = _REQUEST_MODELS[protocol].model_validate(body, extra="allow")
    original = request.model_copy(deep=True)
    async with _get_http_client([path], protocol, expected_body=body) as http_client:
        result = await _coordinator(Molto(http_client=http_client), protocol)(
            body=request, endpoint=protocol, api_key="test-api-key"
        )
        events = [event async for event in result]
        assert_stream_lifecycle(protocol, events)
    await _assert_text_response_parity(collect_sdk_response(protocol, events), protocol)
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
@pytest.mark.parametrize("location", ["root", "nested"])
@pytest.mark.parametrize("model_input", [False, True], ids=["dict", "model"])
async def test_translated_requests_reject_unknown_fields_by_default(
    downstream: Protocol, upstream: Protocol, location: str, model_input: bool
) -> None:
    """Ensures translated requests reject unknown root and nested properties before transport."""
    path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    body = deepcopy(_read_cassette(path)["request"]["body"])
    target = body if location == "root" else _first_message(body, downstream)["content"][0]
    target["provider_unknown"] = True
    request = (
        _REQUEST_MODELS[downstream].model_validate(body, extra="allow") if model_input else body
    )
    requests: list[Request] = []
    async with _get_http_client([], requests=requests) as http_client:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            await _coordinator(Molto(http_client=http_client), downstream)(
                body=request, endpoint=upstream, api_key="test-api-key"
            )
    assert not requests


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_generic_model_input_preserves_default_values(
    downstream: Protocol, upstream: Protocol
) -> None:
    """Ensures defaults on caller-owned request models survive native and translated routes."""
    downstream_path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    upstream_path = _matched_cassette_path("simple_text", upstream, "stream-00.yaml")
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
    request = create_model(
        "DefaultRequest", **{key: (Any, value) for key, value in body.items()}
    )()
    assert not request.model_fields_set
    async with _get_http_client(
        [upstream_path], upstream, translated=downstream != upstream
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=request, endpoint=upstream, api_key="test-api-key"
        )
        events = [event async for event in result]
        assert_stream_lifecycle(downstream, events)
    await _assert_text_response_parity(collect_sdk_response(downstream, events), downstream)


@pytest.mark.parametrize(("downstream", "upstream"), _TRANSLATED_ROUTES)
@pytest.mark.parametrize("model_input", [False, True], ids=["dict", "model"])
async def test_drop_unknown_preserves_translation_and_explicit_extensions(
    downstream: Protocol, upstream: Protocol, model_input: bool
) -> None:
    """Ensures opt-in dropping removes schema extras without changing input or SDK parity."""
    downstream_path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    upstream_path = _matched_cassette_path("simple_text", upstream, "stream-00.yaml")
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
    _add_unknown_fields(body, downstream)
    if downstream != "messages":
        body["stream_options"]["provider_option"] = True
    original = deepcopy(body)
    request = (
        _REQUEST_MODELS[downstream].model_validate(body, extra="allow") if model_input else body
    )
    extensions = {"provider_transport_option": {"value": None}}
    async with _get_http_client(
        [upstream_path], upstream, translated=True, expected_extra_body=extensions
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=request,
            endpoint=upstream,
            api_key="test-api-key",
            drop_unknown=True,
            extra_body=extensions,
        )
        events = [event async for event in result]
        assert_stream_lifecycle(downstream, events)
    await _assert_text_response_parity(collect_sdk_response(downstream, events), downstream)
    assert body == original


def _tool(protocol: Protocol, schema: dict[str, Any]) -> dict[str, Any]:
    if protocol == "messages":
        return {"name": "lookup", "input_schema": schema}
    function = {"name": "lookup", "parameters": schema, "strict": False}
    if protocol == "responses":
        return {"type": "function", **function}
    return {"type": "function", "function": function}


@pytest.mark.parametrize(("downstream", "upstream"), _TRANSLATED_ROUTES)
async def test_drop_unknown_preserves_arbitrary_tool_schema(
    downstream: Protocol, upstream: Protocol
) -> None:
    """Ensures arbitrary JSON Schema keywords and property names are not mistaken for extras."""
    downstream_path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    upstream_path = _matched_cassette_path("simple_text", upstream, "stream-00.yaml")
    schema = {
        "type": "object",
        "properties": {"provider_property": {"type": "string", "x-provider": {"value": None}}},
        "required": ["provider_property"],
        "additionalProperties": False,
        "x-provider-schema": {"custom": True},
    }
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
    body["tools"] = [_tool(downstream, schema)]
    body["tools"][0]["provider_tool_option"] = True
    expected = deepcopy(_read_cassette(upstream_path)["request"]["body"])
    expected["tools"] = [_tool(upstream, schema)]
    original = deepcopy(body)
    async with _get_http_client(
        [upstream_path], upstream, translated=True, expected_body=expected
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body, endpoint=upstream, api_key="test-api-key", drop_unknown=True
        )
        events = [event async for event in result]
    await _assert_text_response_parity(collect_sdk_response(downstream, events), downstream)
    assert body == original


@pytest.mark.parametrize("downstream", PROTOCOLS)
@pytest.mark.parametrize("invalid", ["unsupported-field", "content-type", "role"])
async def test_drop_unknown_retains_semantic_and_shape_validation(
    downstream: Protocol, invalid: str
) -> None:
    """Ensures dropping unknown properties cannot disable known-field or discriminator checks."""
    upstream = "messages" if downstream == "responses" else "responses"
    path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    body = deepcopy(_read_cassette(path)["request"]["body"])
    if invalid == "unsupported-field":
        field, value = {
            "responses": ("store", True),
            "chat/completions": ("n", 2),
            "messages": ("top_k", 5),
        }[downstream]
        body[field] = value
        expected_error = UnsupportedInput
    else:
        message = _first_message(body, downstream)
        if invalid == "content-type":
            message["content"][0]["type"] = "provider-specific-part"
        else:
            message["role"] = "provider-specific-role"
        expected_error = ValidationError
    requests: list[Request] = []
    async with _get_http_client([], requests=requests) as http_client:
        with pytest.raises(expected_error):
            await _coordinator(Molto(http_client=http_client), downstream)(
                body=body, endpoint=upstream, api_key="test-api-key", drop_unknown=True
            )
    assert not requests
