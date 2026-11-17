from dataclasses import replace
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from molto.providers import Provider
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.utils.telemetry import ResolvedRoute, span_name, start_attributes

NATIVE_RESPONSES = ResolvedRoute(
    protocol="responses",
    endpoint="responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
)


def _start(body: dict[str, Any] | BaseModel, **route: Any) -> dict[str, Any]:
    """Map `body` on the native Responses route, with any route fields overridden."""
    return start_attributes(replace(NATIVE_RESPONSES, **route), body)


def test_native_request_maps_every_start_attribute():
    """Asserts that a native request yields the full documented start-attribute set."""
    attributes = _start(
        {
            "model": "gpt-5",
            "input": "Hello!",
            "max_output_tokens": 256,
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": True,
        }
    )

    assert attributes == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-5",
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.temperature": 0.5,
        "gen_ai.request.top_p": 0.9,
        "gen_ai.request.stream": True,
        "server.address": "api.openai.com",
        "server.port": 443,
        "molto.downstream.protocol": "responses",
        "molto.upstream.endpoint": "responses",
        "molto.route": "native",
    }


@pytest.mark.parametrize(
    ("protocol", "body", "max_tokens"),
    [
        (
            "responses",
            OpenAIResponsesCreateRequest(
                model="gpt-5",
                input="Hello!",
                max_output_tokens=256,
                temperature=0.5,
                top_p=0.9,
                stream=True,
            ),
            256,
        ),
        (
            "chat/completions",
            OpenAIChatCompletionsCreateRequest(
                model="gpt-5",
                messages=[{"role": "user", "content": "Hello!"}],
                max_completion_tokens=256,
                temperature=0.5,
                top_p=0.9,
                stream=True,
            ),
            256,
        ),
        (
            "messages",
            AnthropicMessagesCreateRequest(
                model="claude-sonnet-5",
                max_tokens=256,
                messages=[{"role": "user", "content": "Hello!"}],
                temperature=0.5,
                top_p=0.9,
                stream=True,
            ),
            256,
        ),
    ],
)
def test_translated_request_models_map_request_fields(
    protocol: str, body: BaseModel, max_tokens: int
):
    """Ensures each protocol's validated request model is read for every request attribute.

    Translated coordinators pass the validated model rather than a dictionary, so the
    model path must cover each protocol's output-token field and the shared fields.
    """
    attributes = _start(body, protocol=protocol, endpoint="messages")

    assert attributes["gen_ai.request.max_tokens"] == max_tokens
    assert attributes["gen_ai.request.temperature"] == 0.5
    assert attributes["gen_ai.request.top_p"] == 0.9
    assert attributes["gen_ai.request.stream"] is True
    assert attributes["molto.downstream.protocol"] == protocol


def test_route_is_translated_when_endpoint_differs_from_protocol():
    """Asserts that the route is native only when the upstream endpoint matches the protocol."""
    attributes = _start({"model": "gpt-5"}, protocol="messages", endpoint="responses")

    assert attributes["molto.downstream.protocol"] == "messages"
    assert attributes["molto.upstream.endpoint"] == "responses"
    assert attributes["molto.route"] == "translated"


@pytest.mark.parametrize("stream", [False, None])
def test_non_streaming_request_model_omits_stream(stream: bool | None):
    """Ensures `gen_ai.request.stream` is set only for streaming requests, as the convention requires."""
    body = OpenAIResponsesCreateRequest(model="gpt-5", input="Hello!", stream=stream)

    assert "gen_ai.request.stream" not in _start(body)


def test_unset_optional_fields_are_omitted():
    """Asserts that optional request attributes appear only when the request sets them."""
    attributes = _start(
        {"model": "qwen3", "messages": []},
        protocol="chat/completions",
        endpoint="chat/completions",
        provider="vllm",
    )

    assert set(attributes) == {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "server.address",
        "server.port",
        "molto.downstream.protocol",
        "molto.upstream.endpoint",
        "molto.route",
    }


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"max_completion_tokens": 64, "max_tokens": 32}, 64),
        ({"max_tokens": 32}, 32),
        ({"max_completion_tokens": "64", "max_tokens": 32}, 32),
    ],
)
def test_chat_max_tokens_prefers_valid_max_completion_tokens(body: dict, expected: int):
    """Ensures Chat reports `max_completion_tokens`, falling back to the deprecated `max_tokens`.

    A mistyped preferred field must not hide a valid fallback value.
    """
    attributes = _start(
        {"model": "gpt-5", "messages": [], **body},
        protocol="chat/completions",
        endpoint="chat/completions",
    )

    assert attributes["gen_ai.request.max_tokens"] == expected


def test_chat_request_model_reads_deprecated_max_tokens():
    """Ensures the Chat fallback also applies to a validated request model."""
    body = OpenAIChatCompletionsCreateRequest(model="gpt-5", messages=[], max_tokens=32)

    attributes = _start(body, protocol="chat/completions", endpoint="messages")

    assert attributes["gen_ai.request.max_tokens"] == 32


def test_integer_sampling_values_are_reported_as_floats():
    """Ensures temperature and top_p keep the convention's double type for integer input."""
    attributes = _start(
        {"model": "claude-sonnet-5", "max_tokens": 8, "temperature": 1, "top_p": 1},
        protocol="messages",
        endpoint="messages",
        provider="anthropic",
    )

    assert attributes["gen_ai.request.temperature"] == 1.0
    assert isinstance(attributes["gen_ai.request.temperature"], float)
    assert isinstance(attributes["gen_ai.request.top_p"], float)


def test_mistyped_native_fields_are_omitted_without_raising():
    """Asserts that unvalidated native values of the wrong type never become attributes.

    Native bodies reach the SDK unvalidated; telemetry must not reject them or
    report a value that does not match the convention's attribute type. Booleans
    are rejected as numbers even though `bool` subclasses `int`.
    """
    attributes = _start(
        {
            "model": 123,
            "max_output_tokens": True,
            "temperature": "0.5",
            "top_p": True,
            "stream": "true",
        }
    )

    for key in (
        "gen_ai.request.model",
        "gen_ai.request.max_tokens",
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
        "gen_ai.request.stream",
    ):
        assert key not in attributes


def test_empty_model_is_treated_as_absent():
    """Ensures an empty model neither becomes an attribute nor leaves a trailing space in the name."""
    attributes = _start({"model": ""})

    assert "gen_ai.request.model" not in attributes
    assert span_name(attributes) == "chat"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000/v1", {"server.address": "localhost", "server.port": 8000}),
        ("http://127.0.0.1/v1", {"server.address": "127.0.0.1", "server.port": 80}),
        ("https://api.anthropic.com", {"server.address": "api.anthropic.com", "server.port": 443}),
    ],
)
def test_server_attributes_use_explicit_or_scheme_default_port(base_url: str, expected: dict):
    """Ensures `server.port` is always set with `server.address`, as the convention requires."""
    attributes = _start({"model": "gpt-5"}, base_url=base_url)

    assert {key: attributes[key] for key in expected} == expected


@pytest.mark.parametrize("base_url", ["http://[::1", "localhost:8000", ""])
def test_unusable_base_url_omits_server_attributes_without_raising(base_url: str):
    """Asserts that a malformed or host-less base URL leaves its error to the SDK at transport."""
    attributes = _start({"model": "gpt-5"}, base_url=base_url)

    assert "server.address" not in attributes
    assert "server.port" not in attributes


def test_span_name_includes_model_when_present():
    """Ensures spans are named `{operation} {model}`, the GenAI inference-span format."""
    assert span_name({"gen_ai.request.model": "gpt-5"}) == "chat gpt-5"
    assert span_name({}) == "chat"


def test_provider_set_is_pinned_for_gen_ai_naming():
    """Ensures adding a provider forces a decision about its gen_ai.provider.name.

    Molto reports the provider name unchanged. Well-known convention values such as
    `aws.bedrock` or `gcp.gemini` differ from the obvious Molto names, so a new provider
    must either match the convention or add a mapping before this test is updated.
    """
    assert set(get_args(Provider)) == {"openai", "anthropic", "vllm"}
