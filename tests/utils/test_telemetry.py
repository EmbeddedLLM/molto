from dataclasses import replace
from types import SimpleNamespace
from typing import Any, get_args

import openai
import pytest
from anthropic.types import Message
from httpx import Request, Response
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as ChatChoice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from molto.providers import Provider
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.exceptions import AuthenticationError, ContextLengthExceededError
from molto.types.normalized import PartDeltaEvent, TextDelta, ThinkingDelta
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.utils.telemetry import (
    MessagesUsage,
    ResolvedRoute,
    SpanRecorder,
    chat_completions_response_attributes,
    error_attributes,
    exception_event_attributes,
    is_chat_completions_content_delta,
    is_messages_content_delta,
    is_responses_content_delta,
    messages_response_attributes,
    observe,
    responses_response_attributes,
    span_name,
    start_attributes,
)

NATIVE_RESPONSES = ResolvedRoute(
    protocol="responses",
    endpoint="responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
)


def _start(
    body: dict[str, Any] | BaseModel, *, streaming: bool = False, **route: Any
) -> dict[str, Any]:
    """Map `body` on the native Responses route, with any route fields overridden."""
    return start_attributes(replace(NATIVE_RESPONSES, **route), body, streaming=streaming)


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
        },
        streaming=True,
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
    attributes = _start(body, streaming=True, protocol=protocol, endpoint="messages")

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


def test_non_streaming_call_omits_stream():
    """Ensures `gen_ai.request.stream` is set only for streaming calls, as the convention requires.

    The coordinator decides whether a call streams; telemetry never reads `stream` itself.
    """
    body = OpenAIResponsesCreateRequest(model="gpt-5", input="Hello!", stream=True)

    assert "gen_ai.request.stream" not in _start(body, streaming=False)


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


def test_messages_top_k_is_reported_on_native_and_validated_requests():
    """Ensures `top_k`, which only Messages defines, is reported as the convention's integer.

    A translated route rejects `top_k`, so its span must still show the field that
    caused the rejection; that path passes the validated request model.
    """
    native = _start(
        {"model": "claude-sonnet-5", "max_tokens": 8, "top_k": 40},
        protocol="messages",
        endpoint="messages",
        provider="anthropic",
    )
    translated = _start(
        AnthropicMessagesCreateRequest(
            model="claude-sonnet-5", max_tokens=8, messages=[], top_k=40
        ),
        protocol="messages",
        endpoint="responses",
    )

    assert native["gen_ai.request.top_k"] == 40
    assert translated["gen_ai.request.top_k"] == 40


def test_native_body_reports_a_field_its_request_type_does_not_define():
    """Ensures a native body's sampling field is reported as sent, whatever protocol defines it.

    Native passthrough forwards fields its request type does not define, such as `top_k`
    on a vLLM Chat request, so the span must report what was actually sent.
    """
    attributes = _start(
        {"model": "qwen3", "messages": [], "top_k": 20},
        protocol="chat/completions",
        endpoint="chat/completions",
        provider="vllm",
    )

    assert attributes["gen_ai.request.top_k"] == 20


def test_chat_sampling_controls_are_reported_with_convention_types():
    """Ensures Chat penalties, seed, and choice count use the convention's keys and types.

    Penalties are doubles even for integer input, and `seed` and `n` stay integers.
    """
    attributes = _start(
        {
            "model": "gpt-5",
            "messages": [],
            "frequency_penalty": 1,
            "presence_penalty": 0.5,
            "seed": 7,
            "n": 3,
        },
        protocol="chat/completions",
        endpoint="chat/completions",
    )

    assert attributes["gen_ai.request.frequency_penalty"] == 1.0
    assert isinstance(attributes["gen_ai.request.frequency_penalty"], float)
    assert attributes["gen_ai.request.presence_penalty"] == 0.5
    assert attributes["gen_ai.request.seed"] == 7
    assert attributes["gen_ai.request.choice.count"] == 3


def test_default_choice_count_is_omitted():
    """Asserts that `n=1` is not reported, since the convention reports only a count other than one."""
    attributes = _start(
        {"model": "gpt-5", "messages": [], "n": 1},
        protocol="chat/completions",
        endpoint="chat/completions",
    )

    assert "gen_ai.request.choice.count" not in attributes


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
            "top_k": 40.0,
            "frequency_penalty": "0.1",
            "presence_penalty": False,
            "seed": "7",
            "n": True,
        }
    )

    for key in (
        "gen_ai.request.model",
        "gen_ai.request.max_tokens",
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
        "gen_ai.request.top_k",
        "gen_ai.request.frequency_penalty",
        "gen_ai.request.presence_penalty",
        "gen_ai.request.seed",
        "gen_ai.request.choice.count",
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


REQUEST = Request("POST", "https://api.openai.com/v1/responses")


def _raised_from(error: Exception, cause: Exception) -> Exception:
    """Return `error` raised from `cause`, with tracebacks, as Molto's error chain does."""
    try:
        try:
            raise cause
        except Exception as caught:
            raise error from caught
    except Exception as raised:
        return raised


def test_missing_credentials_omit_the_synthetic_status_code():
    """Asserts that the synthetic 401 on a missing-credentials error is not reported as HTTP.

    No HTTP call happens: the 401 exists only on the outward SDK exception, and the
    normalized cause has no response.
    """
    outward = openai.AuthenticationError(
        "Missing credentials.", response=Response(401, request=REQUEST), body=None
    )
    error = _raised_from(
        outward, AuthenticationError("Missing credentials.", response=None, body=None)
    )

    attributes = error_attributes(error, "setup")

    assert attributes == {"error.type": "AuthenticationError", "molto.error.phase": "setup"}


def test_error_on_a_successful_stream_response_omits_the_status_code():
    """Asserts that a failure arriving on an HTTP 200 stream does not report status 200."""
    cause = ContextLengthExceededError(
        "context length exceeded", response=Response(200, request=REQUEST), body=None
    )
    error = _raised_from(openai.APIError("context length exceeded", REQUEST, body=None), cause)

    attributes = error_attributes(error, "stream")

    assert attributes == {
        "error.type": "ContextLengthExceededError",
        "molto.error.phase": "stream",
    }


def test_unrecognized_sdk_error_reports_its_class_and_status():
    """Ensures an error that passes through unchanged keeps its own class and HTTP status."""
    error = openai.RateLimitError(
        "rate limited", response=Response(429, request=REQUEST), body=None
    )

    attributes = error_attributes(error, "setup")

    assert attributes == {
        "error.type": "RateLimitError",
        "http.response.status_code": 429,
        "molto.error.phase": "setup",
    }


def test_exception_event_attributes_carry_no_message_from_the_cause_chain():
    """Asserts that the recorded exception keeps types and frames but no chained message text."""
    error = _raised_from(
        RuntimeError("outer secret-request-marker"),
        ValueError("inner secret-request-marker"),
    )

    attributes = exception_event_attributes(error)

    assert attributes["exception.message"] == "RuntimeError"
    stacktrace = attributes["exception.stacktrace"]
    assert "secret-request-marker" not in stacktrace
    assert "builtins.RuntimeError" in stacktrace
    assert "builtins.ValueError" in stacktrace
    assert "_raised_from" in stacktrace


def test_messages_usage_includes_cached_tokens_in_input():
    """Ensures native Messages input tokens include cache reads and writes, as normalized usage does."""
    usage = MessagesUsage(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=2
    )

    assert usage.attributes() == {
        "gen_ai.usage.input_tokens": 15,
        "gen_ai.usage.output_tokens": 5,
        "gen_ai.usage.cache_read.input_tokens": 3,
        "gen_ai.usage.cache_write.input_tokens": 2,
    }


def test_messages_usage_keeps_thinking_tokens_across_stream_updates():
    """Ensures native Messages reports thinking tokens, and a later delta without details keeps them.

    `message_delta` usage updates the `message_start` snapshot, so a delta that omits
    `output_tokens_details` must not erase a count reported earlier.
    """
    usage = MessagesUsage()
    usage.update(
        SimpleNamespace(
            input_tokens=10,
            output_tokens=50,
            output_tokens_details=SimpleNamespace(thinking_tokens=36),
        )
    )
    usage.update(SimpleNamespace(output_tokens=60, output_tokens_details=None))

    attributes = usage.attributes()

    assert attributes["gen_ai.usage.output_tokens"] == 60
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 36


def test_messages_usage_reports_web_search_requests():
    """Ensures native Messages reports `server_tool_use.web_search_requests` as Molto's search count."""
    usage = MessagesUsage(
        input_tokens=10,
        output_tokens=5,
        server_tool_use=SimpleNamespace(web_search_requests=2),
    )

    assert usage.attributes()["molto.usage.web_search_requests"] == 2


@pytest.mark.parametrize("reasoning_tokens", [0, None])
def test_zero_or_missing_reasoning_tokens_are_omitted(reasoning_tokens: int | None):
    """Asserts that reasoning tokens appear only when non-zero, as the convention recommends."""
    completion = ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model="gpt-5",
        choices=[],
        usage={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    )

    assert "gen_ai.usage.reasoning.output_tokens" not in chat_completions_response_attributes(
        completion
    )


def test_incomplete_responses_response_reports_its_reason():
    """Ensures a native Responses span reports the incomplete reason, since Responses has no finish reason."""
    response = SimpleNamespace(
        id="resp_1",
        model="gpt-5",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=None,
    )

    attributes = responses_response_attributes(response)

    assert attributes["gen_ai.response.finish_reasons"] == ["max_output_tokens"]


def _recorded_span(streaming: bool, record) -> Any:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("test").start_span("chat")
    record(SpanRecorder(span, streaming=streaming))
    span.end()
    (finished,) = exporter.get_finished_spans()
    return finished


def test_failed_responses_stream_event_reports_failed():
    """Ensures a native `response.failed` event records the failed response's end attributes.

    The SDK raises only on a separate `error` event, so a stream ending in
    `response.failed` alone would otherwise look like a success on its span.
    """
    event = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            id="resp_1", model="gpt-5", status="failed", incomplete_details=None, usage=None
        ),
    )

    span = _recorded_span(True, lambda recorder: recorder.record_responses_event(event))

    assert span.attributes["gen_ai.response.id"] == "resp_1"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("failed",)


def _chunk(index: int, *, content: str | None = None, finish_reason: str | None = None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(index=index, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id="chatcmpl-1", model="gpt-5", choices=[choice], usage=None)


def test_chat_stream_reports_every_choice_finish_reason_without_usage():
    """Ensures Chat streams report finish reasons per choice and omit usage the caller did not request.

    Usage appears only with `stream_options.include_usage`, which native passthrough
    never injects.
    """

    def record(recorder: SpanRecorder) -> None:
        recorder.record_chat_completions_chunk(_chunk(1, finish_reason="length"))
        recorder.record_chat_completions_chunk(_chunk(0, finish_reason="stop"))

    span = _recorded_span(True, record)

    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop", "length")
    assert "gen_ai.usage.input_tokens" not in span.attributes


def test_first_token_is_recorded_once_and_only_for_streaming_callers():
    """Ensures `molto.first_token` marks the first content delta, once, for streaming callers only."""

    def record(recorder: SpanRecorder) -> None:
        recorder.record_normalized(PartDeltaEvent(index=0, delta=TextDelta(content_delta="Hi")))
        recorder.record_normalized(PartDeltaEvent(index=0, delta=TextDelta(content_delta="!")))

    assert [event.name for event in _recorded_span(True, record).events] == ["molto.first_token"]
    assert _recorded_span(False, record).events == ()


def test_signature_only_reasoning_delta_is_not_a_first_token():
    """Asserts that a delta replacing opaque reasoning state carries no content for time to first token."""

    def record(recorder: SpanRecorder) -> None:
        recorder.record_normalized(
            PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="opaque"))
        )

    assert _recorded_span(True, record).events == ()


# Native passthrough does not validate upstream output, so these build SDK models
# without validation, as the SDKs do for stream chunks a provider sends.


def test_chat_chunk_choice_without_delta_is_recorded_without_raising():
    """Asserts that a Chat chunk whose choice has no delta still records its finish reason.

    Telemetry must never turn a successful native call into a failure.
    """
    chunk = ChatCompletionChunk.model_construct(
        id="chatcmpl-1",
        model="gpt-5",
        choices=[ChunkChoice.model_construct(index=0, finish_reason="stop")],
        usage=None,
    )

    span = _recorded_span(True, lambda recorder: recorder.record_chat_completions_chunk(chunk))

    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.events == ()


def test_chat_completion_without_choice_indexes_or_choices_is_recorded_without_raising():
    """Asserts that Chat responses missing choice indexes, or all choices, never raise."""
    unindexed = ChatCompletion.model_construct(
        id="chatcmpl-1",
        model="gpt-5",
        choices=[ChatChoice.model_construct(finish_reason="stop")],
        usage=None,
    )
    choiceless = ChatCompletion.model_construct(
        id="chatcmpl-2", model="gpt-5", choices=None, usage=None
    )

    assert chat_completions_response_attributes(unindexed)["gen_ai.response.finish_reasons"] == [
        "stop"
    ]
    assert chat_completions_response_attributes(choiceless)["gen_ai.response.id"] == "chatcmpl-2"


def test_message_without_usage_is_recorded_without_raising():
    """Asserts that a native Messages response without usage reports no usage instead of raising."""
    message = Message.model_construct(id="msg_1", model="claude-sonnet-5", stop_reason="end_turn")

    attributes = messages_response_attributes(message)

    assert attributes["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert "gen_ai.usage.input_tokens" not in attributes


async def test_observe_closes_its_source_when_closed_early():
    """Ensures closing the recording wrapper closes the wrapped stream, releasing its response."""
    closed = []

    async def source():
        try:
            yield "a"
            yield "b"
        finally:
            closed.append(True)

    recorded = []
    wrapped = observe(source(), recorded.append)

    assert await anext(wrapped) == "a"
    await wrapped.aclose()

    assert closed == [True]
    assert recorded == ["a"]


async def test_observe_yields_every_event_when_recording_fails():
    """Asserts that a recorder raising on unexpected output never interrupts the stream.

    Native Messages output is unvalidated, so a non-integer cache count makes usage
    arithmetic raise; the caller must still receive every event.
    """

    async def source():
        yield "a"
        yield "b"

    def record(event: str) -> None:
        MessagesUsage(input_tokens=1, cache_read_input_tokens="2").attributes()

    assert [event async for event in observe(source(), record)] == ["a", "b"]


async def test_observe_propagates_source_errors_unchanged():
    """Asserts that an error from the wrapped stream reaches the caller as the same exception."""
    failure = RuntimeError("upstream failed")

    async def source():
        yield "a"
        raise failure

    with pytest.raises(RuntimeError) as raised:
        async for _ in observe(source(), lambda event: None):
            pass

    assert raised.value is failure


@pytest.mark.parametrize(
    ("event_type", "delta", "expected"),
    [
        ("response.output_text.delta", "Hi", True),
        ("response.reasoning_text.delta", "Thinking", True),
        ("response.reasoning_summary_text.delta", "Summary", True),
        ("response.function_call_arguments.delta", '{"city"', True),
        ("response.custom_tool_call_input.delta", "input", True),
        ("response.mcp_call_arguments.delta", '{"q"', True),
        ("response.refusal.delta", "I can't", True),
        ("response.output_text.delta", "", False),
        ("response.audio.delta", "audio", False),
    ],
)
def test_responses_content_deltas(event_type: str, delta: str, expected: bool):
    """Ensures native Responses first-token detection counts text, reasoning, tool, and refusal deltas."""
    event = SimpleNamespace(type=event_type, delta=delta)

    assert is_responses_content_delta(event) is expected


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"content": "Hi"}, True),
        ({"refusal": "I can't"}, True),
        ({"reasoning_content": "Thinking"}, True),
        (
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"city"'}}]},
            True,
        ),
        ({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f"}}]}, False),
        ({"role": "assistant", "content": ""}, False),
    ],
)
def test_chat_completions_content_deltas(delta: dict, expected: bool):
    """Ensures native Chat first-token detection counts text, refusal, reasoning, and tool arguments.

    A tool-call delta carrying only its name or ID, or an empty role chunk, carries
    no generated content.
    """
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )

    assert is_chat_completions_content_delta(chunk) is expected


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"type": "text_delta", "text": "Hi"}, True),
        ({"type": "thinking_delta", "thinking": "Thinking"}, True),
        ({"type": "input_json_delta", "partial_json": '{"city"'}, True),
        ({"type": "signature_delta", "signature": "opaque"}, False),
        ({"type": "text_delta", "text": ""}, False),
    ],
)
def test_messages_content_deltas(delta: dict, expected: bool):
    """Ensures native Messages first-token detection counts text, thinking, and tool-input deltas only."""
    event = SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(**delta))

    assert is_messages_content_delta(event) is expected
