import pytest
from openai.types.responses import ResponseInputItemParam
from pydantic import TypeAdapter

from molto.providers.openai.request_translation import build_openai_request
from molto.types.ir.portable import InputIR
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.chat_completions.to_ir import prepare_chat_request
from molto.types.openai.responses import OpenAIResponsesRequest
from molto.utils.exceptions import BadInputError

SDK_RESPONSE_INPUT = TypeAdapter(list[ResponseInputItemParam])


def test_chat_request_builds_canonical_openai_payload() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert build_openai_request(request.to_ir(), "chat_completions") == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_chat_request_preparation_removes_responses_unsupported_stream_options() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    chat = prepare_chat_request(request, "chat_completions")
    responses = prepare_chat_request(request, "responses")

    assert chat.stream_options is not None
    assert chat.stream_options.include_usage is True
    assert responses.stream_options is None


def test_responses_request_builds_canonical_openai_payload() -> None:
    request = OpenAIResponsesRequest(model="openai/gpt-test", input="hello")

    assert build_openai_request(request.to_ir(), "responses") == {
        "model": "gpt-test",
        "input": "hello",
    }


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
def test_responses_sdk_payload_rejects_unsupported_penalties(field: str) -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="hello",
        **{field: 0.5},
    )

    with pytest.raises(BadInputError, match=field):
        build_openai_request(request.to_ir(), "responses")


@pytest.mark.parametrize(
    ("endpoint", "openai_request", "expected"),
    [
        (
            "chat_completions",
            OpenAIChatCompletionRequest(
                model="openai/gpt-test",
                messages=[{"role": "user", "content": "hello"}],
                prompt_cache_key="cache-key",
                prompt_cache_retention="24h",
                safety_identifier="safety-id",
                user="legacy-user",
            ),
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": "cache-key",
                "prompt_cache_retention": "24h",
                "safety_identifier": "safety-id",
                "user": "legacy-user",
            },
        ),
        (
            "responses",
            OpenAIResponsesRequest(
                model="openai/gpt-test",
                input="hello",
                prompt_cache_key="cache-key",
                prompt_cache_retention="24h",
                safety_identifier="safety-id",
                user="legacy-user",
            ),
            {
                "model": "gpt-test",
                "input": "hello",
                "prompt_cache_key": "cache-key",
                "prompt_cache_retention": "24h",
                "safety_identifier": "safety-id",
                "user": "legacy-user",
            },
        ),
    ],
)
def test_shared_openai_request_metadata_survives_ir(
    endpoint: str,
    openai_request: OpenAIChatCompletionRequest | OpenAIResponsesRequest,
    expected: dict,
) -> None:
    assert build_openai_request(openai_request.to_ir(), endpoint) == expected


def test_responses_stateless_web_search_history_round_trips_through_ir() -> None:
    history = [
        {"role": "user", "content": "Find the current release."},
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "content": [],
            "encrypted_content": "encrypted-state",
            "status": "completed",
        },
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {
                "type": "search",
                "query": "current release",
                "queries": ["current release"],
                "sources": [{"type": "url", "url": "https://example.com/release"}],
            },
            "status": "completed",
        },
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "The current release is 2.0.",
                    "annotations": [],
                }
            ],
        },
        {"role": "user", "content": "Summarize that."},
    ]
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input=history,
        store=False,
    )

    ir = InputIR.model_validate(request.to_ir().model_dump(mode="python"))

    payload = build_openai_request(ir, "responses")
    assert payload == {
        "model": "gpt-test",
        "store": False,
        "input": history,
    }
    SDK_RESPONSE_INPUT.validate_python(payload["input"])


def test_responses_stateless_code_interpreter_history_round_trips_through_ir() -> None:
    history = [
        {"role": "user", "content": "Calculate 21 * 2."},
        {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "code": "print(21 * 2)",
            "container_id": "container_1",
            "outputs": [{"type": "logs", "logs": "42\n"}],
            "status": "completed",
        },
        {
            "type": "message",
            "id": "msg_2",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The result is 42.", "annotations": []}],
        },
        {"role": "user", "content": "What operation did you use?"},
    ]
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input=history,
        store=False,
    )

    ir = InputIR.model_validate(request.to_ir().model_dump(mode="python"))

    payload = build_openai_request(ir, "responses")
    assert payload == {
        "model": "gpt-test",
        "store": False,
        "input": history,
    }
    SDK_RESPONSE_INPUT.validate_python(payload["input"])


def test_responses_stateless_unsupported_mcp_history_rejects() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input=[
            {
                "type": "mcp_call",
                "id": "mcp_1",
                "server_label": "docs",
                "name": "lookup",
                "arguments": "{}",
                "status": "completed",
                "output": "result",
            },
            {"role": "user", "content": "Summarize that result."},
        ],
        store=False,
    )

    with pytest.raises(BadInputError, match="OpenAIMcpToolCall"):
        request.to_ir()
