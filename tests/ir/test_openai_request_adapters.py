import pytest
from openai.types.responses import ResponseInputItemParam
from pydantic import TypeAdapter

from molto.providers.openai.request_translation import build_openai_request
from molto.types.ir.portable import InputIR
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.chat_completions.to_ir import prepare_chat_request
from molto.types.openai.responses import OpenAIResponsesRequest
from molto.types.openai.responses.to_ir import prepare_responses_request
from molto.utils.exceptions import BadInputError

SDK_RESPONSE_INPUT = TypeAdapter(list[ResponseInputItemParam])


def test_chat_request_builds_canonical_openai_payload() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert build_openai_request(request.to_ir(), "chat/completions") == {
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

    chat = prepare_chat_request(request, "chat/completions")
    responses = prepare_chat_request(request, "responses")

    assert chat.stream_options is not None
    assert chat.stream_options.include_usage is True
    assert chat.include_usage is True
    # Responses always reports usage: no stream_options are forwarded, but the
    # user's include_usage survives for output emission.
    assert responses.stream_options is None
    assert responses.include_usage is True


def test_responses_request_include_usage_forwarded_as_true_to_chat() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="hello",
        stream=True,
        stream_options={"include_usage": False, "include_obfuscation": True},
    )

    chat = prepare_responses_request(request, "chat/completions")

    # The Chat provider always receives include_usage=True; the user's
    # include_obfuscation remains representable and is forwarded. The user's
    # include_usage value survives for output emission.
    assert chat.stream_options is not None
    assert chat.stream_options.include_usage is True
    assert chat.stream_options.include_obfuscation is True
    assert chat.include_usage is False
    assert build_openai_request(chat, "chat/completions")["stream_options"] == {
        "include_obfuscation": True,
        "include_usage": True,
    }


def test_responses_request_include_usage_is_dropped_for_responses() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="hello",
        stream=True,
        stream_options={"include_usage": True, "include_obfuscation": False},
    )

    responses = prepare_responses_request(request, "responses")

    # The Responses endpoint always reports usage; include_usage is dropped
    # while include_obfuscation remains representable.
    assert responses.include_usage is None
    assert responses.stream_options is not None
    assert responses.stream_options.include_usage is None
    assert responses.stream_options.include_obfuscation is False


def test_responses_request_stream_options_dropped_when_only_include_usage() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="hello",
        stream=True,
        stream_options={"include_usage": True},
    )

    responses = prepare_responses_request(request, "responses")

    assert responses.stream_options is None
    assert responses.include_usage is None


def test_responses_request_include_usage_not_forwarded_when_not_streaming() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="hello",
        stream=False,
        stream_options={"include_usage": False},
    )

    chat = prepare_responses_request(request, "chat/completions")

    # include_usage is a streaming-only option; non-stream requests carry no
    # stream_options to the provider, and the usage is always returned.
    assert chat.stream_options is None
    assert chat.include_usage is False


def test_responses_request_builds_canonical_openai_payload() -> None:
    request = OpenAIResponsesRequest(model="openai/gpt-test", input="hello")

    assert build_openai_request(request.to_ir(), "responses") == {
        "model": "gpt-test",
        "input": "hello",
    }


def test_responses_easy_assistant_message_builds_canonical_openai_payload() -> None:
    input_items = [
        {
            "role": "assistant",
            "content": "Earlier answer.",
            "phase": "final_answer",
        }
    ]
    SDK_RESPONSE_INPUT.validate_python(input_items)
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input=input_items,
    )

    assert build_openai_request(request.to_ir(), "responses") == {
        "model": "gpt-test",
        "input": input_items,
    }


def test_responses_reasoning_history_preserves_omitted_content() -> None:
    input_items = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "encrypted-state",
        }
    ]
    SDK_RESPONSE_INPUT.validate_python(input_items)
    request = OpenAIResponsesRequest(model="openai/gpt-test", input=input_items)

    assert build_openai_request(request.to_ir(), "responses") == {
        "model": "gpt-test",
        "input": input_items,
    }


def test_responses_request_preparation_removes_chat_unsupported_controls() -> None:
    request = OpenAIResponsesRequest(
        model="anthropic/claude-test",
        input="hello",
        include=["reasoning.encrypted_content"],
        reasoning={"effort": "high", "summary": "auto"},
        store=False,
    )

    responses = prepare_responses_request(request, "responses")
    chat = prepare_responses_request(request, "chat/completions")

    assert responses.include == ("reasoning.encrypted_content",)
    assert responses.generation.reasoning_summary == "auto"
    assert chat.include is None
    assert chat.generation.reasoning_summary is None
    assert chat.generation.reasoning_effort == "high"
    assert chat.store is False


def test_responses_web_search_is_projected_to_openai_chat_options() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="Find nearby events.",
        tools=[
            {
                "type": "web_search",
                "search_context_size": "high",
                "user_location": {
                    "type": "approximate",
                    "city": "San Francisco",
                    "country": "US",
                },
            },
            {
                "type": "function",
                "name": "save_event",
                "parameters": {"type": "object"},
            },
        ],
    )

    ir = prepare_responses_request(request, "chat/completions")

    assert build_openai_request(ir, "chat/completions") == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Find nearby events."}],
        "web_search_options": {
            "search_context_size": "high",
            "user_location": {
                "type": "approximate",
                "approximate": {"city": "San Francisco", "country": "US"},
            },
        },
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "save_event",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ],
    }


def test_responses_bare_web_search_is_projected_to_openai_chat_options() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="Search the web.",
        tools=[{"type": "web_search"}],
    )

    ir = prepare_responses_request(request, "chat/completions")

    assert build_openai_request(ir, "chat/completions")["web_search_options"] == {}


def test_responses_web_search_rejects_fields_openai_chat_cannot_represent() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="Search the docs.",
        tools=[{"type": "web_search", "filters": {"allowed_domains": ["example.com"]}}],
    )

    with pytest.raises(BadInputError, match="filters"):
        prepare_responses_request(request, "chat/completions")


def test_responses_web_search_on_anthropic_chat_is_rejected() -> None:
    request = OpenAIResponsesRequest(
        model="anthropic/claude-test",
        input="Search the docs.",
        tools=[{"type": "web_search"}],
    )

    ir = prepare_responses_request(request, "chat/completions")

    # The Anthropic compatibility endpoint validates tools as function-only and
    # has no web search field, so hosted web search fails closed on the wire.
    with pytest.raises(BadInputError, match="cannot represent Responses hosted tool"):
        build_openai_request(ir, "chat/completions")


def test_chat_store_survives_ir_and_request_translation() -> None:
    request = OpenAIChatCompletionRequest(
        model="anthropic/claude-test",
        messages=[{"role": "user", "content": "hello"}],
        store=False,
    )

    assert build_openai_request(request.to_ir(), "chat/completions") == {
        "model": "claude-test",
        "store": False,
        "messages": [{"role": "user", "content": "hello"}],
    }


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
def test_responses_sdk_payload_omits_unsupported_penalties(field: str) -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="hello",
        **{field: 0.5},
    )

    payload = build_openai_request(request.to_ir(), "responses")

    # The Responses SDK has no penalty arguments; the adapter drops them and
    # strict capability validation rejects them before provider construction.
    assert field not in payload


@pytest.mark.parametrize(
    ("endpoint", "openai_request", "expected"),
    [
        (
            "chat/completions",
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


NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "crm",
    "description": "CRM tools for customer lookup and order management.",
    "tools": [
        {
            "type": "function",
            "name": "get_customer_profile",
            "description": "Fetch a customer profile by customer ID.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "list_open_orders",
            "description": "List open orders for a customer ID.",
            "defer_loading": True,
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    ],
}


def test_responses_namespace_tool_round_trips_through_ir() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input="List open orders for customer CUST-12345.",
        tools=[NAMESPACE_TOOL],
    )

    ir = InputIR.model_validate(request.to_ir().model_dump(mode="python"))

    payload = build_openai_request(ir, "responses")
    assert payload == {
        "model": "gpt-test",
        "input": "List open orders for customer CUST-12345.",
        "tools": [NAMESPACE_TOOL],
    }


def test_responses_namespaced_function_call_history_round_trips_through_ir() -> None:
    history = [
        {"role": "user", "content": "List open orders for customer CUST-12345."},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "list_open_orders",
            "namespace": "crm",
            "arguments": '{"customer_id":"CUST-12345"}',
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "[]"},
        {"role": "user", "content": "Any priority orders?"},
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


def test_chat_request_rejects_namespaced_function_call_history() -> None:
    request = OpenAIResponsesRequest(
        model="openai/gpt-test",
        input=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "list_open_orders",
                "namespace": "crm",
                "arguments": '{"customer_id":"CUST-12345"}',
            },
        ],
    )

    with pytest.raises(BadInputError, match="namespaced function calls"):
        build_openai_request(request.to_ir(), "chat/completions")
