from __future__ import annotations

from typing import Any

from molto.types.anthropic_messages import AnthropicMessagesStreamEventAdapter
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEventAdapter,
)
from molto.utils.io import load_sse

_RECONSTRUCTIBLE_RESPONSE_FIELDS = {
    "id",
    "incomplete_details",
    "model",
    "object",
    "output",
    "service_tier",
    "status",
    "usage",
}

_RECONSTRUCTIBLE_CHAT_COMPLETIONS_FIELDS = {
    "choices",
    "created",
    "id",
    "model",
    "object",
    "service_tier",
    "usage",
}

_ANTHROPIC_EVENT_ADAPTER = AnthropicMessagesStreamEventAdapter


def assert_contains_expected(actual: Any, expected: Any, path: str = "$") -> None:
    """Assert that actual recursively contains the expected value."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected an object at {path}, got {type(actual)}"
        missing_keys = expected.keys() - actual.keys()
        assert not missing_keys, f"Missing keys at {path}: {sorted(missing_keys)}"
        for key, expected_value in expected.items():
            assert_contains_expected(actual[key], expected_value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"Expected a list at {path}, got {type(actual)}"
        assert len(actual) == len(expected), (
            f"Expected {len(expected)} items at {path}, got {len(actual)}"
        )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            assert_contains_expected(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, f"Value mismatch at {path}: {actual!r} != {expected!r}"


def reconstructable_chat_completions_chunks(
    sse: str,
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate and normalize one Chat Completions SSE chunk for comparison."""
    OpenAIChatCompletionsCreateRequest.model_validate(body)
    chunk = OpenAIChatCompletionsStreamChunk.model_validate(load_sse(sse)).model_dump(
        mode="json",
        exclude_unset=True,
    )
    chunk = {
        key: value
        for key, value in chunk.items()
        if key in _RECONSTRUCTIBLE_CHAT_COMPLETIONS_FIELDS
    }

    choices = chunk["choices"]
    for choice in choices:
        # TODO: Remove this as soon as we support logprobs
        if choice.get("logprobs") in (None, ""):
            choice.pop("logprobs", None)
        if choice["finish_reason"] is not None:
            delta = choice["delta"]
            for field_name in ("content", "reasoning_content"):
                if delta.get(field_name) in (None, ""):
                    delta.pop(field_name, None)

    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return [chunk]

    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    chunk["usage"] = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "prompt_tokens_details": {
            "cached_tokens": prompt_details.get("cached_tokens") or 0,
            "cache_write_tokens": prompt_details.get("cache_write_tokens") or 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": completion_details.get("reasoning_tokens") or 0,
        },
    }
    if not choices:
        return [chunk]

    # Providers may combine terminal usage with the finish chunk or send it next.
    usage_chunk = {**chunk, "choices": []}
    chunk["usage"] = None
    return [chunk, usage_chunk]


def reconstructable_responses_event(
    sse: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Validate a Responses SSE event and retain fields reconstructible from Chat."""
    request = OpenAIResponsesCreateRequest.model_validate(body)
    event = OpenAIResponsesStreamEventAdapter.validate_python(load_sse(sse)).model_dump(
        mode="json",
        exclude_unset=True,
    )
    event.pop("item_id", None)

    item = event.get("item")
    if isinstance(item, dict):
        # Chat results expose neither Responses item IDs nor assistant phases.
        item.pop("id", None)
        item.pop("phase", None)

    response = event.get("response")
    if isinstance(response, dict):
        response = {
            key: value
            for key, value in response.items()
            if key in request.model_fields_set or key in _RECONSTRUCTIBLE_RESPONSE_FIELDS
        }
        # SDK responses may omit this optional field when there is no incomplete reason.
        response.setdefault("incomplete_details", None)
        reasoning = response.get("reasoning")
        if isinstance(reasoning, dict):
            # Chat exposes reasoning effort, but not Responses context, mode, or summary.
            response["reasoning"] = {
                key: value for key, value in reasoning.items() if key == "effort"
            }
        if (
            event["type"] in {"response.created", "response.in_progress"}
            and response.get("service_tier") == "auto"
        ):
            # Native Responses may report an unresolved tier before completion;
            # translated responses use the documented target default throughout.
            response["service_tier"] = "default"
        for output_item in response.get("output") or []:
            output_item.pop("id", None)
            output_item.pop("phase", None)
        event["response"] = response

    return event


def reconstructable_anthropic_event(sse: str) -> dict[str, Any]:
    """Validate an Anthropic SSE event and retain fields available across protocols."""
    event = _ANTHROPIC_EVENT_ADAPTER.validate_python(load_sse(sse)).model_dump(
        mode="json",
        exclude_unset=True,
    )
    if event["type"] == "message_start":
        message = event["message"]
        message.pop("stop_details", None)
        # Other protocols report usage only when the stream ends. Matching the
        # Messages start counts would require buffering its otherwise-live stream.
        usage = message.pop("usage", None)
        if isinstance(usage, dict) and "service_tier" in usage:
            message["usage"] = {"service_tier": usage["service_tier"]}
    elif event["type"] == "message_delta":
        event["delta"].pop("stop_details", None)
    return event
