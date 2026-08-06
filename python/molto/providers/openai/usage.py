from __future__ import annotations

from collections.abc import Mapping

from molto.types.ir.usage import Usage

from .types import OpenAIEndpoint


def _optional_int(values: Mapping[str, object], name: str) -> int | None:
    value = values.get(name)
    return None if value is None else int(value)


def _put(values: dict[str, object], name: str, value: int | None) -> None:
    if value is not None:
        values[name] = value


def usage_from_openai(payload: Mapping[str, object] | None, endpoint: OpenAIEndpoint) -> Usage:
    """Map the common OpenAI usage counters into the portable usage model."""

    if payload is None:
        return Usage()
    if endpoint == "responses":
        input_details = dict(payload.get("input_tokens_details") or {})
        output_details = dict(payload.get("output_tokens_details") or {})
        return Usage(
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cached_input_tokens=int(input_details.get("cached_tokens") or 0),
            cache_write_input_tokens=_optional_int(input_details, "cache_write_tokens"),
            reasoning_output_tokens=int(output_details.get("reasoning_tokens") or 0),
        )

    input_details = dict(payload.get("prompt_tokens_details") or {})
    output_details = dict(payload.get("completion_tokens_details") or {})
    return Usage(
        input_tokens=int(payload.get("prompt_tokens", 0)),
        output_tokens=int(payload.get("completion_tokens", 0)),
        total_tokens=int(payload.get("total_tokens", 0)),
        cached_input_tokens=int(input_details.get("cached_tokens") or 0),
        cache_write_input_tokens=_optional_int(input_details, "cache_write_tokens"),
        reasoning_output_tokens=int(output_details.get("reasoning_tokens") or 0),
        audio_input_tokens=_optional_int(input_details, "audio_tokens"),
        image_input_tokens=_optional_int(input_details, "image_tokens"),
        text_input_tokens=_optional_int(input_details, "text_tokens"),
        audio_output_tokens=_optional_int(output_details, "audio_tokens"),
        image_output_tokens=_optional_int(output_details, "image_tokens"),
        text_output_tokens=_optional_int(output_details, "text_tokens"),
        accepted_prediction_output_tokens=_optional_int(
            output_details, "accepted_prediction_tokens"
        ),
        rejected_prediction_output_tokens=_optional_int(
            output_details, "rejected_prediction_tokens"
        ),
    )


def usage_to_openai(usage: Usage, endpoint: OpenAIEndpoint) -> dict[str, object]:
    """Map portable usage counters to an OpenAI response payload."""

    if endpoint == "responses":
        input_details: dict[str, object] = {"cached_tokens": usage.cached_input_tokens}
        _put(input_details, "cache_write_tokens", usage.cache_write_input_tokens)
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "input_tokens_details": input_details,
            "output_tokens_details": {"reasoning_tokens": usage.reasoning_output_tokens},
        }
    input_details: dict[str, object] = {"cached_tokens": usage.cached_input_tokens}
    _put(input_details, "cache_write_tokens", usage.cache_write_input_tokens)
    output_details: dict[str, object] = {"reasoning_tokens": usage.reasoning_output_tokens}
    _put(input_details, "audio_tokens", usage.audio_input_tokens)
    _put(input_details, "image_tokens", usage.image_input_tokens)
    _put(input_details, "text_tokens", usage.text_input_tokens)
    _put(output_details, "audio_tokens", usage.audio_output_tokens)
    _put(output_details, "image_tokens", usage.image_output_tokens)
    _put(output_details, "text_tokens", usage.text_output_tokens)
    _put(
        output_details,
        "accepted_prediction_tokens",
        usage.accepted_prediction_output_tokens,
    )
    _put(
        output_details,
        "rejected_prediction_tokens",
        usage.rejected_prediction_output_tokens,
    )
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_tokens_details": input_details,
        "completion_tokens_details": output_details,
    }
