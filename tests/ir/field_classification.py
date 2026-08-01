from __future__ import annotations

from enum import Enum


class FrozenFieldOwner(str, Enum):
    IMAGE_GENERATION_CALL = "openai:image_generation_call"
    PORTABLE = "portable"
    OUTPUT_TEXT = "openai:output_text"
    RESPONSE_METADATA = "openai:response_metadata"
    RESPONSES_SNAPSHOT = "openai:responses_snapshot"
    STREAM_METADATA = "openai:stream_metadata"
    TOOL_USAGE = "openai:tool_usage"
    USAGE_DETAILS = "openai:usage_details"
    WEB_SEARCH_CALL = "openai:web_search_call"


_PREFIX_OWNERS: tuple[tuple[str, FrozenFieldOwner], ...] = (
    ("response.output[].message.content[].annotations[]", FrozenFieldOwner.OUTPUT_TEXT),
    ("response.output[].message.content[].logprobs[]", FrozenFieldOwner.OUTPUT_TEXT),
    ("response.output[].web_search_call", FrozenFieldOwner.WEB_SEARCH_CALL),
    ("response.output[].image_generation_call", FrozenFieldOwner.IMAGE_GENERATION_CALL),
    ("response.tools[].web_search", FrozenFieldOwner.RESPONSES_SNAPSHOT),
    ("response.tools[].code_interpreter", FrozenFieldOwner.RESPONSES_SNAPSHOT),
    ("response.tools[].image_generation", FrozenFieldOwner.RESPONSES_SNAPSHOT),
    ("response.usage.input_tokens_details.cache_write_tokens", FrozenFieldOwner.USAGE_DETAILS),
    ("response.tool_usage", FrozenFieldOwner.TOOL_USAGE),
    ("response.billing", FrozenFieldOwner.RESPONSE_METADATA),
    ("response.moderation", FrozenFieldOwner.RESPONSE_METADATA),
    ("response.reasoning.context", FrozenFieldOwner.RESPONSE_METADATA),
    ("response.reasoning.mode", FrozenFieldOwner.RESPONSE_METADATA),
    ("stream.obfuscation", FrozenFieldOwner.STREAM_METADATA),
    ("chat.choices[].logprobs", FrozenFieldOwner.OUTPUT_TEXT),
    ("chat.choices[].message.annotations", FrozenFieldOwner.OUTPUT_TEXT),
    ("chat.choices[].finish_reason", FrozenFieldOwner.RESPONSE_METADATA),
    ("chat.choices[].index", FrozenFieldOwner.RESPONSE_METADATA),
    ("chat.system_fingerprint", FrozenFieldOwner.RESPONSE_METADATA),
)

_PORTABLE_PREFIXES = (
    "response.id",
    "response.status",
    "response.error",
    "response.incomplete_details",
    "response.output[].message",
    "response.output[].reasoning",
    "response.output[].function_call",
    "response.usage",
    "response.previous_response_id",
    "response.conversation",
    "response.instructions",
    "response.max_output_tokens",
    "response.max_tool_calls",
    "response.parallel_tool_calls",
    "response.temperature",
    "response.top_p",
    "response.top_logprobs",
    "response.frequency_penalty",
    "response.presence_penalty",
    "response.tool_choice",
    "response.text",
    "response.truncation",
    "response.tools[].function",
    "stream.response",
    "stream.item_id",
    "stream.output_index",
    "stream.content_index",
    "stream.sequence_number",
    "stream.delta",
    "stream.type",
    "chat.id",
    "chat.choices[].message",
    "chat.choices[].delta",
    "chat.usage",
)

_RESPONSE_METADATA_PREFIXES = (
    "response.object",
    "response.created_at",
    "response.completed_at",
    "response.model",
    "response.service_tier",
    "response.background",
    "response.store",
    "response.metadata",
    "response.prompt_cache_key",
    "response.prompt_cache_retention",
    "response.safety_identifier",
    "response.user",
    "chat.object",
    "chat.created",
    "chat.model",
    "chat.service_tier",
)


def classify_frozen_field(path: str) -> FrozenFieldOwner:
    """Classify a normalized populated SDK leaf path or fail closed."""

    for prefix, owner in _PREFIX_OWNERS:
        if path == prefix or path.startswith(f"{prefix}."):
            return owner
    for prefix in _RESPONSE_METADATA_PREFIXES:
        if path == prefix or path.startswith(f"{prefix}."):
            return FrozenFieldOwner.RESPONSE_METADATA
    for prefix in _PORTABLE_PREFIXES:
        if path == prefix or path.startswith(f"{prefix}."):
            return FrozenFieldOwner.PORTABLE
    raise ValueError(f"Unclassified frozen OpenAI field: {path!r}.")
