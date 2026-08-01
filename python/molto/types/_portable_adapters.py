"""Shared dispatch over portable output items."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from molto.types.ir.output_items import AssistantMessageItem, FunctionCallItem, ReasoningItem

REASONING_SUMMARY = "summary"
REASONING_CONTENT = "content"
REASONING_ENCRYPTED = "encrypted"


def project(
    values: dict[str, Any],
    explicit_fields: Iterable[str],
    *,
    strict: bool = False,
    include_missing: bool = False,
) -> dict[str, Any]:
    """Keep only explicitly present fields when shaping an OpenAI payload."""

    if strict:
        return {field: values[field] for field in explicit_fields}
    if include_missing:
        return {field: values.get(field) for field in explicit_fields}
    explicit = set(explicit_fields)
    return {field: value for field, value in values.items() if field in explicit}


def dispatch_output_item(
    item: Any,
    *,
    assistant_message: Callable[[AssistantMessageItem, list[Any]], Any],
    assistant_content: Callable[[Any, int], Any],
    reasoning_item: Callable[[ReasoningItem, list[tuple[str, Any]]], Any],
    reasoning_content: Callable[[Any], tuple[str, Any]],
    function_call: Callable[[FunctionCallItem], Any],
) -> Any:
    """Dispatch one portable output item to an adapter's wire shaping."""

    if isinstance(item, AssistantMessageItem):
        content = [
            assistant_content(part, content_index)
            for content_index, part in enumerate(item.content)
        ]
        return assistant_message(item, content)
    if isinstance(item, ReasoningItem):
        parts = [reasoning_content(part) for part in item.parts]
        return reasoning_item(item, parts)
    if isinstance(item, FunctionCallItem):
        return function_call(item)
    raise AssertionError(f"Unhandled portable output item type: {type(item).__name__}")
