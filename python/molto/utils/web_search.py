"""Hosted-search wire conversions shared by adapters and upstream clients."""

from typing import Any
from uuid import uuid4

from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    SearchCall,
    SearchResult,
    TextOutput,
    WebCitation,
    WebSearchSource,
)


def messages_text_blocks(
    part: TextOutput,
) -> list[dict[str, Any]]:
    """Split text at citation boundaries into blocks Anthropic accepts.

    Citation boundaries fall on character offsets, so a boundary can isolate
    whitespace or name a zero-length range. Messages rejects an empty text block
    and a block without non-whitespace text, so whitespace-only ranges are merged
    into a neighbouring block and a point annotation is attached to the block that
    contains its offset. Concatenating the returned blocks reproduces the part.
    """
    text = part.content
    if not part.citations:
        return [{"type": "text", "text": text}] if text else []

    boundaries = {0, len(text)}
    for citation in part.citations:
        if citation.start_index == citation.end_index:
            # A point annotation names a position, not a range, so it must not
            # split the text into separate blocks.
            continue
        boundaries.add(citation.start_index)
        boundaries.add(citation.end_index)

    if any(offset < 0 or offset > len(text) for offset in boundaries):
        raise ValueError("Citation offsets fall outside the associated answer text.")
    if any(citation.end_index < citation.start_index for citation in part.citations):
        raise ValueError("Citation end precedes its start.")

    ordered_boundaries = sorted(boundaries)
    # Leading spacing has no earlier block to merge into, so it is carried into
    # the first block that has content.
    ranges: list[list[int]] = []
    pending_start: int | None = None
    for start, end in zip(ordered_boundaries, ordered_boundaries[1:], strict=False):
        if start == end:
            continue
        if not text[start:end].strip():
            if ranges:
                ranges[-1][1] = end
            else:
                pending_start = start
            continue
        ranges.append([start if pending_start is None else pending_start, end])
        pending_start = None

    if not ranges:
        # The part has no splittable content; keep it as one block rather than
        # inventing attribution for a block that carries no text.
        return [{"type": "text", "text": text}] if text else []

    blocks: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(ranges):
        block: dict[str, Any] = {"type": "text", "text": text[start:end]}
        block_citations = [
            citation_to_messages(citation, text=text, block_start=start, block_end=end)
            for citation in part.citations
            if _cites_block(citation, start=start, end=end, is_last=index == len(ranges) - 1)
        ]
        if block_citations:
            block["citations"] = block_citations
        blocks.append(block)
    return blocks


def _cites_block(citation: WebCitation, *, start: int, end: int, is_last: bool) -> bool:
    """Report whether an attribution applies to the block spanning `start`..`end`."""
    if citation.start_index != citation.end_index:
        # A block may extend past the cited range when surrounding whitespace was
        # folded into it, so any overlap associates the attribution.
        return start < citation.end_index and citation.start_index < end
    if start <= citation.start_index < end:
        return True
    # A point annotation at the end of the text belongs to the final block.
    return is_last and citation.start_index == end


def citation_from_responses(annotation: dict[str, Any]) -> WebCitation:
    """Convert a Responses URL annotation with offsets local to its text part."""
    return WebCitation(
        url=annotation["url"],
        title=annotation.get("title"),
        start_index=annotation["start_index"],
        end_index=annotation["end_index"],
    )


def citation_from_messages(citation: dict[str, Any], *, text: str) -> WebCitation:
    """Associate a Messages citation with its whole text block."""
    return WebCitation(
        url=citation["url"],
        title=citation.get("title"),
        start_index=0,
        end_index=len(text),
        cited_text=citation["cited_text"],
        encrypted_index=citation["encrypted_index"],
    )


def citation_to_responses(citation: WebCitation) -> dict[str, Any]:
    """Build a Responses URL annotation, omitting unavailable source context."""
    return {
        "type": "url_citation",
        "url": citation.url,
        "title": citation.title or "",
        "start_index": citation.start_index,
        "end_index": citation.end_index,
    }


def citation_to_messages(
    citation: WebCitation, *, text: str, block_start: int, block_end: int
) -> dict[str, Any]:
    """Build a Messages citation with the documented answer-range/state fallbacks."""
    cited_text = citation.cited_text
    if cited_text is None:
        # A point annotation has no answer characters of its own, so its excerpt
        # falls back to the block that carries it rather than to an empty string.
        cited_text = text[citation.start_index : citation.end_index] or text[block_start:block_end]
    return {
        "type": "web_search_result_location",
        "url": citation.url,
        "title": citation.title if citation.title is not None else "",
        "cited_text": cited_text,
        "encrypted_index": citation.encrypted_index
        if citation.encrypted_index is not None
        else "N/A",
    }


def search_source_from_responses(source: dict[str, Any]) -> WebSearchSource:
    """Retain observed Responses source metadata without inventing retrieved content."""
    return WebSearchSource(
        url=source["url"], title=source.get("title"), page_age=source.get("page_age")
    )


def search_source_from_messages(source: dict[str, Any]) -> WebSearchSource:
    """Retain Messages source metadata and opaque retrieved content."""
    return WebSearchSource(
        url=source["url"],
        title=source.get("title"),
        page_age=source.get("page_age"),
        encrypted_content=source.get("encrypted_content"),
    )


def search_source_to_responses(search_source: WebSearchSource) -> dict[str, Any]:
    """Build the Responses URL-source shape."""
    return {"type": "url", "url": search_source.url}


def search_source_to_messages(search_source: WebSearchSource) -> dict[str, Any]:
    """Build a Messages source using placeholders only for unavailable values."""
    return {
        "type": "web_search_result",
        "url": search_source.url,
        "title": search_source.title if search_source.title is not None else "",
        "encrypted_content": search_source.encrypted_content
        if search_source.encrypted_content is not None
        else "N/A",
        **({"page_age": search_source.page_age} if search_source.page_age is not None else {}),
    }


def search_call_from_responses(item: dict[str, Any]) -> SearchCall:
    """Convert an observed Responses operation or a supplied history item."""
    action = item.get("action") or {}
    return SearchCall(
        id=item["id"],
        action=action.get("type", "search"),
        query=action.get("query"),
        queries=action.get("queries"),
        url=action.get("url"),
        pattern=action.get("pattern"),
        status=item.get("status", "in_progress"),
    )


def search_call_from_messages(block: dict[str, Any], *, strict: bool = True) -> SearchCall:
    """Convert Messages server-search input after its arguments have been collected.

    `strict` validates caller-supplied history. Provider output is normalized with
    `strict=False` so an unknown or partially parsed input shape cannot turn a
    provider defect into a request-validation error.
    """
    search_input = block.get("input")
    if not isinstance(search_input, dict):
        if strict:
            raise UnsupportedInput("Unsupported Anthropic web search input fields.")
        return SearchCall(id=block["id"])
    if strict and set(search_input) - {"query"}:
        raise UnsupportedInput("Unsupported Anthropic web search input fields.")
    return SearchCall(id=block["id"], query=search_input.get("query"))


def search_call_to_responses_action(search_call: SearchCall) -> dict[str, Any]:
    """Build the action shared by Responses item starts and completed items."""
    action: dict[str, Any] = {"type": search_call.action}
    for field in ("query", "queries", "url", "pattern"):
        value = getattr(search_call, field)
        if value is not None:
            action[field] = value
    return action


def search_call_to_responses(search_call: SearchCall) -> dict[str, Any]:
    """Build a Responses invocation item without inventing a linked result."""
    return {
        "id": search_call.id,
        "type": "web_search_call",
        "status": search_call.status,
        "action": search_call_to_responses_action(search_call),
    }


def search_calls_to_messages(search_call: SearchCall) -> list[dict[str, Any]]:
    """Build one Messages invocation per query; callers omit browsing operations."""
    if search_call.action != "search":
        raise UnsupportedInput("Messages cannot represent browsing operations.")
    # Retain the original ID on the first call to link the pooled batch result
    # once. Responses supplies no source attribution for the individual queries.
    return [
        {
            "type": "server_tool_use",
            "id": search_call.id if index == 0 else f"srvtoolu_{uuid4().hex}",
            "name": "web_search",
            "input": {} if query is None else {"query": query},
        }
        for index, query in enumerate(search_call.queries or [search_call.query])
    ]


def search_result_from_responses(item: dict[str, Any]) -> SearchResult:
    """Extract a result after the caller identifies a finished Responses search."""
    sources = (item.get("action") or {}).get("sources")
    return SearchResult(
        tool_call_id=item["id"],
        sources=[search_source_from_responses(source) for source in sources]
        if sources is not None
        else None,
        error="unavailable" if item.get("status") == "failed" else None,
    )


def search_result_from_messages(block: dict[str, Any]) -> SearchResult:
    """Convert a Messages result block, preserving errors and explicit empty results."""
    content = block["content"]
    return SearchResult(
        tool_call_id=block["tool_use_id"],
        sources=[search_source_from_messages(source) for source in content]
        if isinstance(content, list)
        else None,
        error=content.get("error_code") if isinstance(content, dict) else None,
    )


def search_result_to_messages(search_result: SearchResult) -> dict[str, Any]:
    """Build a Messages result; callers omit results with unknown sources and no error."""
    if search_result.error is not None:
        content: Any = {"type": "web_search_tool_result_error", "error_code": search_result.error}
    elif search_result.sources is not None:
        content = [search_source_to_messages(source) for source in search_result.sources]
    else:
        raise ValueError("Cannot represent an unknown search result as a Messages result.")
    return {
        "type": "web_search_tool_result",
        "tool_use_id": search_result.tool_call_id,
        "content": content,
    }
