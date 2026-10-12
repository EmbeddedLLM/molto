#!/usr/bin/env python3
"""Prepare an Anthropic Messages cassette for non-streaming response parity.

The Responses cassette is canonical. This script rewrites the representable
request and JSON response fields in a Messages cassette while preserving its
native wire format and transport metadata. Hosted search uses the planned reduced
Messages contract: observed sources, answer-range citations, and N/A state
placeholders. Missing sources on a completed search are an evidence error.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from molto.utils.io import dump_yaml, json_loads, read_yaml


def _request_body(cassette: dict[str, Any]) -> dict[str, Any]:
    try:
        request_body = cassette["request"]["body"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain request.body.") from exc
    if not isinstance(request_body, dict):
        raise ValueError("Cassette request.body must be an object.")
    return request_body


def _image_content(image_url: Any) -> dict[str, Any]:
    if not isinstance(image_url, str):
        raise ValueError("Responses image content must contain a string image_url field.")

    if not image_url.startswith("data:"):
        return {"type": "image", "source": {"type": "url", "url": image_url}}

    metadata, separator, data = image_url.partition(",")
    if not separator or not metadata.endswith(";base64"):
        raise ValueError("Responses image data URLs must contain base64-encoded data.")
    media_type = metadata.removeprefix("data:").removesuffix(";base64")
    if not media_type or not data:
        raise ValueError("Responses image data URLs must contain a media type and data.")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _message_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("Responses message content must be text or a list of content parts.")

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Responses message content parts must be objects.")
        if part.get("type") == "input_image":
            blocks.append(_image_content(part.get("image_url")))
        elif part.get("type") in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Responses text content must contain a string text field.")
            blocks.extend(
                _cited_text_content(part)
                if part.get("annotations")
                else [{"type": "text", "text": text}]
            )
        else:
            raise ValueError("Responses message content cannot map to Messages.")
    return blocks


def _reasoning_content(item: dict[str, Any]) -> dict[str, str]:
    content = item.get("content") or []
    summary = item.get("summary") or []
    if not isinstance(content, list):
        raise ValueError("Responses reasoning input must contain a content list.")
    if not isinstance(summary, list):
        raise ValueError("Responses reasoning input summary must be a list.")

    reasoning_text: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "reasoning_text":
            raise ValueError("Only raw Responses reasoning text can map to Messages thinking.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Responses reasoning text must be a string.")
        reasoning_text.append(text)

    summary_text: list[str] = []
    for part in summary:
        if not isinstance(part, dict) or part.get("type") != "summary_text":
            raise ValueError("Only Responses reasoning summaries can map to Messages thinking.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Responses reasoning summary text must be a string.")
        summary_text.append(text)

    thinking = {"type": "thinking", "thinking": "".join(reasoning_text or summary_text)}
    encrypted_content = item.get("encrypted_content")
    if encrypted_content is not None:
        if not isinstance(encrypted_content, str):
            raise ValueError("Responses encrypted reasoning content must be a string.")
        thinking["signature"] = encrypted_content
    return thinking


def _function_call_content(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise ValueError("Responses function calls must contain string call_id and name fields.")
    if not isinstance(arguments, str):
        raise ValueError("Responses function call arguments must be a string.")
    try:
        input_object = json_loads(arguments)
    except ValueError as exc:
        raise ValueError("Responses function call arguments must contain valid JSON.") from exc
    if not isinstance(input_object, dict):
        raise ValueError("Responses function call arguments must contain a JSON object.")
    return {
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": input_object,
    }


def _function_call_output_content(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    if not isinstance(call_id, str):
        raise ValueError("Responses function call outputs must contain a string call_id field.")
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": _message_content(item.get("output")),
    }


def _append_assistant_content(
    messages: list[dict[str, Any]], content: list[dict[str, Any]]
) -> None:
    if messages and messages[-1].get("role") == "assistant":
        existing_content = messages[-1].get("content")
        if not isinstance(existing_content, list):
            raise ValueError("Messages assistant content must be a list.")
        existing_content.extend(content)
    else:
        messages.append({"content": content, "role": "assistant"})


def _append_tool_result(messages: list[dict[str, Any]], tool_result: dict[str, Any]) -> None:
    if messages and messages[-1].get("role") == "user":
        existing_content = messages[-1].get("content")
        if (
            isinstance(existing_content, list)
            and existing_content
            and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in existing_content
            )
        ):
            existing_content.append(tool_result)
            return
    messages.append({"content": [tool_result], "role": "user"})


def _messages_from_response_input(response_input: Any) -> list[dict[str, Any]]:
    if isinstance(response_input, str):
        return [{"content": [{"type": "text", "text": response_input}], "role": "user"}]
    if not isinstance(response_input, list):
        raise ValueError("Responses input must be a string or an input-item list.")

    messages: list[dict[str, Any]] = []
    pending_assistant_content: list[dict[str, Any]] = []
    for item in response_input:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects.")
        item_type = item.get("type")
        if item_type == "reasoning":
            pending_assistant_content.append(_reasoning_content(item))
            continue
        if item_type == "web_search_call":
            _append_assistant_content(
                messages, [*pending_assistant_content, *_search_content(item)]
            )
            pending_assistant_content = []
            continue
        if item_type == "function_call":
            _append_assistant_content(
                messages,
                [*pending_assistant_content, _function_call_content(item)],
            )
            pending_assistant_content = []
            continue
        if item_type == "function_call_output":
            if pending_assistant_content:
                raise ValueError(
                    "Responses reasoning input must precede an assistant output item."
                )
            _append_tool_result(messages, _function_call_output_content(item))
            continue

        role = item.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Responses input role {role!r} cannot map to Messages.")
        content = _message_content(item.get("content"))
        if role == "assistant":
            messages.append(
                {"content": [*pending_assistant_content, *content], "role": "assistant"}
            )
            pending_assistant_content = []
        else:
            if pending_assistant_content:
                raise ValueError("Responses reasoning input must precede an assistant message.")
            messages.append({"content": content, "role": "user"})

    if pending_assistant_content:
        raise ValueError("Responses reasoning input must precede an assistant message.")
    return messages


def _web_search_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Map the basic hosted-search declaration without inventing option defaults."""
    if "search_context_size" in tool:
        raise ValueError("Messages cannot translate explicit search_context_size; omit it.")
    unsupported = set(tool) - {"type", "filters", "user_location", "search_context_size"}
    if unsupported:
        raise ValueError(f"Unsupported Responses web search fields: {sorted(unsupported)}")
    result: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search"}
    filters = tool.get("filters")
    if filters is not None:
        if not isinstance(filters, dict) or set(filters) - {"allowed_domains"}:
            raise ValueError("Messages web search supports only allowed_domains filters.")
        if filters.get("allowed_domains") is not None:
            result["allowed_domains"] = deepcopy(filters["allowed_domains"])
    if tool.get("user_location") is not None:
        result["user_location"] = deepcopy(tool["user_location"])
    return result


def _search_content(item: dict[str, Any]) -> list[dict[str, Any]]:
    action = item.get("action")
    if not isinstance(action, dict):
        raise ValueError("Cannot classify a Responses web search call without its action.")
    if action.get("type") in {"open_page", "find_in_page"}:
        return []
    if action.get("type") != "search":
        raise ValueError(f"Unsupported Responses web search action: {action.get('type')!r}")
    queries = action.get("queries")
    if queries is not None and not isinstance(queries, list):
        raise ValueError("Responses search queries must be a list.")
    queries = queries or [action.get("query")]
    if any(query is not None and not isinstance(query, str) for query in queries):
        raise ValueError("Responses search query must be a string when present.")
    call_id = item.get("id")
    if not isinstance(call_id, str):
        raise ValueError("Responses web search calls must contain a string id.")
    # Fixture IDs remain deterministic across rematching, while the first call
    # retains the original ID to link the single pooled result.
    blocks = [
        {
            "type": "server_tool_use",
            "id": call_id if index == 0 else f"{call_id}_query_{index + 1}",
            "name": "web_search",
            "input": {} if query is None else {"query": query},
        }
        for index, query in enumerate(queries)
    ]
    status = item.get("status")
    if status in {"in_progress", "searching"}:
        return blocks
    if status == "failed":
        content: Any = {"type": "web_search_tool_result_error", "error_code": "unavailable"}
    elif status == "completed":
        sources = action.get("sources")
        if not isinstance(sources, list):
            raise ValueError(
                "Completed Responses search is missing action.sources; record with "
                "include=['web_search_call.action.sources']. Missing sources are not empty results."
            )
        content = []
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("url"), str):
                raise ValueError("Responses web search sources must contain a string url.")
            result = {
                "type": "web_search_result",
                "url": source["url"],
                "title": source.get("title") or "",
                "encrypted_content": "N/A",
            }
            if source.get("page_age") is not None:
                result["page_age"] = source["page_age"]
            content.append(result)
    else:
        raise ValueError(f"Unsupported Responses web search status: {status!r}")
    blocks.append({"type": "web_search_tool_result", "tool_use_id": call_id, "content": content})
    return blocks


def _cited_text_content(part: dict[str, Any]) -> list[dict[str, Any]]:
    text = part.get("text")
    if not isinstance(text, str):
        raise ValueError("Responses output text must contain a string text field.")
    annotations = part.get("annotations") or []
    if not isinstance(annotations, list):
        raise ValueError("Responses annotations must be a list.")
    boundaries = {0, len(text)}
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
            raise ValueError("Only Responses URL citations can map to Messages citations.")
        start, end = annotation.get("start_index"), annotation.get("end_index")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end <= len(text)
        ):
            raise ValueError("Responses URL citation range must be within its text part.")
        if not isinstance(annotation.get("url"), str):
            raise ValueError("Responses URL citations must contain a string url.")
        boundaries.update((start, end))
    if not annotations:
        return [{"type": "text", "text": text}]
    blocks = []
    boundaries = sorted(boundaries)
    intervals = set(zip(boundaries, boundaries[1:], strict=False))
    # Point citations retain attribution even though they cover no answer characters.
    intervals.update(
        (annotation["start_index"], annotation["end_index"])
        for annotation in annotations
        if annotation["start_index"] == annotation["end_index"]
    )
    for start, end in sorted(intervals):
        block: dict[str, Any] = {"type": "text", "text": text[start:end]}
        citations = [
            {
                "type": "web_search_result_location",
                "url": annotation["url"],
                "title": annotation.get("title") or "",
                "encrypted_index": "N/A",
                "cited_text": text[annotation["start_index"] : annotation["end_index"]],
            }
            for annotation in annotations
            if annotation["start_index"] <= start
            and end <= annotation["end_index"]
            and (start < end or annotation["start_index"] == annotation["end_index"])
        ]
        if citations:
            block["citations"] = citations
        blocks.append(block)
    return blocks


def _map_search_usage(response: dict[str, Any], usage: dict[str, Any]) -> None:
    # Counter snapshots are authoritative; activity blocks cannot establish usage.
    tool_usage = response.get("tool_usage") or {}
    search_usage = tool_usage.get("web_search") or {}
    count = search_usage.get("num_requests")
    usage.pop("server_tool_use", None)
    if count is not None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Responses web search num_requests must be a nonnegative integer.")
        usage["server_tool_use"] = {"web_search_requests": count, "web_fetch_requests": 0}


def _messages_tools(response_tools: Any) -> list[dict[str, Any]] | None:
    if response_tools is None:
        return None
    if not isinstance(response_tools, list):
        raise ValueError("Responses tools must be a list.")

    tools: list[dict[str, Any]] = []
    for tool in response_tools:
        if isinstance(tool, dict) and tool.get("type") in {"web_search"}:
            tools.append(_web_search_tool(tool))
            continue
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Only Responses function tools can map to Messages tools.")
        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str):
            raise ValueError("Responses function tools must contain a string name.")
        if not isinstance(parameters, dict):
            raise ValueError("Responses function tools must contain an object parameters schema.")

        messages_tool: dict[str, Any] = {"input_schema": deepcopy(parameters), "name": name}
        description = tool.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("Responses function tool descriptions must be strings.")
            messages_tool["description"] = description
        strict = tool.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise ValueError("Responses function tool strict values must be booleans.")
            if strict:
                messages_tool["strict"] = True

        unsupported = {
            key
            for key, value in tool.items()
            if key not in {"description", "name", "parameters", "strict", "type"}
            and value not in (None, False)
        }
        if unsupported:
            raise ValueError(
                f"Unsupported Responses function tool fields for Messages: {sorted(unsupported)}"
            )
        tools.append(messages_tool)
    return tools


def _messages_tool_choice(responses_request: dict[str, Any]) -> dict[str, Any] | None:
    response_choice = responses_request.get("tool_choice")
    if response_choice is None:
        messages_choice = None
    elif (
        response_choice == "required"
        and len(responses_request.get("tools", [])) == 1
        and responses_request["tools"][0].get("type") == "web_search"
    ):
        messages_choice = {"type": "tool", "name": "web_search"}
    elif isinstance(response_choice, str):
        try:
            choice_type = {"auto": "auto", "none": "none", "required": "any"}[response_choice]
        except KeyError as exc:
            raise ValueError(f"Unsupported Responses tool choice: {response_choice!r}.") from exc
        messages_choice = {"type": choice_type}
    elif (
        isinstance(response_choice, dict)
        and response_choice.get("type") == "allowed_tools"
        and response_choice.get("mode") == "required"
        and response_choice.get("tools") == [{"type": "web_search"}]
    ):
        if any(
            tool.get("type") == "function" and tool.get("name") == "web_search"
            for tool in responses_request.get("tools", [])
        ):
            raise ValueError(
                "Messages cannot select hosted search alongside a function named web_search."
            )
        messages_choice = {"type": "tool", "name": "web_search"}
    elif isinstance(response_choice, dict) and response_choice.get("type") == "function":
        name = response_choice.get("name")
        if not isinstance(name, str):
            raise ValueError("A Responses function tool choice must contain a string name.")
        messages_choice = {"name": name, "type": "tool"}
    else:
        raise ValueError("Only Responses function tool choices can map to Messages.")

    parallel_tool_calls = responses_request.get("parallel_tool_calls")
    if parallel_tool_calls is not None:
        if not isinstance(parallel_tool_calls, bool):
            raise ValueError("Responses parallel_tool_calls must be a boolean.")
        if messages_choice is None:
            messages_choice = {"type": "auto"}
        if messages_choice["type"] != "none":
            messages_choice["disable_parallel_tool_use"] = not parallel_tool_calls
    return messages_choice


def _matched_request_body(
    messages_request: dict[str, Any], responses_request: dict[str, Any]
) -> dict[str, Any]:
    request_model = responses_request.get("model")
    if not isinstance(request_model, str):
        raise ValueError("Responses request must contain a string model.")
    stream = responses_request.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise ValueError("Responses request stream must be a boolean when provided.")

    matched_request = deepcopy(messages_request)
    matched_request["messages"] = _messages_from_response_input(responses_request.get("input"))
    matched_request["model"] = request_model
    if stream is None:
        matched_request.pop("stream", None)
    else:
        matched_request["stream"] = stream
    if responses_request.get("max_output_tokens") is not None:
        matched_request["max_tokens"] = responses_request["max_output_tokens"]

    matched_tools = _messages_tools(responses_request.get("tools"))
    if matched_tools is None:
        matched_request.pop("tools", None)
    else:
        matched_request["tools"] = matched_tools
    if matched_tools and responses_request.get("max_tool_calls") is not None:
        for tool in matched_tools:
            if tool.get("type") == "web_search_20250305":
                tool["max_uses"] = responses_request["max_tool_calls"]
    matched_tool_choice = _messages_tool_choice(responses_request)
    if matched_tool_choice is None:
        matched_request.pop("tool_choice", None)
    else:
        matched_request["tool_choice"] = matched_tool_choice

    response_input = responses_request.get("input")
    has_thinking_history = isinstance(response_input, list) and any(
        isinstance(item, dict) and item.get("type") == "reasoning" for item in response_input
    )
    reasoning = responses_request.get("reasoning")
    reasoning_effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    reasoning_summary = reasoning.get("summary") if isinstance(reasoning, dict) else None
    if reasoning_effort == "none":
        matched_request["thinking"] = {"type": "disabled"}
        matched_request.pop("output_config", None)
    elif (
        reasoning_effort is not None
        or reasoning_summary is not None
        or has_thinking_history
        or (
            isinstance(matched_request.get("thinking"), dict)
            and matched_request["thinking"].get("type") == "adaptive"
        )
    ):
        effort = "high" if reasoning_effort is None else reasoning_effort
        if effort == "minimal":
            effort = "low"
        matched_request["thinking"] = {"type": "adaptive", "display": "summarized"}
        existing_output_config = matched_request.get("output_config")
        output_config = (
            {"format": existing_output_config["format"]}
            if isinstance(existing_output_config, dict) and "format" in existing_output_config
            else {}
        )
        output_config["effort"] = effort
        matched_request["output_config"] = output_config
    return matched_request


def _map_input_usage(
    response_usage: dict[str, Any], message_usage: dict[str, Any], cache_ttl: str
) -> None:
    input_details = response_usage.get("input_tokens_details") or {}
    if not isinstance(input_details, dict):
        raise ValueError("Responses input token details must be an object.")

    input_tokens = response_usage["input_tokens"]
    cache_creation_tokens = input_details.get("cache_write_tokens", 0)
    cache_read_tokens = input_details.get("cached_tokens", 0)
    if not all(
        isinstance(value, int)
        for value in (input_tokens, cache_creation_tokens, cache_read_tokens)
    ):
        raise ValueError("Responses input token counts must be integers.")

    message_input_tokens = input_tokens - cache_creation_tokens - cache_read_tokens
    if message_input_tokens < 0:
        raise ValueError("Responses cache tokens exceed total input tokens.")
    message_usage["input_tokens"] = message_input_tokens
    message_usage["cache_creation_input_tokens"] = cache_creation_tokens
    message_usage["cache_read_input_tokens"] = cache_read_tokens
    if isinstance(message_usage.get("cache_creation"), dict):
        # Keep the lifetime breakdown consistent with the canonical cache-write total.
        message_usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": cache_creation_tokens if cache_ttl == "5m" else 0,
            "ephemeral_1h_input_tokens": cache_creation_tokens if cache_ttl == "1h" else 0,
        }


def _map_usage(
    response_usage: dict[str, Any], message_usage: dict[str, Any], cache_ttl: str
) -> None:
    _map_input_usage(response_usage, message_usage, cache_ttl)

    output_tokens = response_usage["output_tokens"]
    total_tokens = response_usage["total_tokens"]
    if not isinstance(output_tokens, int) or not isinstance(total_tokens, int):
        raise ValueError("Responses output and total token counts must be integers.")
    if response_usage["input_tokens"] + output_tokens != total_tokens:
        raise ValueError("Responses total tokens do not equal input plus output tokens.")
    message_usage["output_tokens"] = output_tokens

    output_details = response_usage.get("output_tokens_details") or {}
    if not isinstance(output_details, dict):
        raise ValueError("Responses output token details must be an object.")
    if "reasoning_tokens" in output_details:
        message_output_details = message_usage.setdefault("output_tokens_details", {})
        if not isinstance(message_output_details, dict):
            raise ValueError("Messages output token details must be an object.")
        message_output_details["thinking_tokens"] = output_details["reasoning_tokens"]


def _response_body(cassette: dict[str, Any]) -> dict[str, Any]:
    try:
        response_body = cassette["response"]["body"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain response.body.") from exc
    if not isinstance(response_body, dict):
        raise ValueError("Cassette response.body must be an object.")
    return response_body


def _require_non_streaming(cassette: dict[str, Any]) -> None:
    stream = _request_body(cassette).get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise ValueError("Cassette request stream must be a boolean when provided.")
    if stream:
        raise ValueError("Non-streaming response matching does not accept stream=true.")


def _response_content(response: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses response output must be a list.")

    blocks: list[dict[str, Any]] = []
    has_tool_calls = False
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Responses output items must be objects.")
        item_type = item.get("type")
        if item_type == "reasoning":
            blocks.append(_reasoning_content(item))
            continue
        if item_type == "web_search_call":
            blocks.extend(_search_content(item))
            continue
        if item_type == "function_call":
            tool_use = _function_call_content(item)
            tool_use["caller"] = {"type": "direct"}
            blocks.append(tool_use)
            has_tool_calls = True
            continue
        if item_type != "message" or item.get("role") != "assistant":
            raise ValueError(f"Responses output type {item_type!r} cannot map to Messages.")

        content = item.get("content")
        if not isinstance(content, list):
            raise ValueError("Responses output message content must be a list.")
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                raise ValueError("Only Responses output text can map to Messages.")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Responses output text must contain a string text field.")
            if part.get("logprobs") not in (None, []):
                raise ValueError("Responses output logprobs cannot map to Messages.")
            blocks.extend(_cited_text_content(part))
    return blocks, has_tool_calls


def _stop_reason(
    response: dict[str, Any], *, has_tool_calls: bool, has_search: bool = False
) -> str:
    status = response.get("status")
    if status == "completed":
        return "tool_use" if has_tool_calls else "end_turn"
    if status != "incomplete":
        raise ValueError("Responses response status must be completed or incomplete.")

    incomplete_details = response.get("incomplete_details")
    if incomplete_details is None and has_search:
        return "pause_turn"
    if not isinstance(incomplete_details, dict):
        raise ValueError("Incomplete Responses responses must contain incomplete_details.")
    if incomplete_details.get("reason") == "max_output_tokens":
        return "max_tokens"
    return "end_turn"


def match_messages_response(
    messages_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    """Rewrite one Messages JSON response from a canonical Responses body."""
    _require_non_streaming(messages_cassette)
    _require_non_streaming(responses_cassette)

    messages_request = _request_body(messages_cassette)
    responses_request = _request_body(responses_cassette)
    matched_request = _matched_request_body(messages_request, responses_request)

    message = _response_body(messages_cassette)
    response = _response_body(responses_cassette)
    message_usage = message.get("usage")
    response_usage = response.get("usage")
    if not isinstance(message_usage, dict):
        raise ValueError("Messages response must contain usage.")
    if not isinstance(response_usage, dict):
        raise ValueError("Responses response must contain usage for response matching.")

    response_id = response.get("id")
    model = response.get("model")
    if not isinstance(response_id, str) or not isinstance(model, str):
        raise ValueError("Responses response must contain string id and model fields.")

    has_search = (
        any(tool.get("type") == "web_search" for tool in responses_request.get("tools", []))
        or any(item.get("type") == "web_search_call" for item in response.get("output", []))
        or any(
            block.get("type") in {"server_tool_use", "web_search_tool_result"}
            for block in message.get("content", [])
        )
    )
    content, has_tool_calls = _response_content(response)
    message["id"] = response_id
    message["model"] = model
    message["role"] = "assistant"
    message["type"] = "message"
    message["content"] = content
    message["stop_reason"] = _stop_reason(
        response,
        has_tool_calls=has_tool_calls,
        has_search=any(
            tool.get("type") == "web_search" for tool in responses_request.get("tools", [])
        )
        or any(item.get("type") == "web_search_call" for item in response.get("output", [])),
    )
    message["stop_sequence"] = None
    _map_usage(
        response_usage, message_usage, matched_request.get("cache_control", {}).get("ttl", "5m")
    )
    if has_search:
        _map_search_usage(response, message_usage)
    message_usage["service_tier"] = "standard"

    messages_cassette["request"]["body"] = matched_request


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an Anthropic Messages cassette for non-streaming "
            "protocol parity, including reduced hosted-search projection. The Messages cassette is updated in place "
            "unless --output is provided."
        )
    )
    parser.add_argument("messages_cassette", type=Path)
    parser.add_argument("responses_cassette", type=Path)
    parser.add_argument("--output", type=Path, help="Write to a different YAML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    messages_cassette = read_yaml(str(args.messages_cassette))
    responses_cassette = read_yaml(str(args.responses_cassette))
    if not isinstance(messages_cassette, dict) or not isinstance(responses_cassette, dict):
        raise ValueError("Cassette roots must be YAML mappings.")

    match_messages_response(messages_cassette, responses_cassette)
    output_path = args.output or args.messages_cassette
    dump_yaml(messages_cassette, str(output_path), sort_keys=True)
    print(f"Matched Anthropic Messages response and wrote {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
