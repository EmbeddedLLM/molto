from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from openai import AsyncOpenAI

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.responses import (
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamEvent,
)
from tests.support.io import json_loads, read_yaml

STREAM_PARAMS = dict(argvalues=[True, False], ids=["stream", "non-stream"])


CASSETTE_DIR = Path("tests/cassettes")

RECORDED_HEADERS = {
    "accept",
    "accept-encoding",
    "authorization",
    "cache-control",
    "connection",
    "content-length",
    "content-type",
    "user-agent",
    "x-run-id",
}

# These headers affect how OpenAI interprets the request and are stable across
# the SDK and Molto clients. Other recorded headers are useful diagnostics but
# include credentials, SDK versions, or transport-computed values.
MATCHED_REQUEST_HEADERS = {
    "content-type",
}

EXCLUDED_PLAYBACK_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
}

SECRET_REQUEST_KEYS = {
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "id_token",
}


def sanitize_test_name(name: str) -> str:
    """Return the stable filesystem-safe name for a pytest node name."""
    safe_name = name.replace("[", "-").replace("]", "")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", safe_name)
    safe_name = re.sub(r"-+", "-", safe_name).strip("-")
    return safe_name or "test"


def openai_endpoint_from_path(path: str) -> str:
    """Map an OpenAI request path to the cassette endpoint directory.

    Example:
        openai_endpoint_from_path("/v1/chat/completions")
        # "chat_completions"
    """
    clean_path = path.strip("/")
    if clean_path.startswith("v1/"):
        clean_path = clean_path[len("v1/") :]
    if clean_path.startswith("chat/completions"):
        return "chat_completions"
    return (clean_path.split("/", 1)[0] or "unknown").replace("-", "_")


def mask_authorization(value: str) -> str:
    """Mask an Authorization header value before writing a cassette."""
    if not value:
        return value
    if value.lower().startswith("bearer "):
        return "Bearer ***"
    return "***"


def sanitize_headers(headers: httpx.Headers) -> dict[str, str]:
    """Filter headers to deterministic, shareable cassette metadata.

    Example:
        sanitize_headers(httpx.Headers({"authorization": "Bearer sk-test", "x-no": "1"}))
        # {"authorization": "Bearer ***"}
    """
    filtered: dict[str, str] = {}
    for key, value in sorted(headers.items(), key=lambda item: item[0].lower()):
        lower_key = key.lower()
        if lower_key not in RECORDED_HEADERS:
            continue
        filtered[key] = mask_authorization(value) if lower_key == "authorization" else value
    return filtered


def sanitize_request_body(body: Any) -> Any:
    """Recursively scrub secret-looking request fields before recording."""
    if isinstance(body, dict):
        scrubbed: dict[str, Any] = {}
        for key, value in body.items():
            key_lower = str(key).lower()
            if key_lower in SECRET_REQUEST_KEYS or key_lower.endswith("_api_key"):
                scrubbed[key] = "***"
            else:
                scrubbed[key] = sanitize_request_body(value)
        return scrubbed
    if isinstance(body, list):
        return [sanitize_request_body(item) for item in body]
    return body


def _request_snapshot(request: httpx.Request, body: Any) -> dict[str, Any]:
    """Return the sanitized request fields persisted in and matched against cassettes."""
    return {
        "method": request.method,
        "path": _request_path(request),
        "query_params": _request_query_params(request),
        "headers": sanitize_headers(request.headers),
        "body": sanitize_request_body(body),
    }


def _matched_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return stable, semantically relevant request headers for replay matching."""
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in MATCHED_REQUEST_HEADERS
    }


def _playback_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove transport-managed headers before replaying a response."""
    return {k: v for k, v in headers.items() if k.lower() not in EXCLUDED_PLAYBACK_HEADERS}


def _parse_request_body(body: bytes) -> Any:
    """Parse an HTTP JSON request body, returning an empty dict for no body."""
    if not body:
        return {}
    return json_loads(body)


def _decode_response_body(body: bytes, headers: httpx.Headers) -> Any:
    """Decode a recorded response body according to its content type."""
    content_type = headers.get("content-type", "")
    if "application/json" not in content_type:
        return body.decode("utf-8")
    return json_loads(body)


def _body_to_sse_lines(body: bytes) -> list[str]:
    """Split a streamed response body into newline-preserving SSE lines.

    Example:
        _body_to_sse_lines(b'data: {"type":"response.created"}\n\n')
        # ['data: {"type":"response.created"}\n', '\n']
    """
    text = body.decode("utf-8")
    return [f"{line}\n" for line in text.splitlines()]


def _request_path(request: httpx.Request) -> str:
    """Return a request path with a leading slash."""
    path = request.url.path
    return path if path.startswith("/") else f"/{path}"


def _request_query_params(request: httpx.Request) -> dict[str, str]:
    """Return request query parameters as cassette metadata."""
    return dict(request.url.params.multi_items())


def _is_stream_request(body: Any) -> bool:
    """Return whether a parsed request body asks OpenAI for streaming."""
    return isinstance(body, dict) and body.get("stream") is True


@dataclass
class OpenAICassetteManager:
    """Resolve and persist per-test OpenAI SDK cassettes."""

    node_name: str
    node_id: str
    module_name: str
    record: bool
    root: Path = CASSETTE_DIR
    recorded_paths: set[Path] = field(default_factory=set)

    @classmethod
    def from_pytest_request(cls, request: Any, *, record: bool) -> "OpenAICassetteManager":
        """Build a manager from pytest's fixture request object."""
        node_path = getattr(request.node, "path", None)
        module_name = Path(str(node_path)).stem if node_path is not None else "unknown_module"
        root = Path(str(request.config.rootpath)) / CASSETTE_DIR
        return cls(
            node_name=request.node.name,
            node_id=request.node.nodeid,
            module_name=sanitize_test_name(module_name),
            record=record,
            root=root,
        )

    @property
    def safe_node_name(self) -> str:
        """Return the sanitized test name used in cassette filenames."""
        return sanitize_test_name(self.node_name)

    def cassette_path(self, endpoint: str, cassette_suffix: str | None = None) -> Path:
        """Return the cassette path for an endpoint and optional filename suffix."""
        if cassette_suffix is None:
            filename = f"{self.safe_node_name}.yaml"
        else:
            safe_suffix = sanitize_test_name(cassette_suffix)
            filename = f"{self.safe_node_name}-{safe_suffix}.yaml"
        return self.root / "openai" / endpoint / self.module_name / filename

    def path_for_request(self, request: httpx.Request, cassette_suffix: str | None) -> Path:
        """Return the cassette path for an outgoing OpenAI request."""
        endpoint = openai_endpoint_from_path(_request_path(request))
        return self.cassette_path(endpoint, cassette_suffix)

    def read(self, path: Path) -> dict[str, Any]:
        """Read a YAML cassette or raise a recording hint if it is missing."""
        if not path.exists():
            raise AssertionError(self.missing_cassette_message(path))
        data = read_yaml(path)
        assert isinstance(data, dict), f"Invalid cassette payload at {path}"
        return data

    def write(self, path: Path, cassette: dict[str, Any]) -> None:
        """Write a cassette with stable YAML ordering and remember it for replay."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                cassette,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        self.recorded_paths.add(path)

    def missing_cassette_message(self, path: Path) -> str:
        """Return the actionable error shown when playback has no cassette."""
        return "\n".join(
            [
                "Missing OpenAI cassette.",
                f"path: {path}",
                f"pytest node id: {self.node_id}",
                f"record with: pytest --record-cassettes {self.node_id}",
            ]
        )


class OpenAICassetteTransport(httpx.AsyncBaseTransport):
    """httpx transport that replays or records one cassette filename suffix."""

    def __init__(
        self,
        cassette_manager: OpenAICassetteManager,
        *,
        cassette_suffix: str | None = None,
        playback_only: bool = False,
    ) -> None:
        """Initialize transport state for one cassette suffix."""
        self.cassette_manager = cassette_manager
        self.cassette_suffix = cassette_suffix
        self.playback_only = playback_only
        self._real_transport = None
        if cassette_manager.record and not playback_only:
            self._real_transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Handle an SDK request by replaying or recording the configured cassette suffix."""
        body_bytes = await request.aread()
        request_body = _parse_request_body(body_bytes)

        path = self.cassette_manager.path_for_request(request, self.cassette_suffix)
        if self._should_replay(path):
            return self._response_from_cassette(request, request_body, path)
        return await self._record_response(request, body_bytes, request_body, path)

    async def aclose(self) -> None:
        """Close the underlying live transport when recording used one."""
        if self._real_transport is not None:
            await self._real_transport.aclose()

    def _should_replay(self, path: Path) -> bool:
        """Return whether this request should use an existing cassette."""
        if self.playback_only or not self.cassette_manager.record:
            return True
        return path in self.cassette_manager.recorded_paths

    def _assert_request_matches_cassette(
        self,
        request: httpx.Request,
        request_body: Any,
        path: Path,
        cassette: dict[str, Any],
    ) -> None:
        """Fail with a field-level diff when replay traffic differs from its cassette."""
        recorded = cassette.get("request")
        assert isinstance(recorded, dict), f"Invalid cassette request at {path}"

        actual = _request_snapshot(request, request_body)
        differences: list[str] = []
        for field_name in ("method", "path", "query_params", "body"):
            expected_value = recorded.get(field_name)
            actual_value = actual[field_name]
            if actual_value != expected_value:
                differences.append(
                    f"{field_name}: recorded={expected_value!r}, actual={actual_value!r}"
                )

        recorded_headers = recorded.get("headers") or {}
        assert isinstance(recorded_headers, dict), f"Invalid cassette request headers at {path}"
        expected_headers = _matched_headers(recorded_headers)
        actual_headers = _matched_headers(actual["headers"])
        if actual_headers != expected_headers:
            differences.append(
                f"headers: recorded={expected_headers!r}, actual={actual_headers!r}"
            )

        assert not differences, "\n".join(
            [
                "OpenAI cassette request mismatch.",
                f"path: {path}",
                f"pytest node id: {self.cassette_manager.node_id}",
                *differences,
            ]
        )

    def _response_from_cassette(
        self,
        request: httpx.Request,
        request_body: Any,
        path: Path,
    ) -> httpx.Response:
        """Convert a stored cassette into an httpx response for SDK parsing."""
        cassette = self.cassette_manager.read(path)
        self._assert_request_matches_cassette(request, request_body, path, cassette)
        response = cassette["response"]
        status_code = response["status_code"]
        headers = _playback_headers(response.get("headers") or {})

        if "sse" in response:
            headers.setdefault("content-type", "text/event-stream; charset=utf-8")
            return httpx.Response(
                status_code,
                content="".join(response["sse"]).encode("utf-8"),
                headers=headers,
                request=request,
            )

        body = response.get("body")
        if isinstance(body, str):
            return httpx.Response(
                status_code,
                content=body.encode("utf-8"),
                headers=headers,
                request=request,
            )
        return httpx.Response(status_code, json=body, headers=headers, request=request)

    async def _record_response(
        self,
        request: httpx.Request,
        body_bytes: bytes,
        request_body: Any,
        path: Path,
    ) -> httpx.Response:
        """Forward a request to OpenAI, write a sanitized cassette, and return it."""
        assert self._real_transport is not None
        real_request = httpx.Request(
            request.method,
            request.url,
            headers=request.headers,
            content=body_bytes,
            extensions=request.extensions,
        )
        response = await self._real_transport.handle_async_request(real_request)
        response_body = await response.aread()
        response_headers = sanitize_headers(response.headers)

        cassette: dict[str, Any] = {
            "request": _request_snapshot(request, request_body),
            "response": {
                "status_code": response.status_code,
                "headers": response_headers,
            },
        }

        if _is_stream_request(request_body) and response.status_code == 200:
            cassette["response"]["sse"] = _body_to_sse_lines(response_body)
        else:
            cassette["response"]["body"] = _decode_response_body(response_body, response.headers)

        self.cassette_manager.write(path, cassette)
        return httpx.Response(
            response.status_code,
            content=response_body,
            headers=_playback_headers(dict(response.headers)),
            request=request,
        )


def assert_usage(
    usage: Any,
    *,
    endpoint: str = "responses",
    output_tokens_exact: int | None = None,
    expect_reasoning: bool = False,
) -> None:
    """Assert that a usage object has positive token counts and consistent totals.

    Args:
        usage: Usage object from either endpoint.
        endpoint: "responses" (input_tokens/output_tokens) or "chat"
            (prompt_tokens/completion_tokens).
        output_tokens_exact: If provided, asserts the output token count
            equals this value exactly (Responses only).
        expect_reasoning: If True, also asserts the reasoning token count is
            present and positive.
    """
    if endpoint == "chat":
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        details = usage.completion_tokens_details
    else:
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        details = usage.output_tokens_details
    assert input_tokens > 0
    assert output_tokens > 0
    assert usage.total_tokens == input_tokens + output_tokens
    if output_tokens_exact is not None:
        assert output_tokens == output_tokens_exact
    if expect_reasoning:
        assert details is not None
        assert details.reasoning_tokens > 0


def assert_stream_text_events(events: list[Any]) -> None:
    """Assert the structure of streamed text events.

    For each unique assistant message ID, validates that there is exactly one
    "output_item.added" and one "output_item.done", at least one
    "output_text.delta", and at least one "output_text.done" referencing that
    item ID.

    Args:
        events: Collected stream event objects from a Molto.responses() call.
    """
    from molto.types.openai.responses import (
        OpenAIResponsesStreamOutput,
        OpenAIResponsesStreamText,
    )

    msg_item_added = [
        e
        for e in events
        if isinstance(e, OpenAIResponsesStreamOutput)
        and e.type == "response.output_item.added"
        and hasattr(e.item, "role")
        and e.item.role == "assistant"
    ]
    msg_item_done = [
        e
        for e in events
        if isinstance(e, OpenAIResponsesStreamOutput)
        and e.type == "response.output_item.done"
        and hasattr(e.item, "role")
        and e.item.role == "assistant"
    ]
    assert len(msg_item_added) >= 1
    assert len(msg_item_done) >= 1
    added_ids = [e.item.id for e in msg_item_added]
    done_ids = [e.item.id for e in msg_item_done]
    for aid in added_ids:
        assert aid is not None
        assert aid.startswith("msg_")
    assert added_ids == done_ids

    for msg_id in added_ids:
        deltas = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText)
            and e.type == "response.output_text.delta"
            and e.item_id == msg_id
        ]
        assert len(deltas) >= 1

        done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText)
            and e.type == "response.output_text.done"
            and e.item_id == msg_id
        ]
        assert len(done) == 1, f"expected 1 text_done for item {msg_id!r}, got {len(done)}"

    assert_events_before_terminal(events, "response.output_text.delta")


def assert_events_before_terminal(events: list[Any], event_type: str) -> None:
    """Assert all events of "event_type" appear before the terminal stream event."""
    indices = [i for i, event in enumerate(events) if event.type == event_type]
    assert all(i < len(events) - 1 for i in indices)


def _assert_event_type_sequence_matches(sdk_events: list[Any], molto_events: list[Any]) -> None:
    sdk_types = [e.type for e in sdk_events]
    molto_types = [e.type for e in molto_events]
    assert sdk_types == molto_types, (
        f"Event type sequence mismatch\nSDK:   {sdk_types}\nMolto: {molto_types}"
    )


def assert_output_item_order(events: list[Any]) -> None:
    """Assert every output_item.added event precedes its corresponding output_item.done."""
    added_indices = [i for i, e in enumerate(events) if e.type == "response.output_item.added"]
    done_indices = [i for i, e in enumerate(events) if e.type == "response.output_item.done"]
    assert len(added_indices) == len(done_indices)
    assert all(added_indices[i] < done_indices[i] for i in range(len(added_indices)))


def assert_stream_events_match(sdk_events: list[Any], molto_events: list[Any]) -> None:
    """Assert one-to-one equality of every parsed SDK and Molto Responses event."""
    _assert_event_type_sequence_matches(sdk_events, molto_events)

    for index, (sdk_event, molto_event) in enumerate(zip(sdk_events, molto_events, strict=True)):
        event_type = sdk_event.type
        sdk_payload = _model_payload(sdk_event)
        molto_payload = _model_payload(molto_event)
        assert molto_payload == sdk_payload, (
            f"Responses event mismatch at index {index} ({event_type})\n"
            f"SDK:   {sdk_payload!r}\n"
            f"Molto: {molto_payload!r}"
        )


def _model_payload(value: Any) -> dict[str, Any]:
    """Return populated fields while preserving explicit null versus omission."""
    return value.model_dump(mode="python", exclude_unset=True, warnings=False)


def assert_response_matches(
    sdk_response: Any, molto_response: Any, *, label: str = "Response"
) -> None:
    """Assert one-to-one equality of parsed SDK and Molto response objects."""
    sdk_payload = _model_payload(sdk_response)
    molto_payload = _model_payload(molto_response)
    assert molto_payload == sdk_payload, (
        f"{label} mismatch\nSDK:   {sdk_payload!r}\nMolto: {molto_payload!r}"
    )


def assert_chat_completion_stream_matches(sdk_chunks: list[Any], molto_chunks: list[Any]) -> None:
    """Assert one-to-one equality of every parsed SDK and Molto Chat chunk."""
    assert len(molto_chunks) == len(sdk_chunks), (
        f"Chat chunk count mismatch: SDK={len(sdk_chunks)}, Molto={len(molto_chunks)}"
    )
    for index, (sdk_chunk, molto_chunk) in enumerate(zip(sdk_chunks, molto_chunks, strict=True)):
        sdk_payload = _model_payload(sdk_chunk)
        molto_payload = _model_payload(molto_chunk)
        assert molto_payload == sdk_payload, (
            f"Chat chunk mismatch at index {index}\n"
            f"SDK:   {sdk_payload!r}\nMolto: {molto_payload!r}"
        )


def stream_chat_text(chunks: list[Any]) -> str:
    """Concatenate the assistant text streamed across Chat completion chunks."""
    return "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)


def make_openai_client(
    cassette_manager: OpenAICassetteManager,
    *,
    cassette_suffix: str | None = None,
    cassette_endpoint: str = "responses",
    api_key: str | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client backed by cassette replay/recording."""
    if not cassette_manager.record:
        cassette_manager.read(cassette_manager.cassette_path(cassette_endpoint, cassette_suffix))

    resolved_api_key = ENV_CONFIG.openai_api_key_plain if api_key is None else api_key
    if cassette_manager.record and api_key is None and not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when recording cassettes")
    if not resolved_api_key and not cassette_manager.record:
        resolved_api_key = "test-api-key"
    return AsyncOpenAI(
        http_client=httpx.AsyncClient(
            transport=OpenAICassetteTransport(
                cassette_manager,
                cassette_suffix=cassette_suffix,
            ),
        ),
        api_key=resolved_api_key,
        max_retries=0,
    )


def make_molto_client(
    cassette_manager: OpenAICassetteManager,
    *,
    cassette_suffix: str | None = None,
) -> Molto:
    """Create a Molto instance backed by playback-only cassette transport."""
    return Molto(
        http_client=httpx.AsyncClient(
            transport=OpenAICassetteTransport(
                cassette_manager,
                cassette_suffix=cassette_suffix,
                playback_only=True,
            ),
        ),
        max_retries=0,
    )


@dataclass
class ResponsesParity:
    """One-to-one SDK and Molto results for a single Responses request."""

    sdk_response: Any
    molto_response: OpenAIResponsesResponse
    sdk_events: list[Any]
    molto_events: list[OpenAIResponsesStreamEvent]


async def run_responses_parity(
    cassette_manager: OpenAICassetteManager,
    req_body: dict[str, Any],
    *,
    cassette_suffix: str | None = None,
    terminal: str = "response.completed",
    molto_request_body: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> ResponsesParity:
    """Run one Responses request through the SDK and Molto and assert parity.

    Both clients replay the same recorded cassette. When streaming, the shared
    ceremony is applied to each side: collect events, require the terminal event
    type, check output-item ordering, match the event streams one-to-one, and
    unwrap the terminal snapshot. The final snapshots are matched field-for-field.

    Args:
        cassette_manager: Per-test cassette fixture.
        req_body: SDK request body; must carry the parametrized ``stream`` flag.
        cassette_suffix: Optional cassette filename suffix for multi-recording tests.
        terminal: Expected terminal stream event type (``response.incomplete`` for
            truncated outputs).
        molto_request_body: Optional distinct Molto request body; defaults to
            ``req_body`` with the ``openai/`` model prefix.
        api_key: Optional SDK API key override.
    """
    stream = req_body.get("stream") is True

    sdk_client = make_openai_client(
        cassette_manager,
        cassette_suffix=cassette_suffix,
        api_key=api_key,
    )
    sdk_result = await sdk_client.responses.create(**req_body)
    sdk_events: list[Any] = []
    if stream:
        sdk_events = [event async for event in sdk_result]
        assert sdk_events[-1].type == terminal, (
            f"SDK terminal event {sdk_events[-1].type!r} != {terminal!r}"
        )
        assert_output_item_order(sdk_events)
        sdk_response = sdk_events[-1].response
    else:
        sdk_response = sdk_result

    molto = make_molto_client(cassette_manager, cassette_suffix=cassette_suffix)
    molto_body = (
        molto_request_body if molto_request_body is not None else prepend_model_prefix(req_body)
    )
    molto_result = await molto.responses(OpenAIResponsesRequest(**molto_body))
    molto_events: list[OpenAIResponsesStreamEvent] = []
    if stream:
        molto_events = [event async for event in molto_result]
        assert isinstance(molto_events[-1], OpenAIResponsesStream)
        assert molto_events[-1].type == terminal, (
            f"Molto terminal event {molto_events[-1].type!r} != {terminal!r}"
        )
        assert_output_item_order(molto_events)
        assert_stream_events_match(sdk_events, molto_events)
        molto_response = molto_events[-1].response
    else:
        molto_response = molto_result

    assert isinstance(molto_response, OpenAIResponsesResponse)
    assert_response_matches(sdk_response, molto_response, label="Responses object")
    return ResponsesParity(
        sdk_response=sdk_response,
        molto_response=molto_response,
        sdk_events=sdk_events,
        molto_events=molto_events,
    )


@dataclass
class ChatParity:
    """One-to-one SDK and Molto results for a single Chat Completions request."""

    sdk_response: Any
    molto_response: Any
    sdk_chunks: list[Any]
    molto_chunks: list[Any]


async def run_chat_parity(
    cassette_manager: OpenAICassetteManager,
    req_body: dict[str, Any],
    *,
    cassette_suffix: str | None = None,
    api_key: str | None = None,
) -> ChatParity:
    """Run one Chat Completions request through the SDK and Molto and assert parity.

    Both clients replay the same recorded cassette. When streaming, the chunk
    streams are matched one-to-one; otherwise the final responses are matched
    field-for-field.
    """
    stream = req_body.get("stream") is True

    sdk_client = make_openai_client(
        cassette_manager,
        cassette_suffix=cassette_suffix,
        cassette_endpoint="chat_completions",
        api_key=api_key,
    )
    sdk_result = await sdk_client.chat.completions.create(**req_body)

    molto = make_molto_client(cassette_manager, cassette_suffix=cassette_suffix)
    molto_result = await molto.chat_completions(
        OpenAIChatCompletionRequest(**prepend_model_prefix(req_body))
    )

    sdk_chunks: list[Any] = []
    molto_chunks: list[Any] = []
    if stream:
        sdk_chunks = [chunk async for chunk in sdk_result]
        molto_chunks = [chunk async for chunk in molto_result]
        assert_chat_completion_stream_matches(sdk_chunks, molto_chunks)
        return ChatParity(
            sdk_response=None,
            molto_response=None,
            sdk_chunks=sdk_chunks,
            molto_chunks=molto_chunks,
        )

    assert_response_matches(sdk_result, molto_result, label="Chat response")
    return ChatParity(
        sdk_response=sdk_result,
        molto_response=molto_result,
        sdk_chunks=sdk_chunks,
        molto_chunks=molto_chunks,
    )


def prepend_model_prefix(body: dict[str, Any], prefix: str = "openai/") -> dict[str, Any]:
    """Return a shallow copy of "body" with the model field prefixed."""
    return {**body, "model": prefix + body["model"]}
