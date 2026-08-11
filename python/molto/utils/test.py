from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from openai import AsyncOpenAI

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    validate_openai_responses_stream_event,
)
from molto.utils.io import json_loads, load_sse, read_yaml

STREAM_PARAMS = dict(argvalues=[True, False], ids=["stream", "non-stream"])

_RECONSTRUCTIBLE_RESPONSE_FIELDS = {
    "id",
    "model",
    "object",
    "output",
    "status",
    "usage",
}

# Responses reports the final service tier only at completion, so matching it on
# every Chat chunk would require buffering the stream.
_RECONSTRUCTIBLE_CHAT_COMPLETIONS_FIELDS = {
    "choices",
    "created",
    "id",
    "model",
    "object",
    "usage",
}


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
    """Validate a Responses SSE event and retain its reconstructible fields."""
    request = OpenAIResponsesCreateRequest.model_validate(body)
    event = validate_openai_responses_stream_event(load_sse(sse)).model_dump(
        mode="json",
        exclude_unset=True,
    )
    event.pop("item_id", None)

    item = event.get("item")
    if isinstance(item, dict):
        item.pop("id", None)

    response = event.get("response")
    if isinstance(response, dict):
        response = {
            key: value
            for key, value in response.items()
            if key in request.model_fields_set or key in _RECONSTRUCTIBLE_RESPONSE_FIELDS
        }
        for output_item in response.get("output") or []:
            output_item.pop("id", None)
        event["response"] = response

    return event


def stream_reference_cassette_suffix(stream: bool) -> str | None:
    """Return the extra cassette suffix for Molto/internal streaming references.

    Example:
        stream_reference_cassette_suffix(False)
        # "sse"
    """
    return None if stream else "sse"


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
    """Return the stable filesystem slug for a pytest node name.

    Example:
        sanitize_test_name("test_reasoning[stream-reasoning_no_summary]")
        # "test_reasoning-stream-reasoning_no_summary"
    """
    slug = name.replace("[", "-").replace("]", "")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "test"


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
    """Mask an Authorization header value before writing a cassette.

    Example:
        mask_authorization("Bearer sk-test")
        # "Bearer ***"
    """
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
    """Recursively scrub secret-looking request fields before recording.

    Example:
        sanitize_request_body({"model": "gpt-4.1", "api_key": "sk-test"})
        # {"model": "gpt-4.1", "api_key": "***"}
    """
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
    """Remove transport-managed headers before replaying a response.

    Example:
        _playback_headers({"content-type": "application/json", "content-length": "10"})
        # {"content-type": "application/json"}
    """
    return {k: v for k, v in headers.items() if k.lower() not in EXCLUDED_PLAYBACK_HEADERS}


def _parse_request_body(body: bytes) -> Any:
    """Parse an HTTP JSON request body, returning an empty dict for no body.

    Example:
        _parse_request_body(b'{"stream": true}')
        # {"stream": True}
    """
    if not body:
        return {}
    return json_loads(body)


def _decode_response_body(body: bytes, headers: httpx.Headers) -> Any:
    """Decode a recorded response body according to its content type.

    Example:
        _decode_response_body(b'{"ok": true}', httpx.Headers({"content-type": "application/json"}))
        # {"ok": True}
    """
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
    """Return a request path with a leading slash.

    Example:
        _request_path(httpx.Request("POST", "https://api.openai.com/v1/responses"))
        # "/v1/responses"
    """
    path = request.url.path
    return path if path.startswith("/") else f"/{path}"


def _request_query_params(request: httpx.Request) -> dict[str, str]:
    """Return request query parameters as cassette metadata.

    Example:
        _request_query_params(httpx.Request("GET", "https://example.test/a?limit=1"))
        # {"limit": "1"}
    """
    return dict(request.url.params.multi_items())


def _is_stream_request(body: Any) -> bool:
    """Return whether a parsed request body asks OpenAI for streaming.

    Example:
        _is_stream_request({"stream": True})
        # True
    """
    return isinstance(body, dict) and body.get("stream") is True


@dataclass
class OpenAICassetteManager:
    """Resolve and persist per-test OpenAI SDK cassettes.

    Example:
        manager = OpenAICassetteManager("test_demo[stream]", "tests/test_demo.py::test_demo[stream]", "test_demo", False)
        manager.cassette_path("responses")
        # Path("tests/cassettes/openai/responses/test_demo/test_demo-stream.yaml")
    """

    node_name: str
    node_id: str
    module_name: str
    record: bool
    root: Path = CASSETTE_DIR
    recorded_paths: set[Path] = field(default_factory=set)

    @classmethod
    def from_pytest_request(cls, request: Any, *, record: bool) -> "OpenAICassetteManager":
        """Build a manager from pytest's fixture request object.

        Example:
            # For node name "test_demo[stream]" in tests/test_demo.py:
            OpenAICassetteManager.from_pytest_request(request, record=False).module_name
            # "test_demo"
        """
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
    def node_slug(self) -> str:
        """Return the sanitized test name used in cassette filenames.

        Example:
            OpenAICassetteManager("test_a[one two]", "id", "mod", False).node_slug
            # "test_a-one-two"
        """
        return sanitize_test_name(self.node_name)

    def cassette_path(self, endpoint: str, cassette_suffix: str | None = None) -> Path:
        """Return the cassette path for an endpoint and optional filename suffix.

        Example:
            manager = OpenAICassetteManager("test_a[stream]", "id", "test_a", False)
            manager.cassette_path("responses")
            # Path("tests/cassettes/openai/responses/test_a/test_a-stream.yaml")
        """
        if cassette_suffix is None:
            filename = f"{self.node_slug}.yaml"
        else:
            suffix_slug = sanitize_test_name(cassette_suffix)
            filename = f"{self.node_slug}-{suffix_slug}.yaml"
        return self.root / "openai" / endpoint / self.module_name / filename

    def path_for_request(self, request: httpx.Request, cassette_suffix: str | None) -> Path:
        """Return the cassette path for an outgoing OpenAI request.

        Example:
            manager = OpenAICassetteManager("test_a", "id", "test_a", False)
            request = httpx.Request("POST", "https://api.openai.com/v1/responses")
            manager.path_for_request(request, None)
            # Path("tests/cassettes/openai/responses/test_a/test_a.yaml")
        """
        endpoint = openai_endpoint_from_path(_request_path(request))
        return self.cassette_path(endpoint, cassette_suffix)

    def read(self, path: Path) -> dict[str, Any]:
        """Read a YAML cassette or raise a recording hint if it is missing.

        Example:
            # manager.read(Path("missing.yaml")) raises an AssertionError containing:
            # "pytest --record-cassettes tests/test_demo.py::test_demo"
        """
        if not path.exists():
            raise AssertionError(self.missing_cassette_message(path))
        data = read_yaml(path)
        assert isinstance(data, dict), f"Invalid cassette payload at {path}"
        return data

    def write(self, path: Path, cassette: dict[str, Any]) -> None:
        """Write a cassette with stable YAML ordering and remember it for replay.

        Example:
            manager.write(Path("tests/cassettes/openai/responses/test_a/test_a-stream.yaml"), {"request": {}, "response": {}})
            # path is written and added to manager.recorded_paths
        """
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
        """Return the actionable error shown when playback has no cassette.

        Example:
            manager = OpenAICassetteManager("test_a", "tests/test_a.py::test_a", "test_a", False)
            "pytest --record-cassettes tests/test_a.py::test_a" in manager.missing_cassette_message(Path("x.yaml"))
            # True
        """
        return "\n".join(
            [
                "Missing OpenAI cassette.",
                f"path: {path}",
                f"pytest node id: {self.node_id}",
                f"record with: pytest --record-cassettes {self.node_id}",
            ]
        )


class OpenAICassetteTransport(httpx.AsyncBaseTransport):
    """httpx transport that replays or records one cassette filename suffix.

    Example:
        manager = OpenAICassetteManager("test_a", "id", "test_a", False)
        transport = OpenAICassetteTransport(manager, cassette_suffix="sse", playback_only=True)
        transport.cassette_suffix
        # "sse"
    """

    def __init__(
        self,
        cassette_manager: OpenAICassetteManager,
        *,
        cassette_suffix: str | None = None,
        playback_only: bool = False,
        expected_request_body: dict[str, Any] | None = None,
        assert_attrs: list[str] | None = None,
    ) -> None:
        """Initialize transport state for one cassette suffix.

        Example:
            manager = OpenAICassetteManager("test_a", "id", "test_a", False)
            OpenAICassetteTransport(manager, cassette_suffix=None).cassette_suffix
            # None
        """
        self.cassette_manager = cassette_manager
        self.cassette_suffix = cassette_suffix
        self.playback_only = playback_only
        self.expected_request_body = expected_request_body
        self.assert_attrs = assert_attrs or []
        self._real_transport = None
        if cassette_manager.record and not playback_only:
            self._real_transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Handle an SDK request by replaying or recording the configured cassette suffix.

        Example:
            # In playback mode, POST /v1/responses with cassette_suffix "sse" reads:
            # tests/cassettes/openai/responses/<module>/<test>-sse.yaml
        """
        body_bytes = await request.aread()
        request_body = _parse_request_body(body_bytes)
        self._assert_expected_request(request_body)

        path = self.cassette_manager.path_for_request(request, self.cassette_suffix)
        if self._should_replay(path):
            return self._response_from_cassette(request, request_body, path)
        return await self._record_response(request, body_bytes, request_body, path)

    async def aclose(self) -> None:
        """Close the underlying live transport when recording used one.

        Example:
            # After await transport.aclose(), any owned AsyncHTTPTransport is closed.
        """
        if self._real_transport is not None:
            await self._real_transport.aclose()

    def _assert_expected_request(self, request_body: Any) -> None:
        """Assert selected JSON request fields match the expected request.

        Example:
            transport.expected_request_body = {"model": "gpt-4.1"}
            transport.assert_attrs = ["model"]
            transport._assert_expected_request({"model": "gpt-4.1"})
            # None
        """
        if not self.expected_request_body:
            return

        attrs = list(self.assert_attrs)
        if "stream" in self.expected_request_body and "stream" not in attrs:
            attrs.append("stream")
        if not attrs:
            return

        assert isinstance(request_body, dict), "Expected JSON object request body"
        for attr in attrs:
            assert request_body.get(attr) == self.expected_request_body[attr], (
                f"Expected {attr} to be {self.expected_request_body[attr]!r}, "
                f"got {request_body.get(attr)!r}"
            )

    def _should_replay(self, path: Path) -> bool:
        """Return whether this request should use an existing cassette.

        Example:
            transport = OpenAICassetteTransport(manager, cassette_suffix="sse", playback_only=True)
            transport._should_replay(Path("missing.yaml"))
            # True
        """
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
        """Convert a stored cassette into an httpx response for SDK parsing.

        Example:
            # A cassette response with an "sse" list replays as text/event-stream.
        """
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
        """Forward a request to OpenAI, write a sanitized cassette, and return it.

        Example:
            # A request body {"stream": True} records response["sse"].
            # A request body {"stream": False} records response["body"].
        """
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
    usage: Any, *, output_tokens_exact: int | None = None, expect_reasoning: bool = False
) -> None:
    """Assert that a usage object has positive token counts and consistent totals.

    Args:
        usage: Object with "input_tokens", "output_tokens", and
            "total_tokens" attributes (e.g. an OpenAI CompletionUsage).
        output_tokens_exact: If provided, asserts the output token count
            equals this value exactly.
        expect_reasoning: If True, also asserts that
            "output_tokens_details.reasoning_tokens" is present and positive.
    """
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    if output_tokens_exact is not None:
        assert usage.output_tokens == output_tokens_exact
    if expect_reasoning:
        assert usage.output_tokens_details is not None
        assert usage.output_tokens_details.reasoning_tokens > 0


def assert_stream_text_events(events: list[Any]) -> None:
    """Assert the structure of streamed text events.

    For each unique assistant message ID, validates that there is exactly one
    "output_item.added" and one "output_item.done", at least one
    "output_text.delta", and at least one "output_text.done" referencing that
    item ID.

    Args:
        events: Collected stream event objects from a Molto.chat() call.
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


# TODO: Add a code interpreter stream parity test once a cassette is available.
_SDK_TYPE_MAP: dict[str, str] = {
    "response.reasoning_summary_text.delta": "response.reasoning.delta",
    "response.reasoning_summary_text.done": "response.reasoning.done",
}

# field -> event types that carry it (used for 1:1 payload comparison)
_FIELD_CHECKS: dict[str, frozenset[str]] = {
    "item_id": frozenset(
        {
            "response.output_text.delta",
            "response.output_text.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.reasoning.delta",
            "response.reasoning.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call.completed",
            "response.code_interpreter_call_code.delta",
            "response.code_interpreter_call_code.done",
        }
    ),
    "delta": frozenset(
        {
            "response.output_text.delta",
            "response.function_call_arguments.delta",
            "response.reasoning.delta",
        }
    ),
    "text": frozenset({"response.output_text.done", "response.reasoning.done"}),
    "arguments": frozenset({"response.function_call_arguments.done"}),
}


def _normalize_sdk_event_type(event_type: str) -> str:
    """Return the Molto/OpenResponses event type matching an SDK event type."""
    return _SDK_TYPE_MAP.get(event_type, event_type)


def _normalized_event_types(events: list[Any]) -> list[str]:
    """Return event types after applying SDK-to-Molto naming compatibility."""
    return [_normalize_sdk_event_type(e.type) for e in events]


def _assert_event_type_sequence_matches(sdk_events: list[Any], molto_events: list[Any]) -> None:
    sdk_types = _normalized_event_types(sdk_events)
    molto_types = [e.type for e in molto_events]
    assert sdk_types == molto_types, (
        f"Event type sequence mismatch\nSDK:   {sdk_types}\nMolto: {molto_types}"
    )


def _assert_matching_scalar_fields(event_type: str, sdk_event: Any, molto_event: Any) -> None:
    for field_name, event_types in _FIELD_CHECKS.items():
        if event_type in event_types:
            assert getattr(sdk_event, field_name, None) == getattr(
                molto_event, field_name, None
            ), f"{field_name} mismatch at {event_type}"


def _assert_matching_output_item(event_type: str, sdk_event: Any, molto_event: Any) -> None:
    if event_type not in {"response.output_item.added", "response.output_item.done"}:
        return

    sdk_item = getattr(sdk_event, "item", None)
    molto_item = getattr(molto_event, "item", None)
    if sdk_item is None or molto_item is None:
        return

    assert sdk_item.id == molto_item.id, f"item.id mismatch at {event_type}"
    assert sdk_item.type == molto_item.type, f"item.type mismatch at {event_type}"


def _assert_matching_terminal_response_id(
    event_type: str, sdk_event: Any, molto_event: Any
) -> None:
    if event_type not in {"response.completed", "response.incomplete"}:
        return

    # Only compare response.id for terminal events: Molto uses a provisional UUID
    # for response.created/in_progress that will not match the cassette ID.
    sdk_response_id = getattr(getattr(sdk_event, "response", None), "id", None)
    if sdk_response_id is not None:
        molto_response_id = getattr(getattr(molto_event, "response", None), "id", None)
        assert sdk_response_id == molto_response_id, f"response.id mismatch at {event_type}"


def assert_output_item_order(events: list[Any]) -> None:
    """Assert every output_item.added event precedes its corresponding output_item.done."""
    added_indices = [i for i, e in enumerate(events) if e.type == "response.output_item.added"]
    done_indices = [i for i, e in enumerate(events) if e.type == "response.output_item.done"]
    assert len(added_indices) == len(done_indices)
    assert all(added_indices[i] < done_indices[i] for i in range(len(added_indices)))


def assert_stream_events_match(sdk_events: list[Any], molto_events: list[Any]) -> None:
    """Assert SDK and Molto stream events have the same order and key payloads."""
    _assert_event_type_sequence_matches(sdk_events, molto_events)

    for sdk_event, molto_event in zip(sdk_events, molto_events, strict=True):
        event_type = _normalize_sdk_event_type(sdk_event.type)
        _assert_matching_scalar_fields(event_type, sdk_event, molto_event)
        _assert_matching_output_item(event_type, sdk_event, molto_event)
        _assert_matching_terminal_response_id(event_type, sdk_event, molto_event)


def make_openai_client(
    cassette_manager: OpenAICassetteManager,
    *,
    cassette_suffix: str | None = None,
    cassette_endpoint: str = "responses",
    expected_request_body: dict[str, Any] | None = None,
    assert_attrs: list[str] | None = None,
    api_key: str | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client backed by cassette replay/recording.

    Example:
        client = make_openai_client(manager)
        client.max_retries
        # 0
    """
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
                expected_request_body=expected_request_body,
                assert_attrs=assert_attrs,
            ),
        ),
        api_key=resolved_api_key,
        max_retries=0,
    )


def make_molto_client(
    cassette_manager: OpenAICassetteManager,
    *,
    cassette_suffix: str | None = None,
    expected_request_body: dict[str, Any] | None = None,
    assert_attrs: list[str] | None = None,
) -> Molto:
    """Create a Molto instance backed by playback-only cassette transport.

    Example:
        molto = make_molto_client(manager, cassette_suffix="sse")
        isinstance(molto, Molto)
        # True
    """
    return Molto(
        http_client=httpx.AsyncClient(
            transport=OpenAICassetteTransport(
                cassette_manager,
                cassette_suffix=cassette_suffix,
                playback_only=True,
                expected_request_body=expected_request_body,
                assert_attrs=assert_attrs,
            ),
        ),
    )


def prepend_model_prefix(body: dict[str, Any], prefix: str = "openai/") -> dict[str, Any]:
    """Return a shallow copy of "body" with the model field prefixed.

    Example::

        prepend_model_prefix({"model": "gpt-4o", ...})
        # {"model": "openai/gpt-4o", ...}
    """
    copy_body = deepcopy(body)
    copy_body["model"] = prefix + copy_body["model"]
    return copy_body
