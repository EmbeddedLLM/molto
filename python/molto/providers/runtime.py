"""Shared provider runtime policy.

Every SDK-backed provider follows one documented policy:

- Caller-injected SDK clients and HTTP transports stay caller-owned; Molto
  never closes them.
- SDK-created clients close after the response completes or the stream is
  exhausted, for both stream and non-stream execution.
- Rate-limit counters are parsed from provider response headers when present.
- `provider_code` is attributed whenever the provider error body exposes it.
- Protocol violations inside provider output translation raise `BadInputError`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from molto.utils.exceptions import (
    BadInputError,
    ContextOverflowError,
    ExternalAuthError,
    ForbiddenError,
    ModelOverloadError,
    MoltoException,
    RateLimitExceedError,
    ResourceNotFoundError,
    ServerBusyError,
    UnexpectedError,
)


def owns_sdk_client(*, sdk_client: Any | None, http_client: Any | None) -> bool:
    """Caller-supplied clients and transports stay caller-owned."""

    return sdk_client is None and http_client is None


def _provider_error_dict(exc: Exception) -> dict[str, Any] | None:
    """Extract the provider error object from an SDK exception body, if any."""

    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    return error if isinstance(error, dict) else None


def error_message(exc: Exception) -> str:
    """Prefer the provider error body's message over the raw exception text."""

    error = _provider_error_dict(exc)
    if error is not None and error.get("message"):
        return str(error["message"])
    return str(exc)


def is_context_error(exc: Exception, *, error_codes: frozenset[str] = frozenset()) -> bool:
    """One heuristic with provider-specific error codes layered on top."""

    error = _provider_error_dict(exc)
    if error is not None and error.get("code") in error_codes:
        return True
    message = error_message(exc).lower()
    if "context" in message and any(word in message for word in ("length", "token", "window")):
        return True
    return "too long" in message and "token" in message and "maximum" in message


def rate_limit_fields(exc: Exception) -> tuple[int, int]:
    """Extract request rate-limit counters from a provider error's headers."""

    headers = getattr(getattr(exc, "response", None), "headers", None) or {}

    def _int(name: str) -> int:
        try:
            return int(headers.get(name))
        except (TypeError, ValueError):
            return 0

    return _int("x-ratelimit-limit-requests"), _int("x-ratelimit-remaining-requests")


def map_sdk_error(
    exc: Exception,
    *,
    sdk_exceptions: dict[str, type[Exception]],
    context_error_codes: frozenset[str] = frozenset(),
) -> MoltoException:
    """Map an SDK exception ladder to stable Molto exceptions.

    Both official SDKs expose near-identical exception hierarchies; the table
    is keyed by stable name so providers only supply their exception classes.
    """

    message = error_message(exc)

    def _is(name: str) -> bool:
        cls = sdk_exceptions.get(name)
        return cls is not None and isinstance(exc, cls)

    if is_context_error(exc, error_codes=context_error_codes):
        mapped: MoltoException = ContextOverflowError(message)
    elif _is("authentication"):
        mapped = ExternalAuthError(message)
    elif _is("permission_denied"):
        mapped = ForbiddenError(message)
    elif _is("not_found"):
        mapped = ResourceNotFoundError(message)
    elif _is("bad_request"):
        mapped = BadInputError(message)
    elif _is("rate_limit"):
        limit, remaining = rate_limit_fields(exc)
        mapped = RateLimitExceedError(message, limit=limit, remaining=remaining)
    elif _is("timeout") or _is("connection"):
        mapped = ModelOverloadError(message)
    elif _is("api_status"):
        status_code = getattr(exc, "status_code", None)
        if status_code in (503, 529):
            mapped = ModelOverloadError(message)
        elif isinstance(status_code, int) and status_code >= 500:
            mapped = ServerBusyError(message)
        else:
            mapped = UnexpectedError(f"HTTP {status_code}: {message}")
    else:
        mapped = UnexpectedError(f"{type(exc).__name__}: {message}")

    # Keep the provider diagnostics available without coupling the public
    # exception hierarchy to SDK exception types.
    mapped.status_code = getattr(exc, "status_code", None)
    mapped.request_id = getattr(exc, "request_id", None)
    mapped.provider_body = getattr(exc, "body", None)
    provider_error = _provider_error_dict(exc)
    if provider_error is not None:
        # OpenAI bodies carry `error.code`; Anthropic bodies carry `error.type`.
        mapped.provider_code = provider_error.get("code") or provider_error.get("type")
    return mapped


def dump_sdk_model(value: Any) -> dict[str, Any]:
    """Normalize one SDK model into a portable dict."""

    return value.model_dump(mode="python", exclude_unset=True, warnings=False)


async def sdk_model_stream(stream: AsyncIterator[Any]) -> AsyncIterator[dict[str, Any]]:
    """Normalize an SDK model stream into portable dicts.

    Raw payload iterators (already portable dicts, e.g. preserved SSE frames)
    pass through unchanged.
    """

    async for event in stream:
        yield event if isinstance(event, dict) else dump_sdk_model(event)


async def execute_sdk_call(
    *,
    request: Any,
    build: Callable[[Any], dict[str, Any]],
    create: Callable[[dict[str, Any]], Awaitable[Any]],
    stream_translate: Callable[[AsyncIterator[dict[str, Any]]], AsyncIterator[Any]],
    response_translate: Callable[[dict[str, Any]], AsyncIterator[Any]],
    map_error: Callable[[Exception], MoltoException],
    close: Callable[[], Awaitable[None]],
) -> AsyncIterator[Any]:
    """Run one SDK call under the shared provider execution policy.

    Build and create failures close the client and are mapped to stable Molto
    exceptions; an exception that is already a ``MoltoException`` is re-raised
    unmapped. Streams normalize each SDK model into a portable dict, translate
    it, and close the client once exhausted; non-stream responses translate the
    dumped model and close immediately.
    """

    try:
        native_request = build(request)
        result = await create(native_request)
    except Exception as exc:
        await close()
        if isinstance(exc, MoltoException):
            raise
        raise map_error(exc) from exc

    try:
        if request.stream:
            events = _mapped_sdk_stream(result, map_error=map_error, close=close)
            try:
                async for event in stream_translate(events):
                    yield event
            finally:
                await events.aclose()
            return
        async for event in response_translate(dump_sdk_model(result)):
            yield event
    finally:
        if not request.stream:
            await close()


async def _mapped_sdk_stream(
    stream: Any,
    *,
    map_error: Callable[[Exception], MoltoException],
    close: Callable[[], Awaitable[None]],
) -> AsyncIterator[dict[str, Any]]:
    """Normalize an SDK stream, mapping iteration failures and closing on exit."""

    try:
        async for event in sdk_model_stream(stream):
            yield event
    except Exception as exc:
        raise map_error(exc) from exc
    finally:
        await close()
