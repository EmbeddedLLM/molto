"""Internal OpenAI-compatible HTTP transport for Molto."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pprint import pformat
from textwrap import indent
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from httpx import AsyncClient
from openai import OpenAIError
from pydantic import BaseModel, ValidationError

try:
    from loguru import logger
except ImportError as exc:
    raise ImportError(
        "The internal Molto server requires Loguru. Install it with `python -m pip install loguru`."
    ) from exc

from molto import Molto
from molto.core import default_endpoint_for
from molto.providers.openai.types import OpenAIEndpoint
from molto.types.openai.chat_completions import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)
from molto.types.openai.responses import (
    OpenAIResponsesError,
    OpenAIResponsesRequest,
    OpenAIResponsesStreamEvent,
)
from molto.utils.exceptions import (
    BadInputError,
    ContextOverflowError,
    ExternalAuthError,
    ForbiddenError,
    ModelCapabilityError,
    ModelOverloadError,
    MoltoException,
    RateLimitExceedError,
    ResourceNotFoundError,
    ServerBusyError,
)

RequestModel = TypeVar("RequestModel", bound=BaseModel)
ErrorType = Literal["invalid_request_error", "authentication_error", "server_error"]
_SENSITIVE_PARAMETER_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "password",
        "secret",
        "client_secret",
        "credential",
        "credentials",
        "access_token",
        "bearer_token",
        "token",
    }
)
_MAX_LOGGED_STRING_LENGTH = 4000
_IGNORED_CHAT_REQUEST_FIELDS = frozenset({"logit_bias", "thinking_budget", "chat_template_kwargs"})
# Harness-injected transport metadata, not provider parameters.
_IGNORED_REQUEST_FIELDS = frozenset({"client_metadata"})
_NULL_CHAT_MESSAGE_PLACEHOLDERS = frozenset(
    {"reasoning_content", "reasoning", "tool_calls", "tool_call_id"}
)

# HTTP status / public message / error type / code for each Molto exception family.
_MOLTO_ERRORS: dict[tuple[type[MoltoException], ...], tuple[int, str, ErrorType, str]] = {
    (BadInputError, ModelCapabilityError): (
        400,
        "The request is invalid for the selected model or endpoint.",
        "invalid_request_error",
        "invalid_request",
    ),
    (ContextOverflowError,): (
        400,
        "The request exceeds the selected model's context limit.",
        "invalid_request_error",
        "context_length_exceeded",
    ),
    (ExternalAuthError,): (
        502,
        "The upstream provider rejected the API key.",
        "server_error",
        "upstream_authentication_error",
    ),
    (ForbiddenError,): (
        403,
        "The upstream provider denied access to the requested resource.",
        "server_error",
        "upstream_forbidden",
    ),
    (ResourceNotFoundError,): (
        404,
        "The requested upstream resource was not found.",
        "invalid_request_error",
        "upstream_not_found",
    ),
    (RateLimitExceedError,): (
        429,
        "The upstream provider rate limit was exceeded.",
        "server_error",
        "rate_limit_exceeded",
    ),
    (ModelOverloadError, ServerBusyError): (
        503,
        "The upstream provider is temporarily unavailable.",
        "server_error",
        "upstream_unavailable",
    ),
}


@dataclass(frozen=True)
class ServerSettings:
    """Operator-controlled settings for the internal server."""

    api_key: str | None = None
    provider_base_url: str | None = None
    max_retries: int = 2
    strict: bool = False

    @classmethod
    def from_env(cls) -> ServerSettings:
        """Read server-only settings without exposing their values."""

        raw_max_retries = os.getenv("MOLTO_SERVER_MAX_RETRIES", "2")
        try:
            max_retries = int(raw_max_retries)
        except ValueError as exc:
            raise ValueError("MOLTO_SERVER_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise ValueError("MOLTO_SERVER_MAX_RETRIES must be non-negative.")
        raw_strict = os.getenv("MOLTO_SERVER_STRICT", "")
        return cls(
            api_key=os.getenv("MOLTO_API_KEY") or None,
            provider_base_url=os.getenv("MOLTO_SERVER_BASE_URL") or None,
            max_retries=max_retries,
            strict=raw_strict.strip().lower() in {"1", "true", "yes", "on"},
        )


class _ServerError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: ErrorType,
        code: str,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        self.headers = headers or {}


def _normalize_chat_payload(payload: Any) -> Any:
    """Apply the server's narrow compatibility policy before strict validation."""

    if not isinstance(payload, dict):
        return payload

    normalized = {
        key: value for key, value in payload.items() if key not in _IGNORED_CHAT_REQUEST_FIELDS
    }
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized

    normalized["messages"] = [
        {
            key: value
            for key, value in message.items()
            if not (key in _NULL_CHAT_MESSAGE_PLACEHOLDERS and value is None)
        }
        if isinstance(message, dict)
        else message
        for message in messages
    ]
    return normalized


def _error_payload(
    message: str,
    *,
    error_type: ErrorType,
    code: str,
    param: str | None = None,
) -> dict[str, dict[str, str | None]]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def _safe_params(value: Any) -> Any:
    """Make request parameters readable without logging credentials or huge blobs."""

    if isinstance(value, dict):
        safe: dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in _SENSITIVE_PARAMETER_NAMES or normalized_key.endswith("_api_key"):
                safe[key] = "***"
            else:
                safe[key] = _safe_params(item)
        return safe
    if isinstance(value, list):
        return [_safe_params(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_params(item) for item in value)
    if isinstance(value, str) and len(value) > _MAX_LOGGED_STRING_LENGTH:
        omitted = len(value) - _MAX_LOGGED_STRING_LENGTH
        return f"{value[:_MAX_LOGGED_STRING_LENGTH]}... <{omitted} characters omitted>"
    return value


def _request_block(request: Request, *, detailed: bool = False) -> str:
    params = getattr(request.state, "request_params", {})
    lines = [
        f"  request_id: {request.state.request_id}",
        f"  endpoint: {getattr(request.state, 'endpoint', request.url.path)}",
        f"  X-Molto-Endpoint: {getattr(request.state, 'requested_endpoint', '<omitted>')}",
        f"  resolved_endpoint: {getattr(request.state, 'resolved_endpoint', '<unresolved>')}",
    ]
    if detailed:
        formatted_params = indent(
            pformat(_safe_params(params), sort_dicts=False, width=100), "    "
        )
        lines.append(f"  params:\n{formatted_params}")
    elif isinstance(params, dict):
        lines.append(f"  model: {params.get('model', '<unknown>')}")
    else:
        lines.append("  model: <unknown>")
    return "\n".join(lines)


def _verbose(request: Request) -> bool:
    return bool(getattr(request.app.state, "verbose", False))


def _log_request(request: Request) -> None:
    if _verbose(request):
        logger.info("REQUEST\n{}", _request_block(request, detailed=True))


def _log_response(request: Request, response: BaseModel) -> None:
    if not _verbose(request):
        return
    payload = response.model_dump(mode="json", by_alias=True, exclude_none=False)
    formatted_payload = indent(pformat(_safe_params(payload), sort_dicts=False, width=100), "    ")
    logger.info(
        "RESPONSE\n  request_id: {}\n  endpoint: {}\n  payload:\n{}",
        request.state.request_id,
        request.state.endpoint,
        formatted_payload,
    )


def _log_error(request: Request, exc: Exception, *, traceback: bool = False) -> None:
    event = logger.opt(exception=exc) if traceback else logger
    details = _request_block(request, detailed=_verbose(request))
    exception_detail = str(exc)
    if isinstance(exc, _ServerError):
        exception_detail += f" (type={exc.error_type}, code={exc.code}, param={exc.param})"
    event.error(
        "ERROR\n{}\n  exception: {}: {}",
        details,
        type(exc).__name__,
        exception_detail,
    )


def _error_response(
    request: Request,
    status_code: int,
    message: str,
    *,
    error_type: ErrorType,
    code: str,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request.state.request_id
    return JSONResponse(
        _error_payload(message, error_type=error_type, code=code, param=param),
        status_code=status_code,
        headers=response_headers,
    )


def _molto_error(exc: MoltoException) -> tuple[int, str, ErrorType, str]:
    for exception_types, response in _MOLTO_ERRORS.items():
        if isinstance(exc, exception_types):
            return response
    return 500, "Internal server error.", "server_error", "internal_error"


def _endpoint(
    request: Request,
    body: OpenAIChatCompletionRequest | OpenAIResponsesRequest,
    *,
    default: OpenAIEndpoint,
) -> OpenAIEndpoint:
    value = body.endpoint or request.headers.get("X-Molto-Endpoint")
    if value is None:
        request.state.resolved_endpoint = default
        return default
    if value not in ("chat_completions", "responses"):
        request.state.resolved_endpoint = "<invalid>"
        raise _ServerError(
            400,
            "X-Molto-Endpoint must be 'chat_completions' or 'responses'.",
            error_type="invalid_request_error",
            code="invalid_endpoint",
            param="X-Molto-Endpoint",
        )
    request.state.resolved_endpoint = value
    return cast(OpenAIEndpoint, value)


def _chat_endpoint(
    request: Request,
    body: OpenAIChatCompletionRequest,
) -> OpenAIEndpoint:
    model_name = body.model.removeprefix("openai/")
    is_openai_gpt_5_6 = body.model.startswith("openai/") and (
        model_name == "gpt-5.6" or model_name.startswith("gpt-5.6-")
    )
    if (
        request.headers.get("X-Molto-Endpoint") is None
        and body.endpoint is None
        and is_openai_gpt_5_6
        and body.tools
        and body.reasoning_effort != "none"
    ):
        request.state.requested_endpoint = "responses"
        request.state.resolved_endpoint = "responses"
        return "responses"
    return _endpoint(request, body, default="chat_completions")


def _prepare_payload(request: Request, payload: Any) -> Any:
    """Drop transport metadata before validation and capture the API key.

    ``client_metadata`` is harness-injected and ``api_key`` is a per-request
    credential consumed by the server; neither is a provider parameter, and the
    strict request models reject unknown fields.
    """

    if not isinstance(payload, dict):
        return payload
    api_key = payload.get("api_key")
    request.state.api_key = (
        api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
    )
    return {
        key: value
        for key, value in payload.items()
        if key not in _IGNORED_REQUEST_FIELDS and key != "api_key"
    }


def _request_api_key(request: Request, settings: ServerSettings) -> str | None:
    """Return the configured or per-request credential.

    ``MOLTO_API_KEY`` is operator-controlled and intentionally wins over
    client-supplied credentials. Without it, a body-level ``api_key`` wins over
    ``Authorization: Bearer``.
    """

    if settings.api_key is not None:
        return settings.api_key
    body_key = getattr(request.state, "api_key", None)
    if body_key is not None:
        return body_key
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None
    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


async def _body(request: Request, model: type[RequestModel]) -> RequestModel:
    content_type = request.headers.get("content-type", "").partition(";")[0].lower()
    if content_type != "application/json":
        raise _ServerError(
            400,
            "Content-Type must be application/json.",
            error_type="invalid_request_error",
            code="invalid_content_type",
            param="Content-Type",
        )
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        request.state.request_params = "<invalid JSON>"
        raise _ServerError(
            400,
            "The request body is not valid JSON.",
            error_type="invalid_request_error",
            code="invalid_json",
        ) from exc
    payload = _prepare_payload(request, payload)
    request.state.request_params = payload
    if model is OpenAIChatCompletionRequest:
        payload = _normalize_chat_payload(payload)
    if isinstance(payload, dict):
        requested_model = payload.get("model")
        if isinstance(requested_model, str) and "/" not in requested_model:
            payload = {**payload, "model": f"openai/{requested_model}"}
    try:
        validated = model.model_validate(payload)
    except ValidationError as exc:
        error = exc.errors(include_url=False, include_context=False)[0]
        param = ".".join(str(part) for part in error["loc"]) or None
        raise _ServerError(
            400,
            error["msg"],
            error_type="invalid_request_error",
            code="invalid_request",
            param=param,
        ) from exc
    request.state.request_params = validated.model_dump(
        mode="json",
        by_alias=True,
        exclude_unset=True,
    )
    return validated


async def _close_iterator(iterator: AsyncIterator[Any], *, request: Request) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception as exc:
        _log_error(request, exc, traceback=True)


def _chat_stream_error() -> str:
    payload = _error_payload(
        "The response stream ended because the upstream request failed.",
        error_type="server_error",
        code="upstream_stream_error",
    )
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _chat_sse(
    iterator: AsyncIterator[OpenAIChatCompletionChunk],
    request: Request,
) -> AsyncIterator[str]:
    completed = False
    try:
        async for chunk in iterator:
            _log_response(request, chunk)
            yield chunk.as_chunk()
        completed = True
    except Exception as exc:
        _log_error(request, exc, traceback=True)
        yield _chat_stream_error()
    finally:
        await _close_iterator(iterator, request=request)
    if completed:
        yield "data: [DONE]\n\n"


async def _responses_sse(
    iterator: AsyncIterator[OpenAIResponsesStreamEvent],
    request: Request,
) -> AsyncIterator[str]:
    last_sequence_number = -1
    try:
        async for event in iterator:
            last_sequence_number = event.sequence_number
            _log_response(request, event)
            yield event.as_responses_chunk()
    except Exception as exc:
        _log_error(request, exc, traceback=True)
        yield OpenAIResponsesError(
            sequence_number=last_sequence_number + 1,
            code="upstream_stream_error",
            message="The response stream ended because the upstream request failed.",
            param=None,
        ).as_responses_chunk()
    finally:
        await _close_iterator(iterator, request=request)


async def _serve(
    request: Request,
    *,
    stream: bool,
    execute: Callable[[str | None], Awaitable[Any]],
    to_sse: Callable[[AsyncIterator[Any], Request], AsyncIterator[str]],
) -> Response:
    """Run one gateway call and render it as JSON or an SSE stream."""

    result = await execute(_request_api_key(request, request.app.state.settings))
    if stream:
        return StreamingResponse(
            to_sse(cast(AsyncIterator[Any], result), request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    response = cast(BaseModel, result)
    _log_response(request, response)
    return Response(
        response.model_dump_json(by_alias=True, exclude_none=False),
        media_type="application/json",
    )


def create_app(
    settings: ServerSettings | None = None,
    *,
    gateway: Molto | None = None,
    verbose: bool = False,
) -> FastAPI:
    """Create the internal server without changing the core Molto package."""

    resolved_settings = settings or ServerSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if gateway is not None:
            app.state.gateway = gateway
            app.state.http_client = None
            yield
            return
        async with AsyncClient(timeout=None) as http_client:
            app.state.http_client = http_client
            app.state.gateway = Molto(strict=resolved_settings.strict)
            yield

    app = FastAPI(
        title="Molto internal server",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.verbose = verbose
    app.state.settings = resolved_settings

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        request.state.request_id = uuid4().hex
        request.state.endpoint = request.url.path
        request.state.requested_endpoint = request.headers.get("X-Molto-Endpoint", "<omitted>")
        request.state.resolved_endpoint = "<unresolved>"
        request.state.request_params = "<not parsed>"
        request.state.api_key = None
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(_ServerError)
    async def server_error(request: Request, exc: _ServerError) -> JSONResponse:
        _log_error(request, exc)
        return _error_response(
            request,
            exc.status_code,
            exc.message,
            error_type=exc.error_type,
            code=exc.code,
            param=exc.param,
            headers=exc.headers,
        )

    @app.exception_handler(MoltoException)
    async def molto_error(request: Request, exc: MoltoException) -> JSONResponse:
        status_code, message, error_type, code = _molto_error(exc)
        _log_error(request, exc)
        return _error_response(
            request,
            status_code,
            message,
            error_type=error_type,
            code=code,
        )

    @app.exception_handler(OpenAIError)
    async def sdk_client_error(request: Request, exc: OpenAIError) -> JSONResponse:
        _log_error(request, exc, traceback=True)
        return _error_response(
            request,
            401,
            "No API key was provided. Pass one with the request "
            "(Authorization: Bearer <key> or body `api_key`) or configure "
            "the server's MOLTO_API_KEY or OPENAI_API_KEY.",
            error_type="authentication_error",
            code="missing_api_key",
        )

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> JSONResponse:
        _log_error(request, exc)
        return _error_response(
            request,
            400,
            "The request is invalid for the selected model or endpoint.",
            error_type="invalid_request_error",
            code="invalid_request",
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        _log_error(request, exc, traceback=True)
        return _error_response(
            request,
            500,
            "Internal server error.",
            error_type="server_error",
            code="internal_error",
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        logger.debug("HEALTH\n{}", "  endpoint: /healthz")
        return {"status": "ok"}

    @app.post("/v1/chat/completions", include_in_schema=False)
    async def chat_completions(request: Request) -> Response:
        body = await _body(request, OpenAIChatCompletionRequest)
        endpoint = _chat_endpoint(request, body)
        _log_request(request)

        async def execute(api_key: str | None) -> Any:
            return await request.app.state.gateway.chat_completions(
                body,
                api_key=api_key,
                endpoint=endpoint,
                http_client=request.app.state.http_client,
                base_url=resolved_settings.provider_base_url,
                max_retries=resolved_settings.max_retries,
            )

        return await _serve(request, stream=body.stream, execute=execute, to_sse=_chat_sse)

    @app.post("/v1/responses", include_in_schema=False)
    async def responses(request: Request) -> Response:
        body = await _body(request, OpenAIResponsesRequest)
        endpoint = _endpoint(
            request,
            body,
            default=default_endpoint_for(cast(Any, body.model.partition("/")[0]), "responses"),
        )
        _log_request(request)

        async def execute(api_key: str | None) -> Any:
            return await request.app.state.gateway.responses(
                body,
                api_key=api_key,
                endpoint=endpoint,
                http_client=request.app.state.http_client,
                base_url=resolved_settings.provider_base_url,
                max_retries=resolved_settings.max_retries,
            )

        return await _serve(request, stream=body.stream, execute=execute, to_sse=_responses_sse)

    return app
