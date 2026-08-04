"""Internal OpenAI-compatible HTTP transport for Molto."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pprint import pformat
from textwrap import indent
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from httpx import AsyncClient
from pydantic import BaseModel, ValidationError
from starlette.datastructures import MutableHeaders

try:
    from loguru import logger
except ImportError as exc:
    raise ImportError(
        "The internal Molto server requires Loguru. Install it with `python -m pip install loguru`."
    ) from exc

from molto import Molto
from molto.providers.openai.types import OpenAIEndpoint
from molto.types.openai.chat_completions import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from molto.types.openai.responses import (
    OpenAIResponsesError,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
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
_NULL_CHAT_MESSAGE_PLACEHOLDERS = frozenset(
    {"reasoning_content", "reasoning", "tool_calls", "tool_call_id"}
)


@dataclass(frozen=True)
class ServerSettings:
    """Operator-controlled settings for the internal server."""

    provider_base_url: str | None = None
    max_retries: int = 2

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
        return cls(
            provider_base_url=os.getenv("MOLTO_SERVER_BASE_URL") or None,
            max_retries=max_retries,
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


def _request_block(request: Request) -> str:
    params = _safe_params(getattr(request.state, "request_params", "<not parsed>"))
    formatted_params = indent(pformat(params, sort_dicts=False, width=100), "    ")
    return (
        f"  request_id: {request.state.request_id}\n"
        f"  endpoint: {getattr(request.state, 'endpoint', request.url.path)}\n"
        f"  X-Molto-Upstream: {getattr(request.state, 'requested_upstream', '<omitted>')}\n"
        f"  resolved_upstream: {getattr(request.state, 'resolved_upstream', '<unresolved>')}\n"
        f"  params:\n{formatted_params}"
    )


def _request_summary(request: Request) -> str:
    params = getattr(request.state, "request_params", {})
    model = params.get("model", "<unknown>") if isinstance(params, dict) else "<unknown>"
    return (
        f"  request_id: {request.state.request_id}\n"
        f"  endpoint: {getattr(request.state, 'endpoint', request.url.path)}\n"
        f"  X-Molto-Upstream: {getattr(request.state, 'requested_upstream', '<omitted>')}\n"
        f"  resolved_upstream: {getattr(request.state, 'resolved_upstream', '<unresolved>')}\n"
        f"  model: {model}"
    )


def _verbose(request: Request) -> bool:
    return bool(getattr(request.app.state, "verbose", False))


def _log_request(request: Request) -> None:
    if _verbose(request):
        logger.info("REQUEST\n{}", _request_block(request))


def _log_raw_request(request: Request) -> None:
    if _verbose(request):
        logger.info("RAW REQUEST JSON\n{}", request.state.raw_request_json)


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
    details = _request_block(request) if _verbose(request) else _request_summary(request)
    event.error(
        "ERROR\n{}\n  exception: {}: {}",
        details,
        type(exc).__name__,
        str(exc),
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
    if isinstance(exc, (BadInputError, ModelCapabilityError)):
        return (
            400,
            "The request is invalid for the selected model or upstream.",
            "invalid_request_error",
            "invalid_request",
        )
    if isinstance(exc, ContextOverflowError):
        return (
            400,
            "The request exceeds the selected model's context limit.",
            "invalid_request_error",
            "context_length_exceeded",
        )
    if isinstance(exc, ExternalAuthError):
        return (
            502,
            "The upstream provider rejected the server credentials.",
            "server_error",
            "upstream_authentication_error",
        )
    if isinstance(exc, ForbiddenError):
        return (
            403,
            "The upstream provider denied access to the requested resource.",
            "server_error",
            "upstream_forbidden",
        )
    if isinstance(exc, ResourceNotFoundError):
        return (
            404,
            "The requested upstream resource was not found.",
            "invalid_request_error",
            "upstream_not_found",
        )
    if isinstance(exc, RateLimitExceedError):
        return (
            429,
            "The upstream provider rate limit was exceeded.",
            "server_error",
            "rate_limit_exceeded",
        )
    if isinstance(exc, (ModelOverloadError, ServerBusyError)):
        return (
            503,
            "The upstream provider is temporarily unavailable.",
            "server_error",
            "upstream_unavailable",
        )
    return 500, "Internal server error.", "server_error", "internal_error"


def _upstream(request: Request, *, default: OpenAIEndpoint) -> OpenAIEndpoint:
    value = request.headers.get("X-Molto-Upstream")
    if value is None:
        request.state.resolved_upstream = default
        return default
    if value not in ("chat_completions", "responses"):
        request.state.resolved_upstream = "<invalid>"
        raise _ServerError(
            400,
            "X-Molto-Upstream must be 'chat_completions' or 'responses'.",
            error_type="invalid_request_error",
            code="invalid_upstream",
            param="X-Molto-Upstream",
        )
    request.state.resolved_upstream = value
    return cast(OpenAIEndpoint, value)


def _chat_upstream(
    request: Request,
    body: OpenAIChatCompletionRequest,
) -> OpenAIEndpoint:
    model_name = body.model.removeprefix("openai/")
    is_openai_gpt_5_6 = body.model.startswith("openai/") and (
        model_name == "gpt-5.6" or model_name.startswith("gpt-5.6-")
    )
    if (
        request.headers.get("X-Molto-Upstream") is None
        and is_openai_gpt_5_6
        and body.tools
        and body.reasoning_effort != "none"
    ):
        MutableHeaders(raw=request.scope["headers"])["X-Molto-Upstream"] = "responses"
        request.state.requested_upstream = "responses"
    return _upstream(request, default="chat_completions")


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
    request.state.raw_request_json = (await request.body()).decode("utf-8")
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
    *,
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
    *,
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
            yield
            return
        async with AsyncClient(timeout=None) as http_client:
            app.state.gateway = Molto(
                http_client=http_client,
                base_url=resolved_settings.provider_base_url,
                max_retries=resolved_settings.max_retries,
            )
            yield

    app = FastAPI(
        title="Molto internal server",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.verbose = verbose

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        request.state.request_id = uuid4().hex
        request.state.endpoint = request.url.path
        request.state.requested_upstream = request.headers.get("X-Molto-Upstream", "<omitted>")
        request.state.resolved_upstream = "<unresolved>"
        request.state.request_params = "<not parsed>"
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

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> JSONResponse:
        _log_error(request, exc)
        return _error_response(
            request,
            400,
            "The request is invalid for the selected model or upstream.",
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
        #_log_raw_request(request)
        upstream = _chat_upstream(request, body)
        _log_request(request)
        result = await request.app.state.gateway.chat_completions(body, upstream=upstream)
        if body.stream:
            iterator = cast(AsyncIterator[OpenAIChatCompletionChunk], result)
            return StreamingResponse(
                _chat_sse(iterator, request=request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = cast(OpenAIChatCompletionResponse, result)
        _log_response(request, response)
        return Response(
            response.model_dump_json(by_alias=True, exclude_none=False),
            media_type="application/json",
        )

    @app.post("/v1/responses", include_in_schema=False)
    async def responses(request: Request) -> Response:
        body = await _body(request, OpenAIResponsesRequest)
        upstream = _upstream(request, default="responses")
        _log_request(request)
        result = await request.app.state.gateway.responses(body, upstream=upstream)
        if body.stream:
            iterator = cast(AsyncIterator[OpenAIResponsesStreamEvent], result)
            return StreamingResponse(
                _responses_sse(iterator, request=request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = cast(OpenAIResponsesResponse, result)
        _log_response(request, response)
        return Response(
            response.model_dump_json(by_alias=True, exclude_none=False),
            media_type="application/json",
        )

    return app
