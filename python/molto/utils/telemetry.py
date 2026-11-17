"""OpenTelemetry attribute keys and span-start attributes for coordinator calls.

Only the `Molto` coordinators use this module; adapters and upstream clients stay
telemetry-free. See `docs/observability.md` for the span contract.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from httpx import URL, InvalidURL
from opentelemetry import trace
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from molto.providers import Endpoint, Provider
from molto.version import __version__

# Keys follow the GenAI semantic conventions. They are plain strings rather than
# imports from `opentelemetry-semantic-conventions`, which exposes GenAI keys only
# from a private, Development-status module that lags the current spec.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"
MOLTO_DOWNSTREAM_PROTOCOL = "molto.downstream.protocol"
MOLTO_ROUTE = "molto.route"
MOLTO_UPSTREAM_ENDPOINT = "molto.upstream.endpoint"

# The cross-vendor operation type for every coordinator, not Molto's protocol name:
# backends recognize LLM spans by well-known operation names.
OPERATION_CHAT = "chat"

# Downstream request fields holding the output-token limit, in precedence order.
# Chat Completions still accepts the deprecated `max_tokens`. The first field with an
# integer value wins, so a mistyped preferred field does not hide a valid fallback.
_MAX_TOKENS_FIELDS: dict[Endpoint, tuple[str, ...]] = {
    "responses": ("max_output_tokens",),
    "chat/completions": ("max_completion_tokens", "max_tokens"),
    "messages": ("max_tokens",),
}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def get_tracer() -> trace.Tracer:
    """Return Molto's tracer from the application's provider, a no-op without an SDK."""
    return trace.get_tracer("molto", __version__)


def span_name(attributes: Mapping[str, AttributeValue]) -> str:
    """Name a span `{operation} {model}`, or `{operation}` when the request has no model."""
    model = attributes.get(GEN_AI_REQUEST_MODEL)
    return f"{OPERATION_CHAT} {model}" if model else OPERATION_CHAT


@dataclass(frozen=True)
class ResolvedRoute:
    """Route values a coordinator resolves before transport.

    Attributes:
        protocol: Downstream protocol, from the coordinator.
        endpoint: Effective upstream endpoint.
        provider: Resolved provider, reported unchanged as `gen_ai.provider.name`.
        base_url: Effective upstream base URL: the caller's, or the client's default.
    """

    protocol: Endpoint
    endpoint: Endpoint
    provider: Provider
    base_url: str


def start_attributes(
    route: ResolvedRoute, body: Mapping[str, Any] | BaseModel
) -> dict[str, AttributeValue]:
    """Return the span attributes known before transport.

    Args:
        route: The coordinator's resolved route.
        body: Downstream request: a native dictionary or a validated request model.
    """
    attributes: dict[str, AttributeValue] = {
        GEN_AI_OPERATION_NAME: OPERATION_CHAT,
        GEN_AI_PROVIDER_NAME: route.provider,
        MOLTO_DOWNSTREAM_PROTOCOL: route.protocol,
        MOLTO_UPSTREAM_ENDPOINT: route.endpoint,
        MOLTO_ROUTE: "native" if route.protocol == route.endpoint else "translated",
    }
    # Native bodies are unvalidated dictionaries, so record a field only when it
    # has the expected type; telemetry must never reject or alter a request.
    model = _field(body, "model")
    if isinstance(model, str) and model:
        attributes[GEN_AI_REQUEST_MODEL] = model
    for name in _MAX_TOKENS_FIELDS[route.protocol]:
        value = _field(body, name)
        if _is_int(value):
            attributes[GEN_AI_REQUEST_MAX_TOKENS] = value
            break
    for key, name in (
        (GEN_AI_REQUEST_TEMPERATURE, "temperature"),
        (GEN_AI_REQUEST_TOP_P, "top_p"),
    ):
        value = _field(body, name)
        if _is_int(value) or isinstance(value, float):
            attributes[key] = float(value)
    # Matches the coordinators, which stream only when `stream` is exactly `True`.
    if _field(body, "stream") is True:
        attributes[GEN_AI_REQUEST_STREAM] = True
    attributes.update(_server_attributes(route.base_url))
    return attributes


def _field(body: Mapping[str, Any] | BaseModel, name: str) -> Any:
    if isinstance(body, Mapping):
        return body.get(name)
    return getattr(body, name, None)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _server_attributes(base_url: str) -> dict[str, AttributeValue]:
    # An unparsable URL must surface from the SDK at transport, not from telemetry.
    try:
        url = URL(base_url)
    except InvalidURL:
        return {}
    if not url.host:
        return {}
    port = url.port or _DEFAULT_PORTS.get(url.scheme)
    return {SERVER_ADDRESS: url.host} | ({SERVER_PORT: port} if port else {})
