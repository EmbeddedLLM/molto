from dataclasses import dataclass
from typing import Literal, TypeAlias

Endpoint: TypeAlias = Literal["responses", "chat/completions", "messages"]
Provider: TypeAlias = Literal["anthropic", "openai", "togetherai", "vllm"]
ReasoningRepresentation: TypeAlias = Literal["raw", "summary"]


@dataclass(frozen=True)
class ProviderCapabilities:
    thinking: ReasoningRepresentation


PROVIDER_CAPABILITIES: dict[Provider, ProviderCapabilities] = {
    "anthropic": ProviderCapabilities(thinking="summary"),
    "openai": ProviderCapabilities(thinking="summary"),
    "togetherai": ProviderCapabilities(thinking="raw"),
    "vllm": ProviderCapabilities(thinking="raw"),
}

DEFAULT_PROVIDER_BY_ENDPOINT: dict[Endpoint, Provider] = {
    "responses": "openai",
    "chat/completions": "openai",
    "messages": "anthropic",
}
