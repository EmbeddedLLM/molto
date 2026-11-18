from dataclasses import dataclass
from typing import Literal, TypeAlias

Endpoint: TypeAlias = Literal["responses", "chat/completions", "messages"]
Provider: TypeAlias = Literal[
    "anthropic", "bedrock", "fireworks", "gemini", "openai", "openrouter", "togetherai", "vllm"
]
ChatReasoningField: TypeAlias = Literal["reasoning_content", "reasoning"]
ReasoningRepresentation: TypeAlias = Literal["raw", "summary"]


@dataclass(frozen=True)
class ProviderCapabilities:
    thinking: ReasoningRepresentation
    requires_search_history_state: bool = False
    chat_reasoning_field: ChatReasoningField | None = None
    chat_signature_namespace: str | None = None


PROVIDER_CAPABILITIES: dict[Provider, ProviderCapabilities] = {
    "anthropic": ProviderCapabilities(thinking="summary", requires_search_history_state=True),
    "bedrock": ProviderCapabilities(thinking="summary"),
    "fireworks": ProviderCapabilities(thinking="raw"),
    "openai": ProviderCapabilities(thinking="summary"),
    "gemini": ProviderCapabilities(thinking="summary"),
    "openrouter": ProviderCapabilities(thinking="raw"),
    "togetherai": ProviderCapabilities(thinking="raw"),
    "vllm": ProviderCapabilities(thinking="raw"),
}

DEFAULT_PROVIDER_BY_ENDPOINT: dict[Endpoint, Provider] = {
    "responses": "openai",
    "chat/completions": "openai",
    "messages": "anthropic",
}
