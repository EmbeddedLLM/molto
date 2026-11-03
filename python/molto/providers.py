from dataclasses import dataclass
from typing import Literal, TypeAlias

Endpoint: TypeAlias = Literal["responses", "chat/completions", "messages"]
Provider: TypeAlias = Literal[
    "anthropic", "bedrock", "fireworks", "gemini", "openai", "openrouter", "togetherai", "vllm"
]
ReasoningRepresentation: TypeAlias = Literal["raw", "summary"]


@dataclass(frozen=True)
class ProviderCapabilities:
    thinking: ReasoningRepresentation


PROVIDER_CAPABILITIES: dict[Provider, ProviderCapabilities] = {
    "anthropic": ProviderCapabilities(thinking="summary"),
    "bedrock": ProviderCapabilities(thinking="summary"),
    "fireworks": ProviderCapabilities(thinking="summary"),
    "openai": ProviderCapabilities(thinking="summary"),
    "gemini": ProviderCapabilities(thinking="raw"),
    "openrouter": ProviderCapabilities(thinking="summary"),
    "togetherai": ProviderCapabilities(thinking="raw"),
    "vllm": ProviderCapabilities(thinking="raw"),
}

# These destinations select transport only; they do not certify model support.
PROVIDER_BASE_URLS: dict[str, dict[Endpoint, str]] = {
    "fireworks": {
        "responses": "https://api.fireworks.ai/inference/v1",
        "chat/completions": "https://api.fireworks.ai/inference/v1",
        "messages": "https://api.fireworks.ai/inference",
    },
    "openrouter": {
        "responses": "https://openrouter.ai/api/v1",
        "chat/completions": "https://openrouter.ai/api/v1",
        "messages": "https://openrouter.ai/api",
    },
}

PROVIDER_KEY_NAMES = {
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
    "fireworks": "FIREWORKS_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "togetherai": "TOGETHER_AI_API_KEY",
}

DEFAULT_PROVIDER_BY_ENDPOINT: dict[Endpoint, Provider] = {
    "responses": "openai",
    "chat/completions": "openai",
    "messages": "anthropic",
}
