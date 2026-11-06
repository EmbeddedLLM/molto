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
    # Only the resolved Responses client can opt into explicit wire labels;
    # ambiguous Chat/Messages input still uses `thinking`.
    responses_reasoning: Literal["profile", "wire"] = "profile"
    requires_search_history_state: bool = False


PROVIDER_CAPABILITIES: dict[Provider, ProviderCapabilities] = {
    "anthropic": ProviderCapabilities(thinking="summary", requires_search_history_state=True),
    "bedrock": ProviderCapabilities(thinking="summary"),
    "fireworks": ProviderCapabilities(thinking="summary"),
    "openai": ProviderCapabilities(thinking="summary"),
    "gemini": ProviderCapabilities(thinking="summary"),
    "openrouter": ProviderCapabilities(thinking="summary"),
    "togetherai": ProviderCapabilities(thinking="raw"),
    "vllm": ProviderCapabilities(thinking="raw"),
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
