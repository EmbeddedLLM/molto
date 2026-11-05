"""OpenRouter transports with protocol-specific reasoning interpretation."""

from dataclasses import replace
from typing import Any

from molto.providers import ProviderCapabilities
from molto.types.normalized import Thinking
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import (
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
    _reasoning_part,
)


class OpenRouterChatCompletionsClient(OpenAIChatCompletionsClient):
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    REQUIRED_API_KEY = "OPENROUTER_API_KEY"


class OpenRouterMessagesClient(AnthropicMessagesClient):
    DEFAULT_BASE_URL = "https://openrouter.ai/api"
    REQUIRED_API_KEY = "OPENROUTER_API_KEY"
    BEARER_AUTH = True


class OpenRouterResponsesClient(OpenAIResponsesClient):
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    REQUIRED_API_KEY = "OPENROUTER_API_KEY"

    @classmethod
    def resolve_capabilities(cls, profile: ProviderCapabilities) -> ProviderCapabilities:
        return replace(profile, responses_reasoning="wire")

    def _reasoning_part(self, item: Any) -> Thinking:
        part = _reasoning_part(item)
        format = getattr(item, "format", None)
        # OpenRouter labels raw text with "unknown" rather than an opaque encoding.
        # Signed/summary formats still require an explicit translation contract.
        if format is not None and (format != "unknown" or part.summary or part.signature):
            raise ValueError("Responses reasoning format cannot be translated.")
        return part

    def _validate_reasoning(self, part: Thinking) -> None:
        # Responses already distinguishes raw text from summaries. Thinking owns
        # mixed-representation validation; completion requires signed summaries.
        pass
