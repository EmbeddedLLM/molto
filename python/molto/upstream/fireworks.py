"""Fireworks clients for each explicitly selected upstream protocol."""

from dataclasses import replace

from molto.providers import ProviderCapabilities
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class FireworksResponsesClient(OpenAIResponsesClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"


class FireworksChatCompletionsClient(OpenAIChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"

    @classmethod
    def resolve_capabilities(cls, capabilities: ProviderCapabilities) -> ProviderCapabilities:
        return replace(capabilities, thinking="raw")


class FireworksMessagesClient(AnthropicMessagesClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"
