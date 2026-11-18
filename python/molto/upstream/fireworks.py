"""Fireworks clients using the shared OpenAI protocol implementations."""

from dataclasses import replace

from molto.providers import ProviderCapabilities
from molto.upstream.openai import ChatCompletionsClient, ResponsesClient


class FireworksResponsesClient(ResponsesClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"

    @classmethod
    def resolve_capabilities(cls, profile: ProviderCapabilities) -> ProviderCapabilities:
        return replace(profile, thinking="raw")


class FireworksChatCompletionsClient(ChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"
    OMIT_NULL_MESSAGE_FIELDS = True
    OMIT_INCLUDE_OBFUSCATION = True

    @classmethod
    def resolve_capabilities(cls, profile: ProviderCapabilities) -> ProviderCapabilities:
        return super().resolve_capabilities(replace(profile, thinking="raw"))
