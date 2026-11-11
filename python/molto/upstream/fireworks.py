"""Fireworks clients using the shared OpenAI protocol implementations."""

from molto.upstream.openai import ChatCompletionsClient, ResponsesClient


class FireworksResponsesClient(ResponsesClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"


class FireworksChatCompletionsClient(ChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"
    OMIT_NULL_MESSAGE_FIELDS = True
    OMIT_INCLUDE_OBFUSCATION = True
