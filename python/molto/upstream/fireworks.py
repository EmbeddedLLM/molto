"""Fireworks clients using the shared OpenAI protocol implementations."""

from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class FireworksResponsesClient(OpenAIResponsesClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"


class FireworksChatCompletionsClient(OpenAIChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"
