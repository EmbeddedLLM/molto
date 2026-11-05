"""Fireworks Responses transport."""

from molto.upstream.openai import OpenAIResponsesClient


class FireworksResponsesClient(OpenAIResponsesClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"
