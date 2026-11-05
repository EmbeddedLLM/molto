"""Together Responses transport using the shared OpenAI wire implementation."""

from molto.upstream.openai import OpenAIResponsesClient


class TogetherAIResponsesClient(OpenAIResponsesClient):
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"
