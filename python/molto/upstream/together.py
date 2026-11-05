"""Together clients using the shared OpenAI protocol implementations."""

from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class TogetherAIResponsesClient(OpenAIResponsesClient):
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"


class TogetherAIChatCompletionsClient(OpenAIChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"
