"""Together clients using the shared OpenAI protocol implementations."""

from molto.upstream.openai import ChatCompletionsClient, ResponsesClient


class TogetherAIResponsesClient(ResponsesClient):
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"


class TogetherAIChatCompletionsClient(ChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"
