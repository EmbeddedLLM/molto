"""Bedrock API-key clients; callers supply the regional protocol destination."""

from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class BedrockResponsesClient(OpenAIResponsesClient):
    REQUIRED_API_KEY = "AWS_BEARER_TOKEN_BEDROCK"
    REQUIRES_BASE_URL = True


class BedrockChatCompletionsClient(OpenAIChatCompletionsClient):
    REQUIRED_API_KEY = "AWS_BEARER_TOKEN_BEDROCK"
    REQUIRES_BASE_URL = True
