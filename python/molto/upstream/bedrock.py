"""Bedrock API-key clients; callers supply the regional protocol destination."""

from molto.upstream.openai import ChatCompletionsClient, ResponsesClient


class BedrockResponsesClient(ResponsesClient):
    REQUIRED_API_KEY = "AWS_BEARER_TOKEN_BEDROCK"
    REQUIRES_BASE_URL = True


class BedrockChatCompletionsClient(ChatCompletionsClient):
    REQUIRED_API_KEY = "AWS_BEARER_TOKEN_BEDROCK"
    REQUIRES_BASE_URL = True
