"""vLLM protocol clients sharing the canonical wire implementations."""

from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import ChatCompletionsClient, ResponsesClient


class VLLMResponsesClient(ResponsesClient):
    """vLLM Responses transport and normalization."""


class VLLMChatCompletionsClient(ChatCompletionsClient):
    """vLLM Chat Completions transport and normalization."""


class VLLMMessagesClient(AnthropicMessagesClient):
    """vLLM Messages transport and normalization."""
