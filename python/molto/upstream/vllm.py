"""vLLM protocol clients sharing the canonical wire implementations."""

from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class VLLMResponsesClient(OpenAIResponsesClient):
    """vLLM Responses transport and normalization."""


class VLLMChatCompletionsClient(OpenAIChatCompletionsClient):
    """vLLM Chat Completions transport and normalization."""


class VLLMMessagesClient(AnthropicMessagesClient):
    """vLLM Messages transport and normalization."""
