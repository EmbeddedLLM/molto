"""Explicit identities for existing cross-vendor protocol configurations.

These classes preserve configurable routes, not claims that a vendor hosts a
particular endpoint. Keeping them together avoids circular canonical imports.
"""

from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class OpenAIMessagesClient(AnthropicMessagesClient):
    """Messages wire implementation with the OpenAI provider profile."""


class AnthropicResponsesClient(OpenAIResponsesClient):
    """Responses wire implementation with the Anthropic provider profile."""


class AnthropicChatCompletionsClient(OpenAIChatCompletionsClient):
    """Chat Completions wire implementation with the Anthropic provider profile."""
