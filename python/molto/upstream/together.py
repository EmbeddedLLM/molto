"""Together's documented Chat Completions transport and reasoning dialect."""

from typing import NoReturn

from openai import APIStatusError, BadRequestError

from molto.types.exceptions import ContextLengthExceededError, ModelNotFoundError
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class TogetherAIChatCompletionsClient(OpenAIChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    REASONING_FIELD = "reasoning"
    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"

    def _handle_error(self, error: Exception) -> NoReturn:
        if isinstance(error, BadRequestError) and (
            "is longer than the model's context length" in error.message
        ):
            raise ContextLengthExceededError(
                error.message, response=error.response, body=error.body
            ) from error
        if isinstance(error, APIStatusError) and error.code == "model_not_available":
            raise ModelNotFoundError(
                error.message, response=error.response, body=error.body
            ) from error
        super()._handle_error(error)


class TogetherAIResponsesClient(OpenAIResponsesClient):
    """Explicit custom Responses destination; no hosted Together endpoint is assumed."""

    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"
    REQUIRES_BASE_URL = True


class TogetherAIMessagesClient(AnthropicMessagesClient):
    """Explicit custom Messages destination; no hosted Together endpoint is assumed."""

    REQUIRED_API_KEY = "TOGETHER_AI_API_KEY"
    REQUIRES_BASE_URL = True
