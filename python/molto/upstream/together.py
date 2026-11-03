"""Together's documented Chat Completions transport and reasoning dialect."""

from typing import NoReturn

from openai import APIStatusError, AsyncOpenAI, BadRequestError, OpenAIError

from molto.types.exceptions import ContextLengthExceededError, ModelNotFoundError
from molto.upstream.openai import OpenAIChatCompletionsClient


class TogetherAIChatCompletionsClient(OpenAIChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    REASONING_FIELD = "reasoning"

    def _make_sdk_client(self) -> AsyncOpenAI:
        # The OpenAI SDK otherwise falls back to its own environment credentials.
        if not self._api_key:
            raise OpenAIError("Missing credentials. Set TOGETHER_AI_API_KEY or pass api_key.")
        return super()._make_sdk_client()

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
