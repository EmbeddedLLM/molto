"""Together AI transport using the OpenAI-compatible Chat Completions endpoint."""

from typing import Any, NoReturn

from openai import APIStatusError, AsyncOpenAI, BadRequestError, OpenAIError

from molto.types.exceptions import ContextLengthExceededError, ModelNotFoundError, UnsupportedInput
from molto.types.normalized import TextOutput, Thinking, ToolCall
from molto.upstream.openai import OpenAIChatCompletionsClient


class TogetherAIChatCompletionsClient(OpenAIChatCompletionsClient):
    """Translate Together's `reasoning` field as unsigned raw thinking."""

    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    reasoning_field = "reasoning"

    def _make_sdk_client(self) -> AsyncOpenAI:
        # An OpenAI admin key can bypass the SDK's missing-key check. Together
        # must require its own credential before the shared transport sends HTTP.
        if not self._api_key:
            raise OpenAIError(
                "Missing credentials. Please pass an `api_key` or set the "
                "`TOGETHER_AI_API_KEY` environment variable."
            )
        return super()._make_sdk_client()

    def _handle_error(self, error: Exception) -> NoReturn:
        # Together's recorded context overflow has no error code to distinguish it.
        if (
            isinstance(error, BadRequestError)
            and "is longer than the model's context length" in error.message
        ):
            raise ContextLengthExceededError(
                error.message, response=error.response, body=error.body
            ) from error
        if isinstance(error, APIStatusError) and error.code == "model_not_available":
            raise ModelNotFoundError(
                error.message, response=error.response, body=error.body
            ) from error
        super()._handle_error(error)

    @classmethod
    def _assistant_message(cls, parts: list[Thinking | TextOutput | ToolCall]) -> dict[str, Any]:
        if any(part.summary for part in parts if isinstance(part, Thinking)):
            raise UnsupportedInput(
                "Together Chat Completions cannot represent reasoning summaries."
            )
        return super()._assistant_message(parts)
