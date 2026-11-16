"""Bedrock API-key clients; callers supply the regional protocol destination."""

from anthropic import AsyncAnthropic, omit

from molto.types.exceptions import AuthenticationError
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import ChatCompletionsClient, ResponsesClient


class BedrockResponsesClient(ResponsesClient):
    REQUIRED_API_KEY = "AWS_BEARER_TOKEN_BEDROCK"
    REQUIRES_BASE_URL = True


class BedrockChatCompletionsClient(ChatCompletionsClient):
    REQUIRED_API_KEY = "AWS_BEARER_TOKEN_BEDROCK"
    REQUIRES_BASE_URL = True


class BedrockMessagesClient(AnthropicMessagesClient):
    """Anthropic-native Messages on Bedrock runtime or Mantle's /anthropic route."""

    REQUIRED_API_KEY = "AWS_BEARER_TOKEN_BEDROCK"
    REQUIRES_BASE_URL = True

    def _make_anthropic_client(self) -> AsyncAnthropic:
        if not self._api_key:
            raise AuthenticationError(
                f"Missing credentials. Set {self.REQUIRED_API_KEY} or pass api_key.",
                response=None,
                body=None,
                request=self._make_authentication_request("v1/messages"),
            )
        return AsyncAnthropic(
            api_key=self._api_key,
            # Bedrock Messages authenticates via x-api-key. Do not also send an
            # unrelated ANTHROPIC_AUTH_TOKEN inherited from the SDK environment.
            auth_token="",
            default_headers={"Authorization": omit},
            base_url=self._base_url,
            http_client=self._http_client,
        )
