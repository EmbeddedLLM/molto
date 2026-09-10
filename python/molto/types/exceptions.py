from httpx import Request, Response


class UnsupportedInput(ValueError):
    """A cross-protocol input cannot be translated without losing its semantics."""


class APIException(Exception):
    """Base exception class for normalized API errors."""

    def __init__(
        self,
        message: str,
        *,
        response: Response | None,
        body: dict | str | None,
        request: Request | None = None,
    ) -> None:
        super().__init__(message)
        if response is not None and not isinstance(response, Response):
            raise TypeError(f"Expected `httpx.Response`, got {type(response)}")
        # SDK error bodies are raw text when the provider error is not a JSON
        # object, e.g. an empty router 404 from a wrong URL.
        if body is not None and not isinstance(body, dict | str):
            raise TypeError(f"Expected `dict` or `str`, got {type(body)}")
        self.message = message
        self.response = response
        self.body = body
        self.request = response.request if response is not None else request


class AuthenticationError(APIException):
    """Authentication failed (HTTP 401 or missing SDK credentials)."""


class ContextLengthExceededError(APIException):
    """Context window exceeded (HTTP 400 or SSE; OpenAI code `context_length_exceeded`)."""


class InsufficientCreditsError(APIException):
    """Insufficient credits (403)"""


class ResourceNotFoundError(APIException):
    """Requested resource was not found (404)"""


class ModelNotFoundError(APIException):
    """Model not found (400 or 404)"""
