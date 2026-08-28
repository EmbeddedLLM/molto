from httpx import Response


class UnsupportedInput(ValueError):
    """A cross-protocol input cannot be translated without losing its semantics."""


class APIException(Exception):
    """Base exception class for normalized API errors."""

    def __init__(self, message: str, *, response: Response, body: dict | None) -> None:
        super().__init__(message)
        if not isinstance(response, Response):
            raise TypeError(f"Expected `httpx.Response`, got {type(response)}")
        if body is not None and not isinstance(body, dict):
            raise TypeError(f"Expected `dict`, got {type(body)}")
        self.message = message
        self.response = response
        self.body = body


class AuthenticationError(APIException):
    """Authentication failed (401)"""


class InsufficientCreditsError(APIException):
    """Insufficient credits (403)"""


class ResourceNotFoundError(APIException):
    """Requested resource was not found (404)"""


class ModelNotFoundError(ResourceNotFoundError):
    """Model not found (400 or 404)"""
