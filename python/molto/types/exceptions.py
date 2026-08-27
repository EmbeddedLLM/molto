from httpx import Response


class MoltoException(Exception):
    """Base exception class for Molto errors."""

    def __init__(self, message: str, *, response: Response, body: dict | None) -> None:
        super().__init__(message)
        if not isinstance(response, Response):
            raise TypeError(f"Expected `httpx.Response`, got {type(response)}")
        if body is not None and not isinstance(body, dict):
            raise TypeError(f"Expected `dict`, got {type(body)}")
        self.message = message
        self.response = response
        self.body = body


class AuthenticationError(MoltoException):
    """Authentication failed (401)"""


class InsufficientCreditsError(MoltoException):
    """Insufficient credits (403)"""


class NotFoundError(MoltoException):
    """Requested resource was not found (404)"""


class ModelNotFoundError(NotFoundError):
    """Model not found (400 or 404)"""
