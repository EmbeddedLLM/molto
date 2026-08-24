from functools import wraps

import httpx


def docstring_message(cls):
    """
    Decorates an exception to make its docstring its default message.
    https://stackoverflow.com/a/66491013
    """
    # Must use cls_init name, not cls.__init__ itself, in closure to avoid recursion
    cls_init = cls.__init__

    @wraps(cls.__init__)
    def wrapped_init(self, msg=cls.__doc__, *args, **kwargs):
        cls_init(self, msg, *args, **kwargs)

    cls.__init__ = wrapped_init
    return cls


@docstring_message
class MoltoException(Exception):
    """Base exception class for Molto errors."""

    def __init__(self, message: str, *, response: httpx.Response, body: object | None) -> None:
        super().__init__(message)
        self.message = message
        self.response = response
        self.body = body


@docstring_message
class ExternalAuthError(MoltoException):
    """401: Authentication with external provider failed."""


@docstring_message
class InsufficientCreditsError(MoltoException):
    """403: Insufficient credits."""


@docstring_message
class ModelNotFoundError(MoltoException):
    """404: Model not found."""
