from functools import wraps
from typing import Any


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
    """Base exception class for errors."""

    status_code: int | None = None
    request_id: str | None = None
    provider_code: str | None = None
    provider_body: Any = None


@docstring_message
class ExternalAuthError(MoltoException):
    """Authentication with external provider failed."""


@docstring_message
class ForbiddenError(MoltoException):
    """You do not have access to this resource."""


@docstring_message
class ResourceNotFoundError(MoltoException):
    """Resource with the specified name is not found."""


@docstring_message
class BadInputError(MoltoException):
    """Your input is invalid."""


@docstring_message
class ModelCapabilityError(BadInputError):
    """No model has the specified capabilities."""


@docstring_message
class ContextOverflowError(MoltoException):
    """Model's context length has been exceeded."""


@docstring_message
class UnexpectedError(MoltoException):
    """We ran into an unexpected error."""


@docstring_message
class RateLimitExceedError(MoltoException):
    """The rate limit is exceeded."""

    def __init__(
        self,
        *args,
        limit: int,
        remaining: int,
    ):
        super().__init__(*args)
        self.limit = limit
        self.remaining = remaining


@docstring_message
class ServerBusyError(MoltoException):
    """The server is busy."""


@docstring_message
class ModelOverloadError(MoltoException):
    """The model is overloaded."""
