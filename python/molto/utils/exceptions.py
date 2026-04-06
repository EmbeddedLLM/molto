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


@docstring_message
class AuthorizationError(MoltoException):
    """You do not have the correct credentials."""


@docstring_message
class ExternalAuthError(MoltoException):
    """Authentication with external provider failed."""


@docstring_message
class ForbiddenError(MoltoException):
    """You do not have access to this resource."""


@docstring_message
class UpgradeTierError(MoltoException):
    """You have exhausted the allocations of your subscribed tier. Please upgrade."""


@docstring_message
class BudgetExceededError(MoltoException):
    """You have reached your budget limit. Wait for the next cycle or ask an administrator to increase it."""


@docstring_message
class InsufficientCreditsError(MoltoException):
    """Please ensure that you have sufficient credits."""


@docstring_message
class ResourceNotFoundError(MoltoException):
    """Resource with the specified name is not found."""


@docstring_message
class MethodNotAllowedError(MoltoException):
    """Method is not allowed."""


@docstring_message
class ResourceExistsError(MoltoException):
    """Resource with the specified name already exists."""


@docstring_message
class UnsupportedMediaTypeError(MoltoException):
    """This file type is unsupported."""


@docstring_message
class BadInputError(MoltoException):
    """Your input is invalid."""


class ResponsesAPIError(MoltoException):
    """Responses/OpenAI-style API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_type = str(error_type)
        self.param = param
        self.code = code


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
        reset_at: int,
        used: int | None = None,
        retry_after: int | None = None,
        meta: dict[str, Any] | None = None,
    ):
        super().__init__(*args)
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
        self.used = used
        self.retry_after = retry_after
        self.meta = meta


@docstring_message
class UnavailableError(MoltoException):
    """The requested functionality is unavailable."""


@docstring_message
class ServerBusyError(MoltoException):
    """The server is busy."""


@docstring_message
class ModelOverloadError(MoltoException):
    """The model is overloaded."""
