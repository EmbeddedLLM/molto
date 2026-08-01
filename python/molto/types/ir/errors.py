from __future__ import annotations

from .base import IRModel, ProviderName


class ErrorDetail(IRModel):
    code: str
    message: str
    parameter: str | None = None
    retryable: bool | None = None
    provider: ProviderName | None = None
