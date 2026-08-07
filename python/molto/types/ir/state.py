from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from molto.types.ir.base import IRModel, ProviderName, StableId


class StateReference(IRModel):
    """Opaque state whose meaning and validity remain owned by one provider."""

    id: StableId
    provider: ProviderName
    kind: Literal["response", "conversation", "compaction", "continuation"]
    value: str
    expires_at: int | None = None


class StateReferences(IRModel):
    provider: ProviderName
    values: tuple[StateReference, ...] = ()

    @model_validator(mode="after")
    def _validate_provider_affinity(self) -> StateReferences:
        seen: set[tuple[str, str]] = set()
        for reference in self.values:
            if reference.provider != self.provider:
                raise ValueError("Every state reference must be owned by the declared provider.")
            identity = (reference.kind, reference.id)
            if identity in seen:
                raise ValueError(f"Duplicate state reference {reference.kind}:{reference.id}.")
            seen.add(identity)
        return self
