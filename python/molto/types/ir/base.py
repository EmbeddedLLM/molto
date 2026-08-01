from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

StableId = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
ProviderName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*$")]
ItemIndex = Annotated[int, Field(ge=0)]
ContentIndex = Annotated[int, Field(ge=0)]
SequenceNumber = Annotated[int, Field(ge=0)]


class IRModel(BaseModel):
    """Strict, immutable base for the portable IR."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ItemStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self is not ItemStatus.IN_PROGRESS


class ResponseStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self is not ResponseStatus.IN_PROGRESS


AssistantPhase = Literal["commentary", "final_answer"]
