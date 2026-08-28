from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def _none_to_zero(v: int | None) -> int:
    if v is None:
        return 0
    return v


ZeroIfNoneInt = Annotated[int, BeforeValidator(_none_to_zero)]


class ExcludeNoneModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    def model_dump_populated(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize only populated fields."""
        return super().model_dump(
            *args,
            **kwargs,
            exclude_unset=True,
            exclude_none=True,
            mode="json",
        )

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Serialize only populated fields.

        If possible, use `model_dump_populated` instead.
        This serves as a fallback.
        """
        kwargs.setdefault("exclude_unset", True)
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("mode", "json")
        return super().model_dump(*args, **kwargs)


class ExtraModel(ExcludeNoneModel):
    extra_fields: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Provider-specific or otherwise non-normalized fields "
                "observed while handling output."
            ),
        ),
    ] = {}

    @model_validator(mode="before")
    @classmethod
    def fill_in_extra_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        model_fields = cls.model_fields.keys()
        return {
            **{k: v for k, v in data.items() if k in model_fields},
            "extra_fields": {k: v for k, v in data.items() if k not in model_fields},
        }


class StrictModel(ExcludeNoneModel):
    model_config = ConfigDict(extra="forbid")
