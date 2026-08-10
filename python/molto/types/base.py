from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from molto.configs import ENV_CONFIG


class _Model(BaseModel):
    model_config = ConfigDict(
        extra=ENV_CONFIG.pydantic_extra_behavior,
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize only populated fields."""
        kwargs.setdefault("exclude_unset", True)
        kwargs.setdefault("mode", "json")
        return super().model_dump(*args, **kwargs)

    # @classmethod
    # def model_validate(
    #     cls,
    #     obj: Any,
    #     *,
    #     strict: bool | None = None,
    #     extra: ExtraValues | None = None,
    #     from_attributes: bool | None = None,
    #     context: Any | None = None,
    #     by_alias: bool | None = None,
    #     by_name: bool | None = None,
    # ):
    #     if isinstance(obj, BaseModel):
    #         obj = obj.model_dump()
    #     elif isinstance(obj, dict):
    #         pass
    #     else:
    #         raise ValueError(f"Expected dict or BaseModel, got {type(obj)}")
    #     return super().model_validate(
    #         obj,
    #         strict=strict,
    #         extra=extra,
    #         from_attributes=from_attributes,
    #         context=context,
    #         by_alias=by_alias,
    #         by_name=by_name,
    #     )


class NormalizedModel(_Model):
    extra_fields: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Provider-specific or otherwise non-normalized fields "
                "to preserve or pass through when constructing the upstream request."
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
