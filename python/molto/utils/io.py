from collections.abc import Iterable
from typing import Any

import orjson
from anthropic import BaseModel as AnthropicModel
from openai import BaseModel as OpenAIModel
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from molto.types.base import ExcludeNoneModel
from molto.utils.types import JSONInput, JSONOutput


def request_to_dict(body: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Serialize native input while retaining extensions on Molto request models."""

    def convert(value: Any) -> Any:
        if isinstance(value, (OpenAIModel, AnthropicModel)):
            return value.to_dict(mode="json")
        if isinstance(value, ExcludeNoneModel):
            # Schema-based dumps can omit extras accepted by a per-call override,
            # including keys inside TypedDict fields. Retain those values while
            # preserving Molto's omission of unset and None-valued declared fields.
            fields = {
                key: item
                for key, item in value
                if key in (value.model_extra or {})
                or (key in value.model_fields_set and item is not None)
            }
            return to_jsonable_python(convert(fields))
        elif isinstance(value, BaseModel):
            # Caller models own aliases and serializers; meaningful default values
            # must survive even when the caller did not explicitly set the fields.
            serialized = value.model_dump(mode="json", by_alias=True, exclude_none=True)
            extras = {
                key: item
                for key, item in (value.model_extra or {}).items()
                if key not in serialized
            }
            return {**serialized, **to_jsonable_python(convert(extras))}
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        # SDK array parameters accept iterables. Materialize them once so metadata
        # checks can inspect history without exhausting it before transport.
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [convert(item) for item in value]
        return value

    # Raw dictionaries may carry SDK omission markers that are not JSON values.
    return convert(body)


def json_loads(data: str) -> JSONOutput:
    return orjson.loads(data)


def json_dumps(data: JSONInput, **kwargs) -> str:
    return orjson.dumps(data, **kwargs).decode("utf-8")


def read_json(path: str) -> JSONOutput:
    """Reads a JSON file.

    Args:
        path (str): Path to the file.

    Returns:
        data (JSONOutput): The data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return orjson.loads(f.read())


def read_yaml(path: str) -> JSONOutput:
    """Reads a YAML file.

    Args:
        path (str): Path to the file.

    Returns:
        data (JSONOutput): The data.
    """
    import yaml  # Test-only dependency, so not imported at module scope.

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(
    data: JSONInput,
    path: str,
    allow_unicode: bool = True,
    encoding: str = "utf-8",
    **kwargs,
) -> str:
    """Writes a YAML file.

    Args:
        data (JSONInput): The data.
        path (str): Path to the file.
        allow_unicode (bool, optional): See `yaml.dump`.
        encoding (str, optional): See `yaml.dump`.
        **kwargs: Other keyword arguments to pass into `yaml.dump`.

    Returns:
        path (str): Path to the file.
    """
    import yaml

    # PyYAML ignores `encoding` for text streams, so the file itself must use it.
    with open(path, "w", encoding=encoding) as f:
        yaml.dump(data, f, allow_unicode=allow_unicode, encoding=encoding, **kwargs)
    return path


def load_sse(sse: str) -> dict[str, Any]:
    data = "\n".join(
        line.removeprefix("data:").lstrip()
        for line in sse.splitlines()
        if line.startswith("data:")
    )
    return json_loads(data)


def dumps_sse(data: BaseModel | dict[str, Any], event_type: str = "") -> str:
    if isinstance(data, BaseModel):
        data_json = data.model_dump_json(exclude_unset=True)
    else:
        data_json = json_dumps(data)
    if event_type:
        return f"event: {event_type}\ndata: {data_json}\n\n"
    else:
        return f"data: {data_json}\n\n"
