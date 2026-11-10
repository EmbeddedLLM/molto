from typing import Any

import orjson
from pydantic import BaseModel

from molto.utils.types import JSONInput, JSONOutput


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
