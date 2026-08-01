"""JSON and YAML helpers for the test suite."""

from collections import OrderedDict
from typing import Any, Union

import orjson
import yaml

# Superficial JSON input/output types
# https://github.com/python/typing/issues/182#issuecomment-186684288
JSONOutput = Union[str, int, float, bool, None, dict[str, Any], list[Any]]
# For input, we also accept tuples, ordered dicts etc.
JSONInput = Union[
    str, int, float, bool, None, dict[str, Any], list[Any], tuple[Any, ...], OrderedDict
]


def json_loads(data: str) -> JSONOutput:
    return orjson.loads(data)


def json_dumps(data: JSONInput, **kwargs) -> str:
    return orjson.dumps(data, **kwargs).decode("utf-8")


def read_yaml(path: str) -> JSONOutput:
    """Reads a YAML file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        data (JSONOutput): The data.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
