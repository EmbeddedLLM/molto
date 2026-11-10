from collections import OrderedDict
from typing import Any

# Superficial JSON types; inputs also accept tuples and ordered mappings.
JSONOutput = str | int | float | bool | None | dict[str, Any] | list[Any]
JSONInput = JSONOutput | tuple[Any, ...] | OrderedDict
