import time
from typing import Any

from uuid_utils import uuid7 as _uuid7


def get_non_empty(mapping: dict[str, Any], key: str, default: Any):
    value = mapping.get(key, None)
    return value if value else default


def uuid7_str(prefix: str = "") -> str:
    return f"{prefix}{_uuid7()}"


def get_ttl_hash(seconds: int = 3600) -> int:
    """Return the same value within `seconds` time period"""
    return round(time.time() / max(1, seconds))


def mask_string(x: str | None, *, include_len: bool = True) -> str | None:
    if x is None or x == "":
        return x
    str_len = len(x)
    if str_len < 4:
        return f"{'*' * str_len} ({str_len=})"
    visible_len = min(100, str_len // 5)
    x = f"{x[:visible_len]}***{x[-visible_len:]}"
    return f"{x} ({str_len=})" if include_len else x
