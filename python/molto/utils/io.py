import pickle
from typing import Any

import orjson
import yaml
from httpx import AsyncClient, HTTPStatusError, Response
from pydantic import BaseModel
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from molto.utils.types import JSONInput, JSONOutput

DOC_WHITE_LIST = {
    "application/pdf": [".pdf"],
    "application/xml": [".xml"],
    "application/json": [".json"],
    "application/jsonl": [".jsonl"],
    "application/x-ndjson": [".jsonl"],
    "application/json-lines": [".jsonl"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "text/markdown": [".md"],
    "text/plain": [".txt"],
    "text/html": [".html"],
    "text/tab-separated-values": [".tsv"],
    "text/csv": [".csv"],
    "text/xml": [".xml"],
}
IMAGE_WHITE_LIST = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/gif": [".gif"],
    "image/webp": [".webp"],
}
AUDIO_WHITE_LIST = {
    "audio/mpeg": [".mp3"],
    "audio/wav": [".wav"],
    "audio/x-wav": [".wav"],
    "audio/x-pn-wav": [".wav"],
    "audio/wave": [".wav"],
    "audio/vnd.wav": [".wav"],
    "audio/vnd.wave": [".wav"],
}
UPLOAD_WHITE_LIST = {**DOC_WHITE_LIST, **IMAGE_WHITE_LIST, **AUDIO_WHITE_LIST}
UPLOAD_WHITE_LIST_MIME = set(UPLOAD_WHITE_LIST.keys())
# Use the first MIME for each extension
MIME_WHITE_LIST = {**DOC_WHITE_LIST, **IMAGE_WHITE_LIST, **AUDIO_WHITE_LIST}
EXT_TO_MIME = {}
for mime, exts in MIME_WHITE_LIST.items():
    for ext in exts:
        EXT_TO_MIME[ext] = EXT_TO_MIME.get(ext, mime)


def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def dump_pickle(out_path: str, obj: Any):
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)


def read_json(path: str) -> JSONOutput:
    """Reads a JSON file.

    Args:
        path (str): Path to the file.

    Returns:
        data (JSONOutput): The data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return orjson.loads(f.read())


def dump_json(data: JSONInput, path: str, **kwargs) -> str:
    """Writes a JSON file.

    Args:
        data (JSONInput): The data.
        path (str): Path to the file.
        **kwargs: Other keyword arguments to pass into `orjson.dumps`.

    Returns:
        path (str): Path to the file.
    """
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, **kwargs))
    return path


def json_loads(data: str) -> JSONOutput:
    return orjson.loads(data)


def json_dumps(data: JSONInput, **kwargs) -> str:
    return orjson.dumps(data, **kwargs).decode("utf-8")


def read_yaml(path: str) -> JSONOutput:
    """Reads a YAML file.

    Args:
        path (str): Path to the file.

    Returns:
        data (JSONOutput): The data.
    """
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


def truncate_utf8(text: str, *, max_bytes: int = 50 * 1024) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n\n(Output truncated.)"


def _should_retry_status(response: Response) -> None:
    """Raise exceptions for retryable HTTP status codes."""
    if response.status_code in (429, 502, 503, 504):
        response.raise_for_status()  # This will raise HTTPStatusError


def get_async_client(
    *,
    timeout: float = 30.0,
    follow_redirects: bool = False,  # Prevent redirect-based SSRF
    max_redirects: int = 0,
    retry_config: RetryConfig | None = None,
) -> AsyncClient:
    if retry_config is None:
        retry_config = RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300,
            ),
            # Stop after 5 attempts
            stop=stop_after_attempt(5),
            # Re-raise the last exception if all retries fail
            reraise=True,
        )
    client = AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        max_redirects=max_redirects,
        transport=AsyncTenacityTransport(
            config=retry_config,
            validate_response=_should_retry_status,
        ),
    )
    return client


HTTP_ACLIENT = get_async_client()
