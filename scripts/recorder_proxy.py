#!/usr/bin/env python3
"""
HTTP Proxy with Request/Response Recording
Records all traffic to YAML files for later analysis.

```bash
python scripts/recorder_proxy.py
```
"""

import argparse
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from anyio import CancelScope
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from httpx import URL, AsyncClient
from loguru import logger
from starlette.types import Send
from uuid_utils import uuid7
from yaml import dump as yaml_dump

# Unreviewed captures must never enter the matched fixture corpus.
OUTPUT_DIR = Path("/tmp/molto-recordings")
TARGET_HOST = "https://api.openai.com"
TIMEOUT = 60 * 5  # 5 minutes

# Each exchange is evidence: retries would conceal the original provider failure.
# Native SDK callers must likewise set max_retries=0.
HTTP_CLIENT = AsyncClient(timeout=TIMEOUT)

# Headers to record (whitelist)
RECORDED_HEADERS = {
    "content-type",
    "content-length",
    "authorization",
    "user-agent",
    "accept",
    "accept-encoding",
    "cache-control",
    "connection",
    # Helpful for grouping multi-request tool loops into a single "run"
    "x-run-id",
    "x-request-id",
    "request-id",
    "retry-after",
}

# Headers to exclude when forwarding response (they're handled by httpx/client)
EXCLUDED_RESPONSE_HEADERS = {
    "content-encoding",  # Already decoded by httpx
    "content-length",  # Will be recalculated
    "transfer-encoding",  # Handled by the server
    "connection",  # Let the server handle this
}


def mask_authorization(value: str) -> str:
    """
    Mask Authorization header values in recordings.

    Cassettes are intended to be shareable "golden" artifacts; never write any
    API key material to disk (even partially).
    """
    if not value:
        return value
    lower = value.lower()
    if lower.startswith("bearer "):
        return "Bearer ***"
    return "***"


def filter_headers(headers) -> dict:
    """Filter headers to only include whitelisted ones."""
    return {
        k: v if k.lower() != "authorization" else mask_authorization(v)
        for k, v in headers.items()
        if k.lower() in RECORDED_HEADERS
    }


def filter_response_headers(headers) -> dict:
    """Filter response headers to exclude problematic ones."""
    return {k: v for k, v in headers.items() if k.lower() not in EXCLUDED_RESPONSE_HEADERS}


def save_recording(recording: dict[str, Any], filename: str):
    """Save recording to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    with open(filepath.with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml_dump(recording, f, allow_unicode=True, encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await HTTP_CLIENT.aclose()


app = FastAPI(title="Recording Proxy", lifespan=lifespan)


class RecordingStreamingResponse(StreamingResponse):
    def __init__(self, *args, on_close: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(*args, **kwargs)
        self.on_close = on_close

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            # A disconnect during send leaves the generator suspended at yield.
            # Close it explicitly so its recording is saved before returning.
            with CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await self.on_close()


def decode_body(body: bytes, content_type: str) -> Any:
    """Retain malformed or non-JSON bodies without inventing a JSON representation."""
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type == "application/json" or media_type.endswith("+json"):
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError):
            pass
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body


def mask_secrets(value: Any, secrets: list[str]) -> Any:
    """Mask known request credentials, including credentials echoed by an upstream error."""
    if isinstance(value, dict):
        return {key: mask_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_secrets(item, secrets) for item in value]
    if isinstance(value, (str, bytes)):
        for secret in secrets:
            if isinstance(value, bytes):
                value = value.replace(secret.encode(), b"***")
            else:
                value = value.replace(secret, "***")
    return value


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_request(request: Request, path: str) -> Response:
    """Forward one exchange without retries and retain its observed completion state."""
    request_id = str(uuid7())
    now = datetime.now(timezone.utc)
    filename = f"{now.strftime('%Y-%m-%d')}-{request_id}"
    logger.info("Recording exchange to {}", OUTPUT_DIR / filename)
    target = URL(TARGET_HOST)
    if target.username or target.password or target.query:
        raise ValueError("Configure credentials on the SDK, not in the proxy target URL")
    target_url = f"{TARGET_HOST.rstrip('/')}/{path}"
    secrets = [
        value
        for name, value in request.headers.items()
        if name.lower() in {"authorization", "x-api-key", "api-key"} and value
    ]
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        secrets.append(authorization[7:])
    secrets.extend(
        value
        for name, value in request.query_params.multi_items()
        if name.lower() in {"key", "api_key", "api-key", "access_token"} and value
    )
    body = await request.body()
    parsed_body = decode_body(body, request.headers.get("content-type", ""))
    recording: dict[str, Any] = {
        "timestamp": now.isoformat(),
        "request_id": request_id,
        "filename": filename,
        "upstream_url": target_url,
        "request": {
            "method": request.method,
            "path": f"/{path}",
            "query_params": dict(request.query_params),
            "headers": filter_headers(request.headers),
            "body": parsed_body,
        },
        "response": {"status_code": None, "headers": {}},
    }
    new_headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    upstream_request = HTTP_CLIENT.build_request(
        request.method,
        target_url,
        params=request.query_params.multi_items(),
        headers=new_headers,
        content=body,
        timeout=TIMEOUT,
    )
    try:
        response = await HTTP_CLIENT.send(upstream_request, stream=True)
    except Exception as exc:
        # Exception text can contain credential-bearing URLs. Retain the class only.
        recording["response"]["transport_error"] = type(exc).__name__
        save_recording(mask_secrets(recording, secrets), filename)
        raise
    result = recording["response"]
    result["status_code"] = response.status_code
    result["headers"] = filter_headers(response.headers)
    media_type = response.headers.get("content-type", "")
    if media_type.partition(";")[0].strip().lower() == "text/event-stream":
        result.update(sse=[], stream_status="disconnected", terminal_event=None, error_event=False)

        async def close_stream() -> None:
            await response.aclose()
            save_recording(mask_secrets(recording, secrets), filename)

        async def stream() -> AsyncGenerator[str, None]:
            try:
                async for line in response.aiter_lines():
                    # This is ordered SSE-line evidence, not original TCP/newline bytes.
                    result["sse"].append(f"{line}\n")
                    if line.startswith("event:"):
                        event_type = line.removeprefix("event:").strip()
                        if event_type in {
                            "response.completed",
                            "response.incomplete",
                            "response.failed",
                            "message_stop",
                        }:
                            result["terminal_event"] = event_type
                        if event_type == "error":
                            result["error_event"] = True
                    if line.startswith("data:"):
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            result["terminal_event"] = "[DONE]"
                        else:
                            try:
                                event = json.loads(data)
                            except ValueError:
                                event = None
                            if isinstance(event, dict) and (
                                event.get("type") == "error" or "error" in event
                            ):
                                result["error_event"] = True
                    yield f"{line}\n"
                result["stream_status"] = "complete"
            except Exception as exc:
                result["stream_status"] = "transport_error"
                result["transport_error"] = type(exc).__name__
                raise

        return RecordingStreamingResponse(
            stream(),
            on_close=close_stream,
            status_code=response.status_code,
            headers=filter_response_headers(response.headers),
        )

    chunks = []
    try:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
    except Exception as exc:
        result["transport_error"] = type(exc).__name__
        raise
    finally:
        # Retain bytes received before a read failure as incomplete evidence too.
        response_body = b"".join(chunks)
        result["body"] = decode_body(response_body, media_type)
        result["body_bytes"] = response_body
        with CancelScope(shield=True):
            await response.aclose()
            save_recording(mask_secrets(recording, secrets), filename)
    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=filter_response_headers(response.headers),
    )


def main():
    """Run the recording proxy server."""
    global OUTPUT_DIR, TARGET_HOST

    parser = argparse.ArgumentParser(
        description="HTTP Proxy with Request/Response Recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--target",
        "-t",
        default=TARGET_HOST,
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    TARGET_HOST = args.target.rstrip("/")
    target = URL(TARGET_HOST)
    if target.username or target.password or target.query:
        parser.error("Configure credentials on the SDK, not in the proxy target URL")

    print("Recording Proxy Starting (no retries)...")
    print(f"   Proxy: http://{args.host}:{args.port}")
    print(f"   Target: {TARGET_HOST}")
    print(f"   Output: {OUTPUT_DIR.absolute()}")
    print("   Press CTRL+C to stop\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        # Access logs include the raw query string, which can contain credentials.
        access_log=False,
    )


if __name__ == "__main__":
    main()
