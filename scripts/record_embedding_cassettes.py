#!/usr/bin/env python3
"""
Record matched embedding cassettes through `scripts/recorder_proxy.py` with each provider's SDK.

Run outside pytest from the repository root. The Cohere and Voyage SDKs are not project
dependencies, so add them for this run only:

```bash
uv run --with cohere==7.3.2 --with voyageai==0.6.0 \
    python scripts/record_embedding_cassettes.py --env-file .env
```

Each provider gets its own proxy process. The proxy and the SDKs do not retry,
so each cassette is one real upstream exchange. A failed request is reported and left on disk.
The raw recordings are aligned afterwards by `scripts/match_embedding_cassettes.py`.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import yaml

OUTPUT_DIR = Path("tests/cassettes/embeddings/matched")
QUERY = ["Where do penguins live?"]
DOCUMENTS = ["Penguins live in the Southern Hemisphere.", "Polar bears live in the Arctic."]
DIMENSIONS = 256
# The unbilled Voyage tier allows 3 requests per minute.
VOYAGE_PACING_SECONDS = 21

# Every matched request asks for no truncation: OpenAI rejects over-long input,
# so Cohere and Voyage are asked to do the same.
SCENARIOS = {
    "query-float": (QUERY, "float", "search_query", "query"),
    "query-base64": (QUERY, "base64", "search_query", "query"),
    "documents-float": (DOCUMENTS, "float", "search_document", "document"),
    "documents-base64": (DOCUMENTS, "base64", "search_document", "document"),
}


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            values[name.strip()] = value.strip().strip("\"'")
    return values


def wait_for_port(port: int, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"Recorder proxy did not start on port {port}.")


def record_provider(
    *,
    target: str,
    port: int,
    output_dir: Path,
    filename: str,
    calls: dict[str, Callable[[str], object]],
    secret: str,
) -> None:
    """Run one proxy for `target` and move each new recording into its scenario directory."""
    incoming = Path(tempfile.mkdtemp(prefix=".incoming-", dir=output_dir))
    proxy = subprocess.Popen(
        [
            sys.executable,
            "scripts/recorder_proxy.py",
            *("--target", target),
            *("--output-dir", str(incoming)),
            *("--host", "127.0.0.1"),
            *("--port", str(port)),
        ]
    )
    try:
        wait_for_port(port)
        for scenario, call in calls.items():
            try:
                call(f"http://127.0.0.1:{port}")
            except Exception as error:  # The proxy has already saved the failed exchange.
                print(f"{scenario}/{filename}: {type(error).__name__}")
            (recording,) = incoming.glob("*.yaml")
            raw = recording.read_bytes()
            assert secret.encode() not in raw
            status = yaml.safe_load(raw)["response"]["status_code"]
            destination = output_dir / scenario / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A move keeps the recorded bytes unchanged.
            shutil.move(recording, destination)
            print(f"{scenario}/{filename}: {status}")
    finally:
        proxy.terminate()
        proxy.wait(timeout=30)
        shutil.rmtree(incoming)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--providers", nargs="+", default=["openai", "cohere", "voyageai"])
    args = parser.parse_args()
    env = read_env(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "openai" in args.providers:
        from openai import OpenAI

        key = env["OPENAI_API_KEY"]

        def openai_call(texts, encoding):
            return lambda base: OpenAI(
                api_key=key, base_url=f"{base}/v1", max_retries=0
            ).embeddings.create(
                model="text-embedding-3-small",
                input=texts,
                dimensions=DIMENSIONS,
                encoding_format=encoding,
            )

        record_provider(
            target="https://api.openai.com",
            port=args.port,
            output_dir=args.output_dir,
            filename="openai-embeddings.yaml",
            calls={
                name: openai_call(texts, encoding)
                for name, (texts, encoding, _, _) in SCENARIOS.items()
            },
            secret=key,
        )

    if "cohere" in args.providers:
        import cohere

        key = env["COHERE_API_KEY"]

        def cohere_call(texts, encoding, input_type):
            return lambda base: cohere.ClientV2(api_key=key, base_url=base).embed(
                texts=texts,
                model="embed-v4.0",
                input_type=input_type,
                embedding_types=[encoding],
                output_dimension=DIMENSIONS,
                truncate="NONE",
                request_options={"max_retries": 0},
            )

        record_provider(
            target="https://api.cohere.com",
            port=args.port + 1,
            output_dir=args.output_dir,
            filename="cohere-cohere_embed.yaml",
            calls={
                name: cohere_call(texts, encoding, input_type)
                for name, (texts, encoding, input_type, _) in SCENARIOS.items()
            },
            secret=key,
        )

    if "voyageai" in args.providers:
        import voyageai

        key = env["VOYAGE_API_KEY"]

        # The Voyage SDK always requests base64 on the wire and decodes it locally,
        # so it can only produce the base64 scenarios.
        def voyage_call(texts, input_type):
            def call(base):
                time.sleep(VOYAGE_PACING_SECONDS)
                return voyageai.Client(api_key=key, base_url=f"{base}/v1", max_retries=0).embed(
                    texts,
                    model="voyage-4-lite",
                    input_type=input_type,
                    truncation=False,
                    output_dtype="float",
                    output_dimension=DIMENSIONS,
                )

            return call

        record_provider(
            target="https://api.voyageai.com",
            port=args.port + 2,
            output_dir=args.output_dir,
            filename="voyageai-voyage_embeddings.yaml",
            calls={
                name: voyage_call(texts, voyage_type)
                for name, (texts, encoding, _, voyage_type) in SCENARIOS.items()
                if encoding == "base64"
            },
            secret=key,
        )


if __name__ == "__main__":
    main()
