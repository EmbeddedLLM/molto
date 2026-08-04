"""Command line entry point for the internal Molto server."""

from __future__ import annotations

import argparse
import sys

from app import ServerSettings, create_app
from loguru import logger

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8017


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python extensions/server/run.py",
        description="Run the internal/testing HTTP server extension for Molto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log sanitized request and response payloads",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
    )
    return parser


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=sys.stderr.isatty(),
        backtrace=True,
        diagnose=False,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | <level>{message}</level>"
        ),
    )


def main() -> None:
    args = _parser().parse_args()
    _configure_logging(args.log_level)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")

    settings = ServerSettings.from_env()

    logger.info(
        "SERVER\n  host: {}\n  port: {}",
        args.host,
        args.port,
    )

    import uvicorn

    uvicorn.run(
        create_app(settings, verbose=args.verbose),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        date_header=False,
    )


if __name__ == "__main__":
    main()
