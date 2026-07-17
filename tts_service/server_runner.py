#!/usr/bin/env python3
"""Background runner used by the CLI."""

from __future__ import annotations

import argparse

import uvicorn

from .config import load_config
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeVoice MLX Studio server runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--host", type=str, default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument("--ssl-keyfile", type=str, default=None, help="TLS private key")
    parser.add_argument("--ssl-certfile", type=str, default=None, help="TLS certificate")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(args.config)
    uvicorn.run(
        app,
        host=args.host or config.server.host,
        port=args.port or config.server.port,
        log_level=config.server.log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
