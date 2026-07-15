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
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(args.config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level=config.server.log_level)


if __name__ == "__main__":
    main()
