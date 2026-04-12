#!/usr/bin/env python3
"""
服务器运行器 - 用于后台启动服务
通过 python -m tts_service.server_runner --config x.yaml 调用
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/aaronyang/workspace/speech')

from tts_service.server import create_app
from tts_service.config import load_config
import uvicorn


def main():
    import argparse

    parser = argparse.ArgumentParser(description='TTS Server Runner')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--overrides', type=str, default=None,
                        help='Config overrides as dict string (not recommended)')

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    print(f"Starting server with config: {args.config}")
    print(f"  Host: {config.server.host}:{config.server.port}")
    print(f"  Model: {config.model.name}")

    app = create_app(args.config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level
    )


if __name__ == "__main__":
    main()
