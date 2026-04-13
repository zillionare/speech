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
from tts_service.first_run import maybe_run_first_run
import uvicorn


def main():
    import argparse

    parser = argparse.ArgumentParser(description='TTS Server Runner')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--skip-first-run', action='store_true',
                        help='跳过首次运行向导（使用已有配置）')

    args = parser.parse_args()

    # 首次运行检查与配置
    if args.skip_first_run:
        from tts_service.config import load_config
        config = load_config(args.config)
        from tts_service.first_run import setup_environment
        setup_environment(config)
    else:
        config = maybe_run_first_run(args.config)

    # 如果配置获取失败（用户取消了向导），退出
    if config is None:
        print("配置未完成，退出。")
        sys.exit(1)

    print(f"Starting server with config: {args.config}")
    print(f"  Host: {config.server.host}:{config.server.port}")
    print(f"  Model: {config.model.name}")
    print(f"  Model cache: {config.model.expanded_cache_dir}")
    print(f"  Voices dir: {config.samples.expanded_base_dir}")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置（使用官方 HuggingFace）')}")

    app = create_app(args.config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level
    )


if __name__ == "__main__":
    main()
