#!/usr/bin/env python3
"""
TTS Web Service CLI - 使用 Python Fire

命令:
    python -m tts_service.cli start [--config config.yaml] [--port 8123]
    python -m tts_service.cli stop
    python -m tts_service.cli restart [--config config.yaml]
    python -m tts_service.cli status
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional

try:
    import fire
    FIRE_AVAILABLE = True
except ImportError:
    FIRE_AVAILABLE = False

from .config import load_config
from .first_run import maybe_run_first_run, setup_environment


class ServerManager:
    """服务器进程管理器"""

    def __init__(self, pid_file: str = "/tmp/tts_service.pid"):
        self.pid_file = pid_file

    def get_pid(self) -> Optional[int]:
        """获取服务进程 ID"""
        if not os.path.exists(self.pid_file):
            return None
        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None

    def save_pid(self, pid: int):
        """保存服务进程 ID"""
        os.makedirs(os.path.dirname(self.pid_file), exist_ok=True)
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))

    def remove_pid(self):
        """移除 PID 文件"""
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)

    def is_running(self) -> bool:
        """检查服务是否运行中"""
        pid = self.get_pid()
        if pid is None:
            return False

        # 检查进程是否存在
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            self.remove_pid()
            return False

    def stop(self) -> bool:
        """停止服务"""
        pid = self.get_pid()
        if pid is None:
            print("TTS Service is not running")
            return False

        try:
            # 尝试优雅终止
            os.kill(pid, signal.SIGTERM)

            # 等待进程终止
            for _ in range(30):  # 最多等待 3 秒
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except (OSError, ProcessLookupError):
                    self.remove_pid()
                    print(f"TTS Service stopped (PID: {pid})")
                    return True

            # 强制终止
            os.kill(pid, signal.SIGKILL)
            print(f"TTS Service killed (PID: {pid})")
            self.remove_pid()
            return True

        except (OSError, ProcessLookupError):
            print(f"Process {pid} not found")
            self.remove_pid()
            return False


class TTSCLI:
    """TTS Web Service CLI"""

    def __init__(self):
        self.manager = ServerManager()

    def start(
        self,
        config: str = "config.yaml",
        host: Optional[str] = None,
        port: Optional[int] = None,
        model: Optional[str] = None,
        voices_dir: Optional[str] = None,
        daemon: bool = True,
        no_daemon: bool = False,
        skip_first_run: bool = False
    ):
        """
        启动 TTS 服务

        Args:
            config: 配置文件路径 (默认: config.yaml)
            host: 服务器地址 (覆盖配置文件)
            port: 服务器端口 (覆盖配置文件)
            model: TTS 模型名称 (覆盖配置文件)
            voices_dir: 声音样本目录 (覆盖配置文件)
            daemon: 后台运行 (默认: True)
            no_daemon: 前台运行模式 (与 daemon=False 等效)
            skip_first_run: 跳过首次运行向导
        """
        # 处理 --no-daemon 参数
        if no_daemon:
            daemon = False
        if self.manager.is_running():
            pid = self.manager.get_pid()
            print(f"TTS Service is already running (PID: {pid})")
            return

        # 首次运行配置
        if not skip_first_run:
            cfg = maybe_run_first_run(config)
            if cfg is None:
                print("配置未完成，退出。")
                return
        else:
            # 加载配置
            overrides = {}
            if host:
                overrides['server'] = {'host': host}
            if port:
                overrides.setdefault('server', {})['port'] = port
            if model:
                overrides.setdefault('model', {})['name'] = model
            if voices_dir:
                overrides.setdefault('samples', {})['base_dir'] = voices_dir

            try:
                cfg = load_config(config, **overrides)
            except FileNotFoundError as e:
                print(f"错误：{e}")
                print(f"将在 {config} 创建默认配置...")
                self._create_default_config(config)
                cfg = load_config(config, **overrides)

            # 依然要设置环境变量
            setup_environment(cfg)

        print(f"Starting TTS Service...")
        print(f"  Config: {config}")
        print(f"  Host: {cfg.server.host}:{cfg.server.port}")
        print(f"  Model: {cfg.model.name}")
        print(f"  Voices: {cfg.samples.expanded_base_dir}")

        if daemon:
            # 后台启动 - 首次运行配置已经在前端完成
            cmd = [
                sys.executable, "-m", "tts_service.server_runner",
                "--config", os.path.abspath(config),
                "--skip-first-run"  # 后台模式跳过首次运行检查
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            self.manager.save_pid(proc.pid)
            print(f"TTS Service started (PID: {proc.pid})")
            print(f"Health check: http://{cfg.server.host}:{cfg.server.port}/health")
            print(f"API docs: http://{cfg.server.host}:{cfg.server.port}/docs")
        else:
            # 前台运行
            from .server import create_app
            import uvicorn

            app = create_app(config)
            print(f"Running in foreground mode...")
            uvicorn.run(
                app,
                host=cfg.server.host,
                port=cfg.server.port,
                log_level=cfg.server.log_level
            )

    def stop(self):
        """停止 TTS 服务"""
        self.manager.stop()

    def restart(
        self,
        config: str = "config.yaml",
        host: Optional[str] = None,
        port: Optional[int] = None,
        model: Optional[str] = None,
        voices_dir: Optional[str] = None,
        no_daemon: bool = False
    ):
        """
        重启 TTS 服务

        Args:
            config: 配置文件路径
            host: 服务器地址
            port: 服务器端口
            model: TTS 模型名称
            voices_dir: 声音样本目录
            no_daemon: 前台运行模式
        """
        print("Restarting TTS Service...")
        self.stop()
        time.sleep(1)
        self.start(config, host, port, model, voices_dir, not no_daemon, no_daemon)

    def status(self):
        """显示服务状态"""
        pid = self.manager.get_pid()

        if pid is None:
            print("TTS Service: NOT RUNNING")
            return

        if self.manager.is_running():
            print(f"TTS Service: RUNNING (PID: {pid})")

            # 尝试获取更多信息
            try:
                cfg = load_config("config.yaml")
                print(f"  Health: http://{cfg.server.host}:{cfg.server.port}/health")
                print(f"  API: http://{cfg.server.host}:{cfg.server.port}/docs")
            except:
                pass
        else:
            print(f"TTS Service: NOT RUNNING (stale PID: {pid})")
            self.manager.remove_pid()

    def _create_default_config(self, path: str):
        """创建默认配置文件"""
        default_config = '''# TTS Web Service 配置文件

# 首次运行配置
first_run:
  # HuggingFace 镜像地址（中国大陆推荐用 hf-mirror.com）
  hf_endpoint: "https://hf-mirror.com"
  # 是否跳过首次运行提示
  skip_welcome: false

# 模型配置
model:
  # 模型名称 (HuggingFace model ID)
  name: "mlx-community/VibeVoice-Realtime-0.5B-4bit"
  # 模型缓存目录
  cache_dir: "~/.cache/tts_models"
  # 默认推理步数 (数值越低越快，质量可能降低)
  num_steps: 10
  # CFG scale (控制生成多样性)
  cfg_scale: 1.3

# 声音样本配置
samples:
  # 声音样本目录 (.wav 文件放在这里)
  base_dir: "./voices"
  # 限定允许的声音列表 (为空则允许所有)
  allowed_speakers: []

# 服务器配置
server:
  host: "0.0.0.0"
  port: 8123
  workers: 1
  log_level: "info"

# 默认生成参数
defaults:
  voice: "zh-Aaron_man"
  response_format: "wav"
  speed: 1.0

pid_file: "/tmp/tts_service.pid"
'''
        with open(path, 'w') as f:
            f.write(default_config)
        print(f"Created: {path}")


def main():
    """CLI 入口点"""
    if FIRE_AVAILABLE:
        fire.Fire(TTSCLI)
    else:
        print("Python Fire not installed. Using argparse fallback.")
        print("Install with: pip install python-fire")
        run_argparse_cli()


def run_argparse_cli():
    """使用 argparse 的 CLI 替代方案"""
    import argparse

    parser = argparse.ArgumentParser(description='TTS Web Service CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # start 命令
    start_parser = subparsers.add_parser('start', help='Start TTS service')
    start_parser.add_argument('--config', default='config.yaml', help='Config file path')
    start_parser.add_argument('--host', type=str, help='Server host')
    start_parser.add_argument('--port', type=int, help='Server port')
    start_parser.add_argument('--model', type=str, help='TTS model name')
    start_parser.add_argument('--voices-dir', type=str, help='Voices directory')
    start_parser.add_argument('--daemon', action='store_true', default=True, help='Run in background')
    start_parser.add_argument('--no-daemon', action='store_false', dest='daemon', help='Run in foreground')

    # stop 命令
    subparsers.add_parser('stop', help='Stop TTS service')

    # restart 命令
    restart_parser = subparsers.add_parser('restart', help='Restart TTS service')
    restart_parser.add_argument('--config', default='config.yaml', help='Config file path')
    restart_parser.add_argument('--host', type=str, help='Server host')
    restart_parser.add_argument('--port', type=int, help='Server port')
    restart_parser.add_argument('--model', type=str, help='TTS model name')
    restart_parser.add_argument('--voices-dir', type=str, help='Voices directory')
    restart_parser.add_argument('--no-daemon', action='store_true', dest='no_daemon', help='Run in foreground')

    # status 命令
    subparsers.add_parser('status', help='Show service status')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = TTSCLI()

    if args.command == 'start':
        cli.start(
            config=args.config,
            host=args.host,
            port=args.port,
            model=args.model,
            voices_dir=args.voices_dir,
            daemon=args.daemon
        )
    elif args.command == 'stop':
        cli.stop()
    elif args.command == 'restart':
        cli.restart(
            config=args.config,
            host=args.host,
            port=args.port,
            model=args.model,
            voices_dir=args.voices_dir,
            no_daemon=args.no_daemon if hasattr(args, 'no_daemon') else False
        )
    elif args.command == 'status':
        cli.status()


if __name__ == "__main__":
    main()
