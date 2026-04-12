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
        daemon: bool = True
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
        """
        if self.manager.is_running():
            pid = self.manager.get_pid()
            print(f"TTS Service is already running (PID: {pid})")
            return

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
            print(f"Error: {e}")
            print(f"Creating default config at {config}...")
            self._create_default_config(config)
            cfg = load_config(config, **overrides)

        print(f"Starting TTS Service...")
        print(f"  Config: {config}")
        print(f"  Host: {cfg.server.host}:{cfg.server.port}")
        print(f"  Model: {cfg.model.name}")
        print(f"  Voices: {cfg.samples.expanded_base_dir}")

        if daemon:
            # 后台启动
            cmd = [
                sys.executable, "-m", "tts_service.server_runner",
                "--config", os.path.abspath(config)
            ]
            if host or port or model or voices_dir:
                cmd.append("--overrides")
                cmd.append(str(overrides))

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
        voices_dir: Optional[str] = None
    ):
        """
        重启 TTS 服务

        Args:
            config: 配置文件路径
            host: 服务器地址
            port: 服务器端口
            model: TTS 模型名称
            voices_dir: 声音样本目录
        """
        print("Restarting TTS Service...")
        self.stop()
        time.sleep(1)
        self.start(config, host, port, model, voices_dir)

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
        default_config = '''# TTS Web Service 配置

model:
  name: "mlx-community/VibeVoice-Realtime-0.5B-4bit"
  cache_dir: "~/.cache/tts_models"
  num_steps: 10
  cfg_scale: 1.3

samples:
  base_dir: "./voices"
  allowed_speakers: []

server:
  host: "0.0.0.0"
  port: 8123
  workers: 1
  log_level: "info"

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
            voices_dir=args.voices_dir
        )
    elif args.command == 'status':
        cli.status()


if __name__ == "__main__":
    main()
