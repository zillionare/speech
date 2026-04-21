#!/usr/bin/env python3
"""CLI for the new worktree implementation."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import uvicorn

from .config import load_config
from .server import create_app


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
    """Minimal process manager for the FastAPI app."""

    def __init__(self):
        self.manager = ServerManager()

    def start(
        self,
        config: str = "config.yaml",
        host: Optional[str] = None,
        port: Optional[int] = None,
        daemon: bool = False,
        no_daemon: bool = False,
        reload: bool = True,
    ):
        if no_daemon:
            daemon = False
        if self.manager.is_running():
            pid = self.manager.get_pid()
            print(f"TTS Service is already running (PID: {pid})")
            return

        overrides = {"server": {"host": host, "port": port}}
        cfg = load_config(config, **overrides)

        print(f"Starting TTS Service...")
        print(f"  Config: {config}")
        print(f"  Host: {cfg.server.host}:{cfg.server.port}")
        print(f"  Model: {cfg.model.model_id} ({cfg.model.quantize_bits}-bit)")
        print(f"  Voices: {cfg.voices.expanded_base_dir}")

        if daemon:
            cmd = [
                sys.executable,
                "-m",
                "tts_service.server_runner",
                "--config", os.path.abspath(config),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            self.manager.save_pid(proc.pid)
            print(f"TTS Service started (PID: {proc.pid})")
        else:
            app = create_app(config)
            uvicorn.run(
                app,
                host=cfg.server.host,
                port=cfg.server.port,
                log_level=cfg.server.log_level,
                reload=reload,
                reload_dirs=[os.path.dirname(os.path.dirname(__file__))] if reload else None
            )

    def stop(self):
        """停止 TTS 服务"""
        self.manager.stop()

    def restart(
        self,
        config: str = "config.yaml",
        host: Optional[str] = None,
        port: Optional[int] = None,
        no_daemon: bool = False
    ):
        """
        重启 TTS 服务

        Args:
            config: 配置文件路径
            host: 服务器地址
            port: 服务器端口
            no_daemon: 前台运行模式
        """
        print("Restarting TTS Service...")
        self.stop()
        time.sleep(1)
        self.start(config, host, port, not no_daemon, no_daemon)

    def status(self):
        """显示服务状态"""
        pid = self.manager.get_pid()

        if pid is None:
            print("TTS Service: NOT RUNNING")
            return

        if self.manager.is_running():
            print(f"TTS Service: RUNNING (PID: {pid})")

            try:
                cfg = load_config("config.yaml")
                print(f"  Health: http://{cfg.server.host}:{cfg.server.port}/health")
                print(f"  API: http://{cfg.server.host}:{cfg.server.port}/docs")
            except:
                pass
        else:
            print(f"TTS Service: NOT RUNNING (stale PID: {pid})")
            self.manager.remove_pid()


def main():
    parser = argparse.ArgumentParser(description="VibeVoice MLX Studio CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # start 命令
    start_parser = subparsers.add_parser('start', help='Start TTS service')
    start_parser.add_argument('--config', default='config.yaml', help='Config file path')
    start_parser.add_argument('--host', type=str, help='Server host')
    start_parser.add_argument('--port', type=int, help='Server port')
    start_parser.add_argument('--daemon', action='store_true', help='Run in background')
    start_parser.add_argument('--no-daemon', action='store_true', help='Run in foreground')
    start_parser.add_argument('--reload', action='store_true', default=False, help='Enable auto-reload (default: True via function)')
    start_parser.add_argument('--no-reload', action='store_false', dest='reload', help='Disable auto-reload')

    # stop 命令
    subparsers.add_parser('stop', help='Stop TTS service')

    # restart 命令
    restart_parser = subparsers.add_parser('restart', help='Restart TTS service')
    restart_parser.add_argument('--config', default='config.yaml', help='Config file path')
    restart_parser.add_argument('--host', type=str, help='Server host')
    restart_parser.add_argument('--port', type=int, help='Server port')
    restart_parser.add_argument('--no-daemon', action='store_true', dest='no_daemon', help='Run in foreground')

    # status 命令
    subparsers.add_parser('status', help='Show service status')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = TTSCLI()

    if args.command == 'start':
        cli.start(config=args.config, host=args.host, port=args.port, daemon=args.daemon, no_daemon=args.no_daemon, reload=args.reload if args.reload else True)
    elif args.command == 'stop':
        cli.stop()
    elif args.command == 'restart':
        cli.restart(config=args.config, host=args.host, port=args.port, no_daemon=args.no_daemon if hasattr(args, 'no_daemon') else False)
    elif args.command == 'status':
        cli.status()


if __name__ == "__main__":
    main()
