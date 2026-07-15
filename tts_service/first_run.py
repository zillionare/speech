#!/usr/bin/env python3
"""
首次运行配置模块 - 交互式引导用户完成初始设置
"""

import os
import sys
from pathlib import Path
from typing import Optional

from .config import Config, load_config


# 默认配置值 (推荐使用 VibeVoice 1.5B 8bit 用于生产环境)
DEFAULTS = {
    "model_name": "gafiatulin/vibevoice-1.5b-mlx",  # 1.5B 模型，默认 8bit
    "cache_dir": "~/.speech/models",
    "hf_endpoint": "https://hf-mirror.com",
    "voices_dir": "~/.speech/voices",
    "quantization": "8bit",  # 生产环境推荐 8bit
}

# 首次运行标记文件
FIRST_RUN_MARKER = "~/.speech/.initialized"


def is_first_run() -> bool:
    """检查是否是首次运行"""
    marker = Path(FIRST_RUN_MARKER).expanduser()
    return not marker.exists()


def mark_initialized():
    """标记已完成首次运行配置"""
    marker = Path(FIRST_RUN_MARKER).expanduser()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def prompt_user(prompt: str, default: str = "") -> str:
    """提示用户输入，带有默认值"""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    try:
        user_input = input(full_prompt).strip()
        return user_input if user_input else default
    except (EOFError, KeyboardInterrupt):
        print("\n")
        return default


def prompt_confirm(prompt: str, default: bool = True) -> bool:
    """确认提示"""
    suffix = "Y/n" if default else "y/N"
    full_prompt = f"{prompt} [{suffix}]: "

    try:
        user_input = input(full_prompt).strip().lower()
        if not user_input:
            return default
        return user_input in ('y', 'yes', '是的', '是', '1')
    except (EOFError, KeyboardInterrupt):
        print("\n")
        return default


def create_directories(config: dict):
    """创建必要的目录"""
    dirs_to_create = [
        config["cache_dir"],
        config["voices_dir"],
    ]

    for dir_path in dirs_to_create:
        expanded = Path(dir_path).expanduser()
        expanded.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ 目录已创建/确认: {expanded}")


def generate_config_yaml(config: dict, output_path: str = "config.yaml"):
    """生成配置文件"""
    quantization_line = f'  quantization: "{config.get("quantization", "8bit")}"' if config.get("quantization") else '  quantization: "8bit"  # 可选: "4bit", "8bit", 或 null'
    yaml_content = f"""# TTS Web Service 配置文件
# 生成时间: 首次运行向导

# 首次运行配置
first_run:
  # HuggingFace 镜像地址
  hf_endpoint: "{config['hf_endpoint']}"
  # 是否跳过首次运行提示
  skip_welcome: false

# 模型配置
model:
  # 模型名称 (HuggingFace model ID)
  # 生产推荐: gafiatulin/vibevoice-1.5b-mlx (8bit量化)
  # 更快速: mlx-community/VibeVoice-Realtime-0.5B-4bit
  name: "{config['model_name']}"
  # 模型缓存目录 (~/.speech/models 约需 5-10GB 空间)
  cache_dir: "{config['cache_dir']}"
{quantization_line}
  # 默认推理步数 (数值越低越快，质量可能降低)
  num_steps: 10
  # CFG scale (控制生成多样性)
  cfg_scale: 1.3

# 声音样本配置
samples:
  # 声音样本目录 (.wav 文件放在这里)
  base_dir: "{config['voices_dir']}"
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

# PID 文件路径
pid_file: "/tmp/tts_service.pid"
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"  ✓ 配置文件已生成: {output_path}")


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("🚀 欢迎使用 TTS Web Service")
    print("=" * 60)
    print("\n这是您首次运行本服务，让我们快速完成初始化配置。")
    print("默认配置已针对中国大陆用户优化（使用 hf-mirror.com 镜像）。\n")


def print_summary(config: dict):
    """打印配置摘要"""
    print("\n" + "=" * 60)
    print("📋 配置摘要")
    print("=" * 60)
    print(f"  模型名称: {config['model_name']}")
    print(f"  量化方式: {config.get('quantization', '默认')}")
    print(f"  模型缓存: {Path(config['cache_dir']).expanduser()}")
    print(f"  声音样本: {Path(config['voices_dir']).expanduser()}")
    print(f"  HF镜像:   {config['hf_endpoint']}")
    print("=" * 60)


def run_first_run_wizard(config_path: str = "config.yaml") -> Config:
    """
    运行首次运行配置向导

    Args:
        config_path: 配置文件保存路径

    Returns:
        加载的配置对象
    """
    print_welcome()

    # 收集用户配置
    print("请按回车使用默认值，或直接输入新值：\n")

    user_config = {}

    # 1. HF_ENDPOINT
    print("1. HuggingFace 镜像地址")
    print(f"   说明: 用于下载模型，中国大陆建议使用 hf-mirror.com")
    user_config["hf_endpoint"] = prompt_user(
        "   HF_ENDPOINT", DEFAULTS["hf_endpoint"]
    )

    # 2. 模型名称和量化
    print("\n2. TTS 模型选择")
    print(f"   推荐选项:")
    print(f"   - gafiatulin/vibevoice-1.5b-mlx (生产推荐, 1.5B 8bit)")
    print(f"   - milkey/vibevoice-7b-mlx:4bit (更高质量, 7B 4bit)")
    print(f"   - mlx-community/VibeVoice-Realtime-0.5B-4bit (更快, 0.5B 4bit)")
    user_config["model_name"] = prompt_user(
        "   Model", DEFAULTS["model_name"]
    )

    # 量化设置
    print("\n3. 量化设置")
    if "1.5b" in user_config["model_name"].lower():
        print(f"   1.5B 模型推荐 8bit 量化 (平衡速度和质量)")
        default_quant = "8bit"
    elif "4bit" in user_config["model_name"].lower():
        print(f"   检测到 4bit 模型名称，使用 4bit 量化")
        default_quant = "4bit"
    else:
        print(f"   可选: 4bit (更快), 8bit (推荐), None (无量化)")
        default_quant = DEFAULTS["quantization"]

    quant_input = prompt_user(f"   Quantization (4bit/8bit)", default_quant)
    user_config["quantization"] = quant_input if quant_input in ("4bit", "8bit") else default_quant

    # 4. 模型缓存目录
    print("\n4. 模型缓存目录")
    print(f"   说明: 下载的模型将存储在此目录 (~/.speech/models 约需 5-10GB)")
    user_config["cache_dir"] = prompt_user(
        "   Cache dir", DEFAULTS["cache_dir"]
    )

    # 5. 声音样本目录
    print("\n5. 声音样本目录")
    print(f"   说明: 放置自定义 .wav 声音样本的目录")
    user_config["voices_dir"] = prompt_user(
        "   Voices dir", DEFAULTS["voices_dir"]
    )

    # 显示摘要并确认
    print_summary(user_config)

    if not prompt_confirm("是否保存以上配置？", default=True):
        print("\n配置已取消，将使用默认配置启动。")
        user_config = DEFAULTS.copy()

    # 创建目录
    print("\n📁 创建必要目录...")
    create_directories(user_config)

    # 设置 HF_ENDPOINT 环境变量
    os.environ['HF_ENDPOINT'] = user_config["hf_endpoint"]

    # 保存配置
    generate_config_yaml(user_config, config_path)

    # 标记已初始化
    mark_initialized()

    print("\n" + "=" * 60)
    print("✅ 首次运行配置完成！")
    print("=" * 60)
    print(f"\n配置文件已保存到: {config_path}")
    print("您可以通过编辑该文件修改配置。\n")

    return load_config(config_path)


def setup_environment(config: Config):
    """
    设置运行时环境（非交互式，用于后续启动）

    Args:
        config: 配置对象
    """
    # 设置 HF_ENDPOINT
    if config.first_run.hf_endpoint:
        os.environ['HF_ENDPOINT'] = config.first_run.hf_endpoint

    # 创建必要的目录
    Path(config.model.expanded_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(config.samples.expanded_base_dir).mkdir(parents=True, exist_ok=True)


def maybe_run_first_run(config_path: str = "config.yaml") -> Config:
    """
    检查并可能运行首次运行向导

    Args:
        config_path: 配置文件路径

    Returns:
        配置对象
    """
    # 如果配置文件已存在且不是首次运行，直接加载
    config_exists = Path(config_path).exists()
    first_run = is_first_run()

    if config_exists and not first_run:
        config = load_config(config_path)
        setup_environment(config)
        return config

    # 如果配置文件存在但标记不存在（可能是手动复制了配置），只需标记并设置环境
    if config_exists and first_run:
        print("检测到配置文件，快速初始化...")
        config = load_config(config_path)
        setup_environment(config)
        mark_initialized()

        # 显示简短的欢迎信息
        if not config.first_run.skip_welcome:
            print("\n" + "=" * 60)
            print("🚀 TTS Web Service")
            print("=" * 60)
            print(f"  HF_ENDPOINT: {config.first_run.hf_endpoint}")
            print(f"  模型: {config.model.name}")
            print(f"  模型缓存: {config.model.expanded_cache_dir}")
            print(f"  声音样本: {config.samples.expanded_base_dir}")
            print("=" * 60 + "\n")

        return config

    # 完全首次运行，启动交互式向导
    return run_first_run_wizard(config_path)
