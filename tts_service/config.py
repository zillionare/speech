"""配置管理模块"""

import os
import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """模型配置"""
    name: str = "gafiatulin/vibevoice-1.5b-mlx"  # 默认使用 1.5B 8bit 模型
    # 可选模型:
    # - gafiatulin/vibevoice-1.5b-mlx (8bit 量化)
    # - milkey/vibevoice-7b-mlx:4bit (4bit 量化)
    # - mlx-community/VibeVoice-Realtime-0.5B-4bit (0.5B 4bit)
    cache_dir: str = "~/.speech/models"
    quantization: Optional[str] = None  # 量化模式: "4bit", "8bit" 或 None (使用模型默认)
    num_steps: int = 10
    cfg_scale: float = 1.3

    @property
    def expanded_cache_dir(self) -> str:
        return os.path.expanduser(self.cache_dir)


class SamplesConfig(BaseModel):
    """声音样本配置"""
    base_dir: str = "./voices"
    allowed_speakers: List[str] = Field(default_factory=list)

    @property
    def expanded_base_dir(self) -> str:
        return os.path.expanduser(self.base_dir)


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8123
    workers: int = 1
    log_level: str = "info"


class DefaultsConfig(BaseModel):
    """默认参数配置"""
    voice: str = "zh-Aaron_man"
    response_format: str = "wav"
    speed: float = 1.0


class FirstRunConfig(BaseModel):
    """首次运行配置"""
    hf_endpoint: str = "https://hf-mirror.com"
    skip_welcome: bool = False


class Config(BaseModel):
    """主配置类"""
    first_run: FirstRunConfig = Field(default_factory=FirstRunConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    samples: SamplesConfig = Field(default_factory=SamplesConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    pid_file: str = "/tmp/tts_service.pid"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def update_from_dict(self, overrides: dict):
        """从字典更新配置"""
        for key, value in overrides.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, BaseModel) and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(current, sub_key):
                            setattr(current, sub_key, sub_value)
                else:
                    setattr(self, key, value)


def load_config(config_path: Optional[str] = None, **overrides) -> Config:
    """
    加载配置

    Args:
        config_path: YAML 配置文件路径
        **overrides: 覆盖配置项

    Returns:
        Config 实例
    """
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    if overrides:
        config.update_from_dict(overrides)

    return config
