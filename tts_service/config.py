"""Configuration models for the new vibevoice-mlx based service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model and inference defaults."""

    model_id: str = "gafiatulin/vibevoice-1.5b-mlx"
    quantize_bits: int = 8
    diffusion_steps: int = 10
    cfg_scale: float = 1.3
    max_speech_tokens: int = 200
    seed: int = 42
    use_semantic: bool = True
    use_coreml_semantic: bool = False


class VoicesConfig(BaseModel):
    """Voice library settings."""

    base_dir: str = "./voices"
    default_voice: str = "zh-Bowen_man"
    cache_subdir: str = ".cache"
    bundled_chinese_voices: list[str] = Field(
        default_factory=lambda: [
            "zh-007_man",
            "zh-007_woman",
            "zh-Anchen_man_bgm",
            "zh-Aaron_man",
            "zh-Bowen_man",
            "zh-Xinran_woman",
            "zh-Zxx_man",
        ]
    )

    @property
    def expanded_base_dir(self) -> Path:
        return Path(os.path.expanduser(self.base_dir)).resolve()

    @property
    def cache_dir(self) -> Path:
        return self.expanded_base_dir / self.cache_subdir


class OutputsConfig(BaseModel):
    """Generated audio output settings."""

    base_dir: str = "./outputs"
    history_limit: int = 20

    @property
    def expanded_base_dir(self) -> Path:
        return Path(os.path.expanduser(self.base_dir)).resolve()


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8123
    log_level: str = "info"


class Config(BaseModel):
    """Application configuration root."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    voices: VoicesConfig = Field(default_factory=VoicesConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    pid_file: str = "/tmp/tts_service.pid"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        base_dir = config_path.resolve().parent

        voices_section = data.get("voices")
        if isinstance(voices_section, dict) and voices_section.get("base_dir"):
            voice_base = Path(voices_section["base_dir"]).expanduser()
            if not voice_base.is_absolute():
                voices_section["base_dir"] = str((base_dir / voice_base).resolve())

        outputs_section = data.get("outputs")
        if isinstance(outputs_section, dict) and outputs_section.get("base_dir"):
            output_base = Path(outputs_section["base_dir"]).expanduser()
            if not output_base.is_absolute():
                outputs_section["base_dir"] = str((base_dir / output_base).resolve())

        pid_file = data.get("pid_file")
        if isinstance(pid_file, str) and pid_file:
            pid_path = Path(pid_file).expanduser()
            if not pid_path.is_absolute():
                data["pid_file"] = str((base_dir / pid_path).resolve())

        return cls(**data)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        for section_name, section_value in overrides.items():
            if not hasattr(self, section_name):
                continue
            current = getattr(self, section_name)
            if isinstance(current, BaseModel) and isinstance(section_value, dict):
                for key, value in section_value.items():
                    if hasattr(current, key) and value is not None:
                        setattr(current, key, value)
            elif section_value is not None:
                setattr(self, section_name, section_value)


def load_config(config_path: Optional[str] = None, **overrides: Any) -> Config:
    config = Config.from_yaml(config_path) if config_path and Path(config_path).exists() else Config()
    if overrides:
        config.apply_overrides(overrides)
    return config
