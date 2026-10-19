"""Configuration models for the new vibevoice-mlx based service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LiveConfig(BaseModel):
    """Live podcast settings."""

    tts_max_seconds: int = 60
    tts_timeout_seconds: int = 30
    max_ai_rephrases_per_segment: int = 2


class ModelConfig(BaseModel):
    """Model and inference defaults."""

    model_id: str = "gafiatulin/vibevoice-1.5b-mlx"
    tts_provider: str = ""
    quantize_bits: int = 8
    diffusion_steps: int = 10
    cfg_scale: float = 1.3
    max_speech_tokens: int = 200
    seed: int = 42
    use_semantic: bool = True
    use_coreml_semantic: bool = False
    use_remote_qwen: bool = True
    qwen_base_url: str = "http://192.168.0.102:8000"
    qwen_api_key: str = ""
    qwen_model_id: str = "Qwen3-TTS-12Hz-1.7B-Base-8bit"
    minimax_api_mode: str = "native"
    minimax_api_key: str = ""
    minimax_base_url: str = "https://api.minimaxi.com"
    minimax_model_id: str = "speech-2.8-hd"
    minimax_voice_id: str = "male-qn-qingse"
    minimax_emotion: str = ""
    minimax_speed: float = 1.0
    minimax_volume: float = 1.0
    minimax_pitch: int = 0
    omnivoice_base_url: str = "http://192.168.0.102:8002"
    omnivoice_model_id: str = "k2-fsa/OmniVoice"
    omnivoice_api_key: str = ""
    max_segment_chars: int = 200
    stereo: bool = True
    spatial_jitter: bool = True
    segment_gap_seconds: float = 1.0
    speaker_gap_seconds: float = 1.0


class TTSProviderProfile(BaseModel):
    """One remotely configured TTS connection shown in the Web UI."""

    id: str
    name: str
    provider: str = "qwen"
    api_mode: str = "openai"
    num_steps: int = 16
    ref_audio_max_duration_s: float = 10.0
    base_url: str = ""
    api_key: str = ""
    model_id: str = ""
    voice_id: str = ""
    emotion: str = ""
    speed: float = 1.0
    volume: float = 1.0
    pitch: int = 0


class VoicesConfig(BaseModel):
    """Voice library settings."""

    base_dir: str = "./voices"
    default_voice: str = "zh-Bowen_man"
    cache_subdir: str = ".cache"
    live_speakers: list[str] = Field(default_factory=list)
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


class ASRConfigPydantic(BaseModel):
    """ASR configuration (Pydantic wrapper for YAML persistence)."""

    enabled: bool = False
    backend: str = "qwen3_asr"
    model: str = "mlx-community/whisper-medium-mlx-4bit"
    language: str = "zh"
    chunk_seconds: float = 4.0
    beam_size: int = 1
    vad_filter: bool = True
    compute_type: str = "float16"
    device: str = "auto"
    warmup_on_start: bool = False
    download_endpoint: str = "https://hf-mirror.com"
    cache_dir: str = ""
    qwen3_model_dir: str = "~/.cache/sherpa-onnx/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    qwen3_num_threads: int = 4
    qwen3_max_new_tokens: int = 512
    qwen3_hotwords: str = ""


class ServerConfig(BaseModel):
    """HTTP/HTTPS server configuration."""

    host: str = "0.0.0.0"
    port: int = 8123
    log_level: str = "info"
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None


class Config(BaseModel):
    """Application configuration root."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    voices: VoicesConfig = Field(default_factory=VoicesConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    asr: ASRConfigPydantic = Field(default_factory=ASRConfigPydantic)
    pid_file: str = "/tmp/tts_service.pid"
    tts_providers: list[TTSProviderProfile] = Field(default_factory=list)
    active_tts_provider: str = ""

    def ensure_tts_providers(self) -> None:
        """Create profiles from legacy model fields when no profiles exist."""
        defaults = [
            TTSProviderProfile(
                id="qwen",
                name="Qwen 远程",
                provider="qwen",
                api_mode="openai",
                base_url=self.model.qwen_base_url,
                api_key=self.model.qwen_api_key,
                model_id=self.model.qwen_model_id,
            ),
            TTSProviderProfile(
                id="minimax",
                name="MiniMax",
                provider="minimax",
                api_mode=self.model.minimax_api_mode,
                base_url=self.model.minimax_base_url,
                api_key=self.model.minimax_api_key,
                model_id=self.model.minimax_model_id,
                voice_id=self.model.minimax_voice_id,
                emotion=self.model.minimax_emotion,
                speed=self.model.minimax_speed,
                volume=self.model.minimax_volume,
                pitch=self.model.minimax_pitch,
            ),
            TTSProviderProfile(
                id="local",
                name="本地 VibeVoice",
                provider="local",
                api_mode="local",
                model_id=self.model.model_id,
            ),
            TTSProviderProfile(
                id="omnivoice",
                name="OmniVoice",
                provider="omnivoice",
                api_mode="openai",
                base_url=self.model.omnivoice_base_url,
                api_key=self.model.omnivoice_api_key,
                model_id=self.model.omnivoice_model_id,
            ),
        ]
        if not self.tts_providers:
            self.tts_providers = list(defaults)
        else:
            existing_providers = {profile.provider for profile in self.tts_providers}
            for profile in defaults:
                if profile.provider not in existing_providers:
                    self.tts_providers.append(profile)
                    existing_providers.add(profile.provider)
        if not self.active_tts_provider:
            provider = self.model.tts_provider or (
                "qwen" if self.model.use_remote_qwen else "local"
            )
            selected = next(
                (profile for profile in self.tts_providers if profile.provider == provider),
                self.tts_providers[0],
            )
            self.active_tts_provider = selected.id

    def get_tts_provider(self, profile_id: str | None = None) -> TTSProviderProfile:
        self.ensure_tts_providers()
        target_id = profile_id or self.active_tts_provider
        for profile in self.tts_providers:
            if profile.id == target_id:
                return profile
        raise ValueError(f"TTS provider profile not found: {target_id}")

    def apply_tts_provider(self, profile_id: str) -> TTSProviderProfile:
        profile = self.get_tts_provider(profile_id)
        self.active_tts_provider = profile.id
        self.model.tts_provider = profile.provider
        if profile.provider == "qwen":
            self.model.use_remote_qwen = True
            self.model.qwen_base_url = profile.base_url
            self.model.qwen_api_key = profile.api_key
            self.model.qwen_model_id = profile.model_id
        elif profile.provider == "minimax":
            self.model.use_remote_qwen = True
            self.model.minimax_api_mode = profile.api_mode
            self.model.minimax_api_key = profile.api_key
            self.model.minimax_base_url = profile.base_url
            self.model.minimax_model_id = profile.model_id
            self.model.minimax_voice_id = profile.voice_id
            self.model.minimax_emotion = profile.emotion
            self.model.minimax_speed = profile.speed
            self.model.minimax_volume = profile.volume
            self.model.minimax_pitch = profile.pitch
        elif profile.provider == "omnivoice":
            self.model.use_remote_qwen = True
            self.model.omnivoice_base_url = profile.base_url
            self.model.omnivoice_api_key = profile.api_key
            self.model.omnivoice_model_id = profile.model_id
        elif profile.provider == "local":
            self.model.use_remote_qwen = False
        else:
            raise ValueError(f"Unsupported TTS provider: {profile.provider}")
        return profile

    def sync_tts_provider_from_model(self) -> None:
        """Keep legacy model fields and the active profile in sync."""
        self.ensure_tts_providers()
        profile = self.get_tts_provider()
        if profile.provider == "qwen":
            profile.base_url = self.model.qwen_base_url
            profile.api_key = self.model.qwen_api_key
            profile.model_id = self.model.qwen_model_id
        elif profile.provider == "minimax":
            profile.api_mode = self.model.minimax_api_mode
            profile.api_key = self.model.minimax_api_key
            profile.base_url = self.model.minimax_base_url
            profile.model_id = self.model.minimax_model_id
            profile.voice_id = self.model.minimax_voice_id
            profile.emotion = self.model.minimax_emotion
            profile.speed = self.model.minimax_speed
            profile.volume = self.model.minimax_volume
            profile.pitch = self.model.minimax_pitch
        elif profile.provider == "omnivoice":
            profile.base_url = self.model.omnivoice_base_url
            profile.api_key = self.model.omnivoice_api_key
            profile.model_id = self.model.omnivoice_model_id

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

        server_section = data.get("server")
        if isinstance(server_section, dict):
            for key in ("ssl_keyfile", "ssl_certfile"):
                value = server_section.get(key)
                if not value:
                    continue
                tls_path = Path(value).expanduser()
                if not tls_path.is_absolute():
                    server_section[key] = str((base_dir / tls_path).resolve())

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
    config.ensure_tts_providers()
    if config_path:
        secrets_path = Path(config_path).resolve().parent / ".speech-secrets.yaml"
        if secrets_path.exists():
            try:
                with open(secrets_path, "r", encoding="utf-8") as handle:
                    secrets = yaml.safe_load(handle) or {}
                for key in ("qwen_api_key", "minimax_api_key", "omnivoice_api_key"):
                    api_key = secrets.get(key)
                    if isinstance(api_key, str):
                        setattr(config.model, key, api_key)
                profile_secrets = secrets.get("tts_providers") or {}
                if isinstance(profile_secrets, dict):
                    for profile in config.tts_providers:
                        api_key = profile_secrets.get(profile.id)
                        if isinstance(api_key, str):
                            profile.api_key = api_key
            except (OSError, yaml.YAMLError):
                pass
    env_api_key = os.getenv("MINIMAX_API_KEY")
    if env_api_key:
        config.model.minimax_api_key = env_api_key
    qwen_env_api_key = os.getenv("QWEN_TTS_API_KEY")
    if qwen_env_api_key:
        config.model.qwen_api_key = qwen_env_api_key
    omnivoice_env_api_key = os.getenv("OMNIVOICE_API_KEY")
    if omnivoice_env_api_key:
        config.model.omnivoice_api_key = omnivoice_env_api_key
    for profile in config.tts_providers:
        if profile.provider == "qwen" and config.model.qwen_api_key:
            profile.api_key = config.model.qwen_api_key
        elif profile.provider == "minimax" and config.model.minimax_api_key:
            profile.api_key = config.model.minimax_api_key
        elif profile.provider == "omnivoice" and config.model.omnivoice_api_key:
            profile.api_key = config.model.omnivoice_api_key
    if overrides:
        config.apply_overrides(overrides)
    config.ensure_tts_providers()
    try:
        config.apply_tts_provider(config.active_tts_provider)
    except ValueError:
        pass
    return config


def save_config_to_yaml(config: Config, path: str) -> None:
    """Save configuration back to YAML file."""
    config_path = Path(path)
    config.ensure_tts_providers()
    config.sync_tts_provider_from_model()
    model_data = config.model.model_dump()
    # Keep credentials out of the normal, shareable YAML configuration.
    qwen_api_key = model_data.pop("qwen_api_key", "")
    minimax_api_key = model_data.pop("minimax_api_key", "")
    omnivoice_api_key = model_data.pop("omnivoice_api_key", "")
    data = {
        "model": model_data,
        "voices": config.voices.model_dump(),
        "outputs": config.outputs.model_dump(),
        "server": config.server.model_dump(),
        "live": config.live.model_dump(),
        "asr": config.asr.model_dump(),
        "pid_file": config.pid_file,
        "tts_providers": [
            {
                key: value
                for key, value in profile.model_dump().items()
                if key != "api_key"
            }
            for profile in config.tts_providers
        ],
        "active_tts_provider": config.active_tts_provider,
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle, allow_unicode=True, sort_keys=False, default_flow_style=False)

    secrets_path = config_path.resolve().parent / ".speech-secrets.yaml"
    secret_data = {}
    if qwen_api_key.strip():
        secret_data["qwen_api_key"] = qwen_api_key.strip()
    if minimax_api_key.strip():
        secret_data["minimax_api_key"] = minimax_api_key.strip()
    if omnivoice_api_key.strip():
        secret_data["omnivoice_api_key"] = omnivoice_api_key.strip()
    profile_secrets = {
        profile.id: profile.api_key.strip()
        for profile in config.tts_providers
        if profile.api_key.strip()
    }
    if profile_secrets:
        secret_data["tts_providers"] = profile_secrets
    if secret_data:
        secrets_path.write_text(
            yaml.safe_dump(secret_data, sort_keys=False),
            encoding="utf-8",
        )
        secrets_path.chmod(0o600)
    elif secrets_path.exists():
        secrets_path.unlink()
