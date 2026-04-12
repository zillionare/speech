"""Pydantic 数据模型定义"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """
    单 speaker TTS 请求模型
    兼容 OpenAI TTS API 格式
    """
    model: Optional[str] = Field(
        default=None,
        description="TTS 模型名称"
    )
    input: str = Field(
        ...,
        description="要转换为语音的文本",
        max_length=65536  # 64KB
    )
    voice: Optional[str] = Field(
        default=None,
        description="声音名称"
    )
    response_format: Literal["wav", "mp3", "flac", "ogg"] = Field(
        default="wav",
        description="输出音频格式"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="语速倍率"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
                "input": "Hello, this is a test.",
                "voice": "zh-Aaron_man",
                "response_format": "wav",
                "speed": 1.0
            }
        }


class PodcastRequest(BaseModel):
    """
    多 speaker 播客生成请求模型
    """
    model: Optional[str] = Field(
        default=None,
        description="TTS 模型名称"
    )
    input: str = Field(
        ...,
        description="播客脚本文本，格式：Speaker X: 文本内容",
        max_length=65536  # 64KB
    )
    voice_mapping: Dict[str, str] = Field(
        ...,
        description="Speaker 到声音文件的映射",
        examples=[{"Speaker 1": "zh-Aaron_man", "Speaker 2": "en-Alice_woman"}]
    )
    response_format: Literal["wav", "mp3", "flac"] = Field(
        default="wav",
        description="输出音频格式"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="语速倍率"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
                "input": "Speaker 1: Hello everyone!\nSpeaker 2: Nice to meet you all.",
                "voice_mapping": {"Speaker 1": "zh-Aaron_man", "Speaker 2": "en-Alice_woman"},
                "response_format": "wav",
                "speed": 1.0
            }
        }


class VoiceInfo(BaseModel):
    """声音信息模型"""
    name: str
    path: Optional[str]
    has_text: bool
    text_preview: Optional[str]


class VoiceListResponse(BaseModel):
    """声音列表响应"""
    voices: List[str]


class VoiceDetailResponse(BaseModel):
    """声音详情响应"""
    voices: List[VoiceInfo]


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    message: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model: str
    voices_count: int
    uptime: Optional[float] = None
