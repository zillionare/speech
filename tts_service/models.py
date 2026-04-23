"""Pydantic models shared by the API and the UI."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SpeakerResolution(BaseModel):
    requested_name: str
    resolved_voice: str
    used_default: bool
    transcript_preview: Optional[str] = None


class VoiceInfo(BaseModel):
    speaker: str
    transcript: str
    transcript_preview: str
    cache_ready: bool
    is_default: bool
    audio_url: str


class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]


class GenerationRecord(BaseModel):
    request_id: str
    filename: str
    audio_url: str
    input_text: str
    output_format: Literal["wav", "flac", "ogg"]
    duration_seconds: float
    generation_seconds: float
    resolved_speakers: List[SpeakerResolution]


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=65536)
    output_format: Literal["wav", "flac", "ogg"] = "wav"
    voice: Optional[str] = None
    voice_mapping: Dict[str, str] = Field(default_factory=dict)


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str = Field(..., min_length=1, max_length=65536)
    voice: Optional[str] = None
    response_format: Literal["wav", "flac", "ogg"] = "wav"


class PodcastRequest(BaseModel):
    model: Optional[str] = None
    input: str = Field(..., min_length=1, max_length=65536)
    voice_mapping: Dict[str, str] = Field(default_factory=dict)
    response_format: Literal["wav", "flac", "ogg"] = "wav"


class TranscriptUpdateRequest(BaseModel):
    transcript: str = ""


class HealthResponse(BaseModel):
    status: str
    model: str
    quantize_bits: int
    voices_count: int
    default_voice: str


class AppConfigResponse(BaseModel):
    model: str
    quantize_bits: int
    default_voice: str
    diffusion_steps: int
    use_coreml_semantic: bool


class AppConfigUpdateRequest(BaseModel):
    voices_path: Optional[str] = None
    outputs_path: Optional[str] = None
    default_voice: Optional[str] = None
    diffusion_steps: Optional[int] = None
    quantize_bits: Optional[int] = None
    cfg_scale: Optional[float] = None
    max_speech_tokens: Optional[int] = None
    use_semantic: Optional[bool] = None
    use_coreml_semantic: Optional[bool] = None
    seed: Optional[int] = None


class PruneOutputsRequest(BaseModel):
    keep_count: int = Field(default=3, ge=0)


class PruneOutputsResponse(BaseModel):
    deleted: List[str]
    kept: List[str]
