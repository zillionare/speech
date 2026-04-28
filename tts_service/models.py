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
    segment_count: int = 1
    created_at: Optional[str] = None


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=65536)
    output_format: Literal["wav", "flac", "ogg"] = "wav"
    voice: Optional[str] = None
    voice_mapping: Dict[str, str] = Field(default_factory=dict)
    engine: Optional[Literal["local", "remote"]] = None


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
    cfg_scale: float
    max_speech_tokens: int
    use_semantic: bool
    use_coreml_semantic: bool
    seed: int
    voices_path: str
    outputs_path: str
    use_remote_qwen: bool
    qwen_base_url: str
    max_segment_chars: int
    stereo: bool
    spatial_jitter: bool
    segment_gap_seconds: float
    speaker_gap_seconds: float


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
    use_remote_qwen: Optional[bool] = None
    qwen_base_url: Optional[str] = None
    max_segment_chars: Optional[int] = None
    stereo: Optional[bool] = None
    spatial_jitter: Optional[bool] = None
    segment_gap_seconds: Optional[float] = None
    speaker_gap_seconds: Optional[float] = None


class PruneOutputsRequest(BaseModel):
    keep_count: int = Field(default=3, ge=0)


class PodcastSegment(BaseModel):
    index: int
    text: str
    speaker: str = ""
    tone: str = ""
    speed_modifier: str = ""
    voice_ref: str = ""
    audio_filename: Optional[str] = None
    duration_seconds: float = 0.0
    generation_seconds: float = 0.0
    status: Literal["pending", "generated", "error"] = "pending"


class PodcastProject(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    output_format: Literal["wav", "flac", "ogg"] = "wav"
    segments: List[PodcastSegment]
    merged_audio_filename: Optional[str] = None


class PodcastListResponse(BaseModel):
    podcasts: List[PodcastProject]


class CreatePodcastRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    text: str = Field(..., min_length=1, max_length=65536)
    output_format: Literal["wav", "flac", "ogg"] = "wav"


class UpdateSegmentRequest(BaseModel):
    text: Optional[str] = None
    speaker: Optional[str] = None
    tone: Optional[str] = None
    speed_modifier: Optional[str] = None
    voice_ref: Optional[str] = None


class RegenerateSegmentRequest(BaseModel):
    engine: Optional[Literal["local", "remote"]] = None


class PruneOutputsResponse(BaseModel):
    deleted: List[str]
    kept: List[str]
