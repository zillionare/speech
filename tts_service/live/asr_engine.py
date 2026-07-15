"""Embedded ASR engine.

Stub implementation — methods raise NotImplementedError.
Full spec: .project/specs/spec.md SPEC-003.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRConfig:
    enabled: bool = False
    backend: str = "mlx_whisper"
    model: str = "mlx-community/whisper-small"
    language: str = "zh"
    chunk_seconds: float = 1.0
    beam_size: int = 1
    vad_filter: bool = True
    compute_type: str = "float16"
    device: str = "auto"
    warmup_on_start: bool = False


@dataclass
class ASRResult:
    text: str = ""
    is_final: bool = False
    audio_ms: int = 0
    confidence: float = 0.0
    raw_segments: list = None  # type: ignore[assignment]


class EmbeddedASR:
    """In-process ASR inference engine."""

    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self._failures = 0
        self._degraded = False

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def is_warming(self) -> bool:
        return False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def warmup(self) -> None:
        """Warm up the ASR model."""
        raise NotImplementedError("ASR warmup not implemented")

    async def transcribe_chunk(self, pcm_bytes: bytes) -> ASRResult:
        """Transcribe a chunk of 16kHz mono PCM audio."""
        raise NotImplementedError("ASR transcribe_chunk not implemented")
