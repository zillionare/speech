"""Embedded ASR engine.

In-process ASR inference using mlx_whisper or faster_whisper.
Full spec: .project/specs/spec.md SPEC-003.
ACC-003-1: ASR configuration fields
ACC-003-2: warmup / is_ready / is_warming
ACC-003-3: transcribe_chunk returns ASRResult
ACC-003-4: failure degradation
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_FAILURES_BEFORE_DEGRADED = 5


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
    is_final: bool = True
    audio_ms: int = 0
    confidence: float = 0.0
    raw_segments: list = field(default_factory=list)


class EmbeddedASR:
    """In-process ASR inference engine.

    The ``_model`` attribute is set to a callable ``transcribe_fn(audio_np, cfg) -> dict``
    after warmup. Tests can mock ``_model`` with any callable that returns a dict
    containing ``text`` and ``segments`` keys.
    """

    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self._model: Optional[Callable] = None
        self._lock = threading.Lock()
        self._failures = 0
        self._degraded = False
        self._warming = False

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def is_warming(self) -> bool:
        return self._warming

    @property
    def degraded(self) -> bool:
        return self._degraded

    def warmup(self) -> None:
        """Load the ASR model with a 1-second silent sample."""
        if self._model is not None:
            return
        self._warming = True
        try:
            self._load_model()
        except Exception as exc:
            logger.warning("ASR warmup failed: %s", exc)
        finally:
            self._warming = False

    def _load_model(self) -> None:
        """Load the model based on the configured backend, store transcribe_fn."""
        backend = self.cfg.backend
        if backend == "mlx_whisper":
            try:
                import mlx_whisper

                def _mlx_transcribe(audio_np: np.ndarray, cfg: ASRConfig) -> dict:
                    return mlx_whisper.transcribe(
                        audio_np,
                        path_or_hsp=cfg.model,
                        language=cfg.language,
                        word_timestamps=True,
                    )

                self._model = _mlx_transcribe
            except ImportError:
                logger.warning("mlx_whisper not installed; ASR disabled")
                self._model = None
        elif backend == "faster_whisper":
            try:
                from faster_whisper import WhisperModel

                fw_model = WhisperModel(
                    self.cfg.model,
                    device=self.cfg.device,
                    compute_type=self.cfg.compute_type,
                )

                def _fw_transcribe(audio_np: np.ndarray, cfg: ASRConfig) -> dict:
                    segments, info = fw_model.transcribe(
                        audio_np,
                        language=cfg.language,
                        beam_size=cfg.beam_size,
                        vad_filter=cfg.vad_filter,
                        word_timestamps=True,
                    )
                    seg_list = list(segments)
                    return {
                        "text": " ".join(s.text for s in seg_list),
                        "segments": [
                            {"start": s.start, "end": s.end, "text": s.text}
                            for s in seg_list
                        ],
                    }

                self._model = _fw_transcribe
            except ImportError:
                logger.warning("faster_whisper not installed; ASR disabled")
                self._model = None
        else:
            logger.warning("Unknown ASR backend: %s", backend)
            self._model = None

    def _do_transcribe(self, audio_np: np.ndarray) -> dict:
        """Call the backend's transcribe function and return raw result dict."""
        if self._model is None:
            return {"text": "", "segments": []}
        return self._model(audio_np, self.cfg)

    async def transcribe_chunk(self, pcm_bytes: bytes) -> ASRResult:
        """Transcribe a chunk of 16kHz mono PCM audio (16-bit signed)."""
        if not pcm_bytes:
            return ASRResult(text="", audio_ms=0, confidence=0.0)

        audio_ms = len(pcm_bytes) // 32  # 16kHz * 2 bytes = 32 bytes/ms

        if self._model is None:
            return ASRResult(text="", audio_ms=audio_ms, confidence=0.0)

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._do_transcribe, audio_np)
            text = (result.get("text") or "").strip()
            segments = result.get("segments", [])
            confidence = 1.0 if text else 0.0

            self._failures = 0
            self._degraded = False
            return ASRResult(
                text=text,
                audio_ms=audio_ms,
                confidence=confidence,
                raw_segments=segments,
            )
        except Exception as exc:
            logger.warning("ASR transcribe failed: %s", exc)
            self._failures += 1
            if self._failures >= _MAX_FAILURES_BEFORE_DEGRADED:
                self._degraded = True
            return ASRResult(text="", audio_ms=audio_ms, confidence=0.0)
