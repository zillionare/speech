"""Remote Qwen3 TTS engine via OpenAI-compatible API."""

from __future__ import annotations

import base64
import io
import time
import urllib.request
from pathlib import Path
from typing import Optional

import soundfile as sf

from ..config import Config
from ..models import SpeakerResolution
from ..sample_manager import SampleManager
from .base import BaseEngine, GenerationResult


class QwenRemoteEngine(BaseEngine):
    """Calls a remote Qwen3 TTS service for voice cloning generation."""

    DEFAULT_MODEL = "Qwen3-TTS-12Hz-1.7B-Base-8bit"

    def __init__(self, config: Config, sample_manager: SampleManager):
        self.config = config
        self.sample_manager = sample_manager
        base = getattr(config.model, "qwen_base_url", "http://localhost:8000")
        self.base_url = base.rstrip("/")

    def _resolve_voice(self, voice: Optional[str]) -> tuple[str, Path, str]:
        sample = self.sample_manager.resolve_or_default(voice or self.config.voices.default_voice)
        transcript = sample.txt_path.read_text(encoding="utf-8").strip() if sample.txt_path.exists() else ""
        return sample.speaker, sample.wav_path, transcript

    def generate_single(self, text: str, voice: Optional[str], output_format: str = "wav") -> GenerationResult:
        resolved_name, wav_path, ref_text = self._resolve_voice(voice)
        start = time.perf_counter()
        audio_bytes = self._call_remote(
            text=text,
            wav_path=wav_path,
            ref_text=ref_text,
            output_format=output_format,
        )
        generation_seconds = time.perf_counter() - start
        duration_seconds = self._estimate_duration(audio_bytes)
        return GenerationResult(
            audio_bytes=audio_bytes,
            output_format=output_format,
            generation_seconds=generation_seconds,
            duration_seconds=duration_seconds,
            resolved_speakers=[
                SpeakerResolution(
                    requested_name=voice or resolved_name,
                    resolved_voice=resolved_name,
                    used_default=False,
                    transcript_preview=ref_text[:120] if ref_text else None,
                )
            ],
            segment_count=1,
        )

    def generate_dialogue(
        self,
        text: str,
        output_format: str = "wav",
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
    ) -> GenerationResult:
        max_chars = getattr(self.config.model, "max_segment_chars", 200)
        if len(text) > max_chars:
            return self._generate_with_segmentation(
                text=text,
                output_format=output_format,
                max_chars=max_chars,
                preferred_voice=preferred_voice,
                voice_mapping=voice_mapping,
            )
        # Qwen remote API handles voice cloning via ref_audio; for multi-speaker
        # dialogue we fall back to the preferred/default voice.
        target_voice = preferred_voice or self.config.voices.default_voice
        return self.generate_single(text=text, voice=target_voice, output_format=output_format)

    def _call_remote(
        self,
        text: str,
        wav_path: Path,
        ref_text: str,
        output_format: str,
    ) -> bytes:
        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.DEFAULT_MODEL,
            "input": text,
            "voice": wav_path.stem,
            "ref_audio": audio_b64,
            "ref_text": ref_text,
            "response_format": output_format,
        }

        req = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=str.encode(__import__("json").dumps(payload)),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()

    @staticmethod
    def _estimate_duration(audio_bytes: bytes) -> float:
        try:
            info = sf.info(io.BytesIO(audio_bytes))
            return info.duration
        except Exception:
            return 0.0
