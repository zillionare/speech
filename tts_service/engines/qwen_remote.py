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
from .base import BaseEngine, GenerationResult, _apply_audio_effects


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

    def generate_single(
        self,
        text: str,
        voice: Optional[str],
        output_format: str = "wav",
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> GenerationResult:
        resolved_name, wav_path, ref_text = self._resolve_voice(voice)
        start = time.perf_counter()
        audio_bytes = self._call_remote(
            text=text,
            wav_path=wav_path,
            ref_text=ref_text,
            output_format=output_format,
            instructions=instructions,
            speed=speed,
        )
        generation_seconds = time.perf_counter() - start
        duration_seconds = self._estimate_duration(audio_bytes)
        result = GenerationResult(
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
        return self._post_process(result)

    def generate_dialogue(
        self,
        text: str,
        output_format: str = "wav",
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
        segment_gap: Optional[float] = None,
        speaker_gap: Optional[float] = None,
    ) -> GenerationResult:
        max_chars = getattr(self.config.model, "max_segment_chars", 200)

        # Detect dialogue format (Speaker: prefix or "Speaker N:" prefix)
        is_dialogue = any(
            line.strip().startswith("Speaker ") or ":" in line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        # Use segmentation for long text OR dialogue with multiple speakers
        if len(text) > max_chars or is_dialogue:
            result = self._generate_with_segmentation(
                text=text,
                output_format=output_format,
                max_chars=max_chars,
                preferred_voice=preferred_voice,
                voice_mapping=voice_mapping,
                segment_gap=segment_gap if segment_gap is not None else getattr(self.config.model, "segment_gap_seconds", 1.0),
                speaker_gap=speaker_gap if speaker_gap is not None else getattr(self.config.model, "speaker_gap_seconds", 1.0),
                instructions=instructions,
            )
            return self._post_process(result, speed_override=speed)

        # Short non-dialogue text: single voice fast path
        target_voice = preferred_voice or self.config.voices.default_voice
        return self.generate_single(text=text, voice=target_voice, output_format=output_format, instructions=instructions, speed=speed)

    def _post_process(self, result: GenerationResult, speed_override: Optional[float] = None) -> GenerationResult:
        # Speed is handled natively by the Qwen API; only apply stereo/spatial_jitter here.
        audio_bytes, duration = _apply_audio_effects(
            result.audio_bytes,
            result.output_format,
            speed=1.0,
            stereo=getattr(self.config.model, "stereo", False),
            spatial_jitter=getattr(self.config.model, "spatial_jitter", False),
        )
        return GenerationResult(
            audio_bytes=audio_bytes,
            output_format=result.output_format,
            generation_seconds=result.generation_seconds,
            duration_seconds=duration,
            resolved_speakers=result.resolved_speakers,
            segment_count=result.segment_count,
        )

    def _apply_postprocess_effects(
        self,
        audio_bytes: bytes,
        output_format: str,
        speed: float,
        stereo: bool,
        spatial_jitter: bool,
    ) -> tuple[bytes, float]:
        # Qwen API already applied speed natively; skip ffmpeg atempo.
        return _apply_audio_effects(audio_bytes, output_format, 1.0, stereo, spatial_jitter)

    def _call_remote(
        self,
        text: str,
        wav_path: Path,
        ref_text: str,
        output_format: str,
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
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
        if instructions:
            payload["instructions"] = instructions
        if speed is not None:
            payload["speed"] = speed

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
