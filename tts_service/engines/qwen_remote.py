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
from .base import (
    BaseEngine,
    GenerationResult,
    _apply_audio_effects,
    _concatenate_audio_segments,
    _parse_tagged_dialogue,
)


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
    ) -> GenerationResult:
        resolved_name, wav_path, ref_text = self._resolve_voice(voice)
        start = time.perf_counter()
        audio_bytes = self._call_remote(
            text=text,
            wav_path=wav_path,
            ref_text=ref_text,
            output_format=output_format,
            instructions=instructions,
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
        segment_gap: Optional[float] = None,
        speaker_gap: Optional[float] = None,
    ) -> GenerationResult:
        # Try new tagged dialogue format first: Speaker[tone>>]: text
        tagged_segments, is_tagged = _parse_tagged_dialogue(text)
        if is_tagged:
            return self._generate_tagged_segments(
                tagged_segments,
                output_format=output_format,
                voice_mapping=voice_mapping,
                segment_gap=segment_gap,
                speaker_gap=speaker_gap,
            )

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
            return self._post_process(result)

        # Short non-dialogue text: single voice fast path
        target_voice = preferred_voice or self.config.voices.default_voice
        return self.generate_single(text=text, voice=target_voice, output_format=output_format, instructions=instructions)

    def _generate_tagged_segments(
        self,
        segments: list[dict],
        output_format: str,
        voice_mapping: Optional[dict[str, str]] = None,
        segment_gap: Optional[float] = None,
        speaker_gap: Optional[float] = None,
    ) -> GenerationResult:
        """Generate each tagged segment with its own voice, tone and speed."""
        audio_parts: list[bytes] = []
        total_gen_seconds = 0.0
        total_duration_seconds = 0.0
        all_resolutions: list[SpeakerResolution] = []
        seen_speakers: set[str] = set()

        for seg in segments:
            mapped_voice = voice_mapping.get(seg["speaker"]) if voice_mapping else seg["speaker"]
            resolved = self.sample_manager.resolve(mapped_voice)
            voice = mapped_voice if resolved else self.config.voices.default_voice

            result = self.generate_single(
                text=seg["text"],
                voice=voice,
                output_format=output_format,
            )
            audio_parts.append(result.audio_bytes)
            total_gen_seconds += result.generation_seconds
            total_duration_seconds += result.duration_seconds
            for spk in result.resolved_speakers:
                if spk.resolved_voice not in seen_speakers:
                    seen_speakers.add(spk.resolved_voice)
                    all_resolutions.append(spk)

        gap = segment_gap if segment_gap is not None else getattr(self.config.model, "segment_gap_seconds", 1.0)
        gaps = [0.0] + [gap] * (len(segments) - 1)
        final_audio = _concatenate_audio_segments(audio_parts, output_format, gaps=gaps)
        return GenerationResult(
            audio_bytes=final_audio,
            output_format=output_format,
            generation_seconds=total_gen_seconds,
            duration_seconds=total_duration_seconds,
            resolved_speakers=all_resolutions,
            segment_count=len(segments),
        )

    def _post_process(self, result: GenerationResult) -> GenerationResult:
        audio_bytes, duration = _apply_audio_effects(
            result.audio_bytes,
            result.output_format,
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

    def _call_remote(
        self,
        text: str,
        wav_path: Path,
        ref_text: str,
        output_format: str,
        instructions: Optional[str] = None,
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
