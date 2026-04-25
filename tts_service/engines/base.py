"""Base TTS engine interface and shared utilities."""

from __future__ import annotations

import io
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf

from ..models import SpeakerResolution


@dataclass(slots=True)
class GenerationResult:
    audio_bytes: bytes
    output_format: str
    generation_seconds: float
    duration_seconds: float
    resolved_speakers: list[SpeakerResolution]
    segment_count: int = 1


class BaseEngine(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def generate_single(
        self,
        text: str,
        voice: Optional[str],
        output_format: str = "wav",
    ) -> GenerationResult:
        ...

    @abstractmethod
    def generate_dialogue(
        self,
        text: str,
        output_format: str = "wav",
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
    ) -> GenerationResult:
        ...

    def _generate_with_segmentation(
        self,
        text: str,
        output_format: str,
        max_chars: int,
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
    ) -> GenerationResult:
        """Segment long text, generate each part, concatenate with ffmpeg.

        Caller should ensure len(text) > max_chars; this method also guards
        against a single segment to avoid unnecessary concatenation overhead.
        """
        from ..segmentation import segment_dialogue, segment_long_text

        # Detect if dialogue or plain narration
        is_dialogue = any(
            line.strip().startswith("Speaker ") or ":" in line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        if is_dialogue:
            segments = segment_dialogue(text, max_chars)
        else:
            segments = [
                type("Seg", (), {"text": s, "speaker": None})()
                for s in segment_long_text(text, max_chars)
            ]

        # Single segment: delegate directly to avoid concat overhead
        if len(segments) == 1:
            return self.generate_single(
                text=segments[0].text,
                voice=segments[0].speaker or preferred_voice,
                output_format=output_format,
            )

        audio_parts: list[bytes] = []
        total_gen_seconds = 0.0
        total_duration_seconds = 0.0
        all_resolutions: list[SpeakerResolution] = []
        seen_speakers: set[str] = set()

        for seg in segments:
            result = self.generate_single(
                text=seg.text,
                voice=seg.speaker or preferred_voice,
                output_format=output_format,
            )
            audio_parts.append(result.audio_bytes)
            total_gen_seconds += result.generation_seconds
            total_duration_seconds += result.duration_seconds
            for spk in result.resolved_speakers:
                if spk.resolved_voice not in seen_speakers:
                    seen_speakers.add(spk.resolved_voice)
                    all_resolutions.append(spk)

        final_audio = _concatenate_audio_segments(audio_parts, output_format)
        return GenerationResult(
            audio_bytes=final_audio,
            output_format=output_format,
            generation_seconds=total_gen_seconds,
            duration_seconds=total_duration_seconds,
            resolved_speakers=all_resolutions,
            segment_count=len(segments),
        )

    def generate_with_segmentation_stream(
        self,
        text: str,
        output_format: str,
        max_chars: int,
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
    ):
        """Generator that yields progress dicts and finally a complete result dict.

        Yields:
            {"type": "progress", "current": int, "total": int, "text": str}
            {"type": "complete", "result": GenerationResult}
            {"type": "error", "message": str}
        """
        from ..segmentation import segment_dialogue, segment_long_text

        try:
            is_dialogue = any(
                line.strip().startswith("Speaker ") or ":" in line.strip()
                for line in text.splitlines()
                if line.strip()
            )

            if is_dialogue:
                segments = segment_dialogue(text, max_chars)
            else:
                segments = [
                    type("Seg", (), {"text": s, "speaker": None})()
                    for s in segment_long_text(text, max_chars)
                ]

            if len(segments) == 1:
                yield {"type": "progress", "current": 1, "total": 1, "text": segments[0].text[:60]}
                result = self.generate_single(
                    text=segments[0].text,
                    voice=segments[0].speaker or preferred_voice,
                    output_format=output_format,
                )
                yield {"type": "complete", "result": result}
                return

            audio_parts: list[bytes] = []
            total_gen_seconds = 0.0
            total_duration_seconds = 0.0
            all_resolutions: list[SpeakerResolution] = []
            seen_speakers: set[str] = set()

            for i, seg in enumerate(segments, start=1):
                preview = seg.text[:60] + ("..." if len(seg.text) > 60 else "")
                yield {"type": "progress", "current": i, "total": len(segments), "text": preview}
                result = self.generate_single(
                    text=seg.text,
                    voice=seg.speaker or preferred_voice,
                    output_format=output_format,
                )
                audio_parts.append(result.audio_bytes)
                total_gen_seconds += result.generation_seconds
                total_duration_seconds += result.duration_seconds
                for spk in result.resolved_speakers:
                    if spk.resolved_voice not in seen_speakers:
                        seen_speakers.add(spk.resolved_voice)
                        all_resolutions.append(spk)

            final_audio = _concatenate_audio_segments(audio_parts, output_format)
            complete_result = GenerationResult(
                audio_bytes=final_audio,
                output_format=output_format,
                generation_seconds=total_gen_seconds,
                duration_seconds=total_duration_seconds,
                resolved_speakers=all_resolutions,
                segment_count=len(segments),
            )
            yield {"type": "complete", "result": complete_result}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}


def _concatenate_audio_segments(
    segments: list[bytes],
    output_format: str,
    sample_rate: int = 24000,
) -> bytes:
    """Concatenate multiple audio byte segments using ffmpeg concat demuxer."""
    if len(segments) == 1:
        return segments[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        segment_files: list[Path] = []
        for i, seg_bytes in enumerate(segments):
            seg_path = tmp_path / f"seg_{i:04d}.{output_format}"
            seg_path.write_bytes(seg_bytes)
            segment_files.append(seg_path)

        list_file = tmp_path / "concat_list.txt"
        list_file.write_text(
            "\n".join(f"file '{f.name}'" for f in segment_files),
            encoding="utf-8",
        )

        output_path = tmp_path / f"output.{output_format}"
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        return output_path.read_bytes()


def create_engine(config, sample_manager) -> BaseEngine:
    """Factory: return remote Qwen engine if configured, else local MLX."""
    if getattr(config.model, "use_remote_qwen", False):
        from .qwen_remote import QwenRemoteEngine
        return QwenRemoteEngine(config, sample_manager)
    from .local_vibevoice import LocalVibeVoiceEngine
    return LocalVibeVoiceEngine(config, sample_manager)
