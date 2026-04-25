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
        segment_gap: float = 1.0,
        speaker_gap: float = 1.0,
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

        gaps = _compute_gaps(segments, segment_gap, speaker_gap)
        final_audio = _concatenate_audio_segments(audio_parts, output_format, gaps=gaps)
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
        speed: float = 1.0,
        stereo: bool = False,
        spatial_jitter: bool = False,
        segment_gap: float = 1.0,
        speaker_gap: float = 1.0,
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
                audio_bytes, duration = _apply_audio_effects(
                    result.audio_bytes, output_format, speed, stereo, spatial_jitter
                )
                result = GenerationResult(
                    audio_bytes=audio_bytes,
                    output_format=result.output_format,
                    generation_seconds=result.generation_seconds,
                    duration_seconds=duration,
                    resolved_speakers=result.resolved_speakers,
                    segment_count=result.segment_count,
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

            gaps = _compute_gaps(segments, segment_gap, speaker_gap)
            final_audio = _concatenate_audio_segments(audio_parts, output_format, gaps=gaps)
            final_audio, final_duration = _apply_audio_effects(
                final_audio, output_format, speed, stereo, spatial_jitter
            )
            complete_result = GenerationResult(
                audio_bytes=final_audio,
                output_format=output_format,
                generation_seconds=total_gen_seconds,
                duration_seconds=final_duration,
                resolved_speakers=all_resolutions,
                segment_count=len(segments),
            )
            yield {"type": "complete", "result": complete_result}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}


def _generate_silence(duration_seconds: float, sample_rate: int, output_path: Path) -> None:
    """Generate a silent audio file of given duration."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", str(duration_seconds),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _compute_gaps(segments, segment_gap: float, speaker_gap: float) -> list[float]:
    """Return list of silence durations before each segment.

    gaps[0] is always 0.0. Subsequent gaps depend on whether the speaker
    changed from the previous segment.
    """
    gaps = [0.0]
    for i in range(1, len(segments)):
        prev = getattr(segments[i - 1], "speaker", None)
        curr = getattr(segments[i], "speaker", None)
        if prev != curr and (prev is not None or curr is not None):
            gaps.append(speaker_gap)
        else:
            gaps.append(segment_gap)
    return gaps


def _concatenate_audio_segments(
    segments: list[bytes],
    output_format: str,
    sample_rate: int = 24000,
    gaps: list[float] | None = None,
) -> bytes:
    """Concatenate multiple audio byte segments using ffmpeg concat demuxer.

    If *gaps* is provided, inserts silence of the specified duration (seconds)
    before each segment. gaps[0] is ignored (first segment starts immediately).
    """
    if len(segments) == 1:
        return segments[0]

    gaps = gaps or [0.0] * len(segments)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        all_files: list[Path] = []
        for i, seg_bytes in enumerate(segments):
            if gaps[i] > 0:
                silence_path = tmp_path / f"sil_{i:04d}.wav"
                _generate_silence(gaps[i], sample_rate, silence_path)
                all_files.append(silence_path)
            seg_path = tmp_path / f"seg_{i:04d}.{output_format}"
            seg_path.write_bytes(seg_bytes)
            all_files.append(seg_path)

        list_file = tmp_path / "concat_list.txt"
        list_file.write_text(
            "\n".join(f"file '{f.name}'" for f in all_files),
            encoding="utf-8",
        )

        output_path = tmp_path / f"output.{output_format}"
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        return output_path.read_bytes()


def _build_atempo_chain(speed: float) -> str:
    """Build ffmpeg atempo filter chain for any speed > 0."""
    if speed <= 0:
        return "atempo=1.0"
    if 0.5 <= speed <= 2.0:
        return f"atempo={speed}"
    # Chain multiple atempo filters (each must be in [0.5, 2.0])
    filters = []
    remaining = speed
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining}")
    return ",".join(filters)


def _apply_audio_effects(
    audio_bytes: bytes,
    output_format: str,
    speed: float = 1.0,
    stereo: bool = False,
    spatial_jitter: bool = False,
) -> tuple[bytes, float]:
    """Apply speed and stereo spatial effects via ffmpeg.

    Returns (audio_bytes, duration_seconds).
    """
    if speed == 1.0 and not stereo and not spatial_jitter:
        info = sf.info(io.BytesIO(audio_bytes))
        return audio_bytes, info.duration

    filters = []
    if speed != 1.0:
        filters.append(_build_atempo_chain(speed))
    if stereo:
        filters.append("pan=stereo|c0=c0|c1=c0")
    # NOTE: apulsator was removed because it causes audible periodic volume pulsing
    # (amplitude modulation) which severely degrades TTS quality.  If spatial
    # enhancement is needed in the future, consider aecho (subtle reverb) or
    # chorus instead of AM-based effects.

    af = ",".join(filters)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / f"input.{output_format}"
        input_path.write_bytes(audio_bytes)
        output_path = tmp_path / f"output.{output_format}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-af", af,
            "-c:a", "copy" if (not stereo and not spatial_jitter) else "pcm_s16le",
            str(output_path),
        ]
        # For ogg/flac with re-encoding we need to handle codec properly
        if output_format == "ogg":
            cmd[-2] = "libvorbis"
        elif output_format == "flac":
            cmd[-2] = "flac"
        elif output_format == "wav":
            cmd[-2] = "pcm_s16le"

        subprocess.run(cmd, check=True, capture_output=True)
        result_bytes = output_path.read_bytes()
        info = sf.info(io.BytesIO(result_bytes))
        return result_bytes, info.duration


def create_engine(config, sample_manager) -> BaseEngine:
    """Factory: return remote Qwen engine if configured, else local MLX."""
    if getattr(config.model, "use_remote_qwen", False):
        from .qwen_remote import QwenRemoteEngine
        return QwenRemoteEngine(config, sample_manager)
    from .local_vibevoice import LocalVibeVoiceEngine
    return LocalVibeVoiceEngine(config, sample_manager)
