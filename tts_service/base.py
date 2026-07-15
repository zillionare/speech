"""Base TTS engine interface and shared utilities."""

from __future__ import annotations

import io
import math
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from ..models import SpeakerResolution


# Parse lines like: Speaker: text
_TAGGED_LINE_RE = re.compile(r"^([^:\n]{1,80}):\s*(.+)$")


def _parse_tagged_dialogue(text: str) -> tuple[list[dict], bool]:
    """Parse text in Speaker: text format.

    Returns (segments, is_tagged). Each segment dict has:
        speaker: str
        text: str
    """
    segments: list[dict] = []
    current_meta: dict | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_meta and current_lines:
                segments.append({**current_meta, "text": "\n".join(current_lines)})
                current_meta = None
                current_lines = []
            continue

        match = _TAGGED_LINE_RE.match(stripped)
        if match:
            if current_meta and current_lines:
                segments.append({**current_meta, "text": "\n".join(current_lines)})

            speaker = match.group(1).strip()
            text_part = match.group(2).strip()

            current_meta = {"speaker": speaker}
            current_lines = [text_part]
        else:
            if current_meta:
                current_lines.append(stripped)
            else:
                # First non-empty line is not tagged → not a tagged dialogue
                return [], False

    if current_meta and current_lines:
        segments.append({**current_meta, "text": "\n".join(current_lines)})

    return segments, len(segments) > 0


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary; fallback to common paths."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in (
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ):
        if Path(candidate).exists():
            return candidate
    return "ffmpeg"


_FFMPEG_PATH = _find_ffmpeg()


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
        instructions: Optional[str] = None,
    ) -> GenerationResult:
        ...

    @abstractmethod
    def generate_dialogue(
        self,
        text: str,
        output_format: str = "wav",
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
        instructions: Optional[str] = None,
        segment_gap: Optional[float] = None,
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
        instructions: Optional[str] = None,
    ) -> GenerationResult:
        """Segment long text, generate each part, concatenate with ffmpeg.

        Caller should ensure len(text) > max_chars; this method also guards
        against a single segment to avoid unnecessary concatenation overhead.
        """
        from ..segmentation import segment_dialogue, segment_long_text

        text = _strip_markdown_headings(text)

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
                instructions=instructions,
            )

        audio_parts: list[bytes] = []
        total_gen_seconds = 0.0
        total_duration_seconds = 0.0
        all_resolutions: list[SpeakerResolution] = []
        seen_speakers: set[str] = set()

        for seg in segments:
            mapped_voice = voice_mapping.get(seg.speaker) if voice_mapping and seg.speaker else seg.speaker
            result = self.generate_single(
                text=seg.text,
                voice=mapped_voice or preferred_voice,
                output_format=output_format,
                instructions=instructions,
            )
            audio_parts.append(result.audio_bytes)
            total_gen_seconds += result.generation_seconds
            total_duration_seconds += result.duration_seconds
            for spk in result.resolved_speakers:
                if spk.resolved_voice not in seen_speakers:
                    seen_speakers.add(spk.resolved_voice)
                    all_resolutions.append(spk)

        gaps = _compute_gaps(len(segments), segment_gap)
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
        stereo: bool = False,
        spatial_jitter: bool = False,
        segment_gap: float = 1.0,
        instructions: Optional[str] = None,
    ):
        """Generator that yields progress dicts and finally a complete result dict.

        Yields:
            {"type": "progress", "current": int, "total": int, "text": str}
            {"type": "complete", "result": GenerationResult}
            {"type": "error", "message": str}
        """
        from ..segmentation import segment_dialogue, segment_long_text

        try:
            text = _strip_markdown_headings(text)

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
                    instructions=instructions,
                )
                audio_bytes, duration = self._apply_postprocess_effects(
                    result.audio_bytes, output_format, stereo, spatial_jitter
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
                mapped_voice = voice_mapping.get(seg.speaker) if voice_mapping and seg.speaker else seg.speaker
                result = self.generate_single(
                    text=seg.text,
                    voice=mapped_voice or preferred_voice,
                    output_format=output_format,
                    instructions=instructions,
                )
                audio_parts.append(result.audio_bytes)
                total_gen_seconds += result.generation_seconds
                total_duration_seconds += result.duration_seconds
                for spk in result.resolved_speakers:
                    if spk.resolved_voice not in seen_speakers:
                        seen_speakers.add(spk.resolved_voice)
                        all_resolutions.append(spk)

            gaps = _compute_gaps(len(segments), segment_gap)
            final_audio = _concatenate_audio_segments(audio_parts, output_format, gaps=gaps)
            final_audio, final_duration = self._apply_postprocess_effects(
                final_audio, output_format, stereo, spatial_jitter
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

    def _apply_postprocess_effects(
        self,
        audio_bytes: bytes,
        output_format: str,
        stereo: bool,
        spatial_jitter: bool,
    ) -> tuple[bytes, float]:
        """Hook for subclasses to customize post-processing.

        Default delegates to _apply_audio_effects.
        """
        return _apply_audio_effects(audio_bytes, output_format, stereo, spatial_jitter)


def _generate_silence(duration_seconds: float, sample_rate: int, output_path: Path, channels: int = 1) -> None:
    """Generate a silent audio file of given duration."""
    layout = "mono" if channels == 1 else "stereo"
    cmd = [
        _FFMPEG_PATH, "-y",
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl={layout}",
        "-t", str(duration_seconds),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_markdown_headings(text: str) -> str:
    """Remove markdown heading lines and HTML comments."""
    # Strip inline HTML comments first
    text = _HTML_COMMENT_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            lines.append(line)
        elif stripped.startswith("#") and not stripped.startswith("##") and len(stripped) > 1 and stripped[1] == " ":
            # '# heading' — skip it
            continue
        elif stripped.startswith("##"):
            # '## heading' or more — skip it
            continue
        else:
            lines.append(line)
    return "\n".join(lines)


def _compute_gaps(segment_count: int, segment_gap: float) -> list[float]:
    """Return list of silence durations before each segment.

    gaps[0] is always 0.0. All subsequent gaps use the uniform segment_gap.
    """
    return [0.0] + [segment_gap] * (segment_count - 1)


def _concatenate_audio_segments(
    segments: list[bytes],
    output_format: str,
    sample_rate: int = 24000,
    gaps: list[float] | None = None,
    pre_pauses: list[float] | None = None,
    post_pauses: list[float] | None = None,
    base_gap: float = 0.0,
) -> bytes:
    """Concatenate multiple audio byte segments using ffmpeg concat demuxer.

    Legacy mode: pass *gaps* — inserts silence before each segment.
        gaps[0] is ignored (first segment starts immediately).

    Per-segment pause mode: pass *pre_pauses* and *post_pauses*.
        Silence before segment i is:
            pre_pauses[i]                       (i == 0)
            base_gap + pre_pauses[i] + post_pauses[i-1]   (i > 0)
        Silence after the last segment is post_pauses[-1] (if > 0).
    """
    if len(segments) == 1 and not pre_pauses and not post_pauses and (not gaps or gaps[0] == 0):
        return segments[0]

    # Detect channel count from the first segment so silence matches
    channels = 1
    try:
        info = sf.info(io.BytesIO(segments[0]))
        channels = info.channels
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        all_files: list[Path] = []

        use_pause_mode = pre_pauses is not None or post_pauses is not None

        if use_pause_mode:
            pre_pauses = pre_pauses or [0.0] * len(segments)
            post_pauses = post_pauses or [0.0] * len(segments)
            for i, seg_bytes in enumerate(segments):
                if i == 0:
                    silence_dur = pre_pauses[i]
                else:
                    silence_dur = base_gap + pre_pauses[i] + post_pauses[i - 1]
                if silence_dur > 0:
                    silence_path = tmp_path / f"sil_pre_{i:04d}.wav"
                    _generate_silence(silence_dur, sample_rate, silence_path, channels=channels)
                    all_files.append(silence_path)
                seg_path = tmp_path / f"seg_{i:04d}.{output_format}"
                seg_path.write_bytes(seg_bytes)
                all_files.append(seg_path)
            # Append silence after the last segment if post_pause is set
            if post_pauses and post_pauses[-1] > 0:
                silence_path = tmp_path / f"sil_post_{len(segments) - 1:04d}.wav"
                _generate_silence(post_pauses[-1], sample_rate, silence_path, channels=channels)
                all_files.append(silence_path)
        else:
            gaps = gaps or [0.0] * len(segments)
            for i, seg_bytes in enumerate(segments):
                if gaps[i] > 0:
                    silence_path = tmp_path / f"sil_{i:04d}.wav"
                    _generate_silence(gaps[i], sample_rate, silence_path, channels=channels)
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
            _FFMPEG_PATH,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        return output_path.read_bytes()


def _apply_spatial_jitter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply very slow complementary gain modulation to L/R channels.

    Simulates a speaker's head subtly turning left/right while talking.
    Frequency ~0.15 Hz (6.7 s per cycle), depth ±3% (about ±0.5 dB).
    This is inaudible as an "effect" but adds a gentle sense of movement.
    """
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    t = np.arange(len(audio)) / sr
    freq = 0.15  # ~6.7 seconds per full cycle
    depth = 0.03  # ±3% — barely perceptible, no quality loss

    left_gain = 1.0 + depth * np.sin(2 * math.pi * freq * t)
    right_gain = 1.0 - depth * np.sin(2 * math.pi * freq * t)

    audio[:, 0] *= left_gain
    audio[:, 1] *= right_gain
    return audio


def _apply_audio_effects(
    audio_bytes: bytes,
    output_format: str,
    stereo: bool = False,
    spatial_jitter: bool = False,
) -> tuple[bytes, float]:
    """Apply stereo conversion and spatial jitter.

    Stereo + spatial jitter are done in numpy (lossless quality).

    Returns (audio_bytes, duration_seconds).
    """
    if not stereo and not spatial_jitter:
        info = sf.info(io.BytesIO(audio_bytes))
        return audio_bytes, info.duration

    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    if spatial_jitter:
        audio = _apply_spatial_jitter(audio, sr)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    audio_bytes = buf.read()

    if output_format != "wav":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.wav"
            input_path.write_bytes(audio_bytes)
            output_path = tmp_path / f"output.{output_format}"
            cmd = [
                _FFMPEG_PATH, "-y",
                "-i", str(input_path),
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            audio_bytes = output_path.read_bytes()

    info = sf.info(io.BytesIO(audio_bytes))
    return audio_bytes, info.duration


def create_engine(config, sample_manager) -> BaseEngine:
    """Factory: return remote Qwen engine if configured, else local MLX."""
    if getattr(config.model, "use_remote_qwen", False):
        from .qwen_remote import QwenRemoteEngine
        return QwenRemoteEngine(config, sample_manager)
    from .local_vibevoice import LocalVibeVoiceEngine
    return LocalVibeVoiceEngine(config, sample_manager)
