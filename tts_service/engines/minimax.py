"""MiniMax TTS engine with native and OpenAI-compatible API modes."""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import soundfile as sf

from ..config import Config
from ..models import SpeakerResolution
from ..sample_manager import SampleManager, VoiceSample
from .base import (
    BaseEngine,
    GenerationResult,
    _FFMPEG_PATH,
    _apply_audio_effects,
    _concatenate_audio_segments,
    _parse_tagged_dialogue,
    _strip_markdown_headings,
)


class MiniMaxEngine(BaseEngine):
    """Call MiniMax TTS directly or through a LiteLLM OpenAI endpoint."""

    DEFAULT_SAMPLE_RATE = 32000
    DEFAULT_VOICE_ID = "male-qn-qingse"

    def __init__(self, config: Config, sample_manager: SampleManager):
        self.config = config
        self.sample_manager = sample_manager

    @property
    def sample_rate(self) -> int:
        return self.DEFAULT_SAMPLE_RATE

    def ensure_voice_cache_ready(self, speaker: str) -> VoiceSample:
        """MiniMax stores voice clones remotely; validate the local name only."""
        return self.sample_manager.resolve_or_default(speaker)

    def generate_single(
        self,
        text: str,
        voice: Optional[str],
        output_format: str = "wav",
        instructions: Optional[str] = None,
    ) -> GenerationResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Input text cannot be empty")

        start = time.perf_counter()
        audio_bytes = self._call_remote(normalized_text, voice)
        audio_bytes = self._ensure_wav(audio_bytes)
        audio_bytes, duration = _apply_audio_effects(
            audio_bytes,
            "wav",
            stereo=getattr(self.config.model, "stereo", False),
            spatial_jitter=getattr(self.config.model, "spatial_jitter", False),
        )
        if output_format != "wav":
            audio_bytes = self._convert_format(audio_bytes, output_format)
            duration = self._audio_duration(audio_bytes)

        resolved_voice = self._voice_id(voice)
        return GenerationResult(
            audio_bytes=audio_bytes,
            output_format=output_format,
            generation_seconds=time.perf_counter() - start,
            duration_seconds=duration,
            resolved_speakers=[SpeakerResolution(
                requested_name=voice or resolved_voice,
                resolved_voice=resolved_voice,
                used_default=not bool(voice),
            )],
            segment_count=1,
        )

    def generate_dialogue(
        self,
        text: str,
        output_format: str = "wav",
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
        instructions: Optional[str] = None,
        segment_gap: Optional[float] = None,
    ) -> GenerationResult:
        text = _strip_markdown_headings(text).strip()
        if not text:
            raise ValueError("Input text cannot be empty")

        tagged_segments, is_tagged = _parse_tagged_dialogue(text)
        max_chars = getattr(self.config.model, "max_segment_chars", 200)
        if is_tagged or len(text) > max_chars:
            return self._generate_with_segmentation(
                text=text,
                output_format=output_format,
                max_chars=max_chars,
                preferred_voice=preferred_voice,
                voice_mapping=voice_mapping,
                segment_gap=(
                    segment_gap
                    if segment_gap is not None
                    else getattr(self.config.model, "segment_gap_seconds", 1.0)
                ),
                instructions=instructions,
            )

        return self.generate_single(
            text=text,
            voice=preferred_voice,
            output_format=output_format,
            instructions=instructions,
        )

    def _voice_id(self, voice: Optional[str]) -> str:
        configured = getattr(self.config.model, "minimax_voice_id", "")
        return (configured or voice or self.DEFAULT_VOICE_ID).strip()

    def _call_remote(self, text: str, voice: Optional[str]) -> bytes:
        mode = getattr(self.config.model, "minimax_api_mode", "native").lower()
        if mode == "openai":
            return self._call_openai_compatible(text, voice)
        return self._call_native(text, voice)

    def _call_native(self, text: str, voice: Optional[str]) -> bytes:
        payload = {
            "model": getattr(self.config.model, "minimax_model_id", "speech-2.8-hd"),
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": self._voice_id(voice),
                "speed": self._bounded_float(
                    getattr(self.config.model, "minimax_speed", 1.0), 0.5, 2.0,
                ),
                "vol": self._bounded_float(
                    getattr(self.config.model, "minimax_volume", 1.0), 0.01, 10.0,
                ),
                "pitch": self._bounded_int(
                    getattr(self.config.model, "minimax_pitch", 0), -12, 12,
                ),
            },
            "audio_setting": {
                "sample_rate": self.DEFAULT_SAMPLE_RATE,
                "format": "wav",
                "channel": 1,
            },
            "output_format": "hex",
            "language_boost": "Chinese",
        }
        emotion = getattr(self.config.model, "minimax_emotion", "").strip()
        if emotion:
            payload["voice_setting"]["emotion"] = emotion
        response = self._request_json(self._endpoint("/v1/t2a_v2"), payload)
        base_resp = response.get("base_resp") or {}
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(
                f"MiniMax TTS failed: {base_resp.get('status_msg') or base_resp}"
            )
        encoded_audio = (response.get("data") or {}).get("audio")
        if not encoded_audio:
            raise RuntimeError("MiniMax TTS returned no audio")
        try:
            return bytes.fromhex(encoded_audio)
        except ValueError as exc:
            raise RuntimeError("MiniMax returned invalid hex audio") from exc

    def _call_openai_compatible(self, text: str, voice: Optional[str]) -> bytes:
        payload = {
            "model": getattr(self.config.model, "minimax_model_id", "speech-2.8-hd"),
            "input": text,
            "voice": self._voice_id(voice),
            # The current LiteLLM MiniMax adapter reliably returns MP3;
            # convert it locally so the rest of the service stays WAV-first.
            "response_format": "mp3",
        }
        return self._request_bytes(self._endpoint("/v1/audio/speech"), payload)

    def _request_json(self, url: str, payload: dict) -> dict:
        raw = self._request_bytes(url, payload)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("MiniMax returned invalid JSON") from exc

    def _request_bytes(self, url: str, payload: dict) -> bytes:
        api_key = getattr(self.config.model, "minimax_api_key", "").strip()
        if not api_key:
            raise RuntimeError(
                "MiniMax API key is not configured; set it in 系统设置 or MINIMAX_API_KEY"
            )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"MiniMax HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MiniMax connection failed: {exc.reason}") from exc

    def _endpoint(self, suffix: str) -> str:
        base = getattr(self.config.model, "minimax_base_url", "https://api.minimaxi.com").rstrip("/")
        if base.endswith(suffix):
            return base
        if base.endswith("/v1"):
            return f"{base}{suffix.removeprefix('/v1')}"
        return f"{base}{suffix}"

    @staticmethod
    def _ensure_wav(audio_bytes: bytes) -> bytes:
        try:
            sf.info(io.BytesIO(audio_bytes))
            return audio_bytes
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            source = directory / "source.audio"
            target = directory / "output.wav"
            source.write_bytes(audio_bytes)
            subprocess.run(
                [
                    _FFMPEG_PATH, "-nostdin", "-y", "-i", str(source),
                    "-ar", str(MiniMaxEngine.DEFAULT_SAMPLE_RATE), "-ac", "1",
                    "-c:a", "pcm_s16le", str(target),
                ],
                check=True,
                capture_output=True,
            )
            return target.read_bytes()

    @staticmethod
    def _convert_format(audio_bytes: bytes, output_format: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            source = directory / "source.wav"
            target = directory / f"output.{output_format}"
            source.write_bytes(audio_bytes)
            subprocess.run(
                [_FFMPEG_PATH, "-nostdin", "-y", "-i", str(source), str(target)],
                check=True,
                capture_output=True,
            )
            return target.read_bytes()

    @staticmethod
    def _audio_duration(audio_bytes: bytes) -> float:
        return sf.info(io.BytesIO(audio_bytes)).duration

    @staticmethod
    def _bounded_float(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _bounded_int(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, int(value)))
