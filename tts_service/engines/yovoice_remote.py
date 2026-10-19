"""Remote yovoice engine that forwards generation to a `yovoice serve` HTTP API."""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
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
    _apply_audio_effects,
    _parse_tagged_dialogue,
    _strip_markdown_headings,
)


class YovoiceRemoteEngine(BaseEngine):
    """Uploads a local reference voice to yovoice, then requests generation."""

    DEFAULT_BASE_URL = "http://192.168.0.102:8080"
    DEFAULT_MODEL_ID = "index-2.5-q8"
    DEFAULT_SAMPLE_RATE = 24000

    @property
    def sample_rate(self) -> int:
        return self.DEFAULT_SAMPLE_RATE

    def __init__(self, config: Config, sample_manager: SampleManager):
        self.config = config
        self.sample_manager = sample_manager
        self.base_url = getattr(
            config.model, "yovoice_base_url", self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.model_id = getattr(
            config.model, "yovoice_model_id", self.DEFAULT_MODEL_ID
        )
        # Cache the remote voice id so repeated segments don't re-upload.
        self._voice_cache: dict[tuple, str] = {}

    def ensure_voice_cache_ready(self, speaker: str) -> VoiceSample:
        return self.sample_manager.resolve_or_default(speaker)

    def _resolve_voice(self, voice: Optional[str]) -> tuple[str, Path, str]:
        sample = self.sample_manager.resolve_or_default(
            voice or self.config.voices.default_voice,
        )
        transcript = (
            sample.txt_path.read_text(encoding="utf-8").strip()
            if sample.txt_path.exists()
            else ""
        )
        return sample.speaker, sample.wav_path, transcript

    def generate_single(
        self,
        text: str,
        voice: Optional[str],
        output_format: str = "wav",
        instructions: Optional[str] = None,
    ) -> GenerationResult:
        speaker, wav_path, ref_text = self._resolve_voice(voice)
        start = time.perf_counter()
        audio_bytes = self._call_remote(
            text=text,
            speaker=speaker,
            wav_path=wav_path,
            ref_text=ref_text,
            output_format=output_format,
            instructions=instructions,
        )
        generation_seconds = time.perf_counter() - start
        duration_seconds = self._estimate_duration(audio_bytes)
        if output_format == "wav":
            audio_bytes, duration_seconds = _apply_audio_effects(
                audio_bytes,
                "wav",
                stereo=getattr(self.config.model, "stereo", False),
                spatial_jitter=getattr(self.config.model, "spatial_jitter", False),
            )
        resolved_name = self.sample_manager.resolve_or_default(
            voice or self.config.voices.default_voice,
        ).speaker
        return GenerationResult(
            audio_bytes=audio_bytes,
            output_format=output_format,
            generation_seconds=generation_seconds,
            duration_seconds=duration_seconds,
            resolved_speakers=[SpeakerResolution(
                requested_name=voice or resolved_name,
                resolved_voice=resolved_name,
                used_default=False,
                transcript_preview=ref_text[:120] if ref_text else None,
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

    def _ensure_voice_id(self, speaker: str, wav_path: Path) -> str:
        """Upload the reference wav once and cache the returned voice id."""
        stat = wav_path.stat()
        key = (speaker, str(wav_path), stat.st_mtime_ns, stat.st_size)
        cached = self._voice_cache.get(key)
        if cached:
            return cached

        with open(wav_path, "rb") as handle:
            audio_bytes = handle.read()
        name = urllib.parse.quote(speaker, safe="")
        request = urllib.request.Request(
            f"{self.base_url}/v1/voices?name={name}",
            data=audio_bytes,
            headers=self._upload_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Yovoice HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Yovoice connection failed: {exc.reason}") from exc

        voice_id = payload.get("id") if isinstance(payload, dict) else None
        if not voice_id and isinstance(payload, dict):
            # The docs only state that the saved voice id is returned; the exact
            # nesting is unverified, so also accept a nested voice.id shape.
            nested = payload.get("voice")
            if isinstance(nested, dict):
                voice_id = nested.get("id")
        if not voice_id:
            raise RuntimeError(
                f"Yovoice voice upload returned no id: {str(payload)[:500]}"
            )
        self._voice_cache[key] = str(voice_id)
        return str(voice_id)

    def _call_remote(
        self,
        text: str,
        speaker: str,
        wav_path: Path,
        ref_text: str,
        output_format: str,
        instructions: Optional[str] = None,
    ) -> bytes:
        voice_id = self._ensure_voice_id(speaker, wav_path)
        payload = {
            "text": text,
            "modelId": self.model_id,
            "voiceId": voice_id,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._request_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Yovoice HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Yovoice connection failed: {exc.reason}") from exc

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = getattr(self.config.model, "yovoice_api_token", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _upload_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/octet-stream"}
        token = getattr(self.config.model, "yovoice_api_token", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _estimate_duration(audio_bytes: bytes) -> float:
        try:
            return sf.info(io.BytesIO(audio_bytes)).duration
        except Exception:
            return 0.0
