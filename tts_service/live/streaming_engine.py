"""Streaming TTS proxy.

Wraps a BaseEngine to yield audio in fixed-duration PCM chunks.
Full spec: .project/specs/spec.md SPEC-005.
ACC-005-1: stream_segment yields ~200ms PCM chunks.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from tts_service.engines.base import BaseEngine


class StreamingTTSProxy:
    """Wraps a BaseEngine to yield audio in chunks."""

    def __init__(self, engine: "BaseEngine"):
        self.engine = engine
        self.sr = engine.sample_rate

    async def stream_segment(
        self,
        text: str,
        voice: str,
        chunk_ms: int = 200,
    ) -> AsyncIterator[bytes]:
        """Generate audio and yield fixed-duration PCM chunks.

        Calls engine.generate_single (blocking) in a thread pool, then slices
        the returned WAV into raw PCM chunks of approximately chunk_ms duration.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.engine.generate_single(text=text, voice=voice),
        )

        wav_bytes = result.audio_bytes
        pcm_bytes = self._strip_wav_header(wav_bytes)

        bytes_per_chunk = int(self.sr * chunk_ms / 1000) * 2  # 16-bit = 2 bytes/sample
        if bytes_per_chunk <= 0:
            bytes_per_chunk = len(pcm_bytes)

        offset = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset + bytes_per_chunk]
            offset += bytes_per_chunk
            yield chunk
            await asyncio.sleep(0)

    @staticmethod
    def _strip_wav_header(wav_bytes: bytes) -> bytes:
        """Strip the WAV header and return raw PCM samples."""
        buf = io.BytesIO(wav_bytes)
        try:
            with wave.open(buf, "rb") as w:
                return w.readframes(w.getnframes())
        except Exception:
            return wav_bytes
