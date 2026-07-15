"""Streaming TTS proxy.

Stub implementation — methods raise NotImplementedError.
Full spec: .project/specs/spec.md SPEC-005.
"""

from __future__ import annotations

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
        """Generate audio and yield fixed-duration PCM chunks."""
        raise NotImplementedError("StreamingTTSProxy.stream_segment not implemented")
        # The `yield` below makes this an async generator so `async for` works
        # even though the line above always raises. It is unreachable.
        yield b""
