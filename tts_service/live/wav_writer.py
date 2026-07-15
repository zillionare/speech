"""WAV writing utilities for live recording.

Full implementation — utility functions can be implemented directly.
Spec: .project/specs/spec.md SPEC-010.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> None:
    """Write PCM bytes to a WAV file."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)


class WavStreamWriter:
    """Stream WAV data chunk-by-chunk to disk."""

    def __init__(self, path: Path, sample_rate: int, channels: int = 1):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self._wav: Optional[wave.Wave_write] = None

    def __enter__(self):
        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(self.channels)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.sample_rate)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def write(self, pcm_bytes: bytes) -> None:
        if self._wav is None:
            self._wav = wave.open(str(self.path), "wb")
            self._wav.setnchannels(self.channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self.sample_rate)
        self._wav.writeframes(pcm_bytes)

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None
