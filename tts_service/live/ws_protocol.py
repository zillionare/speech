"""WebSocket frame protocol utilities with typed constructors.

Each frame type has a dedicated constructor function. Tests should call
these constructors, not build raw dicts.

Full spec: .project/specs/spec.md SPEC-006.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


VALID_STATES = {
    "IDLE", "AI_SPEAKING", "RECORDING", "DETECTING",
    "PAUSED", "FINISHED", "ABANDONED", "ERROR",
}
VALID_SOURCES = {"tts", "live"}
VALID_ERROR_CODES = {"ASR_DOWN", "ASR_DEGRADED", "TTS_TIMEOUT", "TTS_ERROR", "INTERNAL"}


class InvalidFrameError(ValueError):
    """Raised when a frame does not match the protocol."""


# ── Server -> Client frame constructors ──────────────────────────────────

def live_state(state: str) -> Dict[str, Any]:
    """Construct a state-change frame."""
    if state not in VALID_STATES:
        raise InvalidFrameError(f"Invalid state: {state}")
    return {"type": "state", "state": state}


def segment_begin(index: int, source: str, speaker: str, text: str) -> Dict[str, Any]:
    """Construct a segment_start frame."""
    if source not in VALID_SOURCES:
        raise InvalidFrameError(f"Invalid source: {source}")
    if index < 0:
        raise InvalidFrameError(f"Invalid index: {index}")
    return {"type": "segment_start", "index": index, "source": source,
            "speaker": speaker, "text": text}


def asr_partial(text: str, audio_ms: int) -> Dict[str, Any]:
    """Construct an asr_partial frame."""
    if audio_ms < 0:
        raise InvalidFrameError(f"Invalid audio_ms: {audio_ms}")
    return {"type": "asr_partial", "text": text, "audio_ms": audio_ms}


def asr_final(text: str, matched_ratio: float) -> Dict[str, Any]:
    """Construct an asr_final frame."""
    if not (0.0 <= matched_ratio <= 1.0):
        raise InvalidFrameError(f"matched_ratio out of range: {matched_ratio}")
    return {"type": "asr_final", "text": text, "matched_ratio": matched_ratio}


def alignment_progress(matched_chars: int, total_chars: int) -> Dict[str, Any]:
    """Construct an alignment frame."""
    if matched_chars < 0 or total_chars < 0:
        raise InvalidFrameError("char counts must be non-negative")
    if matched_chars > total_chars:
        raise InvalidFrameError(f"matched_chars ({matched_chars}) > total_chars ({total_chars})")
    return {"type": "alignment", "matched_chars": matched_chars, "total_chars": total_chars}


def asr_warming(progress: float) -> Dict[str, Any]:
    """Construct an asr_warming frame."""
    if not (0.0 <= progress <= 1.0):
        raise InvalidFrameError(f"progress out of range: {progress}")
    return {"type": "asr_warming", "progress": progress}


def asr_unavailable(reason: str) -> Dict[str, Any]:
    """Construct an asr_unavailable frame."""
    return {"type": "asr_unavailable", "reason": reason}


def asr_degraded(consecutive_failures: int) -> Dict[str, Any]:
    """Construct an asr_degraded frame."""
    if consecutive_failures <= 0:
        raise InvalidFrameError("consecutive_failures must be positive")
    return {"type": "asr_degraded", "consecutive_failures": consecutive_failures}


def audio_info(sample_rate: int, channels: int = 1, bit_depth: int = 16) -> Dict[str, Any]:
    """Construct an audio_info frame (server -> client)."""
    if sample_rate <= 0:
        raise InvalidFrameError(f"Invalid sample_rate: {sample_rate}")
    return {"type": "audio_info", "sample_rate": sample_rate,
            "channels": channels, "bit_depth": bit_depth}


def error_frame(code: str, message: str) -> Dict[str, Any]:
    """Construct an error frame."""
    if code not in VALID_ERROR_CODES:
        raise InvalidFrameError(f"Invalid error code: {code}")
    return {"type": "error", "code": code, "message": message}


# ── Client -> Server frame constructors ──────────────────────────────────

def client_audio_info(sample_rate: int, channels: int = 1, bit_depth: int = 16) -> Dict[str, Any]:
    """Construct a client->server audio_info frame."""
    if sample_rate <= 0:
        raise InvalidFrameError(f"Invalid sample_rate: {sample_rate}")
    return {"type": "audio_info", "sample_rate": sample_rate,
            "channels": channels, "bit_depth": bit_depth}


def client_log(level: str, msg: str) -> Dict[str, Any]:
    """Construct a client_log frame."""
    if level not in {"info", "warn", "error"}:
        raise InvalidFrameError(f"Invalid log level: {level}")
    return {"type": "client_log", "level": level, "msg": msg}


def state_ack(state: str) -> Dict[str, Any]:
    """Construct a state_ack frame (client -> server)."""
    if state not in VALID_STATES:
        raise InvalidFrameError(f"Invalid state: {state}")
    return {"type": "state_ack", "state": state}


# ── Encoding / decoding ──────────────────────────────────────────────────

def encode_json_frame(payload: Dict[str, Any]) -> str:
    """Encode a JSON frame to string."""
    return json.dumps(payload)


def decode_json_frame(data: str) -> Dict[str, Any]:
    """Decode a JSON frame from string."""
    return json.loads(data)


def is_binary_frame(data: Any) -> bool:
    """Check if a received message is a binary (audio) frame."""
    return isinstance(data, (bytes, bytearray))
