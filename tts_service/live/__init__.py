"""Stub package for live podcast modules."""

from tts_service.live.session import (
    LiveState, LiveSession, LiveSessionRegistry,
    IllegalStateTransition, Trigger, get_legal_transitions,
)
from tts_service.live.end_detector import (
    EndDetector, normalize_text, compute_alignment_ratio,
    longest_common_subsequence_length,
)
from tts_service.live.asr_engine import EmbeddedASR, ASRConfig, ASRResult
from tts_service.live.streaming_engine import StreamingTTSProxy
from tts_service.live.wav_writer import write_wav, WavStreamWriter
from tts_service.live.ws_protocol import (
    live_state, segment_begin, asr_partial, asr_final,
    alignment_progress, asr_warming, asr_unavailable, asr_degraded,
    audio_info, error_frame, client_audio_info, client_log, state_ack,
    encode_json_frame, decode_json_frame, is_binary_frame,
    InvalidFrameError,
)

__all__ = [
    "LiveState", "LiveSession", "LiveSessionRegistry",
    "IllegalStateTransition", "Trigger", "get_legal_transitions",
    "EndDetector", "normalize_text", "compute_alignment_ratio",
    "longest_common_subsequence_length",
    "EmbeddedASR", "ASRConfig", "ASRResult",
    "StreamingTTSProxy",
    "write_wav", "WavStreamWriter",
    "live_state", "segment_begin", "asr_partial", "asr_final",
    "alignment_progress", "asr_warming", "asr_unavailable", "asr_degraded",
    "audio_info", "error_frame", "client_audio_info", "client_log", "state_ack",
    "encode_json_frame", "decode_json_frame", "is_binary_frame",
    "InvalidFrameError",
]
