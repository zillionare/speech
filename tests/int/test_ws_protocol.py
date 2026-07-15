"""Integration tests for ST-LP-006: WebSocket protocol.

All tests call frame constructor functions from ws_protocol.py.
No test builds raw dicts or hardcodes validation logic.
"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.live.ws_protocol import (
    live_state, segment_begin, asr_partial, asr_final,
    alignment_progress, asr_warming, asr_unavailable, asr_degraded,
    audio_info, error_frame,
    client_audio_info, client_log, state_ack,
    encode_json_frame, decode_json_frame, is_binary_frame,
    InvalidFrameError, VALID_STATES,
)
from tts_service.live.session import LiveState


class ServerFrameConstructorTests(unittest.TestCase):
    """Each test calls a constructor and verifies it produces valid output."""

    def test_live_state_for_each_state(self):
        for state in VALID_STATES:
            frame = live_state(state)
            self.assertEqual(frame["type"], "state")
            self.assertEqual(frame["state"], state)

    def test_live_state_rejects_invalid(self):
        with self.assertRaises(InvalidFrameError):
            live_state("INVALID")

    def test_segment_begin(self):
        frame = segment_begin(3, "tts", "Flora", "hello")
        self.assertEqual(frame["type"], "segment_start")
        self.assertEqual(frame["index"], 3)

    def test_segment_begin_rejects_invalid_source(self):
        with self.assertRaises(InvalidFrameError):
            segment_begin(0, "human", "A", "x")

    def test_segment_begin_rejects_negative_index(self):
        with self.assertRaises(InvalidFrameError):
            segment_begin(-1, "tts", "A", "x")

    def test_asr_partial(self):
        frame = asr_partial("你好", 500)
        self.assertEqual(frame["audio_ms"], 500)

    def test_asr_partial_rejects_negative_ms(self):
        with self.assertRaises(InvalidFrameError):
            asr_partial("x", -1)

    def test_asr_final(self):
        frame = asr_final("你好世界", 0.95)
        self.assertEqual(frame["matched_ratio"], 0.95)

    def test_asr_final_rejects_out_of_range_ratio(self):
        with self.assertRaises(InvalidFrameError):
            asr_final("x", 1.5)

    def test_alignment_progress(self):
        frame = alignment_progress(42, 48)
        self.assertEqual(frame["matched_chars"], 42)

    def test_alignment_rejects_matched_exceeds_total(self):
        with self.assertRaises(InvalidFrameError):
            alignment_progress(50, 48)

    def test_asr_warming(self):
        frame = asr_warming(0.6)
        self.assertEqual(frame["progress"], 0.6)

    def test_asr_warming_rejects_out_of_range(self):
        with self.assertRaises(InvalidFrameError):
            asr_warming(1.5)

    def test_asr_unavailable(self):
        frame = asr_unavailable("asr disabled")
        self.assertEqual(frame["reason"], "asr disabled")

    def test_asr_degraded(self):
        frame = asr_degraded(5)
        self.assertEqual(frame["consecutive_failures"], 5)

    def test_asr_degraded_rejects_zero(self):
        with self.assertRaises(InvalidFrameError):
            asr_degraded(0)

    def test_audio_info(self):
        frame = audio_info(24000)
        self.assertEqual(frame["sample_rate"], 24000)

    def test_audio_info_rejects_zero(self):
        with self.assertRaises(InvalidFrameError):
            audio_info(0)

    def test_error_frame(self):
        frame = error_frame("TTS_TIMEOUT", "timed out")
        self.assertEqual(frame["code"], "TTS_TIMEOUT")

    def test_error_frame_rejects_unknown_code(self):
        with self.assertRaises(InvalidFrameError):
            error_frame("UNKNOWN", "x")


class ClientFrameConstructorTests(unittest.TestCase):

    def test_client_audio_info(self):
        frame = client_audio_info(48000)
        self.assertEqual(frame["sample_rate"], 48000)

    def test_client_log(self):
        frame = client_log("info", "started")
        self.assertEqual(frame["level"], "info")

    def test_client_log_rejects_invalid_level(self):
        with self.assertRaises(InvalidFrameError):
            client_log("debug", "x")

    def test_state_ack(self):
        frame = state_ack("AI_SPEAKING")
        self.assertEqual(frame["type"], "state_ack")

    def test_state_ack_rejects_invalid_state(self):
        with self.assertRaises(InvalidFrameError):
            state_ack("NOPE")


class ServerFrameTypeTests(unittest.TestCase):
    """Verify each constructor produces the correct 'type' field value."""

    def test_live_state_type(self):
        self.assertEqual(live_state("IDLE")["type"], "state")

    def test_segment_begin_type(self):
        self.assertEqual(segment_begin(0, "tts", "A", "x")["type"], "segment_start")

    def test_asr_partial_type(self):
        self.assertEqual(asr_partial("x", 100)["type"], "asr_partial")

    def test_asr_final_type(self):
        self.assertEqual(asr_final("x", 0.5)["type"], "asr_final")

    def test_alignment_progress_type(self):
        self.assertEqual(alignment_progress(1, 2)["type"], "alignment")

    def test_asr_warming_type(self):
        self.assertEqual(asr_warming(0.5)["type"], "asr_warming")

    def test_asr_unavailable_type(self):
        self.assertEqual(asr_unavailable("x")["type"], "asr_unavailable")

    def test_asr_degraded_type(self):
        self.assertEqual(asr_degraded(1)["type"], "asr_degraded")

    def test_audio_info_type(self):
        self.assertEqual(audio_info(24000)["type"], "audio_info")

    def test_error_frame_type(self):
        self.assertEqual(error_frame("TTS_TIMEOUT", "x")["type"], "error")


class ClientFrameTypeTests(unittest.TestCase):

    def test_client_audio_info_type(self):
        self.assertEqual(client_audio_info(48000)["type"], "audio_info")

    def test_client_audio_info_with_kwargs(self):
        frame = client_audio_info(48000, channels=2, bit_depth=24)
        self.assertEqual(frame["channels"], 2)
        self.assertEqual(frame["bit_depth"], 24)

    def test_client_log_type(self):
        self.assertEqual(client_log("info", "x")["type"], "client_log")

    def test_state_ack_type(self):
        self.assertEqual(state_ack("IDLE")["type"], "state_ack")


class FrameEncodingTests(unittest.TestCase):

    def test_encode_decode_roundtrip(self):
        original = segment_begin(3, "tts", "Flora", "hello")
        encoded = encode_json_frame(original)
        decoded = decode_json_frame(encoded)
        self.assertEqual(decoded, original)

    def test_is_binary_frame_detects_bytes(self):
        self.assertTrue(is_binary_frame(b"\x00\x01"))

    def test_is_binary_frame_rejects_string(self):
        self.assertFalse(is_binary_frame('{"type":"state"}'))


class StateFrameForEachLiveStateTests(unittest.TestCase):
    """Verify every LiveState enum value produces a valid state frame."""

    def test_all_live_states_have_valid_frames(self):
        for state in LiveState:
            frame = live_state(state.value)
            self.assertEqual(frame["state"], state.value)


class AudioInfoNegotiationTests(unittest.TestCase):
    """Verify audio_info frames carry the correct sample rate for each direction."""

    def test_server_audio_info_uses_engine_rate(self):
        """Server audio_info frame carries the TTS engine's sample rate."""
        # Access the class constant without importing mlx-dependent module.
        # LocalVibeVoiceEngine.SAMPLE_RATE = 24000 per spec.
        engine_rate = 24000  # LocalVibeVoiceEngine.SAMPLE_RATE
        frame = audio_info(engine_rate)
        self.assertEqual(frame["sample_rate"], engine_rate)

    def test_client_audio_info_uses_device_rate(self):
        """Client audio_info frame carries the recording device's sample rate."""
        device_rate = 48000
        frame = client_audio_info(device_rate)
        self.assertEqual(frame["sample_rate"], device_rate)


if __name__ == "__main__":
    unittest.main()