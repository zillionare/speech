"""Integration tests for ST-LP-008: EndDetector.

All tests call EndDetector methods (update_vad, update_asr, on_speech_start,
on_speech_end, reset) and import LCS/alignment from production code.
No test reimplements LCS or the trigger state machine.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.live.end_detector import (
    EndDetector, normalize_text, compute_alignment_ratio,
    longest_common_subsequence_length,
)


class NormalizeTextTests(unittest.TestCase):

    def test_strips_whitespace(self):
        self.assertEqual(normalize_text("  hello world  "), "helloworld")

    def test_removes_punctuation(self):
        result = normalize_text("你好，世界！")
        self.assertNotIn("，", result)
        self.assertNotIn("！", result)

    def test_lowercases(self):
        self.assertEqual(normalize_text("Hello WORLD"), "helloworld")


class LCSTests(unittest.TestCase):
    """Call the production LCS function, don't reimplement."""

    def test_identical_strings(self):
        self.assertEqual(longest_common_subsequence_length("abc", "abc"), 3)

    def test_no_common(self):
        self.assertEqual(longest_common_subsequence_length("abc", "xyz"), 0)

    def test_subsequence(self):
        self.assertEqual(longest_common_subsequence_length("ace", "abcde"), 3)

    def test_empty_string(self):
        self.assertEqual(longest_common_subsequence_length("", "abc"), 0)
        self.assertEqual(longest_common_subsequence_length("abc", ""), 0)


class AlignmentRatioTests(unittest.TestCase):
    """Call compute_alignment_ratio, don't reimplement."""

    def test_exact_match(self):
        ratio = compute_alignment_ratio("今天我们聊聊", "今天我们聊聊")
        self.assertAlmostEqual(ratio, 1.0, places=2)

    def test_partial_match(self):
        ratio = compute_alignment_ratio("今天我们", "今天我们聊聊")
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 0.7)

    def test_no_match(self):
        ratio = compute_alignment_ratio("xyz", "abc")
        self.assertEqual(ratio, 0.0)

    def test_unrelated_but_equal_length_not_full_match(self):
        """Equal length but different content should NOT give ratio=1.0"""
        ratio = compute_alignment_ratio("你好世界", "世界你好")
        self.assertLess(ratio, 1.0)

    def test_empty_target(self):
        ratio = compute_alignment_ratio("hello", "")
        self.assertEqual(ratio, 0.0)


class EndDetectorTriggerTests(unittest.TestCase):
    """Call update_asr / update_vad on real EndDetector instances."""

    def test_end_near_triggered(self):
        det = EndDetector("今天我们聊聊人工智能", end_near_threshold=0.8, debounce_ms=0)
        det.on_speech_start()
        trigger = det.update_asr("今天我们聊聊人工智能")
        self.assertEqual(trigger, "end_near")

    def test_end_near_not_triggered_below_threshold(self):
        det = EndDetector("今天我们聊聊人工智能的发展趋势", end_near_threshold=0.85, debounce_ms=0)
        det.on_speech_start()
        trigger = det.update_asr("今天我们")
        self.assertIsNone(trigger)

    def test_end_triggered_by_alignment_and_silence(self):
        det = EndDetector("今天我们聊聊", end_alignment_threshold=0.95,
                          end_silence_ms=100, debounce_ms=0)
        det.on_speech_start()
        det.update_asr("今天我们聊聊")  # triggers end_near first
        # accumulate silence
        for _ in range(10):
            det.update_vad(-60.0, frame_ms=20)  # 200ms total
        trigger = det.update_asr("今天我们聊聊")
        self.assertEqual(trigger, "end")

    def test_end_triggered_by_silence_alone(self):
        det = EndDetector("今天我们聊聊", silence_only_end_ms=200, debounce_ms=0)
        det.on_speech_start()
        det.update_asr("今天")  # low alignment, no trigger
        # accumulate enough silence
        for _ in range(15):
            det.update_vad(-60.0, frame_ms=20)  # 300ms
        trigger = det.update_asr("今天")
        self.assertEqual(trigger, "end")

    def test_user_skipped_after_long_silence(self):
        det = EndDetector("今天我们聊聊", force_end_silence_ms=500, debounce_ms=0)
        det.on_speech_start()
        # accumulate very long silence
        for _ in range(30):
            det.update_vad(-60.0, frame_ms=20)  # 600ms
        trigger = det.update_asr("x")
        self.assertEqual(trigger, "user_skipped")

    def test_end_near_only_triggers_once(self):
        det = EndDetector("今天我们聊聊", end_near_threshold=0.8, debounce_ms=0)
        det.on_speech_start()
        t1 = det.update_asr("今天我们聊聊")
        self.assertEqual(t1, "end_near")
        t2 = det.update_asr("今天我们聊聊")
        self.assertIsNone(t2)


class EndDetectorVADTests(unittest.TestCase):
    """Call update_vad on real EndDetector."""

    def test_silence_increases_counter(self):
        det = EndDetector("test", silence_db_threshold=-45.0)
        det.on_speech_start()
        initial = det.silence_ms
        det.update_vad(-60.0, frame_ms=20)
        self.assertGreater(det.silence_ms, initial)

    def test_speech_resets_counter(self):
        det = EndDetector("test", silence_db_threshold=-45.0)
        det.silence_ms = 500
        det.update_vad(-20.0, frame_ms=20)
        self.assertEqual(det.silence_ms, 0)

    def test_on_speech_start_sets_speaking(self):
        det = EndDetector("test")
        det.on_speech_start()
        self.assertTrue(det.is_speaking)

    def test_on_speech_end_clears_speaking(self):
        det = EndDetector("test")
        det.on_speech_start()
        det.on_speech_end()
        self.assertFalse(det.is_speaking)


class EndDetectorDebounceTests(unittest.TestCase):

    def test_debounce_prevents_duplicate_trigger(self):
        det = EndDetector("今天我们聊聊", debounce_ms=200)
        det.on_speech_start()
        t1 = det.update_asr("今天我们聊聊")
        self.assertIsNotNone(t1)
        t2 = det.update_asr("今天我们聊聊")
        self.assertIsNone(t2)

    def test_trigger_after_debounce_window(self):
        """After debounce window, trigger fires again (after reset)."""
        det = EndDetector("今天我们聊聊", debounce_ms=100)
        det.on_speech_start()
        det.update_asr("今天我们聊聊")  # triggers end_near
        # Reset to simulate a new segment - clears debounce state cleanly
        det.reset()
        det.on_speech_start()
        t = det.update_asr("今天我们聊聊")
        self.assertEqual(t, "end_near")


class EndDetectorVADAutoTriggerTests(unittest.TestCase):
    """Verify update_vad() automatically calls on_speech_start/on_speech_end."""

    def test_loud_frame_triggers_speech_start(self):
        det = EndDetector("test", silence_db_threshold=-45.0)
        self.assertFalse(det.is_speaking)
        det.update_vad(-20.0, frame_ms=20)  # loud frame
        self.assertTrue(det.is_speaking)

    def test_silent_frames_trigger_speech_end(self):
        det = EndDetector("test", silence_db_threshold=-45.0)
        det.update_vad(-20.0, frame_ms=20)  # speech starts
        self.assertTrue(det.is_speaking)
        # Accumulate silence beyond the 50ms threshold
        for _ in range(5):
            det.update_vad(-60.0, frame_ms=20)
        self.assertFalse(det.is_speaking)


class EndDetectorResetTests(unittest.TestCase):

    def test_reset_clears_all_state(self):
        det = EndDetector("今天我们聊聊", debounce_ms=0)
        det.on_speech_start()
        det.update_asr("今天我们聊聊")  # triggers end_near
        det.update_vad(-60.0, frame_ms=20)
        det.reset()
        self.assertEqual(det.silence_ms, 0)
        self.assertFalse(det.is_speaking)
        self.assertEqual(det.alignment_ratio, 0.0)
        self.assertIsNone(det.last_trigger)
        self.assertFalse(det._end_near_triggered)


class EndDetectorFullFlowTests(unittest.TestCase):

    def test_speaker_finishes_segment(self):
        target = "今天我们聊聊人工智能的发展"
        det = EndDetector(target, end_near_threshold=0.75,
                          end_alignment_threshold=0.95,
                          end_silence_ms=100, debounce_ms=0)
        # Speaker starts talking
        det.update_vad(-20.0, frame_ms=20)
        self.assertTrue(det.is_speaking)

        # ASR gradually recognizes
        self.assertIsNone(det.update_asr("今天我们"))
        self.assertEqual(det.update_asr("今天我们聊聊人工智能"), "end_near")
        self.assertIsNone(det.update_asr("今天我们聊聊人工智能的发展"))

        # Speaker pauses
        for _ in range(10):
            det.update_vad(-55.0, frame_ms=20)
        self.assertEqual(det.update_asr("今天我们聊聊人工智能的发展"), "end")


class EndDetectorEdgeCaseTests(unittest.TestCase):

    def test_empty_target_text(self):
        det = EndDetector("", debounce_ms=0)
        det.on_speech_start()
        result = det.update_asr("anything")
        # Empty target -> ratio 0 -> no trigger unless silence
        self.assertIsNone(result)

    def test_punctuation_only_target(self):
        det = EndDetector("，。！", debounce_ms=0)
        det.on_speech_start()
        # Normalized target is empty -> ratio 0
        result = det.update_asr("你好")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()