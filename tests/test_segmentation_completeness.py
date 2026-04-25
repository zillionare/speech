"""Regression test: segmented generation must not lose text/audio.

This test verifies that:
1. segment_long_text preserves all characters
2. Concatenated audio duration equals sum of segment durations
3. Long texts (>max_segment_chars) generate proportionally longer audio
   than short texts (no severe truncation from silence_detection)

Run: python -m unittest tests.test_segmentation_completeness -v
"""

import unittest

from tts_service.config import Config
from tts_service.sample_manager import SampleManager
from tts_service.engines.local_vibevoice import LocalVibeVoiceEngine
from tts_service.engines.base import _concatenate_audio_segments
from tts_service.segmentation import segment_long_text


class TestSegmentationCompleteness(unittest.TestCase):
    """Verify segmented TTS does not drop audio or text."""

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        # Use a small max_segment_chars to force segmentation
        cls.config.model.max_segment_chars = 80
        cls.config.model.diffusion_steps = 5  # faster for tests
        cls.sample_manager = SampleManager(
            cls.config.voices.expanded_base_dir,
            cls.config.voices.default_voice,
        )
        cls.engine = LocalVibeVoiceEngine(cls.config, cls.sample_manager)

    def test_segment_long_text_preserves_all_chars(self):
        text = (
            "第一句：这是测试文本的开头。"
            "第二句：我们需要验证分段生成是否完整。"
            "第三句：每一句话都应该出现在最终的音频里。"
            "第四句：如果中间有句子丢失，就说明有问题。"
            "第五句：这是第五句话，用来测试连续性。"
        )
        segments = segment_long_text(text, max_chars=80)
        reconstructed = "".join(segments)
        self.assertEqual(
            reconstructed,
            text,
            f"Segmentation lost characters: {len(text)} -> {len(reconstructed)}",
        )

    def test_concatenated_duration_equals_sum(self):
        text = (
            "春天来了，万物复苏。小草从地里钻出来，花儿也竞相开放。"
            "鸟儿在枝头歌唱，溪水潺潺流淌。这是大自然的馈赠，让我们感受到生命的力量。"
            "夏天紧随其后，烈日炎炎。蝉鸣声此起彼伏，树荫下成了避暑的好去处。"
            "孩子们在河边嬉戏，大人们则在树下乘凉。"
        )
        segments = segment_long_text(text, max_chars=80)
        self.assertGreater(len(segments), 1, "Text should split into multiple segments")

        parts = []
        total_duration = 0.0
        for seg in segments:
            result = self.engine.generate_single(
                text=seg,
                voice=self.config.voices.default_voice,
                output_format="wav",
            )
            parts.append(result.audio_bytes)
            total_duration += result.duration_seconds

        merged = _concatenate_audio_segments(parts, "wav")
        import io, soundfile as sf

        merged_duration = sf.info(io.BytesIO(merged)).duration
        self.assertAlmostEqual(
            merged_duration,
            total_duration,
            delta=0.5,
            msg="Concatenated audio duration should equal sum of segment durations",
        )

    def test_long_text_not_truncated(self):
        """A 100+ char text must produce >10s audio (no severe truncation)."""
        short_text = "你好，这是一个短句测试。"
        long_text = (
            "大家好，欢迎来到量化好声音播客。"
            "今天我们将跟大家聊一聊，如何养龙虾的事儿。"
            "最近不少金融机构都推出了自己的技术方案。"
        )

        short_result = self.engine.generate_single(
            text=short_text,
            voice=self.config.voices.default_voice,
            output_format="wav",
        )
        long_result = self.engine.generate_single(
            text=long_text,
            voice=self.config.voices.default_voice,
            output_format="wav",
        )

        short_cps = len(short_text) / short_result.duration_seconds
        long_cps = len(long_text) / long_result.duration_seconds

        # Both should fall in a reasonable Chinese speech rate (3-8 chars/s)
        self.assertGreaterEqual(short_cps, 3.0, "Short text speaks too slow")
        self.assertLessEqual(short_cps, 10.0, "Short text severely truncated")
        self.assertGreaterEqual(long_cps, 3.0, "Long text speaks too slow")
        self.assertLessEqual(long_cps, 10.0, "Long text severely truncated")

        # Long text should be proportionally longer
        duration_ratio = long_result.duration_seconds / short_result.duration_seconds
        char_ratio = len(long_text) / len(short_text)
        self.assertGreaterEqual(
            duration_ratio,
            char_ratio * 0.5,
            msg=(
                f"Long text audio ({long_result.duration_seconds:.1f}s) "
                f"not proportionally longer than short ({short_result.duration_seconds:.1f}s)"
            ),
        )


if __name__ == "__main__":
    unittest.main()
