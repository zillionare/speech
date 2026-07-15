"""Unit tests for tts_service.voice_design."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.voice_design import (
    ParseResult,
    TurnDesign,
    VoiceDesignError,
    has_markup,
    parse,
)


class VoiceDesignParseTests(unittest.TestCase):
    def test_empty_input(self) -> None:
        result = parse("")
        self.assertFalse(result.has_markup)
        self.assertEqual(result.turn_designs, [])
        self.assertEqual(result.cleaned_text, "")

    def test_body_without_markup(self) -> None:
        text = "Aaron: 你好。\nAnchen: 你好呀。\n"
        result = parse(text)
        self.assertFalse(result.has_markup)
        self.assertEqual(result.cleaned_text, text)
        self.assertEqual(
            result.turn_designs,
            [
                TurnDesign(speaker="Aaron", instructions=None),
                TurnDesign(speaker="Anchen", instructions=None),
            ],
        )

    def test_frontmatter_default_only(self) -> None:
        text = (
            "---\n"
            "voice-design:\n"
            "  default:\n"
            "    pitch: moderate\n"
            "---\n"
            "Aaron: 你好。\n"
        )
        result = parse(text)
        self.assertTrue(result.has_markup)
        self.assertEqual(result.cleaned_text, "Aaron: 你好。\n")
        self.assertEqual(result.turn_designs, [TurnDesign(speaker="Aaron", instructions="moderate pitch")])

    def test_speaker_override_key_order(self) -> None:
        text = (
            "---\n"
            "voice-design:\n"
            "  default:\n"
            "    pitch: moderate\n"
            "    accent: british\n"
            "  speakers:\n"
            "    Aaron:\n"
            "      style: whisper\n"
            "      accent: american\n"
            "---\n"
            "Aaron: 你好。\n"
        )
        result = parse(text)
        # pitch from default, style from speaker, accent overridden by speaker.
        self.assertEqual(
            result.turn_designs[0].instructions,
            "moderate pitch, whisper, american accent",
        )

    def test_line_level_adds_to_frontmatter(self) -> None:
        text = (
            "---\n"
            "voice-design:\n"
            "  speakers:\n"
            "    Aaron:\n"
            "      pitch: low\n"
            "---\n"
            "Aaron: 第一句。\n"
            "Aaron[style=whisper]: 第二句小声说。\n"
        )
        result = parse(text)
        self.assertEqual(result.turn_designs[0].instructions, "low pitch")
        self.assertEqual(result.turn_designs[1].instructions, "low pitch, whisper")
        # Brackets are stripped from the second speaker tag line.
        self.assertIn("Aaron: 第二句小声说。", result.cleaned_text)
        self.assertNotIn("[style=whisper]", result.cleaned_text)

    def test_line_level_overrides_frontmatter(self) -> None:
        text = (
            "---\n"
            "voice-design:\n"
            "  speakers:\n"
            "    Aaron:\n"
            "      pitch: low\n"
            "---\n"
            "Aaron[pitch=high]: 大声点！\n"
        )
        result = parse(text)
        self.assertEqual(result.turn_designs[0].instructions, "high pitch")

    def test_very_low_pitch_serializes_with_space(self) -> None:
        text = "Aaron[pitch=very-low]: 低沉。\n"
        result = parse(text)
        self.assertEqual(result.turn_designs[0].instructions, "very low pitch")

    def test_unknown_pitch_value_raises(self) -> None:
        text = "Aaron[pitch=tall]: 你好。\n"
        with self.assertRaises(VoiceDesignError) as ctx:
            parse(text)
        self.assertEqual(ctx.exception.line, 1)
        self.assertGreaterEqual(ctx.exception.column, 1)

    def test_unknown_style_value_raises(self) -> None:
        # `excited` was in the v0 spec but is not in OmniVoice's vocabulary.
        text = "Aaron[style=excited]: 激动！\n"
        with self.assertRaises(VoiceDesignError):
            parse(text)

    def test_unknown_top_level_frontmatter_key(self) -> None:
        text = (
            "---\n"
            "voice-design:\n"
            "  foo: bar\n"
            "---\n"
            "Aaron: 你好。\n"
        )
        with self.assertRaises(VoiceDesignError):
            parse(text)

    def test_duplicate_key_in_attrs(self) -> None:
        text = "Aaron[pitch=high, pitch=low]: 你好。\n"
        with self.assertRaises(VoiceDesignError):
            parse(text)

    def test_accent_freeform_chinese(self) -> None:
        text = (
            "---\n"
            "voice-design:\n"
            "  speakers:\n"
            "    Aaron:\n"
            "      accent: 四川话\n"
            "---\n"
            "Aaron: 你好。\n"
        )
        result = parse(text)
        self.assertEqual(result.turn_designs[0].instructions, "四川话")

    def test_mixed_language_rejected(self) -> None:
        # OmniVoice forbids mixing English and Chinese values in one instruct.
        text = (
            "---\n"
            "voice-design:\n"
            "  speakers:\n"
            "    Aaron:\n"
            "      pitch: low\n"
            "      accent: 四川话\n"
            "---\n"
            "Aaron: 你好。\n"
        )
        with self.assertRaises(VoiceDesignError):
            parse(text)

    def test_continuation_lines_inherit(self) -> None:
        text = (
            "---\n"
            "voice-design:\n"
            "  speakers:\n"
            "    Aaron:\n"
            "      pitch: low\n"
            "      accent: american\n"
            "---\n"
            "Aaron: 你好。\n"
            "继续讲下去。\n"
            "Anchen: 你好呀。\n"
        )
        result = parse(text)
        self.assertEqual(len(result.turn_designs), 2)
        self.assertEqual(result.turn_designs[0].speaker, "Aaron")
        self.assertEqual(result.turn_designs[0].instructions, "low pitch, american accent")
        self.assertEqual(result.turn_designs[1].speaker, "Anchen")
        self.assertIsNone(result.turn_designs[1].instructions)

    def test_same_speaker_new_attrs_new_turn(self) -> None:
        text = (
            "Aaron[pitch=low]: 第一。\n"
            "Aaron[pitch=high]: 第二。\n"
        )
        result = parse(text)
        self.assertEqual(len(result.turn_designs), 2)
        self.assertEqual(result.turn_designs[0].instructions, "low pitch")
        self.assertEqual(result.turn_designs[1].instructions, "high pitch")


class HasMarkupTests(unittest.TestCase):
    def test_frontmatter_only(self) -> None:
        text = "---\nvoice-design:\n  default:\n    pitch: low\n---\nAaron: hi.\n"
        self.assertTrue(has_markup(text))

    def test_bracket_only(self) -> None:
        text = "Aaron[pitch=low]: 你好。\n"
        self.assertTrue(has_markup(text))

    def test_prose_with_colons_is_not_markup(self) -> None:
        text = "Note: this is plain prose.\nTime: 10:30. No brackets here.\n"
        self.assertFalse(has_markup(text))


if __name__ == "__main__":
    unittest.main()
