"""Tests for Live Studio loudness normalization and merging."""

from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from tts_service.audio_processing import normalize_wav_loudness
from tts_service.server import _ensure_live_preview_normalized, _merge_live_audio


def _write_tone(path: Path, amplitude: float, sample_rate: int = 24000) -> None:
    seconds = 2
    positions = np.arange(sample_rate * seconds, dtype=np.float64) / sample_rate
    samples = amplitude * np.sin(2 * np.pi * 440 * positions)
    pcm = np.round(samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())


def _read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        samples = np.frombuffer(
            source.readframes(source.getnframes()), dtype="<i2",
        ).astype(np.float64)
    return samples, rate


def _rms_db(samples: np.ndarray) -> float:
    rms = np.sqrt(np.mean(samples ** 2))
    return float(20 * np.log10(rms / 32767))


class LiveAudioProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loudness_normalization_equalizes_quiet_and_loud_speech(self) -> None:
        quiet = self.directory / "quiet.wav"
        loud = self.directory / "loud.wav"
        _write_tone(quiet, 0.03)
        _write_tone(loud, 0.3)

        normalize_wav_loudness(quiet)
        normalize_wav_loudness(loud)

        quiet_pcm, _ = _read_pcm(quiet)
        loud_pcm, _ = _read_pcm(loud)
        self.assertLess(abs(_rms_db(quiet_pcm) - _rms_db(loud_pcm)), 0.5)
        self.assertLessEqual(np.max(np.abs(quiet_pcm)), 32767)
        self.assertLessEqual(np.max(np.abs(loud_pcm)), 32767)

    def test_merge_exports_one_normalized_24khz_mono_wav(self) -> None:
        first = self.directory / "seg_0000.wav"
        second = self.directory / "seg_0001.wav"
        _write_tone(first, 0.03, sample_rate=16000)
        _write_tone(second, 0.3, sample_rate=24000)
        session = {
            "dir": str(self.directory),
            "segments": [
                {"index": 0, "source": "live", "audio_filename": first.name},
                {"index": 1, "source": "tts", "audio_filename": second.name},
            ],
        }

        filename = _merge_live_audio(session)

        self.assertEqual(filename, "final.wav")
        merged, rate = _read_pcm(self.directory / filename)
        self.assertEqual(rate, 24000)
        self.assertAlmostEqual(len(merged) / rate, 4.0, places=2)
        first_pcm, _ = _read_pcm(first)
        second_pcm, _ = _read_pcm(second)
        self.assertLess(abs(_rms_db(first_pcm) - _rms_db(second_pcm)), 0.5)

    def test_preview_normalizes_legacy_segment_before_playback(self) -> None:
        session_id = "legacy-session"
        session_dir = self.directory / "live_studio" / session_id
        session_dir.mkdir(parents=True)
        segment_path = session_dir / "seg_0000.wav"
        _write_tone(segment_path, 0.03)
        metadata = {
            "session_id": session_id,
            "segments": [{"index": 0, "source": "tts", "status": "generated"}],
        }
        (session_dir / "meta.json").write_text(
            json.dumps(metadata), encoding="utf-8",
        )
        before, _ = _read_pcm(segment_path)

        _ensure_live_preview_normalized(
            session_id,
            segment_path.name,
            segment_path,
            {},
            self.directory,
        )

        after, _ = _read_pcm(segment_path)
        self.assertGreater(_rms_db(after), _rms_db(before) + 10)
        saved = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["segments"][0]["loudness_normalized"])


if __name__ == "__main__":
    unittest.main()
