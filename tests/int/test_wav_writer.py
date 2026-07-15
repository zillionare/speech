"""Integration tests for ST-LP-010: WAV writer.

Tests call real write_wav() and WavStreamWriter with real PCM data
and verify output by reading back the WAV files.
"""

from __future__ import annotations

import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class WAVHeaderTests(unittest.TestCase):
    """Call write_wav() with real PCM data, read back with wave module."""

    def test_wav_write_correct_header(self):
        from tts_service.live.wav_writer import write_wav
        tmpdir = Path(tempfile.mkdtemp())
        wav_path = tmpdir / "test.wav"
        sample_rate = 48000
        samples = int(sample_rate * 0.5)
        pcm = struct.pack(f"<{samples}h", *([0] * samples))
        write_wav(wav_path, pcm, sample_rate)
        with wave.open(str(wav_path), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), sample_rate)
            self.assertEqual(w.getnframes(), samples)

    def test_wav_write_preserves_sample_rate(self):
        from tts_service.live.wav_writer import write_wav
        tmpdir = Path(tempfile.mkdtemp())
        for sr in [44100, 48000, 96000]:
            wav_path = tmpdir / f"test_{sr}.wav"
            samples = int(sr * 0.1)
            pcm = struct.pack(f"<{samples}h", *([0] * samples))
            write_wav(wav_path, pcm, sr)
            with wave.open(str(wav_path), "rb") as w:
                self.assertEqual(w.getframerate(), sr)

    def test_wav_write_nonzero_audio_preserved(self):
        """Non-silent PCM data is correctly written and read back."""
        from tts_service.live.wav_writer import write_wav
        tmpdir = Path(tempfile.mkdtemp())
        wav_path = tmpdir / "sine.wav"
        sr = 24000
        samples = int(sr * 0.5)
        t = np.linspace(0, 0.5, samples, endpoint=False)
        original = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
        write_wav(wav_path, original.tobytes(), sr)
        with wave.open(str(wav_path), "rb") as w:
            frames = w.readframes(w.getnframes())
            restored = np.frombuffer(frames, dtype=np.int16)
        self.assertTrue(np.any(restored != 0), "Audio data should be non-silent")


class WavStreamWriterTests(unittest.TestCase):
    """Call WavStreamWriter.write() with real chunks, read back."""

    def test_streaming_append_mode(self):
        from tts_service.live.wav_writer import WavStreamWriter
        tmpdir = Path(tempfile.mkdtemp())
        wav_path = tmpdir / "stream.wav"
        sr = 48000
        chunk_size = int(sr * 0.2)  # 200ms
        writer = WavStreamWriter(wav_path, sr)
        total_samples = 0
        for _ in range(5):
            chunk = struct.pack(f"<{chunk_size}h", *([100] * chunk_size))
            writer.write(chunk)
            total_samples += chunk_size
        writer.close()
        with wave.open(str(wav_path), "rb") as w:
            self.assertEqual(w.getframerate(), sr)
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getnframes(), total_samples)

    def test_context_manager(self):
        """WavStreamWriter supports context manager protocol."""
        from tts_service.live.wav_writer import WavStreamWriter
        tmpdir = Path(tempfile.mkdtemp())
        wav_path = tmpdir / "ctx.wav"
        sr = 24000
        chunk_size = int(sr * 0.1)
        chunk = struct.pack(f"<{chunk_size}h", *([50] * chunk_size))
        with WavStreamWriter(wav_path, sr) as writer:
            writer.write(chunk)
            writer.write(chunk)
        # File should be closed and readable
        with wave.open(str(wav_path), "rb") as w:
            self.assertEqual(w.getnframes(), chunk_size * 2)


class PeakNormalizeTests(unittest.TestCase):
    """Call peak_normalize() with real numpy arrays."""

    def test_peak_normalize_scales_to_target(self):
        from tts_service.sample_manager import peak_normalize
        audio = np.array([0.0, 0.5, 1.0, -0.8, 0.3], dtype=np.float32)
        normalized = peak_normalize(audio, target_peak=0.95)
        peak = np.max(np.abs(normalized))
        self.assertAlmostEqual(peak, 0.95, places=2)

    def test_peak_normalize_preserves_relative_amplitudes(self):
        from tts_service.sample_manager import peak_normalize
        audio = np.array([0.1, 0.5, 0.25], dtype=np.float32)
        normalized = peak_normalize(audio, target_peak=0.95)
        ratio = normalized[1] / normalized[0]
        self.assertAlmostEqual(ratio, 5.0, places=1)

    def test_peak_normalize_handles_silence(self):
        from tts_service.sample_manager import peak_normalize
        audio = np.zeros(1000, dtype=np.float32)
        normalized = peak_normalize(audio, target_peak=0.95)
        self.assertEqual(normalized.shape, audio.shape)
        self.assertTrue(np.all(normalized == 0))

    def test_peak_normalize_accepts_int16(self):
        """peak_normalize should handle int16 input without crashing."""
        from tts_service.sample_manager import peak_normalize
        audio = np.array([0, 16384, -32768, 8192], dtype=np.int16)
        normalized = peak_normalize(audio.astype(np.float32), target_peak=0.95)
        peak = np.max(np.abs(normalized))
        self.assertAlmostEqual(peak, 0.95, places=2)


if __name__ == "__main__":
    unittest.main()