from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.sample_manager import preprocess_reference_audio


def _slice_for_lag(reference: np.ndarray, sample: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        ref_slice = reference[lag:lag + sample.size]
        sample_slice = sample[:ref_slice.size]
    else:
        ref_slice = reference[:sample.size + lag]
        sample_slice = sample[-lag:-lag + ref_slice.size]
    return ref_slice, sample_slice


def _aligned_snr_db(reference: np.ndarray, sample: np.ndarray) -> float:
    gain = float(np.dot(reference, sample) / (np.dot(sample, sample) + 1e-8))
    aligned = sample * gain
    residual = reference - aligned
    signal_power = float(np.mean(reference * reference) + 1e-8)
    noise_power = float(np.mean(residual * residual) + 1e-8)
    return 10.0 * np.log10(signal_power / noise_power)


def _best_aligned_snr_db(reference: np.ndarray, sample: np.ndarray, max_lag: int = 4000) -> float:
    best_score: float | None = None
    best_snr = float("-inf")
    for lag in range(-max_lag, max_lag + 1, 20):
        ref_slice, sample_slice = _slice_for_lag(reference, sample, lag)
        if ref_slice.size < 2000 or sample_slice.size != ref_slice.size:
            continue
        score = float(np.dot(ref_slice, sample_slice))
        if best_score is None or score > best_score:
            best_score = score
            best_snr = _aligned_snr_db(ref_slice, sample_slice)
    return best_snr


class PreprocessReferenceAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 16_000
        self.rng = np.random.default_rng(1234)

    def _clean_reference(self, seconds: float = 1.0) -> np.ndarray:
        time_axis = np.linspace(0.0, seconds, int(self.sample_rate * seconds), endpoint=False)
        envelope = np.sin(np.linspace(0.0, np.pi, time_axis.size, endpoint=False))
        clean = 0.32 * envelope * (
            np.sin(2 * np.pi * 220 * time_axis) + 0.4 * np.sin(2 * np.pi * 440 * time_axis)
        )
        return clean.astype(np.float32)

    def test_preprocess_reference_audio_converts_to_mono_and_trims_edges(self) -> None:
        clean = self._clean_reference()
        lead_noise = 0.02 * self.rng.normal(size=(int(self.sample_rate * 0.3), 2))
        tail_noise = 0.02 * self.rng.normal(size=(int(self.sample_rate * 0.3), 2))
        speech_noise = 0.03 * self.rng.normal(size=clean.shape)
        stereo_speech = np.stack([clean + speech_noise, 0.85 * clean - speech_noise], axis=1)
        stereo_audio = np.concatenate([lead_noise, stereo_speech, tail_noise], axis=0).astype(np.float32)

        processed = preprocess_reference_audio(stereo_audio, self.sample_rate)

        self.assertEqual(processed.ndim, 1)
        self.assertLess(processed.size, int(stereo_audio.shape[0] * 0.8))
        self.assertGreater(processed.size, int(clean.size * 0.7))

    def test_preprocess_reference_audio_improves_signal_to_noise_ratio(self) -> None:
        clean = self._clean_reference(seconds=1.2)
        time_axis = np.linspace(0.0, 1.2, clean.size, endpoint=False)
        hum = 0.05 * np.sin(2 * np.pi * 50 * time_axis)
        noise_left = 0.08 * self.rng.normal(size=clean.shape)
        noise_right = 0.08 * self.rng.normal(size=clean.shape)
        stereo_audio = np.stack([
            clean + hum + noise_left,
            0.9 * clean + hum + noise_right,
        ], axis=1).astype(np.float32)

        processed = preprocess_reference_audio(stereo_audio, self.sample_rate)
        noisy_snr = _best_aligned_snr_db(clean, stereo_audio.mean(axis=1))
        processed_snr = _best_aligned_snr_db(clean, processed)

        self.assertGreater(processed_snr, noisy_snr + 2.0)


if __name__ == "__main__":
    unittest.main()