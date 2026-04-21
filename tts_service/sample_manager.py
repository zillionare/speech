"""Voice library management for {speaker}.wav + {speaker}.txt pairs."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, istft, stft


_MIN_SIGNAL_LEVEL = 1e-4
_TARGET_PEAK = 0.95


def preprocess_reference_audio(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if audio.size == 0:
        raise ValueError("Uploaded audio is empty")
    if not np.any(np.abs(audio) > _MIN_SIGNAL_LEVEL):
        raise ValueError("Uploaded audio is silent")

    audio = audio - float(np.mean(audio))
    audio = _highpass_filter(audio, sample_rate)
    audio = _spectral_denoise(audio, sample_rate)
    audio = _trim_silence(audio, sample_rate)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= _MIN_SIGNAL_LEVEL:
        raise ValueError("Uploaded audio is silent after preprocessing")
    normalized = np.clip(audio / peak * _TARGET_PEAK, -1.0, 1.0)
    return normalized.astype(np.float32, copy=False)


def _highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 70.0) -> np.ndarray:
    if sample_rate <= cutoff_hz * 4 or audio.size < 64:
        return audio
    normalized_cutoff = cutoff_hz / (sample_rate * 0.5)
    b_coeffs, a_coeffs = butter(2, normalized_cutoff, btype="highpass")
    padlen = 3 * (max(len(a_coeffs), len(b_coeffs)) - 1)
    if audio.size <= padlen:
        return audio
    return filtfilt(b_coeffs, a_coeffs, audio).astype(np.float32, copy=False)


def _spectral_denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    nperseg = min(1024, max(256, int(sample_rate * 0.032)))
    if audio.size < nperseg:
        return audio
    noverlap = nperseg * 3 // 4
    _, _, spectrum = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    if spectrum.size == 0:
        return audio

    magnitude = np.abs(spectrum)
    noise_profile = _estimate_noise_profile(audio, magnitude, sample_rate, nperseg, noverlap)
    gain = (magnitude - 0.9 * noise_profile) / (magnitude + 1e-6)
    gain = np.clip(gain, 0.2, 1.0)
    _, restored = istft(spectrum * gain, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    if restored.size < audio.size:
        restored = np.pad(restored, (0, audio.size - restored.size))
    return restored[:audio.size].astype(np.float32, copy=False)


def _estimate_noise_profile(
    audio: np.ndarray,
    magnitude: np.ndarray,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    edge_size = min(int(sample_rate * 0.25), audio.size // 5)
    if edge_size >= nperseg:
        noise_source = np.concatenate([audio[:edge_size], audio[-edge_size:]])
        _, _, noise_spectrum = stft(noise_source, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        if noise_spectrum.size:
            return np.median(np.abs(noise_spectrum), axis=1, keepdims=True)

    frame_energy = np.sqrt(np.mean(magnitude * magnitude, axis=0))
    quiet_frame_count = max(1, min(magnitude.shape[1] // 5, 8))
    quiet_indices = np.argsort(frame_energy)[:quiet_frame_count]
    return np.median(magnitude[:, quiet_indices], axis=1, keepdims=True)


def _trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_size = max(256, int(sample_rate * 0.02))
    hop_size = max(128, frame_size // 2)
    if audio.size < frame_size * 2:
        return audio

    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[::hop_size]
    if frames.size == 0:
        return audio

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise_floor = float(np.percentile(rms, 20))
    peak = float(np.max(np.abs(audio)))
    threshold = max(noise_floor * 2.5, peak * 0.03, _MIN_SIGNAL_LEVEL)
    speech_frames = np.flatnonzero(rms >= threshold)
    if speech_frames.size == 0:
        return audio

    padding = int(sample_rate * 0.12)
    start = max(0, speech_frames[0] * hop_size - padding)
    end = min(audio.size, speech_frames[-1] * hop_size + frame_size + padding)
    if end - start < int(sample_rate * 0.35):
        return audio
    return audio[start:end].astype(np.float32, copy=False)


@dataclass(slots=True)
class VoiceSample:
    speaker: str
    wav_path: Path
    txt_path: Path
    cache_path: Path
    transcript: str
    aliases: set[str]

    @property
    def transcript_preview(self) -> str:
        if not self.transcript:
            return ""
        return self.transcript[:120]


class SampleManager:
    """Manage local voices, transcripts, and embedding cache files."""

    def __init__(self, base_dir: Path, default_voice: str):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / ".cache"
        self.default_voice = default_voice
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._voices: dict[str, VoiceSample] = {}
        self._aliases: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        self._voices.clear()
        self._aliases.clear()
        for wav_path in sorted(self.base_dir.glob("*.wav")):
            speaker = wav_path.stem
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                txt_path.write_text("", encoding="utf-8")
            transcript = txt_path.read_text(encoding="utf-8").strip()
            aliases = self._build_aliases(speaker)
            sample = VoiceSample(
                speaker=speaker,
                wav_path=wav_path,
                txt_path=txt_path,
                cache_path=self.cache_dir / f"{speaker}.safetensors",
                transcript=transcript,
                aliases=aliases,
            )
            self._voices[speaker] = sample
            for alias in aliases:
                self._aliases[alias] = speaker

    def list_samples(self) -> list[VoiceSample]:
        return [self._voices[key] for key in sorted(self._voices)]

    def list_voices(self) -> list[str]:
        return [sample.speaker for sample in self.list_samples()]

    def get(self, speaker: str) -> Optional[VoiceSample]:
        return self._voices.get(speaker)

    def resolve(self, speaker: str) -> Optional[VoiceSample]:
        if not speaker:
            return None
        if speaker in self._voices:
            return self._voices[speaker]
        normalized = self._normalize_alias(speaker)
        resolved = self._aliases.get(normalized)
        if resolved:
            return self._voices[resolved]
        return None

    def resolve_or_default(self, speaker: str) -> VoiceSample:
        resolved = self.resolve(speaker)
        if resolved is not None:
            return resolved
        default_sample = self.resolve(self.default_voice)
        if default_sample is None:
            raise FileNotFoundError(
                f"Default voice '{self.default_voice}' is missing from {self.base_dir}"
            )
        return default_sample

    def add_voice(self, speaker: str, audio_bytes: bytes, transcript: str, overwrite: bool = False) -> VoiceSample:
        safe_speaker = self._sanitize_speaker_name(speaker)
        wav_path = self.base_dir / f"{safe_speaker}.wav"
        txt_path = self.base_dir / f"{safe_speaker}.txt"
        cache_path = self.cache_dir / f"{safe_speaker}.safetensors"
        if wav_path.exists() and not overwrite:
            raise FileExistsError(f"Voice '{safe_speaker}' already exists")
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False, dtype="float32")
        except Exception as exc:
            raise ValueError("Uploaded audio could not be decoded") from exc
        processed_audio = preprocess_reference_audio(audio_data, sample_rate)
        sf.write(wav_path, processed_audio, sample_rate, format="WAV", subtype="PCM_16")
        txt_path.write_text(transcript or "", encoding="utf-8")
        if cache_path.exists():
            cache_path.unlink()
        self.refresh()
        return self._voices[safe_speaker]

    def update_transcript(self, speaker: str, transcript: str) -> VoiceSample:
        sample = self.get(speaker)
        if sample is None:
            raise FileNotFoundError(f"Voice '{speaker}' not found")
        sample.txt_path.write_text(transcript or "", encoding="utf-8")
        self.refresh()
        return self._voices[speaker]

    def delete_voice(self, speaker: str) -> None:
        sample = self.get(speaker)
        if sample is None:
            raise FileNotFoundError(f"Voice '{speaker}' not found")
        sample.wav_path.unlink(missing_ok=True)
        sample.txt_path.unlink(missing_ok=True)
        sample.cache_path.unlink(missing_ok=True)
        self.refresh()

    def voice_exists(self, speaker: str) -> bool:
        return self.resolve(speaker) is not None

    @staticmethod
    def _sanitize_speaker_name(speaker: str) -> str:
        speaker = speaker.strip()
        if not speaker:
            raise ValueError("Speaker name cannot be empty")
        for char in ("/", "\\", ":", "\x00"):
            speaker = speaker.replace(char, "_")
        return speaker

    @staticmethod
    def _normalize_alias(value: str) -> str:
        alias = value.strip().lower().replace(" ", "").replace("-", "")
        for prefix in ("zh", "en", "in", "jp", "kr", "de", "fr", "sp", "pt", "it", "nl", "pl"):
            if alias.startswith(prefix) and len(alias) > len(prefix) + 1:
                alias = alias[len(prefix):]
                break
        for suffix in ("_man", "_woman", "_bgm", "man", "woman", "bgm"):
            alias = alias.removesuffix(suffix)
        return alias.replace("_", "")

    def _build_aliases(self, speaker: str) -> set[str]:
        aliases = {self._normalize_alias(speaker), speaker.lower()}
        if "-" in speaker:
            aliases.add(self._normalize_alias(speaker.split("-", 1)[1]))
        if "_" in speaker:
            aliases.add(self._normalize_alias(speaker.split("_", 1)[0]))
        if "-" in speaker and "_" in speaker:
            aliases.add(self._normalize_alias(speaker.split("-", 1)[1].split("_", 1)[0]))
        return {alias for alias in aliases if alias}
