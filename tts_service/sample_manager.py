"""Voice library management for {speaker}.wav + {speaker}.txt pairs."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Optional

import soundfile as sf


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
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        except Exception as exc:
            raise ValueError("Uploaded audio could not be decoded") from exc
        sf.write(wav_path, audio_data, sample_rate, format="WAV")
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
