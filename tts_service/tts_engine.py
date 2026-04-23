"""vibevoice-mlx based TTS engine used by the API and UI."""

from __future__ import annotations

import io
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import soundfile as sf

import cn2an

from .config import Config
from .models import SpeakerResolution
from .sample_manager import SampleManager, VoiceSample


REPO_ROOT = Path(__file__).resolve().parents[1]
VIBEVOICE_MLX_ROOT = REPO_ROOT / "vibevoice-mlx"
if str(VIBEVOICE_MLX_ROOT) not in sys.path:
    sys.path.insert(0, str(VIBEVOICE_MLX_ROOT))

_DIGIT_RE = re.compile(r"\d+")


def _convert_numbers_to_chinese(text: str) -> str:
    """Replace Arabic numerals with Chinese readings for TTS."""
    return _DIGIT_RE.sub(lambda m: cn2an.an2cn(m.group(0)), text)

from vibevoice_mlx.e2e_pipeline import (  # noqa: E402
    SAMPLE_RATE,
    SPEECH_TOK_COMPRESS_RATIO,
    VOICE_CLONE_SAMPLES,
    VoiceCloneData,
    _detect_tokenizer,
    _load_and_resample,
    _load_semantic_encoder,
    encode_voice_reference,
    save_voice,
    tokenize_text,
)
from vibevoice_mlx.generate import GenerationOptions, generate  # noqa: E402
from vibevoice_mlx.load_weights import load_model  # noqa: E402


@dataclass(slots=True)
class GenerationResult:
    audio_bytes: bytes
    output_format: str
    generation_seconds: float
    duration_seconds: float
    resolved_speakers: list[SpeakerResolution]


class TTSEngine:
    """Direct Python integration with vibevoice-mlx generation flow."""

    DIALOGUE_LINE_RE = re.compile(r"^([^:\n]{1,80}):\s*(.+)$")

    def __init__(self, config: Config, sample_manager: SampleManager):
        self.config = config
        self.sample_manager = sample_manager
        self._runtime_lock = threading.RLock()
        self._generation_lock = threading.RLock()
        self._model = None
        self._model_config = None
        self._tokenizer_name: Optional[str] = None
        self._semantic_fn = None
        self._semantic_reset_fn = None

    def ensure_voice_cache_ready(self, speaker: str) -> VoiceSample:
        sample = self.sample_manager.resolve_or_default(speaker)
        self._ensure_runtime_loaded()
        self._load_voice_embeddings(sample)
        return self.sample_manager.resolve_or_default(sample.speaker)

    def generate_single(self, text: str, voice: Optional[str], output_format: str = "wav") -> GenerationResult:
        return self.generate_dialogue(text=text, output_format=output_format, preferred_voice=voice)

    def generate_dialogue(
        self,
        text: str,
        output_format: str = "wav",
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
    ) -> GenerationResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Input text cannot be empty")
        normalized_text = _convert_numbers_to_chinese(normalized_text)

        with self._generation_lock:
            self._ensure_runtime_loaded()
            default_sample = self.sample_manager.resolve_or_default(self.config.voices.default_voice)
            normalized_script, ordered_samples, resolutions = self._build_dialogue_script(
                normalized_text,
                preferred_voice=preferred_voice,
                voice_mapping=voice_mapping or {},
                default_sample=default_sample,
            )

            speaker_embeds = []
            for sample in ordered_samples:
                embeds = self._load_voice_embeddings(sample)
                speaker_embeds.append((embeds.shape[0], embeds))

            tokenize_result = tokenize_text(
                normalized_script,
                self._tokenizer_name,
                self._model_config,
                speaker_embeds=speaker_embeds,
            )

            input_ids, voice_embeds = self._build_voice_embed_map(tokenize_result)
            if self._semantic_reset_fn is not None:
                self._semantic_reset_fn()

            options = GenerationOptions(
                solver="dpm",
                diffusion_steps=self.config.model.diffusion_steps,
                cfg_scale=self.config.model.cfg_scale,
                max_speech_tokens=self.config.model.max_speech_tokens,
                seed=self.config.model.seed,
            )

            start_time = time.perf_counter()
            audio, _metrics = generate(
                model=self._model,
                input_ids=input_ids,
                opts=options,
                semantic_encoder_fn=self._semantic_fn,
                semantic_reset_fn=self._semantic_reset_fn,
                voice_embeds=voice_embeds,
            )
            generation_seconds = time.perf_counter() - start_time

            audio_bytes = self._encode_audio(audio, output_format)
            duration_seconds = len(audio) / SAMPLE_RATE if len(audio) else 0.0
            return GenerationResult(
                audio_bytes=audio_bytes,
                output_format=output_format,
                generation_seconds=generation_seconds,
                duration_seconds=duration_seconds,
                resolved_speakers=resolutions,
            )

    def _ensure_runtime_loaded(self) -> None:
        with self._runtime_lock:
            if self._model is not None:
                return
            self._model, self._model_config = load_model(
                self.config.model.model_id,
                quantize_bits=self.config.model.quantize_bits,
            )
            self._tokenizer_name = _detect_tokenizer(self.config.model.model_id, self._model_config)
            if self.config.model.use_semantic:
                result = _load_semantic_encoder(
                    self._model,
                    self._model_config,
                    self.config.model.model_id,
                    use_coreml=self.config.model.use_coreml_semantic,
                )
                if result is not None:
                    self._semantic_fn, self._semantic_reset_fn = result

    def _load_voice_embeddings(self, sample: VoiceSample):
        if sample.cache_path.exists():
            from vibevoice_mlx.e2e_pipeline import load_voice

            return load_voice(str(sample.cache_path))

        wav = _load_and_resample(str(sample.wav_path))
        if len(wav) > VOICE_CLONE_SAMPLES:
            wav = wav[:VOICE_CLONE_SAMPLES]
        num_tokens = math.ceil(len(wav) / SPEECH_TOK_COMPRESS_RATIO)
        embeds = encode_voice_reference(wav, num_tokens, self._model, self._model_config, self.config.model.model_id)
        save_voice(str(sample.cache_path), embeds)
        return embeds

    def _build_dialogue_script(
        self,
        text: str,
        preferred_voice: Optional[str],
        voice_mapping: dict[str, str],
        default_sample: VoiceSample,
    ) -> tuple[str, list[VoiceSample], list[SpeakerResolution]]:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        matched_dialogue = [self.DIALOGUE_LINE_RE.match(line.strip()) for line in lines]
        is_dialogue = any(match is not None for match in matched_dialogue)

        if not is_dialogue:
            target_voice = preferred_voice or default_sample.speaker
            resolved = self.sample_manager.resolve(target_voice)
            sample = resolved or default_sample
            resolutions = [
                SpeakerResolution(
                    requested_name=target_voice,
                    resolved_voice=sample.speaker,
                    used_default=resolved is None,
                    transcript_preview=sample.transcript_preview,
                )
            ]
            script = "\n".join(f"Speaker 1: {line.strip()}" for line in lines)
            return script, [sample], resolutions

        speaker_order: list[str] = []
        speaker_indices: dict[str, int] = {}
        speaker_samples: dict[str, VoiceSample] = {}
        resolutions: list[SpeakerResolution] = []
        normalized_lines: list[str] = []
        current_speaker: Optional[str] = None

        for raw_line in lines:
            line = raw_line.strip()
            match = self.DIALOGUE_LINE_RE.match(line)
            if match is None:
                if current_speaker is None:
                    current_speaker = preferred_voice or default_sample.speaker
                    if current_speaker not in speaker_indices:
                        speaker_indices[current_speaker] = len(speaker_order) + 1
                        speaker_order.append(current_speaker)
                        speaker_samples[current_speaker] = default_sample
                        resolutions.append(
                            SpeakerResolution(
                                requested_name=current_speaker,
                                resolved_voice=default_sample.speaker,
                                used_default=True,
                                transcript_preview=default_sample.transcript_preview,
                            )
                        )
                normalized_lines.append(f"Speaker {speaker_indices[current_speaker]}: {line}")
                continue

            requested_name = match.group(1).strip()
            content = match.group(2).strip()
            current_speaker = requested_name
            if requested_name not in speaker_indices:
                speaker_indices[requested_name] = len(speaker_order) + 1
                speaker_order.append(requested_name)
                mapped_voice = voice_mapping.get(requested_name) or requested_name
                resolved = self.sample_manager.resolve(mapped_voice)
                sample = resolved or default_sample
                speaker_samples[requested_name] = sample
                resolutions.append(
                    SpeakerResolution(
                        requested_name=requested_name,
                        resolved_voice=sample.speaker,
                        used_default=resolved is None,
                        transcript_preview=sample.transcript_preview,
                    )
                )
            normalized_lines.append(f"Speaker {speaker_indices[requested_name]}: {content}")

        ordered_samples = [speaker_samples[name] for name in speaker_order]
        return "\n".join(normalized_lines), ordered_samples, resolutions

    def _build_voice_embed_map(self, tokenize_result) -> tuple[list[int], dict[int, mx.array]]:
        if not isinstance(tokenize_result, VoiceCloneData):
            return tokenize_result, {}

        voice_embeds: dict[int, mx.array] = {}
        for speaker in tokenize_result.speakers:
            embeds_mx = mx.array(speaker.cached_embeds).astype(mx.float16)
            for index, position in enumerate(speaker.speech_embed_positions):
                if index < embeds_mx.shape[0]:
                    voice_embeds[position] = embeds_mx[index:index + 1]
        return tokenize_result.input_ids, voice_embeds

    @staticmethod
    def _encode_audio(audio, output_format: str) -> bytes:
        buffer = io.BytesIO()
        fmt = output_format.lower()
        if fmt == "wav":
            sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
        elif fmt == "flac":
            sf.write(buffer, audio, SAMPLE_RATE, format="FLAC")
        elif fmt == "ogg":
            sf.write(buffer, audio, SAMPLE_RATE, format="OGG", subtype="VORBIS")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        buffer.seek(0)
        return buffer.read()
