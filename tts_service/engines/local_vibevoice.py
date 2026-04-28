"""Local VibeVoice MLX TTS engine."""

from __future__ import annotations

import io
import math
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import soundfile as sf

import cn2an

from ..config import Config
from ..models import SpeakerResolution
from ..sample_manager import SampleManager, VoiceSample
from .base import (
    BaseEngine,
    GenerationResult,
    _apply_audio_effects,
    _concatenate_audio_segments,
    _parse_tagged_dialogue,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VIBEVOICE_MLX_ROOT = REPO_ROOT / "vibevoice-mlx"
if str(VIBEVOICE_MLX_ROOT) not in sys.path:
    sys.path.insert(0, str(VIBEVOICE_MLX_ROOT))

_DIGITS = "零一二三四五六七八九"

_YEAR_RE = re.compile(r"\b(\d{4})\s*年")
_CODE_RE = re.compile(r"(?:编号|第|No\.?|API|版本|型号|代码|序列号|密码|身份证号|电话|手机|工号|学号|房间号|门牌号|航班号|车次|订单号|快递单号|ISBN|ISSN)(?:[:：是为])?\s*(\d+)")
_NUM_RE = re.compile(r"\d+")


def _convert_numbers_to_chinese(text: str) -> str:
    def replace_year(m: re.Match) -> str:
        return "".join(_DIGITS[int(d)] for d in m.group(1)) + "年"

    def replace_code(m: re.Match) -> str:
        full = m.group(0)
        digits = m.group(1)
        prefix = full[: full.rfind(digits)]
        return prefix + "".join(_DIGITS[int(d)] for d in digits)

    text = _YEAR_RE.sub(replace_year, text)
    text = _CODE_RE.sub(replace_code, text)
    text = _NUM_RE.sub(lambda m: cn2an.an2cn(m.group(0)), text)
    return text


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


class LocalVibeVoiceEngine(BaseEngine):
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

    def generate_single(
        self,
        text: str,
        voice: Optional[str],
        output_format: str = "wav",
        instructions: Optional[str] = None,
    ) -> GenerationResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Input text cannot be empty")
        normalized_text = _convert_numbers_to_chinese(normalized_text)

        with self._generation_lock:
            self._ensure_runtime_loaded()
            default_sample = self.sample_manager.resolve_or_default(self.config.voices.default_voice)
            resolved = self.sample_manager.resolve(voice or self.config.voices.default_voice)
            sample = resolved or default_sample
            script = f"Speaker 1: {normalized_text}"
            resolutions = [
                SpeakerResolution(
                    requested_name=voice or sample.speaker,
                    resolved_voice=sample.speaker,
                    used_default=resolved is None,
                    transcript_preview=sample.transcript_preview,
                )
            ]
            result = self._generate_one(script, [sample], resolutions, output_format)
            return self._post_process(result)

    def generate_dialogue(
        self,
        text: str,
        output_format: str = "wav",
        preferred_voice: Optional[str] = None,
        voice_mapping: Optional[dict[str, str]] = None,
        instructions: Optional[str] = None,
        segment_gap: Optional[float] = None,
        speaker_gap: Optional[float] = None,
    ) -> GenerationResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Input text cannot be empty")
        normalized_text = _convert_numbers_to_chinese(normalized_text)

        # Try new tagged dialogue format first: Speaker[tone>>]: text
        tagged_segments, is_tagged = _parse_tagged_dialogue(normalized_text)
        if is_tagged:
            return self._generate_tagged_segments(
                tagged_segments,
                output_format=output_format,
                voice_mapping=voice_mapping,
                segment_gap=segment_gap,
                speaker_gap=speaker_gap,
            )

        max_chars = getattr(self.config.model, "max_segment_chars", 200)
        if len(normalized_text) > max_chars:
            result = self._generate_with_segmentation(
                text=normalized_text,
                output_format=output_format,
                max_chars=max_chars,
                preferred_voice=preferred_voice,
                voice_mapping=voice_mapping,
                segment_gap=segment_gap if segment_gap is not None else getattr(self.config.model, "segment_gap_seconds", 1.0),
                speaker_gap=speaker_gap if speaker_gap is not None else getattr(self.config.model, "speaker_gap_seconds", 1.0),
                instructions=instructions,
            )
            return self._post_process(result)

        with self._generation_lock:
            self._ensure_runtime_loaded()
            default_sample = self.sample_manager.resolve_or_default(self.config.voices.default_voice)
            normalized_script, ordered_samples, resolutions = self._build_dialogue_script(
                normalized_text,
                preferred_voice=preferred_voice,
                voice_mapping=voice_mapping or {},
                default_sample=default_sample,
            )
            result = self._generate_one(normalized_script, ordered_samples, resolutions, output_format)
            return self._post_process(result)

    def _generate_tagged_segments(
        self,
        segments: list[dict],
        output_format: str,
        voice_mapping: Optional[dict[str, str]] = None,
        segment_gap: Optional[float] = None,
        speaker_gap: Optional[float] = None,
    ) -> GenerationResult:
        """Generate each tagged segment with its own voice.

        Local MLX does not support instructions (tone) or speed modifiers.
        """
        audio_parts: list[bytes] = []
        total_gen_seconds = 0.0
        total_duration_seconds = 0.0
        all_resolutions: list[SpeakerResolution] = []
        seen_speakers: set[str] = set()

        for seg in segments:
            mapped_voice = voice_mapping.get(seg["speaker"]) if voice_mapping else seg["speaker"]
            resolved = self.sample_manager.resolve(mapped_voice)
            voice = mapped_voice if resolved else self.config.voices.default_voice

            result = self.generate_single(
                text=seg["text"],
                voice=voice,
                output_format=output_format,
            )
            audio_parts.append(result.audio_bytes)
            total_gen_seconds += result.generation_seconds
            total_duration_seconds += result.duration_seconds
            for spk in result.resolved_speakers:
                if spk.resolved_voice not in seen_speakers:
                    seen_speakers.add(spk.resolved_voice)
                    all_resolutions.append(spk)

        gap = segment_gap if segment_gap is not None else getattr(self.config.model, "segment_gap_seconds", 1.0)
        gaps = [0.0] + [gap] * (len(segments) - 1)
        final_audio = _concatenate_audio_segments(audio_parts, output_format, gaps=gaps)
        return GenerationResult(
            audio_bytes=final_audio,
            output_format=output_format,
            generation_seconds=total_gen_seconds,
            duration_seconds=total_duration_seconds,
            resolved_speakers=all_resolutions,
            segment_count=len(segments),
        )

    def _post_process(self, result: GenerationResult) -> GenerationResult:
        audio_bytes, duration = _apply_audio_effects(
            result.audio_bytes,
            result.output_format,
            stereo=getattr(self.config.model, "stereo", False),
            spatial_jitter=getattr(self.config.model, "spatial_jitter", False),
        )
        return GenerationResult(
            audio_bytes=audio_bytes,
            output_format=result.output_format,
            generation_seconds=result.generation_seconds,
            duration_seconds=duration,
            resolved_speakers=result.resolved_speakers,
            segment_count=result.segment_count,
        )

    def _generate_one(
        self,
        script: str,
        ordered_samples: list[VoiceSample],
        resolutions: list[SpeakerResolution],
        output_format: str,
    ) -> GenerationResult:
        """Run a single MLX generation for the given dialogue script."""
        speaker_embeds = []
        for sample in ordered_samples:
            embeds = self._load_voice_embeddings(sample)
            speaker_embeds.append((embeds.shape[0], embeds))

        tokenize_result = tokenize_text(
            script,
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
            silence_detection=False,
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
        paragraphs: list[list[str]] = []
        current_para: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                current_para.append(line.rstrip())
            else:
                if current_para:
                    paragraphs.append(current_para)
                    current_para = []
        if current_para:
            paragraphs.append(current_para)

        if not paragraphs:
            raise ValueError("Input text cannot be empty")

        para_has_speaker: list[bool] = []
        para_speaker_name: list[Optional[str]] = []
        for para in paragraphs:
            first = para[0].strip()
            match = self.DIALOGUE_LINE_RE.match(first)
            if match:
                para_has_speaker.append(True)
                para_speaker_name.append(match.group(1).strip())
            else:
                para_has_speaker.append(False)
                para_speaker_name.append(None)

        is_dialogue = any(para_has_speaker)

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
            script = "\n".join(
                f"Speaker 1: {line.strip()}" for para in paragraphs for line in para
            )
            return script, [sample], resolutions

        para_final_speaker: list[str] = []
        for i, has_speaker in enumerate(para_has_speaker):
            if has_speaker:
                para_final_speaker.append(para_speaker_name[i])
            else:
                found: Optional[str] = None
                for j in range(i - 1, -1, -1):
                    if para_has_speaker[j]:
                        found = para_speaker_name[j]
                        break
                if found is None:
                    found = preferred_voice or default_sample.speaker
                para_final_speaker.append(found)

        speaker_order: list[str] = []
        speaker_indices: dict[str, int] = {}
        speaker_samples: dict[str, VoiceSample] = {}
        resolutions: list[SpeakerResolution] = []

        for speaker in para_final_speaker:
            if speaker not in speaker_indices:
                speaker_indices[speaker] = len(speaker_order) + 1
                speaker_order.append(speaker)
                mapped_voice = voice_mapping.get(speaker) or speaker
                resolved = self.sample_manager.resolve(mapped_voice)
                sample = resolved or default_sample
                speaker_samples[speaker] = sample
                resolutions.append(
                    SpeakerResolution(
                        requested_name=speaker,
                        resolved_voice=sample.speaker,
                        used_default=resolved is None,
                        transcript_preview=sample.transcript_preview,
                    )
                )

        normalized_lines: list[str] = []
        for para, speaker in zip(paragraphs, para_final_speaker):
            first = para[0].strip()
            match = self.DIALOGUE_LINE_RE.match(first)
            if match:
                lines = [match.group(2).strip()] + [line.strip() for line in para[1:]]
            else:
                lines = [line.strip() for line in para]
            for line in lines:
                normalized_lines.append(f"Speaker {speaker_indices[speaker]}: {line}")

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
