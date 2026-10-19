"""FastAPI app for the new worktree implementation."""

from __future__ import annotations

import io
import json
import uuid
import copy
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import wave

import asyncio
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .audio_processing import normalize_wav_loudness
from .config import TTSProviderProfile, load_config, save_config_to_yaml
from .models import (
    AppConfigResponse,
    AppConfigUpdateRequest,
    BgmListResponse,
    BgmTrack,
    CreatePodcastRequest,
    GenerateRequest,
    GenerationRecord,
    HealthResponse,
    PodcastListResponse,
    PodcastProject,
    PodcastRequest,
    PruneOutputsRequest,
    PruneOutputsResponse,
    RegenerateSegmentRequest,
    SpeechRequest,
    TranscriptUpdateRequest,
    UpdateGapRequest,
    UpdateSegmentRequest,
    VoiceInfo,
    VoiceListResponse,
    LiveStartRequest,
    LiveStartResponse,
    LiveStopResponse,
    TTSProviderProfileRequest,
    TTSProviderTestRequest,
)
from .podcast_manager import PodcastManager
from .sample_manager import SampleManager, VoiceSample
from .tts_engine import create_engine

import logging

logger = logging.getLogger(__name__)


def _voice_info(sample: VoiceSample, default_voice: str) -> VoiceInfo:
    return VoiceInfo(
        speaker=sample.speaker,
        transcript=sample.transcript,
        transcript_preview=sample.transcript_preview,
        cache_ready=sample.cache_path.exists(),
        is_default=sample.speaker == default_voice,
        audio_url=f"/api/voices/{sample.speaker}/audio",
    )


def _configured_tts_provider(config) -> str:
    provider = getattr(config.model, "tts_provider", "")
    if provider in {"local", "qwen", "minimax", "omnivoice", "yovoice"}:
        return provider
    return "qwen" if getattr(config.model, "use_remote_qwen", False) else "local"


def _tts_profile_response(profile: TTSProviderProfile, active_id: str) -> dict:
    return {
        "id": profile.id,
        "name": profile.name,
        "provider": profile.provider,
        "api_mode": profile.api_mode,
        "base_url": profile.base_url,
        "model_id": profile.model_id,
        "voice_id": profile.voice_id,
        "emotion": profile.emotion,
        "speed": profile.speed,
        "volume": profile.volume,
        "pitch": profile.pitch,
        "api_key_configured": bool(profile.api_key),
        "active": profile.id == active_id,
    }


def _resolve_engine(request_engine: str | None, config, sample_manager):
    """Return the appropriate engine based on request override."""
    if request_engine is None:
        return create_engine(config, sample_manager)
    is_remote = getattr(config.model, "use_remote_qwen", False)
    if request_engine == "remote" and _configured_tts_provider(config) == "minimax":
        from .engines.minimax import MiniMaxEngine
        return MiniMaxEngine(config, sample_manager)
    if request_engine == "remote" and _configured_tts_provider(config) == "omnivoice":
        from .engines.omnivoice import OmniVoiceRemoteEngine
        return OmniVoiceRemoteEngine(config, sample_manager)
    if request_engine == "remote" and _configured_tts_provider(config) == "yovoice":
        from .engines.yovoice_remote import YovoiceRemoteEngine
        return YovoiceRemoteEngine(config, sample_manager)
    if request_engine == "remote" and not is_remote:
        from .engines.qwen_remote import QwenRemoteEngine
        return QwenRemoteEngine(config, sample_manager)
    elif request_engine == "local" and is_remote:
        from .engines.local_vibevoice import LocalVibeVoiceEngine
        return LocalVibeVoiceEngine(config, sample_manager)
    return create_engine(config, sample_manager)


def _discard_live_audio(session: dict, start_index: int) -> None:
    """Discard segment files and statuses from a restart point onward."""
    session_dir = Path(session["dir"])
    for index, segment in enumerate(session["segments"]):
        if index < start_index:
            continue
        filename = segment.get("audio_filename") or f"seg_{index:04d}.wav"
        (session_dir / filename).unlink(missing_ok=True)
        segment.pop("audio_filename", None)
        segment.pop("loudness_normalized", None)
        segment["status"] = "pending"
    (session_dir / "final.wav").unlink(missing_ok=True)


def _merge_live_audio(session: dict) -> Optional[str]:
    """Loudness-normalize and merge all available segments into one WAV."""
    session_dir = Path(session["dir"])
    target_rate = 24000
    chunks: list[np.ndarray] = []
    for segment in session["segments"]:
        index = segment.get("index", 0)
        filename = segment.get("audio_filename") or f"seg_{index:04d}.wav"
        path = session_dir / filename
        if not path.exists():
            continue
        if not segment.get("loudness_normalized"):
            normalize_wav_loudness(path)
            segment["loudness_normalized"] = True
            segment["audio_filename"] = filename
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            channels = source.getnchannels()
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples = samples.astype(np.float32)
        if rate != target_rate and len(samples) > 1:
            old_positions = np.linspace(0.0, 1.0, len(samples))
            new_length = max(1, round(len(samples) * target_rate / rate))
            samples = np.interp(
                np.linspace(0.0, 1.0, new_length), old_positions, samples,
            )
        chunks.append(np.clip(samples, -32768, 32767).astype("<i2"))
    if not chunks:
        return None

    output = session_dir / "final.wav"
    merged = np.concatenate(chunks)
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(target_rate)
        target.writeframes(merged.tobytes())
    normalize_wav_loudness(output)
    return output.name


def _persist_live_session_segments(session: dict) -> None:
    """Persist segment processing markers without changing session metadata."""
    meta_path = Path(session["dir"]) / "meta.json"
    if not meta_path.exists():
        return
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["segments"] = session["segments"]
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
        )
    except (OSError, TypeError, json.JSONDecodeError):
        logger.warning("Unable to persist Live Studio segment metadata")


def _ensure_live_preview_normalized(
    session_id: str,
    filename: str,
    target: Path,
    sessions: dict[str, dict],
    outputs_dir: Path,
) -> None:
    """Normalize a segment on first preview, including legacy sessions."""
    if filename == "final.wav":
        return

    session = sessions.get(session_id)
    if session is None:
        session_dir = (outputs_dir / "live_studio" / session_id).resolve()
        meta_path = session_dir / "meta.json"
        if not meta_path.exists():
            return
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        segments = metadata.get("segments", [])
        for segment in segments:
            index = segment.get("index", 0)
            expected = segment.get("audio_filename") or f"seg_{index:04d}.wav"
            if expected == filename and not segment.get("loudness_normalized"):
                normalize_wav_loudness(target)
                segment["audio_filename"] = filename
                segment["loudness_normalized"] = True
                metadata["segments"] = segments
                meta_path.write_text(
                    json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
                )
            return
        return

    for segment in session["segments"]:
        index = segment.get("index", 0)
        expected = segment.get("audio_filename") or f"seg_{index:04d}.wav"
        if expected != filename:
            continue
        if not segment.get("loudness_normalized"):
            normalize_wav_loudness(target)
            segment["audio_filename"] = filename
            segment["loudness_normalized"] = True
            _persist_live_session_segments(session)
        return


def create_app(config_path: Optional[str] = None) -> FastAPI:
    config = load_config(config_path)
    sample_manager = SampleManager(config.voices.expanded_base_dir, config.voices.default_voice)
    engine = create_engine(config, sample_manager)
    static_dir = Path(__file__).resolve().parent / "static"
    # 新界面（搬来的 yovoice 工作台）构建产物；缺失时自动回落到旧界面。
    web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    outputs_dir = config.outputs.expanded_base_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    history: deque[GenerationRecord] = deque(maxlen=config.outputs.history_limit)
    live_speakers_list = list(getattr(config.voices, "live_speakers", []) or [])
    podcast_manager = PodcastManager(outputs_dir / "podcasts", outputs_dir, live_speakers=live_speakers_list)
    bgm_dir = outputs_dir / "bgm"
    bgm_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Speech Studio",
        description="Qwen-TTS primary dialogue generation and voice management UI with local MLX fallback",
        version="0.3.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    no_cache_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @app.get("/legacy", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/legacy/", response_class=HTMLResponse, include_in_schema=False)
    def legacy_index() -> FileResponse:
        return FileResponse(static_dir / "index.html", headers=no_cache_headers)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> FileResponse:
        if (web_dist / "index.html").is_file():
            return FileResponse(web_dist / "index.html", headers=no_cache_headers)
        return FileResponse(static_dir / "index.html", headers=no_cache_headers)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            model=config.model.model_id,
            quantize_bits=config.model.quantize_bits,
            voices_count=len(sample_manager.list_samples()),
            default_voice=config.voices.default_voice,
        )

    @app.get("/api/config", response_model=AppConfigResponse)
    def get_config() -> AppConfigResponse:
        sample_manager.refresh()
        # Return the actual effective default voice (fallback if configured one is missing)
        effective_default = config.voices.default_voice
        if sample_manager.resolve(effective_default) is None:
            samples = sample_manager.list_samples()
            if samples:
                effective_default = samples[0].speaker
        return AppConfigResponse(
            model=config.model.model_id,
            tts_provider=_configured_tts_provider(config),
            quantize_bits=config.model.quantize_bits,
            default_voice=effective_default,
            diffusion_steps=config.model.diffusion_steps,
            cfg_scale=config.model.cfg_scale,
            max_speech_tokens=config.model.max_speech_tokens,
            use_semantic=config.model.use_semantic,
            use_coreml_semantic=config.model.use_coreml_semantic,
            seed=config.model.seed,
            voices_path=str(config.voices.base_dir),
            outputs_path=str(config.outputs.base_dir),
            use_remote_qwen=getattr(config.model, "use_remote_qwen", False),
            qwen_base_url=getattr(config.model, "qwen_base_url", ""),
            qwen_model_id=getattr(config.model, "qwen_model_id", ""),
            qwen_api_key_configured=bool(getattr(config.model, "qwen_api_key", "")),
            minimax_api_mode=getattr(config.model, "minimax_api_mode", "native"),
            minimax_api_key_configured=bool(getattr(config.model, "minimax_api_key", "")),
            minimax_base_url=getattr(config.model, "minimax_base_url", ""),
            minimax_model_id=getattr(config.model, "minimax_model_id", ""),
            minimax_voice_id=getattr(config.model, "minimax_voice_id", ""),
            minimax_emotion=getattr(config.model, "minimax_emotion", ""),
            minimax_speed=getattr(config.model, "minimax_speed", 1.0),
            minimax_volume=getattr(config.model, "minimax_volume", 1.0),
            minimax_pitch=getattr(config.model, "minimax_pitch", 0),
            max_segment_chars=getattr(config.model, "max_segment_chars", 200),
            stereo=getattr(config.model, "stereo", False),
            spatial_jitter=getattr(config.model, "spatial_jitter", False),
            segment_gap_seconds=getattr(config.model, "segment_gap_seconds", 1.0),
            speaker_gap_seconds=getattr(config.model, "speaker_gap_seconds", 1.0),
        )

    @app.post("/api/config", response_model=AppConfigResponse)
    def update_config(request: AppConfigUpdateRequest) -> AppConfigResponse:
        overrides = {}
        if request.voices_path is not None:
            overrides["voices"] = {"base_dir": request.voices_path}
        if request.outputs_path is not None:
            overrides["outputs"] = {"base_dir": request.outputs_path}
        if request.default_voice is not None:
            overrides["voices"] = overrides.get("voices", {})
            overrides["voices"]["default_voice"] = request.default_voice
        if request.diffusion_steps is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["diffusion_steps"] = request.diffusion_steps
        if request.quantize_bits is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["quantize_bits"] = request.quantize_bits
        if request.cfg_scale is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["cfg_scale"] = request.cfg_scale
        if request.max_speech_tokens is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["max_speech_tokens"] = request.max_speech_tokens
        if request.use_semantic is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["use_semantic"] = request.use_semantic
        if request.use_coreml_semantic is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["use_coreml_semantic"] = request.use_coreml_semantic
        if request.seed is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["seed"] = request.seed
        if request.use_remote_qwen is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["use_remote_qwen"] = request.use_remote_qwen
        if request.qwen_base_url is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["qwen_base_url"] = request.qwen_base_url
        if request.qwen_model_id is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["qwen_model_id"] = request.qwen_model_id
        if request.qwen_api_key is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["qwen_api_key"] = request.qwen_api_key
        if request.tts_provider is not None:
            if request.tts_provider not in {"local", "qwen", "minimax", "omnivoice", "yovoice"}:
                raise HTTPException(status_code=400, detail="Invalid TTS provider")
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["tts_provider"] = request.tts_provider
        if request.minimax_api_mode is not None:
            if request.minimax_api_mode not in {"native", "openai"}:
                raise HTTPException(status_code=400, detail="Invalid MiniMax API mode")
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_api_mode"] = request.minimax_api_mode
        if request.minimax_api_key is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_api_key"] = request.minimax_api_key
        if request.minimax_base_url is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_base_url"] = request.minimax_base_url
        if request.minimax_model_id is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_model_id"] = request.minimax_model_id
        if request.minimax_voice_id is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_voice_id"] = request.minimax_voice_id
        if request.minimax_emotion is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_emotion"] = request.minimax_emotion
        if request.minimax_speed is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_speed"] = request.minimax_speed
        if request.minimax_volume is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_volume"] = request.minimax_volume
        if request.minimax_pitch is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["minimax_pitch"] = request.minimax_pitch
        if request.max_segment_chars is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["max_segment_chars"] = request.max_segment_chars
        if request.stereo is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["stereo"] = request.stereo
        if request.spatial_jitter is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["spatial_jitter"] = request.spatial_jitter
        if request.segment_gap_seconds is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["segment_gap_seconds"] = request.segment_gap_seconds
        if request.speaker_gap_seconds is not None:
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["speaker_gap_seconds"] = request.speaker_gap_seconds

        config.apply_overrides(overrides)
        config.ensure_tts_providers()
        if any(
            value is not None
            for value in (
                request.qwen_base_url,
                request.qwen_model_id,
                request.qwen_api_key,
            )
        ):
            for profile in config.tts_providers:
                if profile.provider == "qwen":
                    profile.base_url = config.model.qwen_base_url
                    profile.model_id = config.model.qwen_model_id
                    profile.api_key = config.model.qwen_api_key
        if any(
            value is not None
            for value in (
                request.minimax_api_mode,
                request.minimax_api_key,
                request.minimax_base_url,
                request.minimax_model_id,
                request.minimax_voice_id,
                request.minimax_emotion,
                request.minimax_speed,
                request.minimax_volume,
                request.minimax_pitch,
            )
        ):
            for profile in config.tts_providers:
                if profile.provider == "minimax":
                    profile.api_mode = config.model.minimax_api_mode
                    profile.api_key = config.model.minimax_api_key
                    profile.base_url = config.model.minimax_base_url
                    profile.model_id = config.model.minimax_model_id
                    profile.voice_id = config.model.minimax_voice_id
                    profile.emotion = config.model.minimax_emotion
                    profile.speed = config.model.minimax_speed
                    profile.volume = config.model.minimax_volume
                    profile.pitch = config.model.minimax_pitch
        if config_path:
            save_config_to_yaml(config, config_path)
        # Sync sample_manager with any changed voice settings
        sample_manager.update_settings(
            base_dir=config.voices.expanded_base_dir,
            default_voice=config.voices.default_voice,
        )
        return get_config()

    @app.get("/api/tts/providers")
    def list_tts_providers() -> list[dict]:
        config.ensure_tts_providers()
        return [
            _tts_profile_response(profile, config.active_tts_provider)
            for profile in config.tts_providers
        ]

    @app.post("/api/tts/providers")
    def save_tts_provider(request: TTSProviderProfileRequest) -> dict:
        config.ensure_tts_providers()
        profile_id = request.id or f"{request.provider}-{uuid.uuid4().hex[:8]}"
        existing = next(
            (profile for profile in config.tts_providers if profile.id == profile_id),
            None,
        )
        api_key = request.api_key
        if api_key is None and existing is not None:
            api_key = existing.api_key
        profile = TTSProviderProfile(
            id=profile_id,
            name=request.name.strip(),
            provider=request.provider,
            api_mode=request.api_mode,
            base_url=request.base_url.strip(),
            api_key=(api_key or "").strip(),
            model_id=request.model_id.strip(),
            voice_id=request.voice_id.strip(),
            emotion=request.emotion.strip(),
            speed=request.speed,
            volume=request.volume,
            pitch=request.pitch,
        )
        if existing is None:
            config.tts_providers.append(profile)
        else:
            config.tts_providers[
                config.tts_providers.index(existing)
            ] = profile
        if request.active or not config.active_tts_provider:
            config.active_tts_provider = profile.id
        if config.active_tts_provider == profile.id:
            config.apply_tts_provider(profile.id)
        if config_path:
            save_config_to_yaml(config, config_path)
        return _tts_profile_response(profile, config.active_tts_provider)

    @app.post("/api/tts/providers/{provider_id}/activate")
    def activate_tts_provider(provider_id: str) -> dict:
        config.ensure_tts_providers()
        try:
            profile = config.apply_tts_provider(provider_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if config_path:
            save_config_to_yaml(config, config_path)
        return _tts_profile_response(profile, config.active_tts_provider)

    @app.delete("/api/tts/providers/{provider_id}")
    def delete_tts_provider(provider_id: str) -> dict:
        config.ensure_tts_providers()
        if provider_id == config.active_tts_provider:
            raise HTTPException(status_code=409, detail="不能删除当前启用的 TTS 配置")
        before = len(config.tts_providers)
        config.tts_providers = [
            profile for profile in config.tts_providers if profile.id != provider_id
        ]
        if len(config.tts_providers) == before:
            raise HTTPException(status_code=404, detail="TTS 配置不存在")
        if config_path:
            save_config_to_yaml(config, config_path)
        return {"id": provider_id, "deleted": True}

    @app.post("/api/tts/providers/{provider_id}/test")
    async def test_tts_provider(
        provider_id: str, request: TTSProviderTestRequest,
    ) -> dict:
        config.ensure_tts_providers()
        try:
            profile = config.get_tts_provider(provider_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        test_config = copy.deepcopy(config)
        test_config.tts_providers = [copy.deepcopy(profile)]
        test_config.active_tts_provider = profile.id
        test_config.apply_tts_provider(profile.id)
        target_engine = create_engine(test_config, sample_manager)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: target_engine.generate_single(
                    text=request.text,
                    voice=None,
                    output_format="wav",
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "id": profile.id,
            "name": profile.name,
            "ok": True,
            "generation_seconds": result.generation_seconds,
            "duration_seconds": result.duration_seconds,
        }

    @app.get("/api/voices", response_model=VoiceListResponse)
    def list_api_voices() -> VoiceListResponse:
        sample_manager.refresh()
        samples = sample_manager.list_samples()
        voices = [_voice_info(sample, config.voices.default_voice) for sample in samples]
        # If configured default voice doesn't exist, mark the first voice as default
        if voices and not any(v.is_default for v in voices):
            voices[0].is_default = True
        return VoiceListResponse(voices=voices)

    @app.post("/api/voices", response_model=VoiceInfo)
    async def upload_voice(
        speaker: Optional[str] = Form(None),
        transcript: str = Form(""),
        overwrite: bool = Form(False),
        audio_file: UploadFile = File(...),
    ) -> VoiceInfo:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")
        # Derive speaker from filename if not provided
        effective_speaker = speaker
        if not effective_speaker:
            effective_speaker = Path(audio_file.filename).stem
        audio_bytes = await audio_file.read()
        try:
            sample = sample_manager.add_voice(speaker=effective_speaker, audio_bytes=audio_bytes, transcript=transcript, overwrite=overwrite)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _voice_info(sample, config.voices.default_voice)

    @app.put("/api/voices/{speaker}/transcript", response_model=VoiceInfo)
    def update_transcript(speaker: str, request: TranscriptUpdateRequest) -> VoiceInfo:
        sample = sample_manager.update_transcript(speaker, request.transcript)
        return _voice_info(sample, config.voices.default_voice)

    @app.post("/api/voices/{speaker}/cache", response_model=VoiceInfo)
    def warm_voice_cache(speaker: str) -> VoiceInfo:
        sample = engine.ensure_voice_cache_ready(speaker)
        return _voice_info(sample, config.voices.default_voice)

    @app.delete("/api/voices/{speaker}")
    def delete_voice(speaker: str) -> dict[str, str]:
        if speaker == config.voices.default_voice:
            raise HTTPException(status_code=400, detail="Cannot delete the configured default voice")
        try:
            sample_manager.delete_voice(speaker)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted", "speaker": speaker}

    @app.post("/api/outputs/prune", response_model=PruneOutputsResponse)
    def prune_outputs(request: PruneOutputsRequest) -> PruneOutputsResponse:
        """Delete oldest generated audio files, keeping only the most recent *keep_count*."""
        audio_exts = {".wav", ".flac", ".ogg"}
        files = [f for f in outputs_dir.iterdir() if f.is_file() and f.suffix.lower() in audio_exts]
        files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
        to_keep = files_sorted[:request.keep_count]
        to_delete = files_sorted[request.keep_count:]

        deleted_names: list[str] = []
        for f in to_delete:
            f.unlink(missing_ok=True)
            deleted_names.append(f.name)

        # Sync in-memory history
        keep_set = {f.name for f in to_keep}
        for _ in range(len(history)):
            record = history.popleft()
            if record.filename in keep_set:
                history.append(record)

        return PruneOutputsResponse(
            deleted=deleted_names,
            kept=[f.name for f in to_keep],
        )

    @app.get("/api/voices/{speaker}/audio")
    def get_voice_audio(speaker: str) -> FileResponse:
        sample = sample_manager.resolve(speaker)
        if sample is None:
            raise HTTPException(status_code=404, detail="Voice not found")
        return FileResponse(sample.wav_path, media_type="audio/wav")

    @app.get("/api/generations", response_model=list[GenerationRecord])
    def list_generations() -> list[GenerationRecord]:
        return list(history)

    @app.get("/api/outputs/{filename}")
    def get_generated_audio(filename: str) -> FileResponse:
        output_path = (outputs_dir / filename).resolve()
        if output_path.parent != outputs_dir.resolve() or not output_path.exists():
            raise HTTPException(status_code=404, detail="Output not found")
        return FileResponse(output_path)

    def _store_generation(request_text: str, output_format: str, result) -> GenerationRecord:
        request_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{request_id}.{output_format}"
        output_path = outputs_dir / filename
        output_path.write_bytes(result.audio_bytes)
        record = GenerationRecord(
            request_id=request_id,
            filename=filename,
            audio_url=f"/api/outputs/{filename}",
            input_text=request_text,
            output_format=output_format,
            duration_seconds=result.duration_seconds,
            generation_seconds=result.generation_seconds,
            resolved_speakers=result.resolved_speakers,
            segment_count=getattr(result, "segment_count", 1),
            created_at=datetime.now().isoformat(),
        )
        history.appendleft(record)
        return record

    @app.post("/api/generate", response_model=GenerationRecord)
    def generate_audio(request: GenerateRequest) -> GenerationRecord:
        target_engine = engine
        if request.engine is not None:
            is_remote = getattr(config.model, "use_remote_qwen", False)
            if request.engine == "remote" and not is_remote:
                from .engines.qwen_remote import QwenRemoteEngine
                target_engine = QwenRemoteEngine(config, sample_manager)
            elif request.engine == "local" and is_remote:
                from .engines.local_vibevoice import LocalVibeVoiceEngine
                target_engine = LocalVibeVoiceEngine(config, sample_manager)
        segment_gap = getattr(config.model, "segment_gap_seconds", 1.0)
        try:
            result = target_engine.generate_dialogue(
                text=request.text,
                output_format=request.output_format,
                preferred_voice=request.voice,
                voice_mapping=request.voice_mapping,
                segment_gap=segment_gap,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _store_generation(request.text, request.output_format, result)

    @app.post("/api/generate/stream")
    def generate_audio_stream(request: GenerateRequest) -> StreamingResponse:
        target_engine = engine
        if request.engine is not None:
            is_remote = getattr(config.model, "use_remote_qwen", False)
            if request.engine == "remote" and not is_remote:
                from .engines.qwen_remote import QwenRemoteEngine
                target_engine = QwenRemoteEngine(config, sample_manager)
            elif request.engine == "local" and is_remote:
                from .engines.local_vibevoice import LocalVibeVoiceEngine
                target_engine = LocalVibeVoiceEngine(config, sample_manager)

        max_chars = getattr(config.model, "max_segment_chars", 200)
        stereo = getattr(config.model, "stereo", False)
        spatial_jitter = getattr(config.model, "spatial_jitter", False)
        segment_gap = getattr(config.model, "segment_gap_seconds", 1.0)

        def event_generator():
            for event in target_engine.generate_with_segmentation_stream(
                text=request.text,
                output_format=request.output_format,
                max_chars=max_chars,
                preferred_voice=request.voice,
                voice_mapping=request.voice_mapping,
                stereo=stereo,
                spatial_jitter=spatial_jitter,
                segment_gap=segment_gap,
            ):
                if event["type"] == "complete":
                    result = event["result"]
                    record = _store_generation(request.text, request.output_format, result)
                    import json
                    yield f"event: complete\ndata: {json.dumps(record.model_dump(), default=str)}\n\n"
                elif event["type"] == "error":
                    import json
                    yield f"event: error\ndata: {json.dumps({'message': event['message']})}\n\n"
                else:
                    import json
                    yield f"event: progress\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/v1/voices", response_model=VoiceListResponse)
    def list_openai_voices() -> VoiceListResponse:
        return list_api_voices()

    @app.get("/v1/voices/details", response_model=VoiceListResponse)
    def list_openai_voice_details() -> VoiceListResponse:
        return list_api_voices()

    @app.post("/v1/audio/speech")
    def create_speech(request: SpeechRequest) -> StreamingResponse:
        result = engine.generate_single(
            text=request.input,
            voice=request.voice or config.voices.default_voice,
            output_format=request.response_format,
        )
        return StreamingResponse(
            io.BytesIO(result.audio_bytes),
            media_type=f"audio/{request.response_format}",
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Generation-Time": f"{result.generation_seconds:.2f}",
            },
        )

    @app.post("/v1/audio/podcast")
    def create_podcast(request: PodcastRequest) -> StreamingResponse:
        result = engine.generate_dialogue(
            text=request.input,
            output_format=request.response_format,
            voice_mapping=request.voice_mapping,
        )
        return StreamingResponse(
            io.BytesIO(result.audio_bytes),
            media_type=f"audio/{request.response_format}",
            headers={
                "Content-Disposition": f"attachment; filename=podcast.{request.response_format}",
                "X-Generation-Time": f"{result.generation_seconds:.2f}",
            },
        )

    # ── Tone voices ─────────────────────────────────────────

    @app.get("/api/tone-voices/{speaker}")
    def list_tone_voices(speaker: str) -> list[VoiceInfo]:
        sample_manager.refresh()
        return [_voice_info(s, config.voices.default_voice) for s in sample_manager.list_tone_voices(speaker)]

    # ── Podcasts ────────────────────────────────────────────

    @app.post("/api/podcasts", response_model=PodcastProject)
    def create_podcast_project(request: CreatePodcastRequest) -> PodcastProject:
        return podcast_manager.create_project(
            title=request.title,
            text=request.text,
            output_format=request.output_format,
            gap_seconds=request.gap_seconds,
        )

    @app.get("/api/podcasts", response_model=PodcastListResponse)
    def list_podcast_projects() -> PodcastListResponse:
        return PodcastListResponse(podcasts=podcast_manager.list_projects())

    @app.get("/api/podcasts/{project_id}", response_model=PodcastProject)
    def get_podcast_project(project_id: str) -> PodcastProject:
        project = podcast_manager.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Podcast not found")
        return project

    @app.delete("/api/podcasts/{project_id}")
    def delete_podcast_project(project_id: str) -> dict[str, str]:
        if not podcast_manager.delete_project(project_id):
            raise HTTPException(status_code=404, detail="Podcast not found")
        return {"status": "deleted", "project_id": project_id}

    @app.put("/api/podcasts/{project_id}/gap", response_model=PodcastProject)
    def update_podcast_gap(
        project_id: str,
        request: UpdateGapRequest,
    ) -> PodcastProject:
        try:
            return podcast_manager.update_gap(project_id, request.gap_seconds)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/podcasts/{project_id}/segments/{index}", response_model=PodcastProject)
    def update_podcast_segment(
        project_id: str,
        index: int,
        request: UpdateSegmentRequest,
    ) -> PodcastProject:
        try:
            return podcast_manager.update_segment(
                project_id=project_id,
                index=index,
                text=request.text,
                speaker=request.speaker,
                tone=request.tone,
                pre_pause=request.pre_pause,
                post_pause=request.post_pause,
                bgm_filename=request.bgm_filename,
                bgm_position=request.bgm_position,
                bgm_volume=request.bgm_volume,
                bgm_fade_in=request.bgm_fade_in,
                bgm_fade_out=request.bgm_fade_out,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/podcasts/{project_id}/segments/{index}/insert", response_model=PodcastProject)
    def insert_podcast_segment(project_id: str, index: int) -> PodcastProject:
        try:
            return podcast_manager.insert_segment(project_id, index)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/podcasts/{project_id}/segments/{index}", response_model=PodcastProject)
    def delete_podcast_segment(project_id: str, index: int) -> PodcastProject:
        try:
            return podcast_manager.delete_segment(project_id, index)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/podcasts/{project_id}/segments/{index}/regenerate", response_model=PodcastProject)
    def regenerate_podcast_segment(
        project_id: str,
        index: int,
        request: RegenerateSegmentRequest,
    ) -> PodcastProject:
        target_engine = _resolve_engine(request.engine, config, sample_manager)
        try:
            return podcast_manager.regenerate_segment(
                project_id=project_id,
                index=index,
                engine=target_engine,
                sample_manager=sample_manager,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/podcasts/{project_id}/generate-all", response_model=PodcastProject)
    def generate_all_podcast_segments(
        project_id: str,
        request: RegenerateSegmentRequest,
    ) -> PodcastProject:
        target_engine = _resolve_engine(request.engine, config, sample_manager)
        try:
            return podcast_manager.generate_all_pending(
                project_id=project_id,
                engine=target_engine,
                sample_manager=sample_manager,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── BGM Library ──────────────────────────────────────────

    @app.get("/api/bgm", response_model=BgmListResponse)
    def list_bgm_tracks() -> BgmListResponse:
        tracks: list[BgmTrack] = []
        for f in sorted(bgm_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}:
                try:
                    import soundfile as sf
                    info = sf.info(str(f))
                    tracks.append(BgmTrack(filename=f.name, duration_seconds=info.duration))
                except Exception:
                    tracks.append(BgmTrack(filename=f.name, duration_seconds=0.0))
        return BgmListResponse(tracks=tracks)

    @app.post("/api/bgm")
    async def upload_bgm(audio_file: UploadFile = File(...)) -> BgmTrack:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")
        safe_name = Path(audio_file.filename).name
        allowed_exts = {".mp3", ".wav", ".flac", ".ogg"}
        if Path(safe_name).suffix.lower() not in allowed_exts:
            raise HTTPException(status_code=400, detail="Invalid audio file format. Allowed: mp3, wav, flac, ogg")
        target_path = bgm_dir / safe_name
        content = await audio_file.read()
        target_path.write_bytes(content)
        try:
            import soundfile as sf
            info = sf.info(str(target_path))
            return BgmTrack(filename=safe_name, duration_seconds=info.duration)
        except Exception:
            return BgmTrack(filename=safe_name, duration_seconds=0.0)

    def _resolve_bgm_path(filename: str) -> Path:
        target_path = (bgm_dir / filename).resolve()
        try:
            target_path.relative_to(bgm_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="BGM track not found")
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="BGM track not found")
        return target_path

    @app.delete("/api/bgm/{filename}")
    def delete_bgm(filename: str) -> dict[str, str]:
        target_path = _resolve_bgm_path(filename)
        target_path.unlink()
        return {"status": "deleted", "filename": filename}

    @app.get("/api/bgm/{filename}")
    def get_bgm_audio(filename: str) -> FileResponse:
        target_path = _resolve_bgm_path(filename)
        return FileResponse(target_path)

    @app.post("/api/podcasts/{project_id}/merge")
    def merge_podcast(project_id: str) -> dict[str, str]:
        try:
            filename = podcast_manager.merge_project(project_id, bgm_dir=bgm_dir)
            return {
                "status": "merged",
                "filename": filename,
                "audio_url": f"/api/outputs/{filename}",
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/podcasts/{project_id}/audio/{filename}")
    def get_podcast_segment_audio(project_id: str, filename: str) -> FileResponse:
        audio_dir = outputs_dir / "podcasts" / project_id
        audio_path = (audio_dir / filename).resolve()
        if audio_path.parent != audio_dir.resolve() or not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio segment not found")
        return FileResponse(audio_path, media_type="audio/wav")

    # ── Live session services ───────────────────────────────────────────
    from .live.session import (
        LiveSessionRegistry, LiveState, Trigger, IllegalStateTransition,
    )
    from .live.asr_engine import ASRConfig

    live_registry = LiveSessionRegistry(
        asr_config=ASRConfig(**config.asr.model_dump())
    )

    def _asr_status() -> dict:
        asr = live_registry.get_asr()
        status = asr.progress
        status.update({
            "enabled": bool(getattr(config.asr, "enabled", False)),
            "ready": asr.is_ready,
            "warming": asr.is_warming,
            "model": config.asr.model,
        })
        return status

    @app.get("/api/asr/status")
    def get_asr_status() -> dict:
        return _asr_status()

    # ── Live Studio (independent online broadcast) ─────────────────────
    # A self-contained live broadcast: AI segments are TTS-generated and
    # streamed to the client; LIVE (human) segments open the mic only
    # AFTER the previous AI segment finishes playing on the client.

    live_studio_sessions: dict[str, dict] = {}

    @app.post("/api/live/start")
    async def start_live_studio(request: Request) -> dict:
        """Start a live studio session from a tagged dialogue script.

        Parses the script into segments, marks segments whose speaker is in
        config.voices.live_speakers as LIVE (human), the rest as AI.
        Returns a session_id to connect the WebSocket driver.
        """
        body = await request.json()
        text = (body.get("text") or "").strip()
        title = (body.get("title") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        from .engines.base import _parse_tagged_dialogue
        tagged, is_tagged = _parse_tagged_dialogue(text)
        if not is_tagged:
            # Plain narration -> single AI segment
            tagged = [{"speaker": None, "text": text}]

        live_speakers = set(getattr(config.voices, "live_speakers", []) or [])
        segs = []
        for i, seg in enumerate(tagged):
            speaker = seg.get("speaker") or ""
            resolved_speaker = speaker or config.voices.default_voice
            is_live = bool(speaker) and speaker in live_speakers
            segs.append({
                "index": i,
                "text": seg["text"].strip(),
                "speaker": speaker,
                "resolved_voice": resolved_speaker,
                "source": "live" if is_live else "tts",
                "status": "pending",
            })

        asr_required = bool(getattr(config.asr, "enabled", False)) and any(
            seg["source"] == "live" for seg in segs
        )
        if asr_required:
            live_registry.get_asr().start_warmup()

        session_id = uuid.uuid4().hex[:12]
        session_dir = outputs_dir / "live_studio" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        if not title and segs:
            title = segs[0]["text"].strip()[:30]
        if not title:
            title = session_id

        live_studio_sessions[session_id] = {
            "session_id": session_id,
            "title": title,
            "segments": segs,
            "cursor": 0,
            "state": "IDLE",
            "dir": str(session_dir),
            "stop_requested": False,
            "prefetch_locks": {},
        }
        # Persist metadata for sidebar history
        import json as _json
        (session_dir / "meta.json").write_text(
            _json.dumps({
                "session_id": session_id,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "segments": segs,
                "segment_count": len(segs),
                "first_text": segs[0]["text"][:50] if segs else "",
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "session_id": session_id,
            "title": title,
            "segments": segs,
            "live_speakers": sorted(live_speakers),
            "asr_required": asr_required,
            "asr": _asr_status(),
        }

    @app.websocket("/ws/live/{session_id}")
    async def ws_live_studio(ws: WebSocket, session_id: str) -> None:
        """Drive a live studio session: AI -> play -> (human record) -> next.

        Protocol (server -> client, JSON text unless noted):
            {"type":"state","state":"AI_SPEAKING"|"RECORDING"|...}
            {"type":"segment_start","index":i,"source":"tts"|"live",
             "speaker":"...","text":"...","resolved_voice":"..."}
            {"type":"audio","index":i}           # followed by binary WAV frame
             {"type":"record_start","index":i,"target_text":"..."}
             {"type":"asr_partial","text":"..."}
             {"type":"alignment","matched":n,"total":n,"ratio":r}
             {"type":"restart","to_index":i,"from_index":j}
             {"type":"segment_done","index":i,"source":"tts"|"live",
              "audio_url":"/api/live/{sid}/audio/{i}"}
            {"type":"finished"}
            {"type":"error","message":"..."}

        Client -> server:
             {"type":"ai_finished","index":i}     # client finished playing AI wav
             {"type":"record_done","index":i}     # human finished recording
             {"type":"restart_from","index":i}    # restart from a clicked segment
             binary frames = PCM16 mono 16kHz mic audio during RECORDING
        """
        session = live_studio_sessions.get(session_id)
        if session is None:
            await ws.close(code=1008, reason="SESSION_NOT_FOUND")
            return
        await ws.accept()

        # Bind driver
        if session.get("_driver_ws") is not None:
            await ws.close(code=1008, reason="driver already connected")
            return
        session["_driver_ws"] = ws
        session["_driver_task"] = None

        async def send_json(frame: dict) -> None:
            try:
                await ws.send_text(json.dumps(frame, ensure_ascii=False))
            except Exception:
                pass

        async def send_bytes(data: bytes) -> None:
            try:
                await ws.send_bytes(data)
            except Exception:
                pass

        asr = live_registry.get_asr()

        async def send_asr_progress() -> None:
            """Forward model download/load progress while the WS is open."""
            if not config.asr.enabled:
                await send_json({
                    "type": "asr_unavailable",
                    "reason": "ASR 未启用，真人段使用手动结束",
                })
                return
            last = None
            while True:
                status = _asr_status()
                comparable = (
                    status.get("status"), status.get("progress"),
                    status.get("downloaded_bytes"), status.get("total_bytes"),
                    status.get("message"), status.get("error"),
                )
                if comparable != last:
                    await send_json({"type": "asr_progress", **status})
                    last = comparable
                if status["ready"] or status["status"] == "error":
                    return
                await asyncio.sleep(0.25)

        async def wait_for_asr() -> bool:
            if not config.asr.enabled:
                return False
            while asr.is_warming:
                await asyncio.sleep(0.25)
            return asr.is_ready

        # Record buffers per segment index
        record_buffers: dict[int, bytearray] = {}
        asr_buffer = bytearray()
        asr_silence_ms = 0
        asr_chunk_bytes = max(32000, int(16000 * 2 * config.asr.chunk_seconds))
        record_state = {
            "transcript": "",
            "generation": 0,
            "auto_done": False,
            "auto_done_index": -1,
        }
        asr_queue: asyncio.Queue = asyncio.Queue()
        auto_done_event = asyncio.Event()

        async def run_driver():
            segs = session["segments"]
            idx = 0

            async def restart_from(target_index: int, from_index: int) -> None:
                for discarded_index in range(target_index, len(segs)):
                    record_buffers.pop(discarded_index, None)
                asr_buffer.clear()
                record_state["generation"] += 1
                record_state["transcript"] = ""
                record_state["auto_done"] = False
                record_state["auto_done_index"] = -1
                auto_done_event.clear()
                _discard_live_audio(session, target_index)
                await send_json({
                    "type": "restart",
                    "to_index": target_index,
                    "from_index": from_index,
                })

            try:
                while idx < len(segs):
                    if session.get("stop_requested"):
                        break
                    seg = segs[idx]
                    session["cursor"] = idx
                    text = seg["text"]

                    await send_json({
                        "type": "segment_start",
                        "index": idx,
                        "source": seg["source"],
                        "speaker": seg["speaker"],
                        "text": text,
                        "resolved_voice": seg["resolved_voice"],
                    })

                    if seg["source"] == "tts":
                        session["state"] = "AI_SPEAKING"
                        await send_json({"type": "state", "state": "AI_SPEAKING"})
                        target_engine = _resolve_engine(session.get("engine_name"), config, sample_manager)
                        wav_path = Path(session["dir"]) / f"seg_{idx:04d}.wav"

                        # Use a per-segment lock so prefetch and runtime generation
                        # don't race on the same WAV file.
                        prefetch_locks = session.setdefault("prefetch_locks", {})
                        lock = prefetch_locks.setdefault(idx, asyncio.Lock())
                        async with lock:
                            if not (seg.get("audio_filename") and wav_path.exists()):
                                try:
                                    loop = asyncio.get_event_loop()
                                    result = await loop.run_in_executor(
                                        None,
                                        lambda: target_engine.generate_single(
                                            text=text,
                                            voice=seg["resolved_voice"],
                                            output_format="wav",
                                        ),
                                    )
                                except Exception as exc:
                                    # Configuration / engine failure: report and STOP.
                                    # Do not silently skip - the user must fix the
                                    # underlying issue (e.g. unreachable remote, bad
                                    # voice, model load failure) before resuming.
                                    seg["status"] = "error"
                                    session["state"] = "ERROR"
                                    await send_json({
                                        "type": "error",
                                        "message": f"第 {idx + 1} 段 TTS 生成失败：{exc}。请检查引擎配置/声音后重新开始。",
                                    })
                                    await send_json({"type": "state", "state": "ERROR"})
                                    return

                                wav_path.write_bytes(result.audio_bytes)
                                seg["audio_filename"] = f"seg_{idx:04d}.wav"
                            if not seg.get("loudness_normalized"):
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(
                                    None, normalize_wav_loudness, wav_path,
                                )
                                seg["loudness_normalized"] = True
                            seg["status"] = "generated"

                        # Send audio frame: JSON header then binary WAV
                        audio_bytes = wav_path.read_bytes()
                        await send_json({"type": "audio", "index": idx,
                                         "sample_rate": target_engine.sample_rate})
                        await send_bytes(audio_bytes)

                        # Wait for playback completion or a manual restart.
                        ai_result = await _wait_for(ws, lambda f: (
                            f.get("type") == "ai_finished" and f.get("index") == idx
                        ))
                        if ai_result.get("type") == "restart_from":
                            await restart_from(ai_result["index"], idx)
                            idx = ai_result["index"]
                            continue
                    else:
                        # LIVE (human) segment: open mic
                        if config.asr.enabled and not await wait_for_asr():
                            status = _asr_status()
                            session["state"] = "ERROR"
                            await send_json({
                                "type": "error",
                                "message": status.get("error") or "ASR 模型未就绪，请检查模型下载配置后重试。",
                            })
                            await send_json({"type": "state", "state": "ERROR"})
                            return
                        session["state"] = "RECORDING"
                        asr_buffer.clear()
                        record_state["generation"] += 1
                        record_state["transcript"] = ""
                        record_state["auto_done"] = False
                        record_state["auto_done_index"] = -1
                        auto_done_event.clear()
                        await send_json({"type": "state", "state": "RECORDING"})
                        await send_json({"type": "record_start", "index": idx,
                                         "target_text": text,
                                         "asr_enabled": config.asr.enabled,
                                         "asr_ready": asr.is_ready})
                        # Wait for the user to finish this segment or restart elsewhere.
                        record_result = await _wait_for(ws, lambda f: (
                            f.get("type") == "record_done" and f.get("index") == idx
                        ))
                        if record_result.get("type") == "restart_from":
                            await restart_from(record_result["index"], idx)
                            idx = record_result["index"]
                            continue
                        record_state["generation"] += 1
                        auto_done_event.clear()
                        # Flush buffered PCM for this segment to a WAV
                        pcm = bytes(record_buffers.pop(idx, bytearray()))
                        if pcm:
                            wav_path = Path(session["dir"]) / f"seg_{idx:04d}.wav"
                            from .live.wav_writer import write_wav
                            write_wav(wav_path, pcm, sample_rate=16000, channels=1)
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None, normalize_wav_loudness, wav_path,
                            )
                            seg["audio_filename"] = f"seg_{idx:04d}.wav"
                            seg["loudness_normalized"] = True
                            seg["status"] = "recorded"
                        else:
                            seg["status"] = "missing"

                    audio_url = None
                    if seg.get("audio_filename"):
                        audio_url = f"/api/live/{session_id}/audio/{seg['audio_filename']}"
                    await send_json({
                        "type": "segment_done",
                        "index": idx,
                        "source": seg["source"],
                        "audio_url": audio_url,
                    })
                    idx += 1

                loop = asyncio.get_event_loop()
                final_filename = await loop.run_in_executor(
                    None, _merge_live_audio, session,
                )
                final_url = (
                    f"/api/live/{session_id}/audio/{final_filename}?download=true"
                    if final_filename else None
                )
                session["state"] = "FINISHED"
                await send_json({"type": "state", "state": "FINISHED"})
                await send_json({"type": "finished", "audio_url": final_url})
            except Exception as exc:
                session["state"] = "ERROR"
                await send_json({"type": "error", "message": str(exc)})

        # Helper to await a matching JSON frame from the client, while
        # routing binary mic frames into record_buffers.
        async def _wait_for(websocket, predicate):
            while True:
                if (
                    auto_done_event.is_set()
                    and record_state["auto_done_index"] != session.get("cursor", 0)
                ):
                    auto_done_event.clear()
                receive_task = asyncio.create_task(websocket.receive())
                auto_task = asyncio.create_task(auto_done_event.wait())
                timeout_task = asyncio.create_task(asyncio.sleep(0.5))
                done, _ = await asyncio.wait(
                    {receive_task, auto_task, timeout_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if auto_task in done and auto_done_event.is_set():
                    receive_task.cancel()
                    timeout_task.cancel()
                    await asyncio.gather(receive_task, timeout_task, return_exceptions=True)
                    return {
                        "type": "record_done",
                        "index": session.get("cursor", 0),
                        "auto": True,
                    }
                if timeout_task in done:
                    # No frame for 500 ms; the speaker likely paused or finished.
                    # Flush any trailing ASR audio so the final words can be
                    # transcribed and auto-completion can fire.
                    receive_task.cancel()
                    auto_task.cancel()
                    await asyncio.gather(receive_task, auto_task, return_exceptions=True)
                    if config.asr.enabled and asr.is_ready and asr_buffer:
                        cur = session.get("cursor", 0)
                        remainder = bytes(asr_buffer)
                        asr_buffer.clear()
                        padded = remainder + b"\x00" * (asr_chunk_bytes - len(remainder))
                        await asr_queue.put((
                            cur, record_state["generation"], padded,
                            len(record_buffers.get(cur, b"")) // 32,
                        ))
                        asr_silence_ms = 0
                    continue
                timeout_task.cancel()
                auto_task.cancel()
                await asyncio.gather(auto_task, timeout_task, return_exceptions=True)
                msg = receive_task.result()
                if msg["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if "bytes" in msg and msg["bytes"] is not None:
                    # Mic PCM frame: buffer under current live cursor
                    cur = session.get("cursor", 0)
                    record_buffers.setdefault(cur, bytearray()).extend(msg["bytes"])
                    if config.asr.enabled and asr.is_ready:
                        asr_buffer.extend(msg["bytes"])
                        while len(asr_buffer) >= asr_chunk_bytes:
                            chunk = bytes(asr_buffer[:asr_chunk_bytes])
                            del asr_buffer[:asr_chunk_bytes]
                            await asr_queue.put((
                                cur, record_state["generation"], chunk,
                                len(record_buffers.get(cur, b"")) // 32,
                            ))
                        # After a short silence, flush any trailing audio so
                        # the ASR can transcribe the final words and trigger
                        # auto-completion without waiting for manual record_done.
                        if asr_buffer:
                            samples = np.frombuffer(msg["bytes"], dtype=np.int16)
                            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                            if rms < 500:
                                asr_silence_ms += len(msg["bytes"]) // 32
                            else:
                                asr_silence_ms = 0
                            if asr_silence_ms >= 500:
                                remainder = bytes(asr_buffer)
                                asr_buffer.clear()
                                padded = remainder + b"\x00" * (asr_chunk_bytes - len(remainder))
                                await asr_queue.put((
                                    cur, record_state["generation"], padded,
                                    len(record_buffers.get(cur, b"")) // 32,
                                ))
                                asr_silence_ms = 0
                    if (
                        record_state["auto_done"]
                        and record_state["auto_done_index"] == cur
                    ):
                        return {"type": "record_done", "index": cur, "auto": True}
                    continue
                if "text" in msg and msg["text"] is not None:
                    try:
                        frame = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    if frame.get("type") == "pause":
                        previous_state = session["state"]
                        if previous_state != "PAUSED":
                            session["state"] = "PAUSED"
                            await send_json({"type": "state", "state": "PAUSED"})
                            while True:
                                paused_msg = await websocket.receive()
                                if paused_msg["type"] == "websocket.disconnect":
                                    raise WebSocketDisconnect()
                                if not paused_msg.get("text"):
                                    continue
                                try:
                                    paused_frame = json.loads(paused_msg["text"])
                                except json.JSONDecodeError:
                                    continue
                                if paused_frame.get("type") == "resume":
                                    break
                                if paused_frame.get("type") == "restart_from":
                                    target_index = paused_frame.get("index")
                                    if isinstance(target_index, int) and 0 <= target_index < len(session["segments"]):
                                        return {"type": "restart_from", "index": target_index}
                            session["state"] = previous_state
                            await send_json({"type": "state", "state": previous_state})
                        continue
                    if frame.get("type") == "resume":
                        continue
                    if frame.get("type") == "restart_from":
                        target_index = frame.get("index")
                        if isinstance(target_index, int) and 0 <= target_index < len(session["segments"]):
                            return {"type": "restart_from", "index": target_index}
                        continue
                    # Optional: surface ASR partials to client
                    if frame.get("type") == "asr_partial":
                        await send_json({"type": "asr_partial", "text": frame.get("text", "")})
                        continue
                    if predicate(frame):
                        if (
                            frame.get("type") == "record_done"
                            and config.asr.enabled
                            and asr.is_ready
                            and asr_buffer
                        ):
                            remainder = bytes(asr_buffer)
                            asr_buffer.clear()
                            padded = remainder + b"\x00" * (asr_chunk_bytes - len(remainder))
                            await asr_queue.put((
                                session.get("cursor", 0),
                                record_state["generation"],
                                padded,
                                len(record_buffers.get(session.get("cursor", 0), b"")) // 32,
                            ))
                        if frame.get("type") == "record_done" and config.asr.enabled:
                            await asr_queue.join()
                        return frame

        async def asr_worker() -> None:
            """Transcribe queued chunks without blocking WebSocket receive."""
            from .live.end_detector import compute_alignment_ratio, tokenize_text
            while True:
                cur, generation, chunk, audio_ms = await asr_queue.get()
                try:
                    if generation != record_state["generation"]:
                        continue
                    result = await asr.transcribe_chunk(chunk)
                    if generation != record_state["generation"] or not result.text:
                        continue
                    chunk_text = result.text.strip()
                    if record_state["transcript"]:
                        record_state["transcript"] += " " + chunk_text
                    else:
                        record_state["transcript"] = chunk_text
                    await send_json({
                        "type": "asr_partial",
                        "text": record_state["transcript"],
                        "audio_ms": audio_ms,
                    })
                    ratio = compute_alignment_ratio(
                        record_state["transcript"], session["segments"][cur]["text"],
                    )
                    await send_json({
                        "type": "alignment",
                        "ratio": ratio,
                    })
                    # Require the speaker to have produced at least as many
                    # tokens as the target, so that a single missing English
                    # word doesn't trigger auto-completion prematurely.
                    target_tokens = tokenize_text(session["segments"][cur]["text"])
                    asr_tokens = tokenize_text(record_state["transcript"])
                    if (ratio >= 0.85
                            and len(asr_tokens) >= len(target_tokens)
                            and not record_state["auto_done"]):
                        record_state["auto_done"] = True
                        record_state["auto_done_index"] = cur
                        await send_json({
                            "type": "record_auto_done",
                            "index": cur,
                            "ratio": ratio,
                        })
                        auto_done_event.set()
                finally:
                    asr_queue.task_done()

        # Download/load progress, ASR, and the segment driver run concurrently.
        asr_progress_task = asyncio.create_task(send_asr_progress())
        asr_worker_task = asyncio.create_task(asr_worker()) if config.asr.enabled else None
        driver_task = asyncio.create_task(run_driver())

        # Keep the WS open; receive loop is handled inside run_driver via _wait_for.
        # We still need to pump receive() to detect disconnects when the driver
        # is NOT waiting (e.g. during TTS generation). We do a lightweight wait.
        try:
            await driver_task
        except WebSocketDisconnect:
            session["stop_requested"] = True
        finally:
            asr_progress_task.cancel()
            if asr_worker_task is not None:
                asr_worker_task.cancel()
            session["_driver_ws"] = None

    @app.get("/api/live/{session_id}/audio/{filename}")
    def get_live_studio_audio(
        session_id: str, filename: str, download: bool = False,
    ) -> FileResponse:
        session_dir = (outputs_dir / "live_studio" / session_id).resolve()
        target = (session_dir / filename).resolve()
        try:
            target.relative_to(session_dir)
        except ValueError:
            raise HTTPException(status_code=404, detail="Audio not found")
        if not target.exists():
            raise HTTPException(status_code=404, detail="Audio not found")
        _ensure_live_preview_normalized(
            session_id, filename, target, live_studio_sessions, outputs_dir,
        )
        download_name = (
            f"live-studio-{session_id}.wav" if filename == "final.wav" else filename
        )
        return FileResponse(
            target,
            media_type="audio/wav",
            filename=download_name if download else None,
        )

    @app.post("/api/live/{session_id}/stop")
    def stop_live_studio(session_id: str) -> dict:
        session = live_studio_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        session["stop_requested"] = True
        session["state"] = "FINISHED"
        return {"session_id": session_id, "state": "FINISHED"}

    @app.get("/api/live/sessions")
    def list_live_sessions() -> list:
        """List all live studio sessions from disk."""
        import json as _json
        base = outputs_dir / "live_studio"
        if not base.exists():
            return []
        sessions = []
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                    has_final = (d / "final.wav").exists()
                    sessions.append({
                        "session_id": meta.get("session_id", d.name),
                        "title": meta.get("title") or meta.get("first_text", "") or d.name,
                        "created_at": meta.get("created_at", ""),
                        "segment_count": meta.get("segment_count", 0),
                        "first_text": meta.get("first_text", ""),
                        "has_final": has_final,
                    })
                except Exception:
                    pass
        return sessions

    @app.get("/api/live/{session_id}")
    def get_live_session(session_id: str) -> dict:
        """Load a live studio session from disk (or memory) with current audio URLs."""
        import json as _json
        session_dir = (outputs_dir / "live_studio" / session_id).resolve()
        try:
            session_dir.relative_to((outputs_dir / "live_studio").resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session path")
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        meta_path = session_dir / "meta.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        try:
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"无法读取会话元数据: {exc}") from exc

        segs = meta.get("segments", [])
        # Keep in-memory session in sync so PATCH/WS can operate after a refresh
        session = live_studio_sessions.get(session_id)
        if session is None:
            has_final = (session_dir / "final.wav").exists()
            session = {
                "session_id": session_id,
                "title": meta.get("title", meta.get("first_text", "")),
                "segments": segs,
                "cursor": 0,
                "state": "FINISHED" if has_final else "IDLE",
                "dir": str(session_dir),
                "stop_requested": False,
                "prefetch_locks": {},
            }
            live_studio_sessions[session_id] = session

        response_segments = []
        for seg in segs:
            idx = seg.get("index", 0)
            filename = seg.get("audio_filename") or f"seg_{idx:04d}.wav"
            path = session_dir / filename
            has_audio = path.exists()
            source = seg.get("source", "tts")
            status = seg.get("status", "pending")
            if has_audio:
                status = "recorded" if source == "live" else "generated"
            audio_url = f"/api/live/{session_id}/audio/{filename}" if has_audio else None
            response_segments.append({
                "index": idx,
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker", ""),
                "resolved_voice": seg.get("resolved_voice", ""),
                "source": source,
                "status": status,
                "audio_filename": filename if has_audio else None,
                "audio_url": audio_url,
            })

        has_final = (session_dir / "final.wav").exists()
        return {
            "session_id": session_id,
            "title": session.get("title") or meta.get("title") or meta.get("first_text", ""),
            "created_at": meta.get("created_at", ""),
            "state": session.get("state", "IDLE"),
            "segments": response_segments,
            "has_final": has_final,
            "final_audio_url": f"/api/live/{session_id}/audio/final.wav?download=true" if has_final else None,
        }

    @app.post("/api/live/{session_id}/export")
    def export_live_session(session_id: str) -> dict:
        """Rebuild and return one merged, loudness-normalized WAV."""
        session = live_studio_sessions.get(session_id)
        if session is None:
            get_live_session(session_id)
            session = live_studio_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")

        filename = _merge_live_audio(session)
        if filename is None:
            raise HTTPException(status_code=409, detail="NO_AUDIO_SEGMENTS")
        return {
            "session_id": session_id,
            "filename": filename,
            "audio_url": f"/api/live/{session_id}/audio/{filename}?download=true",
        }

    @app.post("/api/live/{session_id}/prefetch")
    async def prefetch_live_tts(session_id: str, request: Request) -> dict:
        """Prefetch TTS audio for all AI segments in the background."""
        session = live_studio_sessions.get(session_id)
        session_dir = (outputs_dir / "live_studio" / session_id).resolve()
        if session is None:
            try:
                session_dir.relative_to((outputs_dir / "live_studio").resolve())
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid session path")
            meta_path = session_dir / "meta.json"
            if not meta_path.exists():
                raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
            import json as _json
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            has_final = (session_dir / "final.wav").exists()
            session = {
                "session_id": session_id,
                "title": meta.get("title", meta.get("first_text", "")),
                "segments": meta.get("segments", []),
                "cursor": 0,
                "state": "FINISHED" if has_final else "IDLE",
                "dir": str(session_dir),
                "stop_requested": False,
                "prefetch_locks": {},
            }
            live_studio_sessions[session_id] = session

        body = await request.json()
        engine_name = body.get("engine")
        target_engine = _resolve_engine(engine_name, config, sample_manager)

        async def _prefetch() -> None:
            segs = session["segments"]
            for seg in segs:
                if seg.get("source") != "tts":
                    continue
                idx = seg.get("index", 0)
                wav_path = Path(session["dir"]) / f"seg_{idx:04d}.wav"
                prefetch_locks = session.setdefault("prefetch_locks", {})
                lock = prefetch_locks.setdefault(idx, asyncio.Lock())
                async with lock:
                    if seg.get("audio_filename") and wav_path.exists():
                        continue
                    try:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda: target_engine.generate_single(
                                text=seg["text"],
                                voice=seg["resolved_voice"],
                                output_format="wav",
                            ),
                        )
                        wav_path.write_bytes(result.audio_bytes)
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, normalize_wav_loudness, wav_path,
                        )
                        seg["audio_filename"] = f"seg_{idx:04d}.wav"
                        seg["loudness_normalized"] = True
                        seg["status"] = "generated"
                    except Exception as exc:
                        logger.warning("Prefetch failed for segment %s: %s", idx, exc)

        asyncio.create_task(_prefetch())
        return {"session_id": session_id, "prefetch": "started"}

    @app.delete("/api/live/{session_id}")
    def delete_live_session(session_id: str) -> dict:
        import shutil
        session_dir = (outputs_dir / "live_studio" / session_id).resolve()
        try:
            session_dir.relative_to((outputs_dir / "live_studio").resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session path")
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        shutil.rmtree(session_dir, ignore_errors=True)
        live_studio_sessions.pop(session_id, None)
        return {"session_id": session_id, "deleted": True}

    @app.patch("/api/live/{session_id}/segment/{index}")
    async def update_live_segment(session_id: str, index: int, request: Request) -> dict:
        """Update the text of a segment before or during a live session."""
        session = live_studio_sessions.get(session_id)
        session_dir = (outputs_dir / "live_studio" / session_id).resolve()
        if session is None:
            # Load from disk so edits can happen after a page refresh
            try:
                session_dir.relative_to((outputs_dir / "live_studio").resolve())
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid session path")
            meta_path = session_dir / "meta.json"
            if not meta_path.exists():
                raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
            import json as _json
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            has_final = (session_dir / "final.wav").exists()
            session = {
                "session_id": session_id,
                "title": meta.get("title", meta.get("first_text", "")),
                "segments": meta.get("segments", []),
                "cursor": 0,
                "state": "FINISHED" if has_final else "IDLE",
                "dir": str(session_dir),
                "stop_requested": False,
                "prefetch_locks": {},
            }
            live_studio_sessions[session_id] = session

        segs = session["segments"]
        if index < 0 or index >= len(segs):
            raise HTTPException(status_code=404, detail="SEGMENT_NOT_FOUND")
        body = await request.json()
        new_text = (body.get("text") or "").strip()
        if new_text:
            # If an AI segment's text changed, invalidate any prefetched audio
            # so the driver regenerates it with the new text.
            old_text = segs[index].get("text", "")
            if old_text != new_text and segs[index].get("source") == "tts":
                old_filename = segs[index].get("audio_filename")
                if old_filename:
                    old_path = Path(session["dir"]) / old_filename
                    old_path.unlink(missing_ok=True)
                segs[index]["audio_filename"] = None
                segs[index].pop("loudness_normalized", None)
                segs[index]["status"] = "pending"
            segs[index]["text"] = new_text
            # Persist so the edit survives a refresh
            meta_path = Path(session["dir"]) / "meta.json"
            if meta_path.exists():
                try:
                    import json as _json
                    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["segments"] = segs
                    meta_path.write_text(_json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
        return {"session_id": session_id, "index": index, "text": segs[index]["text"]}

    # ── Legacy Live Podcast routes (kept for compatibility) ────────────

    def _check_no_live_segments(project) -> bool:
        """Return True if project has at least one live segment."""
        from .models import SegmentSource
        return any(seg.source == SegmentSource.LIVE for seg in project.segments)

    @app.post("/api/podcasts/{project_id}/live/start", status_code=201,
              response_model=LiveStartResponse)
    def start_live_session(project_id: str, request: LiveStartRequest) -> LiveStartResponse:
        project = podcast_manager.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="PROJECT_NOT_FOUND")
        if not _check_no_live_segments(project):
            raise HTTPException(status_code=409, detail="NO_LIVE_SEGMENTS")

        segments = project.segments
        session = live_registry.start_live_session(project_id, segments)

        # Determine initial state based on first segment
        from .models import SegmentSource
        first_seg = segments[0] if segments else None
        if first_seg and first_seg.source == SegmentSource.LIVE:
            session.transition(Trigger.START_RECORDING)
        else:
            session.transition(Trigger.START_AI)

        live_count = sum(1 for s in segments if s.source == SegmentSource.LIVE)
        _persist_session(session)
        return LiveStartResponse(
            session_id=session.session_id,
            project_id=project_id,
            state=session.state.value,
            segment_count=len(segments),
            live_segment_count=live_count,
            asr_enabled=getattr(config.asr, "enabled", False),
            asr_ready=live_registry.get_asr().is_ready,
        )

    @app.post("/api/podcasts/{project_id}/live/{session_id}/stop",
              response_model=LiveStopResponse)
    def stop_live_session(project_id: str, session_id: str) -> LiveStopResponse:
        session = live_registry.get(project_id, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        if not session.can_accept_command():
            raise HTTPException(status_code=409, detail="SESSION_TERMINATED")
        live_registry.stop_live_session(project_id, session_id)
        recorded = [i for i, seg in enumerate(session.segments)
                    if seg.source.value == "live" and seg.audio_filename]
        return LiveStopResponse(
            session_id=session_id,
            state=session.state.value,
            recorded_segments=recorded,
            merged=False,
        )

    @app.post("/api/asr/warmup")
    def warmup_asr() -> dict:
        asr = live_registry.get_asr()
        asr.start_warmup()
        return _asr_status()

    @app.post("/api/podcasts/{project_id}/live/{session_id}/redo/{index}")
    def redo_live_segment(project_id: str, session_id: str, index: int) -> dict:
        """ACC-013-1: Redo a live segment by moving cursor back and deleting old WAV."""
        session = live_registry.get(project_id, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        if index < 0 or index >= len(session.segments):
            raise HTTPException(status_code=400, detail="SEGMENT_INDEX_OUT_OF_RANGE")
        seg = session.segments[index]
        if seg.source.value != "live":
            raise HTTPException(status_code=400, detail="ONLY_LIVE_SEGMENTS_CAN_REDO")
        # Delete old WAV if exists
        audio_dir = outputs_dir / "podcasts" / project_id
        old_wav = audio_dir / f"live_{index:04d}.wav"
        deleted = False
        if old_wav.exists():
            old_wav.unlink()
            deleted = True
        # Move cursor back
        session.cursor = index
        return {"session_id": session_id, "index": index, "previous_wav_deleted": deleted}

    @app.post("/api/podcasts/{project_id}/live/{session_id}/resume")
    def resume_live_session(project_id: str, session_id: str) -> dict:
        """ACC-016-1: Resume a persisted session from disk."""
        # Check if session is in memory
        session = live_registry.get(project_id, session_id)
        if session is not None:
            if session.can_accept_command():
                return {"session_id": session_id, "state": session.state.value,
                        "cursor": session.cursor, "resumed_from_disk": False}
            raise HTTPException(status_code=409, detail="SESSION_TERMINATED")
        # Try loading from disk
        session_json = outputs_dir / "podcasts" / project_id / "live_sessions" / f"{session_id}.json"
        if not session_json.exists():
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        data = json.loads(session_json.read_text(encoding="utf-8"))
        # Reconstruct session in memory
        project = podcast_manager.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="PROJECT_NOT_FOUND")
        from .live.session import LiveSession, LiveState
        restored = LiveSession(
            session_id=session_id,
            project_id=project_id,
            cursor=data.get("cursor", 0),
            state=LiveState(data.get("state", "IDLE")),
            segments=project.segments,
            started_at=data.get("started_at", 0.0),
        )
        live_registry._sessions[(project_id, session_id)] = restored
        return {"session_id": session_id, "state": restored.state.value,
                "cursor": restored.cursor, "resumed_from_disk": True}

    def _persist_session(session):
        """Write session state to live_sessions/{sid}.json (ACC-016-1)."""
        session_dir = outputs_dir / "podcasts" / session.project_id / "live_sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_json = session_dir / f"{session.session_id}.json"
        import time as _time
        data = {
            "cursor": session.cursor,
            "state": session.state.value,
            "captured_segments": list(session.audio_buffer.keys()),
            "started_at": session.started_at,
            "last_save_at": _time.time(),
        }
        session_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @app.websocket("/ws/podcasts/{project_id}/live/{session_id}")
    async def ws_live(ws: WebSocket, project_id: str, session_id: str, role: str = "observe"):
        """ACC-006-1: WebSocket handler for live podcast sessions."""
        session = live_registry.get(project_id, session_id)
        if session is None:
            await ws.close(code=1008, reason="SESSION_NOT_FOUND")
            return

        await ws.accept()

        # Assign driver role
        if role == "driver":
            if hasattr(session, "_driver_ws") and session._driver_ws is not None:
                await ws.close(code=1008, reason="driver already connected")
                return
            session._driver_ws = ws
        if not hasattr(session, "_ws_clients"):
            session._ws_clients = set()
        session._ws_clients.add(ws)

        # Send initial state frame
        from .live.ws_protocol import live_state, encode_json_frame
        await ws.send_text(encode_json_frame(live_state(session.state.value)))

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"] is not None:
                    # Binary PCM frame from client (driver only)
                    if role != "driver":
                        await ws.close(code=1003, reason="observe role cannot send audio")
                        break
                    pcm = msg["bytes"]
                    if not hasattr(session, "audio_buffer"):
                        session.audio_buffer = {}
                    buf = session.audio_buffer.setdefault(session.cursor, bytearray())
                    buf.extend(pcm)
                elif "text" in msg and msg["text"] is not None:
                    # JSON control frame
                    try:
                        data = json.loads(msg["text"])
                        if data.get("type") == "client_log":
                            pass  # Log if needed
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            pass
        finally:
            session._ws_clients.discard(ws)
            if getattr(session, "_driver_ws", None) is ws:
                session._driver_ws = None

    from .yovoice_api import YovoiceBackend, register_yovoice_api

    ui_state_dir = Path(config_path).resolve().parent if config_path else outputs_dir
    yovoice_backend = YovoiceBackend(
        config=config,
        sample_manager=sample_manager,
        podcast_manager=podcast_manager,
        outputs_dir=outputs_dir,
        engine_provider=lambda: _resolve_engine(None, config, sample_manager),
        preferences_path=ui_state_dir / "preferences.json",
        live_speakers=live_speakers_list,
    )
    register_yovoice_api(app, yovoice_backend)

    # 静态挂载放在最后：只在没有命中任何 API 路由时才接管（/api、/ws 已在前注册）。
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app
