"""FastAPI app for the new worktree implementation."""

from __future__ import annotations

import io
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .models import (
    AppConfigResponse,
    GenerateRequest,
    GenerationRecord,
    HealthResponse,
    PodcastRequest,
    SpeechRequest,
    TranscriptUpdateRequest,
    VoiceInfo,
    VoiceListResponse,
)
from .sample_manager import SampleManager, VoiceSample
from .tts_engine import TTSEngine


def _voice_info(sample: VoiceSample, default_voice: str) -> VoiceInfo:
    return VoiceInfo(
        speaker=sample.speaker,
        transcript=sample.transcript,
        transcript_preview=sample.transcript_preview,
        cache_ready=sample.cache_path.exists(),
        is_default=sample.speaker == default_voice,
        audio_url=f"/api/voices/{sample.speaker}/audio",
    )


def create_app(config_path: Optional[str] = None) -> FastAPI:
    config = load_config(config_path)
    sample_manager = SampleManager(config.voices.expanded_base_dir, config.voices.default_voice)
    engine = TTSEngine(config, sample_manager)
    static_dir = Path(__file__).resolve().parent / "static"
    outputs_dir = config.outputs.expanded_base_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    history: deque[GenerationRecord] = deque(maxlen=config.outputs.history_limit)

    app = FastAPI(
        title="VibeVoice MLX Studio",
        description="vibevoice-mlx based dialogue generation and voice management UI",
        version="0.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

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
        return AppConfigResponse(
            model=config.model.model_id,
            quantize_bits=config.model.quantize_bits,
            default_voice=config.voices.default_voice,
            diffusion_steps=config.model.diffusion_steps,
            use_coreml_semantic=config.model.use_coreml_semantic,
        )

    @app.get("/api/voices", response_model=VoiceListResponse)
    def list_api_voices() -> VoiceListResponse:
        sample_manager.refresh()
        return VoiceListResponse(
            voices=[_voice_info(sample, config.voices.default_voice) for sample in sample_manager.list_samples()]
        )

    @app.post("/api/voices", response_model=VoiceInfo)
    async def upload_voice(
        speaker: str = Form(...),
        transcript: str = Form(""),
        overwrite: bool = Form(False),
        audio_file: UploadFile = File(...),
    ) -> VoiceInfo:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")
        audio_bytes = await audio_file.read()
        sample = sample_manager.add_voice(speaker=speaker, audio_bytes=audio_bytes, transcript=transcript, overwrite=overwrite)
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
        sample_manager.delete_voice(speaker)
        return {"status": "deleted", "speaker": speaker}

    @app.get("/api/voices/{speaker}/audio")
    def get_voice_audio(speaker: str) -> FileResponse:
        sample = sample_manager.get(speaker)
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
        filename = f"{request_id}.{output_format}"
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
        )
        history.appendleft(record)
        return record

    @app.post("/api/generate", response_model=GenerationRecord)
    def generate_audio(request: GenerateRequest) -> GenerationRecord:
        try:
            result = engine.generate_dialogue(
                text=request.text,
                output_format=request.output_format,
                preferred_voice=request.voice,
                voice_mapping=request.voice_mapping,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _store_generation(request.text, request.output_format, result)

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

    return app
