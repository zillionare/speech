"""FastAPI app for the new worktree implementation."""

from __future__ import annotations

import io
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config, save_config_to_yaml
from .models import (
    AppConfigResponse,
    AppConfigUpdateRequest,
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
)
from .podcast_manager import PodcastManager
from .sample_manager import SampleManager, VoiceSample
from .tts_engine import create_engine


def _voice_info(sample: VoiceSample, default_voice: str) -> VoiceInfo:
    return VoiceInfo(
        speaker=sample.speaker,
        transcript=sample.transcript,
        transcript_preview=sample.transcript_preview,
        cache_ready=sample.cache_path.exists(),
        is_default=sample.speaker == default_voice,
        audio_url=f"/api/voices/{sample.speaker}/audio",
    )


def _resolve_engine(request_engine: str | None, config, sample_manager):
    """Return the appropriate engine based on request override."""
    if request_engine is None:
        return create_engine(config, sample_manager)
    is_remote = getattr(config.model, "use_remote_qwen", False)
    if request_engine == "remote" and not is_remote:
        from .engines.qwen_remote import QwenRemoteEngine
        return QwenRemoteEngine(config, sample_manager)
    elif request_engine == "local" and is_remote:
        from .engines.local_vibevoice import LocalVibeVoiceEngine
        return LocalVibeVoiceEngine(config, sample_manager)
    return create_engine(config, sample_manager)


def create_app(config_path: Optional[str] = None) -> FastAPI:
    config = load_config(config_path)
    sample_manager = SampleManager(config.voices.expanded_base_dir, config.voices.default_voice)
    engine = create_engine(config, sample_manager)
    static_dir = Path(__file__).resolve().parent / "static"
    outputs_dir = config.outputs.expanded_base_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    history: deque[GenerationRecord] = deque(maxlen=config.outputs.history_limit)
    podcast_manager = PodcastManager(outputs_dir / "podcasts", outputs_dir)

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

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            static_dir / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
        )

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
        if config_path:
            save_config_to_yaml(config, config_path)
        # Sync sample_manager with any changed voice settings
        sample_manager.update_settings(
            base_dir=config.voices.expanded_base_dir,
            default_voice=config.voices.default_voice,
        )
        return get_config()

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
        speaker: str = Form(...),
        transcript: str = Form(""),
        overwrite: bool = Form(False),
        audio_file: UploadFile = File(...),
    ) -> VoiceInfo:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")
        audio_bytes = await audio_file.read()
        try:
            sample = sample_manager.add_voice(speaker=speaker, audio_bytes=audio_bytes, transcript=transcript, overwrite=overwrite)
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
                voice_ref=request.voice_ref,
            )
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

    @app.post("/api/podcasts/{project_id}/merge")
    def merge_podcast(project_id: str) -> dict[str, str]:
        try:
            filename = podcast_manager.merge_project(project_id)
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

    return app
