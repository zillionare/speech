"""FastAPI Web 服务"""

import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import io

from .config import Config, load_config
from .tts_engine import TTSEngine
from .models import (
    SpeechRequest, PodcastRequest, VoiceListResponse,
    VoiceDetailResponse, ErrorResponse, HealthResponse
)


# 全局变量
engine: Optional[TTSEngine] = None
config: Optional[Config] = None
start_time: float = 0.0


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    创建 FastAPI 应用

    Args:
        config_path: 配置文件路径

    Returns:
        FastAPI 应用实例
    """
    global config, engine, start_time

    # 加载配置
    config = load_config(config_path)

    # 创建引擎 (延迟加载模型)
    engine = TTSEngine(config)

    start_time = time.time()

    # 生命周期管理
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        print("TTS Server starting...")
        print(f"Model: {config.model.name}")
        print(f"Voices directory: {config.samples.expanded_base_dir}")
        print(f"Available voices: {len(engine.list_voices())}")
        yield
        print("TTS Server shutting down...")

    app = FastAPI(
        title="TTS Web Service",
        description="基于 mlx-audio 的文本转语音服务",
        version="0.1.0",
        lifespan=lifespan
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === API 路由 ===

    @app.get("/")
    async def root():
        """根路径重定向到文档"""
        return {"message": "TTS Web Service", "docs": "/docs"}

    @app.get("/health")
    async def health():
        """健康检查"""
        uptime = time.time() - start_time if start_time else 0
        return HealthResponse(
            status="healthy",
            model=config.model.name,
            voices_count=len(engine.list_voices()),
            uptime=uptime
        )

    @app.get("/v1/voices")
    async def list_voices():
        """
        获取可用声音列表

        Returns:
            VoiceListResponse: 声音名称列表
        """
        voices = engine.list_voices()
        return VoiceListResponse(voices=voices)

    @app.get("/v1/voices/details")
    async def list_voices_details():
        """
        获取声音详细信息列表

        Returns:
            VoiceDetailResponse: 声音详细信息列表
        """
        from .models import VoiceInfo
        voices = []
        for name in engine.list_voices():
            info = engine.get_voice_info(name)
            voices.append(VoiceInfo(**info))
        return VoiceDetailResponse(voices=voices)

    @app.post("/v1/audio/speech")
    async def create_speech(request: SpeechRequest):
        """
        单 speaker TTS 生成

        Args:
            request: SpeechRequest

        Returns:
            音频文件流
        """
        global engine

        if len(request.input) > 65536:
            raise HTTPException(
                status_code=400,
                detail="Input text exceeds 64KB limit"
            )

        # 使用默认 voice
        voice = request.voice or config.defaults.voice
        # 检查 voice 是否有效（本地文件或内置 voice）
        is_local = engine.sample_manager.voice_exists(voice)
        is_builtin = hasattr(engine, 'BUILTIN_VOICES') and voice in engine.BUILTIN_VOICES
        if not is_local and not is_builtin:
            available = engine.list_voices()
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice}' not found. Available: {available}"
            )

        try:
            print(f"Generating speech: voice={voice}, length={len(request.input)} chars")
            start = time.time()

            audio_data = engine.generate_single(
                text=request.input,
                voice=voice,
                output_format=request.response_format,
                speed=request.speed
            )

            duration = time.time() - start
            print(f"Generated: {len(audio_data)} bytes in {duration:.2f}s")

            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type=f"audio/{request.response_format}",
                headers={
                    "X-Generation-Time": f"{duration:.2f}",
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
                }
            )

        except Exception as e:
            print(f"TTS generation failed: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"TTS generation failed: {str(e)}"
            )

    @app.post("/v1/audio/podcast")
    async def create_podcast(request: PodcastRequest):
        """
        多 speaker 播客生成

        Args:
            request: PodcastRequest

        Returns:
            合并后的音频文件流
        """
        global engine

        if len(request.input) > 65536:
            raise HTTPException(
                status_code=400,
                detail="Input text exceeds 64KB limit"
            )

        # 验证所有 voice 是否有效（本地文件或内置 voice）
        for speaker, voice in request.voice_mapping.items():
            is_local = engine.sample_manager.voice_exists(voice)
            is_builtin = hasattr(engine, 'BUILTIN_VOICES') and voice in engine.BUILTIN_VOICES
            if not is_local and not is_builtin:
                available = engine.list_voices()
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice '{voice}' for speaker '{speaker}' not found. Available: {available}"
                )

        try:
            print(f"Generating podcast: mapping={request.voice_mapping}, length={len(request.input)} chars")
            start = time.time()

            audio_data = engine.generate_podcast(
                text=request.input,
                voice_mapping=request.voice_mapping,
                output_format=request.response_format,
                speed=request.speed
            )

            duration = time.time() - start
            print(f"Generated podcast: {len(audio_data)} bytes in {duration:.2f}s")

            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type=f"audio/{request.response_format}",
                headers={
                    "X-Generation-Time": f"{duration:.2f}",
                    "Content-Disposition": f"attachment; filename=podcast.{request.response_format}"
                }
            )

        except Exception as e:
            print(f"Podcast generation failed: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Podcast generation failed: {str(e)}"
            )

    # 错误处理
    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message=str(exc),
                details=traceback.format_exc() if config.server.log_level == "debug" else None
            ).dict()
        )

    return app


import io
