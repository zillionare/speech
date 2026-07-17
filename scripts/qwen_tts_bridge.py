"""Small single-request Qwen3-TTS bridge for the remote vMLX host.

The bundled mlx-audio server currently runs Qwen3-TTS continuous batching in
a worker thread, while MLX GPU arrays are thread-bound. This bridge keeps
generation on the event-loop thread and exposes the same endpoint consumed by
QwenRemoteEngine.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from asyncio import Lock
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from mlx_audio.audio_io import write as write_audio
from mlx_audio.utils import load_model


MODEL_PATH = os.environ.get(
    "QWEN_TTS_MODEL",
    "/Volumes/share/data/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
)

app = FastAPI(title="Qwen3-TTS Bridge")
generation_lock = Lock()
model = load_model(MODEL_PATH)


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    response_format: str = "wav"
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.0
    max_tokens: int = 1200


def _reference_path(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if value.startswith("/") and Path(value).exists():
        return value, None
    try:
        data = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ref_audio: {exc}") from exc
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.write(data)
    handle.close()
    return handle.name, handle.name


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "model": MODEL_PATH}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_PATH, "object": "model"}]}


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest) -> Response:
    ref_path, temporary_path = _reference_path(request.ref_audio)
    try:
        async with generation_lock:
            results = list(model.generate(
                text=request.input,
                # Base voice cloning uses ref_audio/ref_text rather than a
                # named speaker. CustomVoice remains available if requested.
                voice=None if ref_path and request.ref_text else request.voice,
                ref_audio=ref_path,
                ref_text=request.ref_text,
                lang_code="auto",
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=max(request.repetition_penalty, 1.5) if ref_path else request.repetition_penalty,
                max_tokens=request.max_tokens,
                stream=False,
                verbose=False,
            ))
            chunks = [
                np.asarray(result.audio.tolist(), dtype=np.float32)
                if hasattr(result.audio, "tolist")
                else np.asarray(result.audio, dtype=np.float32)
                for result in results
            ]
            if not chunks:
                raise HTTPException(status_code=500, detail="Qwen3-TTS returned no audio")
            audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            sample_rate = results[0].sample_rate
            output = io.BytesIO()
            write_audio(output, audio, sample_rate, format=request.response_format)
            return Response(
                content=output.getvalue(),
                media_type=f"audio/{request.response_format}",
            )
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
