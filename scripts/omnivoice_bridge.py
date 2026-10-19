"""OmniVoice TTS bridge (MLX) running on the remote Apple Silicon host.

Exposes an OpenAI-compatible /v1/audio/speech endpoint plus the extra
ref_audio / ref_text fields used for zero-shot voice cloning.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from mlx_audio.audio_io import write as write_audio
from mlx_audio.tts.utils import load_model

MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
SAMPLE_RATE = 24000
NUM_STEPS = int(os.environ.get("OMNIVOICE_NUM_STEPS", "16"))

app = FastAPI(title="OmniVoice Bridge")
generation_lock = asyncio.Lock()
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL)
    return _model


def _as_np(audio) -> np.ndarray:
    if hasattr(audio, "tolist"):
        return np.asarray(audio.tolist(), dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


class SpeechRequest(BaseModel):
    model: str = MODEL
    input: str
    voice: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    response_format: str = "wav"
    instructions: str | None = None
    speed: float = 1.0
    language: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model": MODEL}


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest) -> Response:
    temporary_path: str | None = None
    if request.ref_audio:
        try:
            data = base64.b64decode(request.ref_audio, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid ref_audio: {exc}") from exc
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.write(data)
        handle.close()
        temporary_path = handle.name

    try:
        kwargs = {}
        if temporary_path:
            kwargs["ref_audio"] = temporary_path
        if request.ref_text:
            kwargs["ref_text"] = request.ref_text
        if request.instructions:
            kwargs["instruct"] = request.instructions
        language = (request.language or "chinese").strip().lower()
        if language and language not in ("auto", "none"):
            kwargs["language"] = language

        model = get_model()
        # MLX GPU arrays are thread-bound; keep generation on the event loop.
        async with generation_lock:
            results = list(model.generate(
                text=request.input,
                num_steps=NUM_STEPS,
                **kwargs,
            ))
        if not results:
            raise HTTPException(status_code=500, detail="OmniVoice returned no audio")
        audio = _as_np(results[0].audio)
        if audio.ndim == 2:
            audio = audio[..., 0]
        audio = audio.astype(np.float32)

        output_format = request.response_format if request.response_format != "mp3" else "wav"
        output = io.BytesIO()
        write_audio(output, audio, SAMPLE_RATE, format=output_format)
        return Response(
            content=output.getvalue(),
            media_type=f"audio/{output_format}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OmniVoice generation failed: {exc}") from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)