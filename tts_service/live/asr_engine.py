"""Embedded ASR engine.

In-process ASR inference using mlx_whisper or faster_whisper.
Full spec: .project/specs/spec.md SPEC-003.
ACC-003-1: ASR configuration fields
ACC-003-2: warmup / is_ready / is_warming
ACC-003-3: transcribe_chunk returns ASRResult
ACC-003-4: failure degradation
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_FAILURES_BEFORE_DEGRADED = 5


def _to_simplified(text: str) -> str:
    """Normalize ASR display text to simplified Chinese when available."""
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except ImportError:
        return text


@dataclass
class ASRConfig:
    enabled: bool = False
    backend: str = "qwen3_asr"
    model: str = "mlx-community/whisper-medium-mlx-4bit"
    language: str = "zh"
    chunk_seconds: float = 4.0
    beam_size: int = 1
    vad_filter: bool = True
    compute_type: str = "float16"
    device: str = "auto"
    warmup_on_start: bool = False
    download_endpoint: str = "https://hf-mirror.com"
    cache_dir: str = ""
    qwen3_model_dir: str = "~/.cache/sherpa-onnx/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    qwen3_num_threads: int = 4
    qwen3_max_new_tokens: int = 512
    qwen3_hotwords: str = ""


@dataclass
class ASRResult:
    text: str = ""
    is_final: bool = True
    audio_ms: int = 0
    confidence: float = 0.0
    raw_segments: list = field(default_factory=list)


class EmbeddedASR:
    """In-process ASR inference engine.

    The ``_model`` attribute is set to a callable ``transcribe_fn(audio_np, cfg) -> dict``
    after warmup. Tests can mock ``_model`` with any callable that returns a dict
    containing ``text`` and ``segments`` keys.
    """

    def __init__(self, cfg: ASRConfig, progress_callback: Optional[Callable[[dict], None]] = None):
        self.cfg = cfg
        self._model: Optional[Callable] = None
        self._lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._progress_callback = progress_callback
        self._progress = {
            "status": "idle",
            "progress": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "message": "等待 ASR 模型",
            "error": None,
        }
        self._failures = 0
        self._degraded = False
        self._warming = False
        self._warmup_thread: Optional[threading.Thread] = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def is_warming(self) -> bool:
        return self._warming

    @property
    def progress(self) -> dict:
        with self._progress_lock:
            return dict(self._progress)

    def _set_progress(self, **updates) -> None:
        with self._progress_lock:
            self._progress.update(updates)
            snapshot = dict(self._progress)
        if self._progress_callback is not None:
            try:
                self._progress_callback(snapshot)
            except Exception:
                logger.debug("ASR progress callback failed", exc_info=True)

    @property
    def degraded(self) -> bool:
        return self._degraded

    def warmup(self) -> None:
        """Load the ASR model with a 1-second silent sample."""
        with self._lock:
            if self._model is not None or self._warming:
                return
            self._warming = True
        self._run_warmup()

    def _run_warmup(self) -> None:
        """Perform the blocking model download/load after warming is claimed."""
        self._set_progress(
            status="loading",
            progress=0.0,
            message="正在准备 ASR 模型",
            error=None,
        )
        try:
            self._load_model()
            self._set_progress(
                status="ready",
                progress=1.0,
                message="ASR 模型已就绪",
                error=None,
            )
        except Exception as exc:
            logger.warning("ASR warmup failed: %s", exc)
            self._set_progress(
                status="error",
                message=f"ASR 模型加载失败：{exc}",
                error=str(exc),
            )
        finally:
            self._warming = False

    def start_warmup(self) -> bool:
        """Start model loading in a daemon thread and return immediately."""
        with self._lock:
            if self._model is not None or self._warming:
                return False
            self._warming = True
        self._warmup_thread = threading.Thread(
            target=self._run_warmup,
            name="asr-model-warmup",
            daemon=True,
        )
        self._warmup_thread.start()
        return True

    def _download_model(self) -> str:
        """Download a Hugging Face model through the configured mirror."""
        model_path = Path(os.path.expanduser(self.cfg.model))
        if model_path.exists():
            return str(model_path.resolve())

        cache_root = Path(os.path.expanduser(self.cfg.cache_dir or "~/.cache/speech/asr"))
        repo_id = self.cfg.model
        safe_repo = repo_id.replace("/", "--")
        cached_root = cache_root / safe_repo
        cached_models = [
            path for path in cached_root.glob("*")
            if path.is_dir()
            and (path / "config.json").exists()
            and ((path / "weights.safetensors").exists() or (path / "weights.npz").exists())
        ]
        if cached_models:
            cached_model = max(cached_models, key=lambda path: path.stat().st_mtime)
            self._set_progress(
                status="loading",
                progress=1.0,
                downloaded_bytes=sum(item.stat().st_size for item in cached_model.iterdir() if item.is_file()),
                total_bytes=0,
                message="使用本地缓存加载 ASR 模型",
                error=None,
            )
            return str(cached_model)

        from huggingface_hub import HfApi

        endpoint = self.cfg.download_endpoint or "https://hf-mirror.com"
        api = HfApi(endpoint=endpoint)
        repo_info = api.model_info(repo_id)
        revision = repo_info.sha
        if not revision:
            raise RuntimeError(f"无法获取 ASR 模型 revision: {repo_id}")

        patterns = {"config.json"}
        if self.cfg.backend == "mlx_whisper":
            patterns.update({"weights.safetensors", "weights.npz"})
        else:
            patterns.update({item.rfilename for item in repo_info.siblings})

        file_infos = api.get_paths_info(
            repo_id, paths=sorted(patterns), revision=revision,
        )
        if not file_infos:
            raise RuntimeError(f"ASR 模型没有可下载文件: {repo_id}")

        model_dir = cache_root / safe_repo / revision
        model_dir.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(
            int(info.size or (info.lfs.size if info.lfs else 0) or 0)
            for info in file_infos
        )
        downloaded_bytes = sum(
            path.stat().st_size
            for info in file_infos
            if (path := model_dir / info.path).exists()
        )

        for info in file_infos:
            target = model_dir / info.path
            expected = int(info.size or (info.lfs.size if info.lfs else 0) or 0)
            if target.exists() and (not expected or target.stat().st_size == expected):
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".incomplete")
            if partial.exists():
                partial.unlink()
            url = "/".join([
                endpoint.rstrip("/"),
                repo_id,
                "resolve",
                revision,
                urllib.parse.quote(info.path, safe="/"),
            ])
            self._set_progress(
                status="downloading",
                progress=(downloaded_bytes / total_bytes if total_bytes else 0.0),
                downloaded_bytes=downloaded_bytes,
                total_bytes=total_bytes,
                message=f"正在从 hf-mirror 下载 {info.path}",
                error=None,
            )
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SpeechStudio-ASR/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded_bytes += len(chunk)
                    self._set_progress(
                        status="downloading",
                        progress=(downloaded_bytes / total_bytes if total_bytes else 0.0),
                        downloaded_bytes=downloaded_bytes,
                        total_bytes=total_bytes,
                        message=f"正在下载 ASR 模型 {downloaded_bytes}/{total_bytes} bytes",
                        error=None,
                    )
            if expected and partial.stat().st_size != expected:
                raise RuntimeError(
                    f"ASR 模型文件下载不完整: {info.path} "
                    f"({partial.stat().st_size}/{expected} bytes)"
                )
            os.replace(partial, target)

        return str(model_dir)

    def _load_model(self) -> None:
        """Load the model based on the configured backend, store transcribe_fn."""
        backend = self.cfg.backend
        if backend == "qwen3_asr":
            try:
                import sherpa_onnx

                self._set_progress(status="loading", message="正在加载 Qwen3-ASR 模型")
                model_dir = Path(os.path.expanduser(self.cfg.qwen3_model_dir))
                if not model_dir.exists():
                    raise RuntimeError(
                        f"Qwen3-ASR 模型目录不存在: {model_dir}。"
                        f"请从 https://github.com/k2-fsa/sherpa-onnx/releases 下载 "
                        f"sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 并解压到该目录。"
                    )
                recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                    conv_frontend=str(model_dir / "conv_frontend.onnx"),
                    encoder=str(model_dir / "encoder.int8.onnx"),
                    decoder=str(model_dir / "decoder.int8.onnx"),
                    tokenizer=str(model_dir / "tokenizer"),
                    num_threads=self.cfg.qwen3_num_threads,
                    max_new_tokens=self.cfg.qwen3_max_new_tokens,
                    hotwords=self.cfg.qwen3_hotwords,
                )

                def _qwen3_transcribe(audio_np: np.ndarray, cfg: ASRConfig) -> dict:
                    stream = recognizer.create_stream()
                    stream.accept_waveform(16000, audio_np)
                    recognizer.decode_stream(stream)
                    text = stream.result.text.strip()
                    return {"text": text, "segments": []}

                self._model = _qwen3_transcribe
            except ImportError:
                raise RuntimeError("sherpa-onnx 未安装，无法使用 Qwen3-ASR。请 pip install sherpa-onnx")
        elif backend == "mlx_whisper":
            try:
                import mlx_whisper

                self._set_progress(status="loading", message="正在加载 ASR 模型到内存")
                model_path = self._download_model()

                def _mlx_transcribe(audio_np: np.ndarray, cfg: ASRConfig) -> dict:
                    return mlx_whisper.transcribe(
                        audio_np,
                        path_or_hf_repo=model_path,
                        language=cfg.language,
                        word_timestamps=False,
                    )

                self._model = _mlx_transcribe
            except ImportError:
                raise RuntimeError("mlx_whisper 未安装，无法使用本地 ASR")
        elif backend == "faster_whisper":
            try:
                from faster_whisper import WhisperModel

                model_path = self._download_model()
                fw_model = WhisperModel(
                    model_path,
                    device=self.cfg.device,
                    compute_type=self.cfg.compute_type,
                )

                def _fw_transcribe(audio_np: np.ndarray, cfg: ASRConfig) -> dict:
                    segments, info = fw_model.transcribe(
                        audio_np,
                        language=cfg.language,
                        beam_size=cfg.beam_size,
                        vad_filter=cfg.vad_filter,
                        word_timestamps=True,
                    )
                    seg_list = list(segments)
                    return {
                        "text": " ".join(s.text for s in seg_list),
                        "segments": [
                            {"start": s.start, "end": s.end, "text": s.text}
                            for s in seg_list
                        ],
                    }

                self._model = _fw_transcribe
            except ImportError:
                raise RuntimeError("faster-whisper 未安装，无法使用 ASR")
        else:
            raise RuntimeError(f"未知 ASR backend: {backend}")

    def _do_transcribe(self, audio_np: np.ndarray) -> dict:
        """Call the backend's transcribe function and return raw result dict."""
        if self._model is None:
            return {"text": "", "segments": []}
        return self._model(audio_np, self.cfg)

    async def transcribe_chunk(self, pcm_bytes: bytes) -> ASRResult:
        """Transcribe a chunk of 16kHz mono PCM audio (16-bit signed)."""
        if not pcm_bytes:
            return ASRResult(text="", audio_ms=0, confidence=0.0)

        audio_ms = len(pcm_bytes) // 32  # 16kHz * 2 bytes = 32 bytes/ms

        if self._model is None:
            return ASRResult(text="", audio_ms=audio_ms, confidence=0.0)

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._do_transcribe, audio_np)
            text = _to_simplified((result.get("text") or "").strip())
            segments = [
                {**segment, "text": _to_simplified(segment.get("text", ""))}
                for segment in result.get("segments", [])
            ]
            confidence = 1.0 if text else 0.0

            self._failures = 0
            self._degraded = False
            return ASRResult(
                text=text,
                audio_ms=audio_ms,
                confidence=confidence,
                raw_segments=segments,
            )
        except Exception as exc:
            logger.warning("ASR transcribe failed: %s", exc)
            self._failures += 1
            if self._failures >= _MAX_FAILURES_BEFORE_DEGRADED:
                self._degraded = True
            return ASRResult(text="", audio_ms=audio_ms, confidence=0.0)
