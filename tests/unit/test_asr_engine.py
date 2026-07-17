"""Unit tests for ACC-003-1~4: EmbeddedASR (SPEC-003 / ST-LP-003).

Tests cover:
- ACC-003-1: ASR configuration fields exist on Config
- ACC-003-2: warmup() sets is_ready, is_warming toggles
- ACC-003-3: transcribe_chunk returns ASRResult with text/confidence/audio_ms
- ACC-003-4: failure returns empty text + confidence=0; repeated failures set degraded

All tests use mocked backends (no mlx_whisper / faster_whisper installed).
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.live.asr_engine import EmbeddedASR, ASRConfig, ASRResult


class ASRConfigTests(unittest.TestCase):
    """ACC-003-1: ASR configuration fields exist on Config."""

    def test_config_has_asr_section(self):
        """ACC-003-1: Config has an asr section with expected fields."""
        from tts_service.config import Config
        cfg = Config()
        self.assertFalse(cfg.asr.enabled)
        self.assertEqual(cfg.asr.backend, "qwen3_asr")
        self.assertEqual(cfg.asr.language, "zh")
        self.assertEqual(cfg.asr.model, "mlx-community/whisper-medium-mlx-4bit")

    def test_asr_config_defaults(self):
        cfg = ASRConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.backend, "qwen3_asr")
        self.assertEqual(cfg.model, "mlx-community/whisper-medium-mlx-4bit")
        self.assertEqual(cfg.language, "zh")
        self.assertEqual(cfg.chunk_seconds, 4.0)
        self.assertEqual(cfg.beam_size, 1)
        self.assertTrue(cfg.vad_filter)


class EmbeddedASRWarmupTests(unittest.TestCase):
    """ACC-003-2: warmup() sets is_ready, is_warming toggles."""

    def test_initial_state_not_ready(self):
        asr = EmbeddedASR(ASRConfig())
        self.assertFalse(asr.is_ready)
        self.assertFalse(asr.is_warming)

    def test_warmup_sets_ready(self):
        """warmup() with a mock backend should set is_ready=True."""
        asr = EmbeddedASR(ASRConfig(backend="mlx_whisper"))
        with patch.object(asr, "_load_model") as mock_load:
            def _fake_load():
                asr._model = MagicMock(return_value={"text": "", "segments": []})
            mock_load.side_effect = _fake_load
            asr.warmup()
        self.assertTrue(asr.is_ready)
        self.assertFalse(asr.is_warming)

    def test_warmup_handles_import_error(self):
        """If backend library is not installed, warmup should not raise."""
        asr = EmbeddedASR(ASRConfig(backend="nonexistent_backend"))
        asr.warmup()  # should not raise
        self.assertFalse(asr.is_ready)

    def test_download_uses_hf_mirror_and_reports_progress(self):
        """Model downloads use the configured mirror and expose byte progress."""
        from types import SimpleNamespace
        from unittest.mock import patch
        import tempfile

        cache_dir = Path(tempfile.mkdtemp())
        asr = EmbeddedASR(ASRConfig(
            model="mlx-community/whisper-medium-mlx-4bit",
            download_endpoint="https://hf-mirror.com",
            cache_dir=str(cache_dir),
        ))
        api_instance = SimpleNamespace(
            model_info=lambda repo: SimpleNamespace(
                sha="revision-1",
                siblings=[SimpleNamespace(rfilename="config.json")],
            ),
            get_paths_info=lambda repo, paths, revision: [
                SimpleNamespace(path="config.json", size=4, lfs=None),
            ],
        )

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size):
                if not hasattr(self, "done"):
                    self.done = True
                    return b"{}  "
                return b""

        with patch("huggingface_hub.HfApi", return_value=api_instance) as api, \
             patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            result = asr._download_model()
            self.assertTrue(result.endswith("mlx-community--whisper-medium-mlx-4bit/revision-1"))
            self.assertEqual(api.call_args.kwargs["endpoint"], "https://hf-mirror.com")
            self.assertIn("hf-mirror.com", urlopen.call_args.args[0].full_url)
            self.assertEqual(asr.progress["status"], "downloading")
            self.assertEqual(asr.progress["progress"], 1.0)

    def test_download_uses_complete_local_cache_without_metadata_request(self):
        """A complete cache must work when the configured mirror is unavailable."""
        import tempfile
        from unittest.mock import patch

        cache_dir = Path(tempfile.mkdtemp())
        model_dir = cache_dir / "mlx-community--whisper-medium-mlx-4bit" / "revision-1"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "weights.npz").write_bytes(b"weights")
        asr = EmbeddedASR(ASRConfig(
            model="mlx-community/whisper-medium-mlx-4bit",
            download_endpoint="https://hf-mirror.com",
            cache_dir=str(cache_dir),
        ))

        with patch("huggingface_hub.HfApi") as api:
            result = asr._download_model()

        self.assertEqual(result, str(model_dir))
        api.assert_not_called()
        self.assertEqual(asr.progress["message"], "使用本地缓存加载 ASR 模型")


class EmbeddedASRTranscribeTests(unittest.TestCase):
    """ACC-003-3: transcribe_chunk returns ASRResult."""

    def test_transcribe_returns_asr_result(self):
        """transcribe_chunk returns ASRResult with text and audio_ms."""
        asr = EmbeddedASR(ASRConfig())
        def _mock_transcribe(audio_np, cfg):
            return {"text": "hello world", "segments": [{"start": 0.0, "end": 1.0}]}
        asr._model = _mock_transcribe
        pcm = b"\x00\x00" * 16000  # 1s of silence at 16kHz
        result = asyncio.run(asr.transcribe_chunk(pcm))
        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.text, "hello world")
        self.assertGreater(result.audio_ms, 0)
        self.assertGreater(result.confidence, 0)

    def test_transcribe_converts_traditional_chinese_to_simplified(self):
        asr = EmbeddedASR(ASRConfig())
        asr._model = lambda audio_np, cfg: {
            "text": "繁體中文",
            "segments": [{"text": "這是一段"}],
        }
        result = asyncio.run(asr.transcribe_chunk(b"\x00\x00" * 16000))
        self.assertEqual(result.text, "繁体中文")
        self.assertEqual(result.raw_segments[0]["text"], "这是一段")

    def test_transcribe_empty_pcm(self):
        """Empty PCM returns empty text, confidence=0."""
        asr = EmbeddedASR(ASRConfig())
        asr._model = lambda audio_np, cfg: {"text": "", "segments": []}
        result = asyncio.run(asr.transcribe_chunk(b""))
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)


class EmbeddedASRFailureTests(unittest.TestCase):
    """ACC-003-4: failure returns empty text + confidence=0; degraded after 5 failures."""

    def test_transcribe_exception_returns_empty(self):
        """If model raises, return ASRResult(text="", confidence=0)."""
        asr = EmbeddedASR(ASRConfig())
        def _raise(audio_np, cfg):
            raise RuntimeError("GPU error")
        asr._model = _raise
        pcm = b"\x00\x00" * 16000
        result = asyncio.run(asr.transcribe_chunk(pcm))
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)

    def test_repeated_failures_set_degraded(self):
        """5 consecutive failures should set degraded=True."""
        asr = EmbeddedASR(ASRConfig())
        def _raise(audio_np, cfg):
            raise RuntimeError("GPU error")
        asr._model = _raise
        pcm = b"\x00\x00" * 16000
        for _ in range(5):
            asyncio.run(asr.transcribe_chunk(pcm))
        self.assertTrue(asr.degraded)

    def test_success_resets_failures(self):
        """A successful transcription resets the failure counter."""
        asr = EmbeddedASR(ASRConfig())
        call_count = [0]
        def _flaky(audio_np, cfg):
            call_count[0] += 1
            if call_count[0] <= 4:
                raise RuntimeError("err")
            return {"text": "ok", "segments": []}
        asr._model = _flaky
        pcm = b"\x00\x00" * 16000
        for _ in range(4):
            asyncio.run(asr.transcribe_chunk(pcm))
        self.assertFalse(asr.degraded)
        asyncio.run(asr.transcribe_chunk(pcm))  # 5th call succeeds
        self.assertFalse(asr.degraded)


if __name__ == "__main__":
    unittest.main()
