"""Unit tests for the remote yovoice engine (no real network access)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tts_service.config import Config
from tts_service.engines.yovoice_remote import YovoiceRemoteEngine
from tts_service.sample_manager import SampleManager


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


class _Router:
    """Fake urlopen that answers the /v1/voices and /v1/generate endpoints."""

    def __init__(self, upload_body: bytes, generate_body: bytes):
        self.upload_body = upload_body
        self.generate_body = generate_body
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if "/v1/voices" in request.full_url:
            return _Response(self.upload_body)
        if "/v1/generate" in request.full_url:
            return _Response(self.generate_body)
        raise AssertionError(f"unexpected URL: {request.full_url}")

    def by_path(self, path: str) -> list:
        return [r for r in self.requests if path in r.full_url]


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    sf.write(output, np.zeros(2400, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _make_engine(token: str = "") -> YovoiceRemoteEngine:
    config = Config()
    config.model.yovoice_base_url = "http://127.0.0.1:8080"
    config.model.yovoice_model_id = "index-2.5-q8"
    config.model.yovoice_api_token = token
    config.model.stereo = False
    config.model.spatial_jitter = False
    voices_dir = Path(tempfile.mkdtemp())
    (voices_dir / "Aaron.txt").write_text("这是参考文本。", encoding="utf-8")
    sf.write(voices_dir / "Aaron.wav", np.zeros(2400, dtype=np.float32), 24000)
    manager = SampleManager(voices_dir, "Aaron")
    return YovoiceRemoteEngine(config, manager)


class YovoiceEngineTests(unittest.TestCase):
    def test_upload_once_and_reuse_voice_id(self):
        engine = _make_engine()
        router = _Router(
            json.dumps({"id": "voice-123"}).encode(),
            _wav_bytes(),
        )
        with patch("tts_service.engines.yovoice_remote.urllib.request.urlopen", router):
            engine.generate_single("你好。", None, "wav")
            engine.generate_single("再见。", None, "wav")

        uploads = router.by_path("/v1/voices")
        generates = router.by_path("/v1/generate")
        self.assertEqual(len(uploads), 1)
        self.assertEqual(len(generates), 2)
        self.assertEqual(uploads[0].full_url, "http://127.0.0.1:8080/v1/voices?name=Aaron.normal")

    def test_generate_payload_contains_text_model_and_voice(self):
        engine = _make_engine()
        router = _Router(
            json.dumps({"id": "voice-123"}).encode(),
            _wav_bytes(),
        )
        with patch("tts_service.engines.yovoice_remote.urllib.request.urlopen", router):
            result = engine.generate_single("第一句。", None, "wav")
            engine.generate_single("第二句。", None, "wav")

        payload = json.loads(router.by_path("/v1/generate")[0].data.decode())
        self.assertEqual(payload["text"], "第一句。")
        self.assertEqual(payload["modelId"], "index-2.5-q8")
        self.assertEqual(payload["voiceId"], "voice-123")
        self.assertEqual(result.output_format, "wav")

    def test_authorization_header_is_sent(self):
        token = "a" * 40
        engine = _make_engine(token=token)
        router = _Router(
            json.dumps({"id": "voice-123"}).encode(),
            _wav_bytes(),
        )
        with patch("tts_service.engines.yovoice_remote.urllib.request.urlopen", router):
            engine.generate_single("你好。", None, "wav")

        self.assertEqual(router.by_path("/v1/voices")[0].headers["Authorization"], f"Bearer {token}")
        self.assertEqual(router.by_path("/v1/generate")[0].headers["Authorization"], f"Bearer {token}")

    def test_http_failure_surfaces_as_runtime_error(self):
        engine = _make_engine()

        def boom(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 500, "boom", {}, io.BytesIO(b"server error"),
            )

        with patch("tts_service.engines.yovoice_remote.urllib.request.urlopen", side_effect=boom):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate_single("你好。", None, "wav")
        self.assertIn("Yovoice HTTP 500", str(ctx.exception))

    def test_nested_voice_id_is_accepted(self):
        engine = _make_engine()
        router = _Router(
            json.dumps({"voice": {"id": "nested-42"}}).encode(),
            _wav_bytes(),
        )
        with patch("tts_service.engines.yovoice_remote.urllib.request.urlopen", router):
            engine.generate_single("你好。", None, "wav")

        payload = json.loads(router.by_path("/v1/generate")[0].data.decode())
        self.assertEqual(payload["voiceId"], "nested-42")


if __name__ == "__main__":
    unittest.main()
