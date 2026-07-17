"""End-to-end tests for the Live Studio WS flow.

Unlike test_live_podcast_flow.py (which only exercises the in-process
LiveSessionRegistry state machine), these tests drive the real HTTP +
WebSocket stack: POST /api/live/start -> WS /ws/live/{sid} -> full
frame exchange (segment_start -> audio -> ai_finished -> record_start ->
record_done -> finished) using the local TTS engine.

This is the test that catches WS driver-loop bugs, audio framing issues,
and AI/live alternation deadlocks that state-machine-only tests miss.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class _StubEngine:
    """Minimal TTS stub: returns 1s of 24kHz silence as WAV bytes."""
    sample_rate = 24000

    def generate_single(self, text, voice, output_format="wav",
                        instructions=None):
        import io
        import wave
        import numpy as np
        from tts_service.engines.base import GenerationResult
        from tts_service.models import SpeakerResolution
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes((np.zeros(24000, dtype=np.int16)).tobytes())
        return GenerationResult(
            audio_bytes=buf.getvalue(),
            output_format="wav",
            generation_seconds=0.01,
            duration_seconds=1.0,
            resolved_speakers=[SpeakerResolution(
                requested_name=voice or "default",
                resolved_voice=voice or "default",
                used_default=False,
            )],
        )


def _make_app(tmpdir: Path, asr_enabled: bool = False):
    """Build an app whose local engine is stubbed out."""
    from tts_service.server import create_app
    config_path = tmpdir / "config.yaml"
    config_path.write_text(f"""
model:
  model_id: test
  quantize_bits: 8
  diffusion_steps: 10
  cfg_scale: 1.0
  max_speech_tokens: 200
  seed: 42
  use_semantic: false
  use_coreml_semantic: false
  use_remote_qwen: false
  qwen_base_url: ""
  max_segment_chars: 200
  stereo: false
  spatial_jitter: false
  segment_gap_seconds: 1.0
  speaker_gap_seconds: 1.0
voices:
  base_dir: {tmpdir / "voices"}
  default_voice: Bowen
  cache_subdir: .cache
  live_speakers: ["Aaron"]
outputs:
  base_dir: {tmpdir / "outputs"}
  history_limit: 5
server:
  host: 127.0.0.1
  port: 8123
  log_level: warning
asr:
  enabled: {str(asr_enabled).lower()}
live:
  tts_max_seconds: 60
  tts_timeout_seconds: 30
  max_ai_rephrases_per_segment: 2
pid_file: {tmpdir / "test.pid"}
""", encoding="utf-8")
    app = create_app(str(config_path))
    # Monkey-patch the engine factory so generate_single uses the stub.
    from tts_service import server as srv
    # The app closure captured `engine` and `sample_manager`; we patch the
    # _resolve_engine helper by injecting a stub through the module.
    original_resolve = srv._resolve_engine
    def _stub_resolve(request_engine, config, sm):
        return _StubEngine()
    srv._resolve_engine = _stub_resolve
    return app, lambda: setattr(srv, "_resolve_engine", original_resolve)


class LiveStudioE2ETests(unittest.TestCase):
    """Real HTTP+WS end-to-end for the Live Studio feature."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.app, self._restore = _make_app(self.tmpdir)

    def tearDown(self):
        self._restore()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_asr_status_is_exposed_without_loading_model(self):
        """The UI can inspect ASR state without blocking on model download."""
        from fastapi.testclient import TestClient
        with TestClient(self.app) as client:
            response = client.get("/api/asr/status")
            self.assertEqual(response.status_code, 200)
            status = response.json()
            self.assertFalse(status["enabled"])
            self.assertFalse(status["ready"])
            self.assertEqual(status["status"], "idle")

    def test_full_ai_live_ai_alternation(self):
        """AI segment -> human segment -> AI segment completes via WS."""
        from fastapi.testclient import TestClient
        with TestClient(self.app) as client:
            resp = client.post("/api/live/start", json={
                "text": "Bowen: 第一句AI。\nAaron: 第二句真人。\nBowen: 第三句AI。",
            })
            self.assertEqual(resp.status_code, 200)
            sid = resp.json()["session_id"]
            segs = resp.json()["segments"]
            self.assertEqual([s["source"] for s in segs], ["tts", "live", "tts"])

            with client.websocket_connect(f"/ws/live/{sid}") as ws:
                received = []
                current_idx = -1
                while len(received) < 50:
                    msg = ws.receive()
                    if msg.get("bytes") is not None:
                        # AI audio frame -> acknowledge playback done
                        self.assertGreater(len(msg["bytes"]), 100)
                        ws.send_text(json.dumps({
                            "type": "ai_finished", "index": current_idx}))
                        continue
                    text = msg.get("text")
                    if not text:
                        break
                    frame = json.loads(text)
                    received.append(frame)
                    if frame["type"] == "segment_start":
                        current_idx = frame["index"]
                    elif frame["type"] == "state" and frame["state"] == "RECORDING":
                        # Send 1s of PCM16 silence for the live segment
                        ws.send_bytes(b"\x00\x00" * 16000)
                        ws.send_text(json.dumps({
                            "type": "record_done", "index": current_idx}))
                    elif frame["type"] == "finished":
                        break

            types = [f["type"] for f in received]
            # Must have gone through all 3 segments
            self.assertIn("finished", types)
            segment_done = [f for f in received if f["type"] == "segment_done"]
            self.assertEqual(len(segment_done), 3)
            for frame in segment_done:
                self.assertTrue(frame["audio_url"])
                audio = client.get(frame["audio_url"])
                self.assertEqual(audio.status_code, 200)
                self.assertIn("audio/wav", audio.headers["content-type"])
            finished = [f for f in received if f["type"] == "finished"][-1]
            self.assertTrue(finished["audio_url"])
            merged = client.get(finished["audio_url"])
            self.assertEqual(merged.status_code, 200)
            self.assertIn("attachment", merged.headers.get("content-disposition", ""))
            seg_starts = [f for f in received if f["type"] == "segment_start"]
            self.assertEqual(len(seg_starts), 3)
            self.assertEqual(seg_starts[0]["source"], "tts")
            self.assertEqual(seg_starts[1]["source"], "live")
            self.assertEqual(seg_starts[2]["source"], "tts")
            # State must have entered RECORDING for the live segment
            states = [f["state"] for f in received
                      if f["type"] == "state"]
            self.assertIn("AI_SPEAKING", states)
            self.assertIn("RECORDING", states)
            self.assertIn("FINISHED", states)

    def test_tts_failure_halts_session(self):
        """When TTS raises, the session reports an error and STOPS.

        It must NOT silently skip and continue (that would hide config
        problems from the user).
        """
        from fastapi.testclient import TestClient

        # Replace the stub with one that always fails.
        class _BoomEngine:
            sample_rate = 24000
            def generate_single(self, **kw):
                raise RuntimeError("engine unreachable")

        from tts_service import server as srv
        original = srv._resolve_engine
        srv._resolve_engine = lambda *a, **k: _BoomEngine()
        try:
            with TestClient(self.app) as client:
                resp = client.post("/api/live/start", json={
                    "text": "Bowen: 会失败。",
                })
                sid = resp.json()["session_id"]
                with client.websocket_connect(f"/ws/live/{sid}") as ws:
                    frames = []
                    for _ in range(10):
                        msg = ws.receive()
                        text = msg.get("text")
                        if not text:
                            break
                        f = json.loads(text)
                        frames.append(f)
                        if f["type"] == "error":
                            break
                    types = [f["type"] for f in frames]
                    self.assertIn("error", types)
                    # Must NOT have finished or advanced past the failure
                    self.assertNotIn("finished", types)
                    self.assertNotIn("segment_done", types)
                    # Error message should be user-actionable
                    err = [f for f in frames if f["type"] == "error"][0]
                    self.assertIn("失败", err["message"])
        finally:
            srv._resolve_engine = original

    def test_asr_accumulates_chunks_and_auto_completes(self):
        """A final ASR chunk completes a fully matched human segment."""
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        from tts_service.live.asr_engine import ASRResult

        class _FakeASR:
            def __init__(self, _cfg):
                self._calls = 0
                self._ready = True

            @property
            def is_ready(self):
                return self._ready

            @property
            def is_warming(self):
                return False

            @property
            def progress(self):
                return {
                    "status": "ready",
                    "progress": 1.0,
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "message": "ready",
                    "error": None,
                }

            def start_warmup(self):
                return True

            async def transcribe_chunk(self, _pcm):
                self._calls += 1
                text = (
                    "大家好，这一期的播客，是关于 vibe"
                    if self._calls == 1 else "coding"
                )
                return ASRResult(text=text, audio_ms=1000)

        with patch("tts_service.live.asr_engine.EmbeddedASR", _FakeASR):
            app, restore = _make_app(self.tmpdir, asr_enabled=True)
            try:
                with TestClient(app) as client:
                    resp = client.post("/api/live/start", json={
                        "text": "Aaron: 大家好，这一期的播客，是关于 vibe coding\nBowen: 人人都想 vibe coding",
                    })
                    self.assertEqual(resp.status_code, 200)
                    sid = resp.json()["session_id"]

                    speech_frame = b"\x40\x1f" * 4096
                    silence_frame = b"\x00\x00" * 4096
                    frames = []
                    with client.websocket_connect(f"/ws/live/{sid}") as ws:
                        current_idx = -1
                        while len(frames) < 80:
                            msg = ws.receive()
                            if msg.get("bytes") is not None:
                                ws.send_text(json.dumps({
                                    "type": "ai_finished", "index": current_idx,
                                }))
                                continue
                            frame = json.loads(msg["text"])
                            frames.append(frame)
                            if frame["type"] == "segment_start":
                                current_idx = frame["index"]
                            elif frame["type"] == "state" and frame["state"] == "RECORDING":
                                for _ in range(8):
                                    ws.send_bytes(speech_frame)
                                for _ in range(8):
                                    ws.send_bytes(silence_frame)
                            elif frame["type"] == "finished":
                                break

                    self.assertIn("record_auto_done", [f["type"] for f in frames])
                    partials = [f["text"] for f in frames if f["type"] == "asr_partial"]
                    self.assertTrue(any("大家好" in text and "coding" in text for text in partials))
                    starts = [f for f in frames if f["type"] == "segment_start"]
                    self.assertEqual([f["source"] for f in starts], ["live", "tts"])
                    self.assertIn("finished", [f["type"] for f in frames])
            finally:
                restore()

    def test_pause_and_resume_during_ai_playback(self):
        """Pause suspends the driver until resume, without losing the segment."""
        from fastapi.testclient import TestClient

        with TestClient(self.app) as client:
            resp = client.post("/api/live/start", json={"text": "Bowen: 可暂停的 AI 段"})
            sid = resp.json()["session_id"]
            with client.websocket_connect(f"/ws/live/{sid}") as ws:
                received = []
                while len(received) < 20:
                    msg = ws.receive()
                    if msg.get("bytes") is not None:
                        continue
                    frame = json.loads(msg["text"])
                    received.append(frame)
                    if frame["type"] == "audio":
                        ws.receive()  # binary WAV
                        ws.send_text(json.dumps({"type": "pause"}))
                    elif frame["type"] == "state" and frame["state"] == "PAUSED":
                        ws.send_text(json.dumps({"type": "resume"}))
                    elif frame["type"] == "state" and frame["state"] == "AI_SPEAKING":
                        if any(f["type"] == "state" and f["state"] == "PAUSED" for f in received):
                            ws.send_text(json.dumps({"type": "ai_finished", "index": 0}))
                    elif frame["type"] == "finished":
                        break

            states = [f.get("state") for f in received if f["type"] == "state"]
            self.assertIn("PAUSED", states)
            self.assertIn("AI_SPEAKING", states)
            self.assertIn("finished", [f["type"] for f in received])

    def test_manual_restart_from_segment_discards_future_audio(self):
        """A restart command returns to the selected segment."""
        from fastapi.testclient import TestClient
        with TestClient(self.app) as client:
            response = client.post("/api/live/start", json={
                "text": "Bowen: 第一段 AI\nAaron: 真人段\nBowen: 后续 AI",
            })
            sid = response.json()["session_id"]
            frames = []
            restarted = False
            with client.websocket_connect(f"/ws/live/{sid}") as ws:
                current = -1
                while len(frames) < 40:
                    msg = ws.receive()
                    if msg.get("bytes") is not None:
                        if restarted:
                            ws.send_text(json.dumps({"type": "ai_finished", "index": current}))
                        continue
                    frame = json.loads(msg["text"])
                    frames.append(frame)
                    if frame["type"] == "segment_start":
                        current = frame["index"]
                    elif frame["type"] == "audio" and not restarted:
                        restarted = True
                        ws.send_text(json.dumps({"type": "restart_from", "index": 1}))
                    elif frame["type"] == "state" and frame["state"] == "RECORDING":
                        ws.send_text(json.dumps({"type": "record_done", "index": current}))
                    elif frame["type"] == "finished":
                        break

            restarts = [f for f in frames if f["type"] == "restart"]
            self.assertEqual(len(restarts), 1)
            self.assertEqual(restarts[0]["to_index"], 1)
            self.assertIn("finished", [f["type"] for f in frames])


if __name__ == "__main__":
    unittest.main()
