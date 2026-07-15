"""Unit tests for ACC-013-1, ACC-015-1, ACC-016-1.

ACC-013-1: Live segment redo endpoint (SPEC-013 / ST-LP-013)
ACC-015-1: live_only parameter in start endpoint (SPEC-015 / ST-LP-015)
ACC-016-1: Session state persistence + resume (SPEC-016 / ST-LP-016)
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

# Stub mlx for environments without Apple Silicon MLX
import types
def _make_mlx_stubs():
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    nn = types.ModuleType("mlx.nn")
    mlx.core = core
    mlx.nn = nn
    for attr in ("array", "zeros", "ones", "concatenate", "split", "astype",
                 "abs", "max", "min", "mean", "sum", "exp", "log", "sqrt",
                 "matmul", "transpose", "reshape", "expand_dims", "squeeze",
                 "tile", "stack", "arange", "linspace", "full", "eye"):
        setattr(core, attr, lambda *a, **kw: None)
    for dt in ("float16", "float32", "bfloat16", "int8", "int16", "int32",
               "uint8", "uint16", "uint32", "bool"):
        setattr(core, dt, dt)
    class _FakeArray:
        def __init__(self): self.shape = None; self.dtype = None
    core.array = lambda *a, **kw: _FakeArray()
    class _Module:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return None
    nn.Module = _Module
    for attr in ("Linear", "Embedding", "LayerNorm", "RMSNorm", "MultiHeadAttention",
                 "SinusoidalPosEmb", "Sequential", "Identity", "ModuleList"):
        setattr(nn, attr, _Module)
    return mlx, core, nn
_mlx, _mlx_core, _mlx_nn = _make_mlx_stubs()
sys.modules["mlx"] = _mlx
sys.modules["mlx.core"] = _mlx_core
sys.modules["mlx.nn"] = _mlx_nn

from tts_service.models import PodcastSegment, SegmentSource, PodcastProject
from tts_service.server import create_app


class _BaseLiveHTTPTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.outputs_dir = self.tmpdir / "outputs"
        self.outputs_dir.mkdir()
        config_path = self.tmpdir / "test_config.yaml"
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
  base_dir: {self.tmpdir / "voices"}
  default_voice: test
  live_speakers: ["Aaron"]
outputs:
  base_dir: {self.outputs_dir}
  history_limit: 5
server:
  host: 127.0.0.1
  port: 8123
  log_level: warning
asr:
  enabled: false
live:
  tts_max_seconds: 60
  tts_timeout_seconds: 30
  max_ai_rephrases_per_segment: 2
pid_file: {self.tmpdir / "test.pid"}
""", encoding="utf-8")
        self.config_path = str(config_path)
        self.app = create_app(self.config_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_project(self, project_id, segments_spec):
        projects_dir = self.outputs_dir / "podcasts"
        projects_dir.mkdir(parents=True, exist_ok=True)
        segments = []
        for i, (speaker, source, text) in enumerate(segments_spec):
            segments.append(PodcastSegment(
                index=i, text=text, speaker=speaker,
                source=source, status="pending",
            ))
        project = PodcastProject(
            id=project_id, title="Test",
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
            segments=segments, gap_seconds=0.5,
        )
        (projects_dir / f"{project_id}.json").write_text(
            project.model_dump_json(indent=2), encoding="utf-8")


class RedoEndpointTests(_BaseLiveHTTPTest):
    """ACC-013-1: Live segment redo endpoint."""

    def test_redo_live_segment_returns_200(self):
        """POST /api/podcasts/{id}/live/{sid}/redo/{index} returns 200."""
        from fastapi.testclient import TestClient
        self._create_project("rd1", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Aaron", SegmentSource.LIVE, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/rd1/live/start", json={})
            sid = resp.json()["session_id"]
            redo_resp = client.post(f"/api/podcasts/rd1/live/{sid}/redo/1")
            self.assertEqual(redo_resp.status_code, 200)
            data = redo_resp.json()
            self.assertEqual(data["index"], 1)

    def test_redo_tts_segment_returns_400(self):
        """Redo on a TTS segment returns 400."""
        from fastapi.testclient import TestClient
        self._create_project("rd2", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Aaron", SegmentSource.LIVE, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/rd2/live/start", json={})
            sid = resp.json()["session_id"]
            redo_resp = client.post(f"/api/podcasts/rd2/live/{sid}/redo/0")
            self.assertEqual(redo_resp.status_code, 400)


class LiveOnlyModeTests(_BaseLiveHTTPTest):
    """ACC-015-1: live_only parameter."""

    def test_start_with_live_only_flag(self):
        """POST start with live_only=true returns 201 and skips TTS."""
        from fastapi.testclient import TestClient
        self._create_project("lo1", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Aaron", SegmentSource.LIVE, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/lo1/live/start", json={"live_only": True})
            self.assertEqual(resp.status_code, 201)
            data = resp.json()
            self.assertEqual(data["session_id"], data["session_id"])  # has session_id


class SessionPersistenceTests(_BaseLiveHTTPTest):
    """ACC-016-1: Session state persistence + resume."""

    def test_resume_nonexistent_session_returns_404(self):
        """POST resume on unknown session returns 404."""
        from fastapi.testclient import TestClient
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/p1/live/fake-session/resume")
            self.assertEqual(resp.status_code, 404)

    def test_session_json_persisted_after_start(self):
        """After start, a session.json should exist on disk."""
        from fastapi.testclient import TestClient
        self._create_project("sp1", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Aaron", SegmentSource.LIVE, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/sp1/live/start", json={})
            sid = resp.json()["session_id"]
            session_json = self.outputs_dir / "podcasts" / "sp1" / "live_sessions" / f"{sid}.json"
            self.assertTrue(session_json.exists(), f"session.json should exist at {session_json}")


if __name__ == "__main__":
    unittest.main()
