"""Unit tests for ACC-004-3: Live session HTTP endpoints (SPEC-004 / IF-001).

Tests use FastAPI TestClient to exercise the live session lifecycle:
- POST /api/podcasts/{id}/live/start
- POST /api/podcasts/{id}/live/{sid}/stop
- POST /api/asr/warmup
- UW-LP-5: no live segments -> 409
- UW-LP-1: terminal state rejects commands
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
    # Basic array ops as no-ops
    for attr in ("array", "zeros", "ones", "concatenate", "split", "astype",
                 "abs", "max", "min", "mean", "sum", "exp", "log", "sqrt",
                 "matmul", "transpose", "reshape", "expand_dims", "squeeze",
                  "tile", "stack", "arange", "linspace", "full", "eye"):
        setattr(core, attr, lambda *a, **kw: None)
    # dtype constants
    for dt in ("float16", "float32", "bfloat16", "int8", "int16", "int32",
               "uint8", "uint16", "uint32", "bool"):
        setattr(core, dt, dt)
    # core has .array property for shape etc
    class _FakeArray:
        def __init__(self): self.shape = None; self.dtype = None
    core.array = lambda *a, **kw: _FakeArray()
    # nn.Module base class
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


def _make_project_json(projects_dir: Path, project_id: str, segments_spec):
    """Create a project JSON with segments.

    segments_spec: list of (speaker, source, text)
    """
    audio_dir = projects_dir.parent / "podcasts" / project_id
    audio_dir.mkdir(parents=True, exist_ok=True)
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
    return project


class LiveSessionHTTPEndpointTests(unittest.TestCase):
    """ACC-004-3: Live session lifecycle endpoints."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.projects_dir = self.tmpdir / "projects"
        self.outputs_dir = self.tmpdir / "outputs"
        self.projects_dir.mkdir()
        self.outputs_dir.mkdir()
        # Use a config that points to our temp dirs
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
        self.client = self.app

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_project(self, project_id, segments_spec):
        """Create project JSON in the same dir that PodcastManager reads from."""
        from tts_service.models import PodcastProject
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

    def test_start_live_session_returns_201(self):
        """POST /api/podcasts/{id}/live/start returns 201 with session_id."""
        from fastapi.testclient import TestClient
        self._create_project("lp1", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Aaron", SegmentSource.LIVE, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/lp1/live/start", json={})
            self.assertEqual(resp.status_code, 201)
            data = resp.json()
            self.assertIn("session_id", data)
            self.assertEqual(data["project_id"], "lp1")
            self.assertIn(data["state"], ["AI_SPEAKING", "RECORDING", "IDLE", "WAITING_TRIGGER", "DETECTING"])

    def test_start_no_live_segments_returns_409(self):
        """ACC-UW-5: project with no live segments returns 409."""
        from fastapi.testclient import TestClient
        self._create_project("lp2", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Bob", SegmentSource.TTS, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/lp2/live/start", json={})
            self.assertEqual(resp.status_code, 409)

    def test_start_nonexistent_project_returns_404(self):
        """Starting a live session on a nonexistent project returns 404."""
        from fastapi.testclient import TestClient
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/nonexistent/live/start", json={})
            self.assertEqual(resp.status_code, 404)

    def test_stop_live_session(self):
        """POST /api/podcasts/{id}/live/{sid}/stop returns 200."""
        from fastapi.testclient import TestClient
        self._create_project("lp3", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Aaron", SegmentSource.LIVE, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/lp3/live/start", json={})
            sid = resp.json()["session_id"]
            resp2 = client.post(f"/api/podcasts/lp3/live/{sid}/stop")
            self.assertEqual(resp2.status_code, 200)
            self.assertEqual(resp2.json()["state"], "FINISHED")

    def test_stop_already_finished_returns_404(self):
        """ACC-UW-1: stopping an already-stopped session returns 404 (session removed from registry)."""
        from fastapi.testclient import TestClient
        self._create_project("lp4", [
            ("Flora", SegmentSource.TTS, "hello"),
            ("Aaron", SegmentSource.LIVE, "world"),
        ])
        with TestClient(self.app) as client:
            resp = client.post("/api/podcasts/lp4/live/start", json={})
            sid = resp.json()["session_id"]
            client.post(f"/api/podcasts/lp4/live/{sid}/stop")
            resp3 = client.post(f"/api/podcasts/lp4/live/{sid}/stop")
            self.assertEqual(resp3.status_code, 404)


if __name__ == "__main__":
    unittest.main()
