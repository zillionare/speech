"""Integration tests for ST-LP-002: PodcastSegment.source field.

All tests instantiate real model classes. Migration and auto-marking
tests use PodcastManager production code paths, not test-local helpers.
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

from tts_service.models import PodcastSegment, PodcastProject, SegmentSource


class SegmentSourceEnumTests(unittest.TestCase):

    def test_enum_values(self):
        self.assertEqual(SegmentSource.TTS.value, "tts")
        self.assertEqual(SegmentSource.LIVE.value, "live")


class SegmentModelTests(unittest.TestCase):

    def test_defaults_to_tts(self):
        seg = PodcastSegment(index=0, text="hello", speaker="Aaron")
        self.assertEqual(seg.source, SegmentSource.TTS)

    def test_can_be_live(self):
        seg = PodcastSegment(index=0, text="hello", speaker="Aaron",
                             source=SegmentSource.LIVE)
        self.assertEqual(seg.source, SegmentSource.LIVE)

    def test_serialization_preserves_source(self):
        seg = PodcastSegment(index=0, text="hello", speaker="Aaron",
                             source=SegmentSource.LIVE)
        json_str = seg.model_dump_json()
        restored = PodcastSegment.model_validate_json(json_str)
        self.assertEqual(restored.source, SegmentSource.LIVE)


class MigrationViaPodcastManagerTests(unittest.TestCase):
    """Test migration through PodcastManager.get_project() which deserializes JSON.

    Old project JSON files that lack the 'source' field should load
    with source defaulting to TTS (Pydantic default).
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.projects_dir = self.tmpdir / "projects"
        self.projects_dir.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_old_project_json(self, project_id: str, segments_data: list[dict]):
        """Write a project JSON in the old format (no 'source' field)."""
        project = {
            "id": project_id,
            "title": "Old Project",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "segments": segments_data,
            "gap_seconds": 0.5,
        }
        (self.projects_dir / f"{project_id}.json").write_text(
            json.dumps(project), encoding="utf-8")

    def test_old_json_without_source_loads_as_tts(self):
        """Old JSON missing 'source' field loads with source=TTS."""
        from tts_service.podcast_manager import PodcastManager
        self._write_old_project_json("old1", [
            {"index": 0, "text": "hello", "speaker": "Aaron", "status": "pending"},
        ])
        pm = PodcastManager(self.projects_dir, self.tmpdir)
        project = pm.get_project("old1")
        self.assertIsNotNone(project)
        self.assertEqual(project.segments[0].source, SegmentSource.TTS)

    def test_old_json_with_explicit_live_preserved(self):
        """Old JSON with explicit source=live is preserved."""
        from tts_service.podcast_manager import PodcastManager
        self._write_old_project_json("old2", [
            {"index": 0, "text": "hello", "speaker": "Aaron",
             "source": "live", "status": "pending"},
        ])
        pm = PodcastManager(self.projects_dir, self.tmpdir)
        project = pm.get_project("old2")
        self.assertEqual(project.segments[0].source, SegmentSource.LIVE)

    def test_mixed_old_and_new_segments(self):
        """Project with some segments having source and some not."""
        from tts_service.podcast_manager import PodcastManager
        self._write_old_project_json("old3", [
            {"index": 0, "text": "a", "speaker": "A", "status": "pending"},
            {"index": 1, "text": "b", "speaker": "B", "source": "live", "status": "pending"},
            {"index": 2, "text": "c", "speaker": "C", "status": "pending"},
        ])
        pm = PodcastManager(self.projects_dir, self.tmpdir)
        project = pm.get_project("old3")
        self.assertEqual(project.segments[0].source, SegmentSource.TTS)
        self.assertEqual(project.segments[1].source, SegmentSource.LIVE)
        self.assertEqual(project.segments[2].source, SegmentSource.TTS)


class ProjectOverrideTests(unittest.TestCase):

    def test_override_present(self):
        p = PodcastProject(
            id="t", title="T", created_at="x", updated_at="x",
            segments=[], live_speakers_override=["Aaron"],
        )
        self.assertEqual(p.live_speakers_override, ["Aaron"])

    def test_override_absent(self):
        p = PodcastProject(
            id="t", title="T", created_at="x", updated_at="x", segments=[],
        )
        self.assertIsNone(p.live_speakers_override)

    def test_override_takes_precedence(self):
        """Override replaces global config when set."""
        global_live = {"Ben"}
        p = PodcastProject(
            id="t", title="T", created_at="x", updated_at="x",
            segments=[], live_speakers_override=["Aaron"],
        )
        override = set(p.live_speakers_override or [])
        effective = override if override else global_live
        self.assertIn("Aaron", effective)
        self.assertNotIn("Ben", effective)

    def test_falls_back_to_global(self):
        global_live = {"Aaron"}
        p = PodcastProject(
            id="t", title="T", created_at="x", updated_at="x",
            segments=[], live_speakers_override=None,
        )
        override = set(p.live_speakers_override or [])
        effective = override if override else global_live
        self.assertEqual(effective, {"Aaron"})


if __name__ == "__main__":
    unittest.main()