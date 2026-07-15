"""Integration tests for ST-LP-009: Multi-source audio merge.

Generates real WAV fixtures, calls PodcastManager.merge_project(),
and validates the output file.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.models import PodcastSegment, SegmentSource


def _make_wav(path: Path, sample_rate: int, duration_s: float, freq: float = 440.0):
    """Generate a real WAV file with a sine wave."""
    samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    sf.write(str(path), audio, sample_rate, format="WAV", subtype="PCM_16")


class MergeWithRealPodcastManagerTests(unittest.TestCase):
    """Call PodcastManager.merge_project() with real WAV files."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.projects_dir = self.tmpdir / "projects"
        self.outputs_dir = self.tmpdir / "outputs"
        self.projects_dir.mkdir()
        self.outputs_dir.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_project_with_audio(self, project_id, segments_spec):
        """Create a project JSON + audio files on disk.

        segments_spec: list of (source, sample_rate, duration_s)
        """
        from tts_service.models import PodcastProject
        import json

        audio_dir = self.outputs_dir / "podcasts" / project_id
        audio_dir.mkdir(parents=True, exist_ok=True)

        segments = []
        for i, (source, sr, dur) in enumerate(segments_spec):
            if source == SegmentSource.TTS:
                fname = f"seg_{i:04d}.wav"
                status = "generated"
            else:
                fname = f"live_{i:04d}.wav"
                status = "generated"
            wav_path = audio_dir / fname
            _make_wav(wav_path, sr, dur)
            segments.append(PodcastSegment(
                index=i, text=f"seg{i}", speaker="A",
                source=source, status=status, audio_filename=fname,
            ))

        project = PodcastProject(
            id=project_id, title="Test", created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00", segments=segments, gap_seconds=0.5,
        )
        proj_path = self.projects_dir / f"{project_id}.json"
        proj_path.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        return project

    def test_merge_all_tts_segments(self):
        """All-TTS project merges via existing path."""
        from tts_service.podcast_manager import PodcastManager
        pm = PodcastManager(self.projects_dir, self.outputs_dir)
        self._create_project_with_audio("p1", [
            (SegmentSource.TTS, 24000, 1.0),
            (SegmentSource.TTS, 24000, 1.0),
        ])
        filename = pm.merge_project("p1")
        self.assertTrue(filename.startswith("p1"))
        merged_path = self.outputs_dir / filename
        self.assertTrue(merged_path.exists())
        info = sf.info(str(merged_path))
        # 2 segments × 1s + 1 gap × 0.5s = 2.5s minimum
        self.assertGreater(info.duration, 2.3)

    def test_merge_mixed_tts_and_live(self):
        """Mixed TTS + live project merges successfully."""
        from tts_service.podcast_manager import PodcastManager
        pm = PodcastManager(self.projects_dir, self.outputs_dir)
        self._create_project_with_audio("p2", [
            (SegmentSource.TTS, 24000, 1.0),
            (SegmentSource.LIVE, 48000, 1.0),  # different sample rate
            (SegmentSource.TTS, 24000, 1.0),
        ])
        filename = pm.merge_project("p2")
        merged_path = self.outputs_dir / filename
        self.assertTrue(merged_path.exists())
        info = sf.info(str(merged_path))
        # 3 segments × 1s + 2 gaps × 0.5s = 4s minimum
        self.assertGreater(info.duration, 3.5)

    def test_merge_missing_live_segment_uses_silence(self):
        """Missing live WAV is substituted with silence."""
        from tts_service.podcast_manager import PodcastManager
        from tts_service.models import PodcastProject
        import json

        audio_dir = self.outputs_dir / "podcasts" / "p3"
        audio_dir.mkdir(parents=True, exist_ok=True)
        # Only create seg_0000.wav, skip live_0001.wav
        _make_wav(audio_dir / "seg_0000.wav", 24000, 1.0)
        _make_wav(audio_dir / "seg_0002.wav", 24000, 1.0)

        segments = [
            PodcastSegment(index=0, text="a", speaker="A", source=SegmentSource.TTS,
                          status="generated", audio_filename="seg_0000.wav"),
            PodcastSegment(index=1, text="b", speaker="B", source=SegmentSource.LIVE,
                          status="pending", audio_filename="live_0001.wav"),
            PodcastSegment(index=2, text="c", speaker="A", source=SegmentSource.TTS,
                          status="generated", audio_filename="seg_0002.wav"),
        ]
        project = PodcastProject(
            id="p3", title="Test", created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00", segments=segments, gap_seconds=0.5,
        )
        (self.projects_dir / "p3.json").write_text(
            project.model_dump_json(indent=2), encoding="utf-8")

        pm = PodcastManager(self.projects_dir, self.outputs_dir)
        # merge_project should handle missing live file: use silence placeholder
        # and produce a valid merged output (per ST-LP-009.E)
        filename = pm.merge_project("p3")
        merged_path = self.outputs_dir / filename
        self.assertTrue(merged_path.exists(), "merge should succeed with silence placeholder")
        info = sf.info(str(merged_path))
        # 2 real segments (1s each) + 1 silence placeholder (1s) + gaps
        self.assertGreaterEqual(info.duration, 2.5)

    def test_merge_output_filename_format(self):
        """Merged file is named {project_id}_merged.wav"""
        from tts_service.podcast_manager import PodcastManager
        pm = PodcastManager(self.projects_dir, self.outputs_dir)
        self._create_project_with_audio("p4", [(SegmentSource.TTS, 24000, 0.5)])
        filename = pm.merge_project("p4")
        self.assertEqual(filename, "p4_merged.wav")


if __name__ == "__main__":
    unittest.main()