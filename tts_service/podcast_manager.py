"""Podcast project manager: segmented generation with per-segment editing."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .engines.base import (
    _compute_gaps,
    _parse_tagged_dialogue,
    _concatenate_audio_segments,
)
from .models import PodcastProject, PodcastSegment
from .segmentation import segment_dialogue, segment_long_text


class PodcastManager:
    """Manage podcast projects with segmented audio generation."""

    def __init__(self, projects_dir: Path, outputs_dir: Path):
        self.projects_dir = Path(projects_dir)
        self.outputs_dir = Path(outputs_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    # ── CRUD ────────────────────────────────────────────────

    def create_project(self, title: str, text: str, output_format: str = "wav", gap_seconds: float = 1.0) -> PodcastProject:
        project_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        segments = self._text_to_segments(text)
        project = PodcastProject(
            id=project_id,
            title=title,
            created_at=now,
            updated_at=now,
            output_format=output_format,
            segments=segments,
            gap_seconds=gap_seconds,
        )
        self._save_project(project)
        # Create project-specific audio directory
        self._project_audio_dir(project_id).mkdir(parents=True, exist_ok=True)
        return project

    def get_project(self, project_id: str) -> Optional[PodcastProject]:
        path = self._project_path(project_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return PodcastProject(**data)

    def list_projects(self) -> list[PodcastProject]:
        projects: list[PodcastProject] = []
        for path in sorted(self.projects_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                projects.append(PodcastProject(**data))
            except Exception:
                continue
        return projects

    def delete_project(self, project_id: str) -> bool:
        path = self._project_path(project_id)
        if not path.exists():
            return False
        # Delete audio files
        audio_dir = self._project_audio_dir(project_id)
        if audio_dir.exists():
            for f in audio_dir.iterdir():
                f.unlink(missing_ok=True)
            audio_dir.rmdir()
        path.unlink(missing_ok=True)
        return True

    # ── Project-level editing ───────────────────────────────

    def update_gap(self, project_id: str, gap_seconds: float) -> PodcastProject:
        project = self._get_or_raise(project_id)
        project.gap_seconds = gap_seconds
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return project

    # ── Segment editing ─────────────────────────────────────

    def update_segment(
        self,
        project_id: str,
        index: int,
        text: Optional[str] = None,
        speaker: Optional[str] = None,
        tone: Optional[str] = None,
        voice_ref: Optional[str] = None,
    ) -> PodcastProject:
        project = self._get_or_raise(project_id)
        if index < 0 or index >= len(project.segments):
            raise ValueError(f"Segment index {index} out of range")
        seg = project.segments[index]
        if text is not None:
            seg.text = text
            # Text changed → mark as pending (needs regeneration)
            seg.status = "pending"
            seg.audio_filename = None
        if speaker is not None:
            seg.speaker = speaker
        if tone is not None:
            seg.tone = tone
        if voice_ref is not None:
            seg.voice_ref = voice_ref
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return project

    def regenerate_segment(
        self,
        project_id: str,
        index: int,
        engine,
        sample_manager,
    ) -> PodcastProject:
        """Regenerate a single segment using the provided engine."""
        project = self._get_or_raise(project_id)
        if index < 0 or index >= len(project.segments):
            raise ValueError(f"Segment index {index} out of range")
        seg = project.segments[index]

        # Resolve voice
        voice = seg.voice_ref or seg.speaker or sample_manager.default_voice
        resolved = sample_manager.resolve(voice)
        if resolved is None:
            resolved = sample_manager.resolve_or_default(seg.speaker or sample_manager.default_voice)

        try:
            result = engine.generate_single(
                text=seg.text,
                voice=resolved.speaker,
                output_format=project.output_format,
            )
        except Exception as exc:
            seg.status = "error"
            project.updated_at = datetime.now().isoformat()
            self._save_project(project)
            raise

        # Save audio file
        audio_dir = self._project_audio_dir(project_id)
        filename = f"seg_{index:04d}.{project.output_format}"
        audio_path = audio_dir / filename
        audio_path.write_bytes(result.audio_bytes)

        seg.audio_filename = filename
        seg.duration_seconds = result.duration_seconds
        seg.generation_seconds = result.generation_seconds
        seg.status = "generated"
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return project

    def generate_all_pending(
        self,
        project_id: str,
        engine,
        sample_manager,
    ) -> PodcastProject:
        """Generate all pending segments sequentially."""
        project = self._get_or_raise(project_id)
        for i, seg in enumerate(project.segments):
            if seg.status != "pending":
                continue
            self.regenerate_segment(project_id, i, engine, sample_manager)
        return self.get_project(project_id) or project

    # ── Merge ───────────────────────────────────────────────

    def merge_project(self, project_id: str) -> str:
        """Concatenate all generated segments into a single audio file.

        Returns the filename of the merged audio.
        """
        project = self._get_or_raise(project_id)
        audio_dir = self._project_audio_dir(project_id)
        audio_parts: list[bytes] = []
        for seg in project.segments:
            if seg.audio_filename:
                path = audio_dir / seg.audio_filename
                if path.exists():
                    audio_parts.append(path.read_bytes())
        if not audio_parts:
            raise ValueError("No generated audio segments to merge")

        gaps = _compute_gaps(len(audio_parts), project.gap_seconds)
        merged = _concatenate_audio_segments(audio_parts, project.output_format, gaps=gaps)
        filename = f"{project_id}_merged.{project.output_format}"
        output_path = self.outputs_dir / filename
        output_path.write_bytes(merged)
        project.merged_audio_filename = filename
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return filename

    # ── Helpers ─────────────────────────────────────────────

    def _project_path(self, project_id: str) -> Path:
        return self.projects_dir / f"{project_id}.json"

    def _project_audio_dir(self, project_id: str) -> Path:
        return self.outputs_dir / "podcasts" / project_id

    def _save_project(self, project: PodcastProject) -> None:
        path = self._project_path(project.id)
        path.write_text(project.model_dump_json(indent=2), encoding="utf-8")

    def _get_or_raise(self, project_id: str) -> PodcastProject:
        project = self.get_project(project_id)
        if project is None:
            raise FileNotFoundError(f"Podcast project '{project_id}' not found")
        return project

    @staticmethod
    def _text_to_segments(text: str) -> list[PodcastSegment]:
        """Parse tagged dialogue or plain text into podcast segments."""
        tagged, is_tagged = _parse_tagged_dialogue(text)
        if is_tagged:
            return [
                PodcastSegment(
                    index=i,
                    text=seg["text"],
                    speaker=seg["speaker"],
                    voice_ref=seg["speaker"],
                )
                for i, seg in enumerate(tagged)
            ]
        # Plain text: segment by sentence boundaries
        plain_segments = segment_long_text(text)
        return [
            PodcastSegment(
                index=i,
                text=s,
                voice_ref="",
            )
            for i, s in enumerate(plain_segments)
        ]
