"""Podcast project manager: segmented generation with per-segment editing."""

from __future__ import annotations

import io
import json
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf

from .engines.base import (
    _FFMPEG_PATH,
    _parse_tagged_dialogue,
    _concatenate_audio_segments,
    _generate_silence,
)
from .models import PodcastProject, PodcastSegment
from .segmentation import segment_dialogue, segment_long_text


# Strip YAML frontmatter, HTML comments, and markdown headings
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _preprocess_podcast_text(text: str) -> str:
    """Clean markdown artifacts before segment parsing.

    Removes:
        - YAML frontmatter (--- ... ---)
        - HTML comments (<!-- ... -->)
        - Markdown heading lines (# Heading)
    Preserves all other line breaks.
    """
    text = _FRONTMATTER_RE.sub("", text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _MD_HEADING_RE.sub("", text)
    # Collapse consecutive blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
        text = _preprocess_podcast_text(text)
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
        pre_pause: Optional[float] = None,
        post_pause: Optional[float] = None,
        bgm_filename: Optional[str] = None,
        bgm_position: Optional[str] = None,
        bgm_volume: Optional[float] = None,
        bgm_fade_in: Optional[float] = None,
        bgm_fade_out: Optional[float] = None,
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
        if pre_pause is not None:
            seg.pre_pause = pre_pause
        if post_pause is not None:
            seg.post_pause = post_pause
        if bgm_filename is not None:
            seg.bgm_filename = bgm_filename
        if bgm_position is not None:
            seg.bgm_position = bgm_position
        if bgm_volume is not None:
            seg.bgm_volume = bgm_volume
        if bgm_fade_in is not None:
            seg.bgm_fade_in = bgm_fade_in
        if bgm_fade_out is not None:
            seg.bgm_fade_out = bgm_fade_out
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

        # Resolve voice (speaker name auto-resolves to {speaker}.normal if no tone suffix)
        resolved = sample_manager.resolve(seg.speaker)
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

    def insert_segment(self, project_id: str, index: int) -> PodcastProject:
        """Insert a new empty segment after the given index."""
        project = self._get_or_raise(project_id)
        if index < -1 or index >= len(project.segments):
            raise ValueError(f"Segment index {index} out of range")
        # Default speaker: use the speaker of the segment at index, or default voice
        default_speaker = ""
        if 0 <= index < len(project.segments):
            default_speaker = project.segments[index].speaker
        new_seg = PodcastSegment(
            index=0,
            text="",
            speaker=default_speaker,
        )
        insert_at = index + 1 if index >= 0 else len(project.segments)
        project.segments.insert(insert_at, new_seg)
        # Re-index all segments
        for i, seg in enumerate(project.segments):
            seg.index = i
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return project

    def delete_segment(self, project_id: str, index: int) -> PodcastProject:
        """Delete a segment and its audio file, then re-index."""
        project = self._get_or_raise(project_id)
        if index < 0 or index >= len(project.segments):
            raise ValueError(f"Segment index {index} out of range")
        seg = project.segments[index]
        # Delete audio file if exists
        if seg.audio_filename:
            audio_path = self._project_audio_dir(project_id) / seg.audio_filename
            audio_path.unlink(missing_ok=True)
        project.segments.pop(index)
        # Re-index all segments
        for i, seg in enumerate(project.segments):
            seg.index = i
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

    def merge_project(self, project_id: str, bgm_dir: Optional[Path] = None) -> str:
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

        pre_pauses = [seg.pre_pause for seg in project.segments if seg.audio_filename]
        post_pauses = [seg.post_pause for seg in project.segments if seg.audio_filename]

        # If any segment has BGM, process per-segment with BGM
        segs_with_audio = [seg for seg in project.segments if seg.audio_filename]
        has_bgm = any(
            seg.bgm_filename and bgm_dir and (bgm_dir / seg.bgm_filename).exists()
            for seg in segs_with_audio
        )

        if has_bgm:
            merged = self._merge_with_bgm(
                segs_with_audio, audio_parts, project.output_format,
                pre_pauses=pre_pauses, post_pauses=post_pauses,
                base_gap=project.gap_seconds, bgm_dir=bgm_dir,
            )
        else:
            merged = _concatenate_audio_segments(
                audio_parts, project.output_format,
                pre_pauses=pre_pauses, post_pauses=post_pauses,
                base_gap=project.gap_seconds,
            )

        filename = f"{project_id}_merged.{project.output_format}"
        output_path = self.outputs_dir / filename
        output_path.write_bytes(merged)
        project.merged_audio_filename = filename
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return filename

    def _merge_with_bgm(
        self,
        segments,
        audio_parts: list[bytes],
        output_format: str,
        pre_pauses: list[float],
        post_pauses: list[float],
        base_gap: float,
        bgm_dir: Optional[Path],
    ) -> bytes:
        """Merge segments with per-segment BGM using ffmpeg concat demuxer."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            all_files: list[Path] = []
            for i, (seg, seg_bytes) in enumerate(zip(segments, audio_parts)):
                # Pre-pause / base_gap + post_pause from previous segment
                if i == 0:
                    silence_dur = pre_pauses[i]
                else:
                    silence_dur = base_gap + pre_pauses[i] + post_pauses[i - 1]
                if silence_dur > 0:
                    silence_path = tmp_path / f"sil_pre_{i:04d}.wav"
                    _generate_silence(silence_dur, 24000, silence_path, channels=1)
                    all_files.append(silence_path)

                # BGM before segment
                if seg.bgm_filename and bgm_dir and (bgm_dir / seg.bgm_filename).exists():
                    if seg.bgm_position == "before":
                        bgm_path = self._process_bgm(
                            bgm_dir / seg.bgm_filename,
                            seg.bgm_volume, seg.bgm_fade_in, seg.bgm_fade_out,
                            tmp_path / f"bgm_{i:04d}.{output_format}",
                        )
                        if bgm_path:
                            all_files.append(bgm_path)

                # Speech segment
                seg_path = tmp_path / f"seg_{i:04d}.{output_format}"
                seg_path.write_bytes(seg_bytes)
                all_files.append(seg_path)

                # BGM after segment
                if seg.bgm_filename and bgm_dir and (bgm_dir / seg.bgm_filename).exists():
                    if seg.bgm_position == "after":
                        bgm_path = self._process_bgm(
                            bgm_dir / seg.bgm_filename,
                            seg.bgm_volume, seg.bgm_fade_in, seg.bgm_fade_out,
                            tmp_path / f"bgm_after_{i:04d}.{output_format}",
                        )
                        if bgm_path:
                            all_files.append(bgm_path)

            # Post-pause after last segment
            if post_pauses:
                if post_pauses[-1] > 0:
                    silence_path = tmp_path / f"sil_post_{len(segments) - 1:04d}.wav"
                    _generate_silence(post_pauses[-1], 24000, silence_path, channels=1)
                    all_files.append(silence_path)

            list_file = tmp_path / "concat_list.txt"
            list_file.write_text(
                "\n".join(f"file '{f.name}'" for f in all_files),
                encoding="utf-8",
            )
            output_path = tmp_path / f"output.{output_format}"
            cmd = [
                _FFMPEG_PATH, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path.read_bytes()

    @staticmethod
    def _process_bgm(
        bgm_path: Path,
        volume: float,
        fade_in: float,
        fade_out: float,
        output_path: Path,
    ) -> Path | None:
        """Apply volume and fade filters to a BGM track."""
        try:
            info = sf.info(str(bgm_path))
            duration = info.duration
        except Exception:
            return None
        filters = [f"volume={volume}"]
        if fade_in > 0:
            filters.append(f"afade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            fade_start = max(0, duration - fade_out)
            filters.append(f"afade=t=out:st={fade_start}:d={fade_out}")
        cmd = [
            _FFMPEG_PATH, "-y",
            "-i", str(bgm_path),
            "-af", ",".join(filters),
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

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
                )
                for i, seg in enumerate(tagged)
            ]
        # Plain text: segment by sentence boundaries
        plain_segments = segment_long_text(text)
        return [
            PodcastSegment(
                index=i,
                text=s,
            )
            for i, s in enumerate(plain_segments)
        ]
