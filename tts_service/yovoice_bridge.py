"""把 Speech Studio 的播客作品映射到界面（yovoice workbench）的 state 模型。

搬进来的界面（web/）原本只认桌面宿主（Go）与浏览器预览两套后端。这里给出第三套：
以 `PodcastProject` / `PodcastSegment` 为唯一真源，把作品渲染成界面认识的 `Draft`
（kind=story：subtitles + AudioTimeline），并把界面改回来的草稿还原成作品字段。
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Iterable, Optional

from .engines.base import _parse_tagged_dialogue
from .models import PodcastProject, PodcastSegment, SegmentSource
from .segmentation import segment_long_text

# 媒体文件名里用 `~` 标记子目录：`podcast~<作品 ID>~<文件>` 走 outputs/podcasts/<作品 ID>/。
MEDIA_SEGMENT_PREFIX = "podcast"
MEDIA_SEPARATOR = "~"
# 生成记录 ID 同时也是媒体文件名，界面据此直接从 clips 还原出音频。
_SEGMENT_TOKEN_RE = re.compile(
    rf"^{MEDIA_SEGMENT_PREFIX}{MEDIA_SEPARATOR}([^{MEDIA_SEPARATOR}]+){MEDIA_SEPARATOR}(.+)$"
)

ACTIVITY_GENERATE = "@yovoice.activity.generate"
ACTIVITY_COMPLETED = "@yovoice.activity.completed"
ACTIVITY_CANCELLED = "@yovoice.activity.cancelled"
ACTIVITY_FAILED = "@yovoice.activity.failed"

ERROR_BUSY = "@yovoice.error.busy"
ERROR_GENERATE_FAILED = "@yovoice.error.generateFailed"
ERROR_REQUEST_INVALID = "@yovoice.error.requestInvalid"


def now_iso() -> str:
    return datetime.now().isoformat()


def speaker_id(name: str) -> str:
    """说话人 ID 由名字派生，保证每次重建状态都稳定。"""
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"spk-{digest}"


def cue_id(project_id: str, index: int) -> str:
    return f"cue-{project_id}-{index}"


def segment_token(project_id: str, filename: str) -> str:
    return f"{MEDIA_SEGMENT_PREFIX}{MEDIA_SEPARATOR}{project_id}{MEDIA_SEPARATOR}{filename}"


def parse_segment_token(token: str) -> Optional[tuple[str, str]]:
    matched = _SEGMENT_TOKEN_RE.match(token or "")
    return (matched.group(1), matched.group(2)) if matched else None


# 与 web/src/shared/workbench.ts 的 createDraft 对齐的默认合成参数。
def default_settings() -> dict[str, Any]:
    return {
        "modelId": "index-2.5-q8",
        "voiceId": None,
        "mode": "text",
        "emotionVoiceId": None,
        "emotionText": "",
        "inferEmotion": False,
        "emotionStrength": 0.6,
        "emotions": [0, 0, 0, 0, 0, 0, 0, 0.5],
        "randomEmotion": False,
        "language": "zh",
        "speed": 1,
        "temperature": 0.8,
        "topP": 0.8,
        "topK": 30,
        "repetitionPenalty": 10,
        "maxTokens": 1500,
        "intervalSilenceMs": 200,
        "doSample": True,
        "numBeams": 3,
        "lengthPenalty": 0,
        "seed": None,
    }


def draft_text(speakers: list[dict[str, Any]], cues: list[dict[str, Any]]) -> str:
    """把台词还原成 `Speaker: text` 文本，供文本导出与旧界面使用。"""
    names = {s["id"]: (s.get("sourceName") or "").strip() for s in speakers}
    lines = []
    for cue in cues:
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        name = names.get(cue.get("speakerId") or "", "")
        lines.append(f"{name}: {text}" if name else text)
    return "\n\n".join(lines)


# ── 作品 → 草稿 ─────────────────────────────────────────────


def project_to_draft(
    project: PodcastProject,
    *,
    voice_names: Iterable[str] = (),
    live_speakers: Iterable[str] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """返回 (草稿, 该作品派生的生成记录)。"""
    known_voices = set(voice_names)
    live = set(live_speakers)

    speakers: list[dict[str, Any]] = []
    by_name: dict[str, str] = {}
    for segment in project.segments:
        name = (segment.speaker or "").strip()
        key = name.lower()
        if key in by_name:
            continue
        sid = speaker_id(name)
        by_name[key] = sid
        settings = default_settings()
        settings["speaker"] = name
        settings["voiceId"] = name if name in known_voices else None
        speakers.append({"id": sid, "sourceName": name, "settings": settings})

    cues: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    accepted: list[str] = []
    has_audio = False

    position = 0.0
    previous_post = 0.0
    gap = float(project.gap_seconds or 0.0)
    for index, segment in enumerate(project.segments):
        name = (segment.speaker or "").strip()
        sid = by_name.get(name.lower(), speaker_id(name))
        position += float(segment.pre_pause or 0.0) + (gap + previous_post if index else 0.0)
        duration = float(segment.duration_seconds or 0.0)
        cues.append({
            "id": cue_id(project.id, index),
            "start": round(position, 3),
            "end": round(position + duration, 3),
            "text": segment.text,
            "speakerId": sid,
        })
        if segment.audio_filename and segment.status in ("generated", "recorded"):
            token = segment_token(project.id, segment.audio_filename)
            generation = {
                "id": token,
                "title": _generation_title(name, segment.text) or project.title,
                "fileName": token,
                "createdAt": project.updated_at or now_iso(),
                "duration": duration,
                "settings": {**default_settings(), "id": project.id, "title": project.title, "text": segment.text, "speaker": name},
                "segment": {
                    "cueId": cue_id(project.id, index),
                    "speakerId": sid,
                    "speakerName": name,
                    "batchId": project.id,
                    "index": index,
                },
            }
            generations.append(generation)
            clips.append({
                "id": f"clip-{project.id}-{index}",
                "generationId": token,
                "start": round(position, 3),
                "offset": 0.0,
                "duration": duration,
            })
            accepted.append(token)
            has_audio = True
        position += duration
        previous_post = float(segment.post_pause or 0.0)

    draft: dict[str, Any] = {
        **default_settings(),
        "id": project.id,
        "kind": "story",
        "title": project.title,
        "text": draft_text(speakers, cues),
        "createdAt": project.created_at,
        "updatedAt": project.updated_at,
        "subtitles": {"speakers": speakers, "cues": cues},
        "timeline": {
            "tracks": [{
                "id": f"lane-{project.id}",
                "name": project.title or "对白",
                "muted": False,
                "clips": clips,
            }],
            "assets": [],
            "acceptedGenerations": accepted,
            "regenerateMode": "ripple",
        },
        "liveSpeakers": sorted(name for name in live),
        "hasAudio": has_audio,
    }
    return draft, generations


def _generation_title(speaker: str, text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return speaker or ""
    return f"{speaker} · {cleaned[:32]}" if speaker else cleaned[:32]


# ── 草稿 → 作品 ─────────────────────────────────────────────


def draft_to_project(draft: dict[str, Any], *, existing: Optional[PodcastProject] = None) -> dict[str, Any]:
    """把界面里的草稿还原成可保存的播客作品字段。"""
    existing_segments = list(existing.segments) if existing else []
    subtitles = draft.get("subtitles") or {}
    speakers = {s.get("id"): (s.get("sourceName") or "").strip() for s in subtitles.get("speakers") or []}
    cues = list(subtitles.get("cues") or [])

    if not cues:
        segments = _text_to_segments(draft.get("text") or "", existing_segments)
    else:
        cues.sort(key=lambda cue: (float(cue.get("start") or 0.0), str(cue.get("id") or "")))
        clips = _collect_clips(draft)
        by_filename = {s.audio_filename: s for s in existing_segments if s.audio_filename}
        segments = []
        for index, cue in enumerate(cues):
            start = float(cue.get("start") or 0.0)
            clip = next(
                (item for item in clips if abs(float(item.get("start") or 0.0) - start) < 0.02),
                None,
            )
            filename: Optional[str] = None
            duration = 0.0
            if clip is not None:
                parsed = parse_segment_token(str(clip.get("generationId") or ""))
                if parsed:
                    filename = parsed[1]
                    duration = float(clip.get("duration") or 0.0)
            carry = by_filename.get(filename) if filename else None
            if carry is None and index < len(existing_segments):
                carry = existing_segments[index]
            name = speakers.get(cue.get("speakerId") or "", "")
            segments.append(_segment_fields(index, cue, name, filename, duration, carry))

    project_id = str(draft.get("id") or "")
    return {
        "id": project_id,
        "title": (draft.get("title") or "").strip() or "未命名作品",
        "createdAt": draft.get("createdAt") or now_iso(),
        "updatedAt": now_iso(),
        "segments": segments,
    }


def _collect_clips(draft: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        clip
        for lane in (draft.get("timeline") or {}).get("tracks") or []
        for clip in lane.get("clips") or []
    ]


def _segment_fields(
    index: int,
    cue: dict[str, Any],
    speaker: str,
    filename: Optional[str],
    duration: float,
    carry: Optional[PodcastSegment],
) -> dict[str, Any]:
    source = carry.source if carry else SegmentSource.TTS
    return {
        "index": index,
        "text": cue.get("text") or "",
        "speaker": speaker,
        "tone": carry.tone if carry else "",
        "speed_modifier": carry.speed_modifier if carry else "",
        "audio_filename": filename,
        "duration_seconds": duration or (carry.duration_seconds if carry else 0.0),
        "generation_seconds": carry.generation_seconds if carry else 0.0,
        "status": ("recorded" if source == SegmentSource.LIVE else "generated") if filename else "pending",
        "pre_pause": carry.pre_pause if carry else 0.0,
        "post_pause": carry.post_pause if carry else 0.0,
        "bgm_filename": carry.bgm_filename if carry else None,
        "bgm_position": carry.bgm_position if carry else "before",
        "bgm_volume": carry.bgm_volume if carry else 0.15,
        "bgm_fade_in": carry.bgm_fade_in if carry else 2.0,
        "bgm_fade_out": carry.bgm_fade_out if carry else 3.0,
        "source": source.value if isinstance(source, SegmentSource) else str(source),
    }


def _text_to_segments(text: str, existing_segments: list[PodcastSegment]) -> list[dict[str, Any]]:
    tagged, is_tagged = _parse_tagged_dialogue(text)
    if is_tagged:
        entries = [(entry["speaker"], entry["text"]) for entry in tagged]
    else:
        entries = [("", entry) for entry in segment_long_text(text)]
    segments = []
    for index, (speaker, body) in enumerate(entries):
        carry = existing_segments[index] if index < len(existing_segments) else None
        segments.append(_segment_fields(index, {"text": body}, speaker, None, 0.0, carry))
    return segments


# ── 状态与资源 ──────────────────────────────────────────────


def build_state(
    *,
    projects: list[PodcastProject],
    voices: list[dict[str, Any]],
    preferences: dict[str, Any],
    live_speakers: Iterable[str] = (),
    activity: Optional[dict[str, Any]] = None,
    runtime_backend: Optional[str] = None,
    runtime_version: Optional[str] = None,
) -> dict[str, Any]:
    voice_names = [voice["id"] for voice in voices]
    drafts: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for project in projects:
        draft, generations = project_to_draft(project, voice_names=voice_names, live_speakers=live_speakers)
        drafts.append(draft)
        history.extend(generations)
    drafts.sort(key=lambda item: (item.get("updatedAt") or ""), reverse=True)
    history.sort(key=lambda item: (item.get("createdAt") or ""), reverse=True)
    return {
        "characters": [],
        "previews": [],
        "drafts": drafts,
        "voices": voices,
        "models": [],
        "history": history,
        "preferences": preferences,
        "runtimePath": None,
        "runtimeBackend": runtime_backend,
        "runtimeVersion": runtime_version,
        "activity": activity,
        "modelActivity": None,
        "aiProviders": [],
        "aiProviderID": "",
    }


def build_assets(
    *,
    projects: list[PodcastProject],
    voices: list[dict[str, Any]],
    voice_durations: dict[str, float],
) -> list[dict[str, Any]]:
    """资源页的一行：生成音频、音色与作品导入的素材。"""
    assets: list[dict[str, Any]] = []
    for project in projects:
        for segment in project.segments:
            if not segment.audio_filename:
                continue
            token = segment_token(project.id, segment.audio_filename)
            assets.append({
                "kind": "segment",
                "id": token,
                "name": _generation_title((segment.speaker or "").strip(), segment.text) or project.title,
                "media": "outputs",
                "fileName": token,
                "projectId": project.id,
                "projectTitle": project.title,
                "projectKind": "story",
                "createdAt": project.updated_at or now_iso(),
                "duration": float(segment.duration_seconds or 0.0),
                "bytes": _estimate_bytes(float(segment.duration_seconds or 0.0)),
                "users": [project.title],
            })
        if project.merged_audio_filename:
            assets.append({
                "kind": "generation",
                "id": project.merged_audio_filename,
                "name": f"{project.title} · 合成",
                "media": "outputs",
                "fileName": project.merged_audio_filename,
                "projectId": project.id,
                "projectTitle": project.title,
                "projectKind": "story",
                "createdAt": project.updated_at or now_iso(),
                "duration": sum(float(seg.duration_seconds or 0.0) for seg in project.segments),
                "bytes": -1,
                "users": [],
            })
    for voice in voices:
        assets.append({
            "kind": "voice",
            "id": voice["id"],
            "name": voice.get("name") or voice["id"],
            "media": "voices",
            "fileName": voice.get("fileName") or "",
            "createdAt": "",
            "duration": voice_durations.get(voice["id"], 0.0),
            "bytes": _estimate_bytes(voice_durations.get(voice["id"], 0.0)),
            "users": [],
        })
    assets.sort(key=lambda item: (item.get("createdAt") or ""), reverse=True)
    return assets


def _estimate_bytes(seconds: float) -> int:
    # 与界面预览一致：按 24 kHz / 16 位 / 单声道 WAV 估算。
    return int(seconds * 48000) + 44
