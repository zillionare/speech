"""Split long text into sentence-boundary segments for chunked TTS generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Chinese sentence-ending punctuation
_SENTENCE_END_RE = re.compile(r"[。！？…]+")
# Clause boundaries for secondary splitting
_CLAUSE_RE = re.compile(r"[，；、]")

_DEFAULT_MAX_CHARS = 200


@dataclass(slots=True)
class TextSegment:
    text: str
    speaker: Optional[str] = None


def _split_by_sentences(text: str) -> list[str]:
    """Split text by sentence-ending punctuation, keeping delimiters."""
    parts = _SENTENCE_END_RE.split(text)
    delimiters = _SENTENCE_END_RE.findall(text)
    segments: list[str] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            if i < len(delimiters):
                segments.append(part + delimiters[i])
            else:
                segments.append(part)
    return segments


def _merge_short_segments(segments: list[str], max_chars: int) -> list[str]:
    """Merge consecutive short segments until max_chars is reached."""
    merged: list[str] = []
    current = ""
    for seg in segments:
        if len(current) + len(seg) <= max_chars:
            current += seg
        else:
            if current:
                merged.append(current)
            current = seg
    if current:
        merged.append(current)
    return merged


def segment_long_text(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    """Segment long text by sentence boundaries.

    Splits by sentence endings first, then merges short sentences.
    If a single sentence exceeds max_chars, splits by clause boundaries.
    """
    sentences = _split_by_sentences(text)
    result: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            result.append(sent)
            continue
        # Try clause-level split
        clauses = _CLAUSE_RE.split(sent)
        current = ""
        for clause in clauses:
            if len(current) + len(clause) <= max_chars:
                current += clause
            else:
                if current:
                    result.append(current)
                current = clause
        if current:
            result.append(current)
    return _merge_short_segments(result, max_chars)


_DIALOGUE_LINE_RE = re.compile(r"^([^:\n]{1,80}):\s*(.+)$")


def segment_dialogue(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[TextSegment]:
    """Parse dialogue with Speaker: prefixes and segment each speaker's text.

    Preserves the same speaker-inheritance rules as the engine's dialogue parser.
    """
    paragraphs: list[list[str]] = []
    current_para: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            current_para.append(line.rstrip())
        else:
            if current_para:
                paragraphs.append(current_para)
                current_para = []
    if current_para:
        paragraphs.append(current_para)

    if not paragraphs:
        raise ValueError("Input text cannot be empty")

    # Determine speaker for each paragraph
    para_speaker: list[Optional[str]] = []
    para_texts: list[list[str]] = []
    for para in paragraphs:
        first = para[0].strip()
        match = _DIALOGUE_LINE_RE.match(first)
        if match:
            para_speaker.append(match.group(1).strip())
            para_texts.append([match.group(2).strip()] + [l.strip() for l in para[1:]])
        else:
            para_speaker.append(None)
            para_texts.append([l.strip() for l in para])

    # Propagate speakers backward for narration paragraphs
    for i in range(len(para_speaker)):
        if para_speaker[i] is None:
            for j in range(i - 1, -1, -1):
                if para_speaker[j] is not None:
                    para_speaker[i] = para_speaker[j]
                    break

    # Flatten to segments with speakers
    segments: list[TextSegment] = []
    for speaker, texts in zip(para_speaker, para_texts):
        full_text = " ".join(t for t in texts if t)
        if not full_text:
            continue
        if len(full_text) <= max_chars:
            segments.append(TextSegment(text=full_text, speaker=speaker))
        else:
            sub_segments = segment_long_text(full_text, max_chars)
            for sub in sub_segments:
                segments.append(TextSegment(text=sub, speaker=speaker))

    return segments
