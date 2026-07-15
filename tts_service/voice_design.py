"""Voice Design markup preprocessor.

Parses optional YAML frontmatter and optional per-line ``Speaker[k=v, ...]:`` brackets
on top of the normal ``Speaker: text`` dialogue format. Emits a cleaned body, a
list of per-turn instruction strings, and validation errors.

This module is intentionally self-contained: stdlib + PyYAML only. No import from
the engines layer or from ``config``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import yaml


PITCH_VALUES = ("very-low", "low", "moderate", "high", "very-high")
STYLE_VALUES = ("whisper",)
DESIGN_KEYS = ("pitch", "style", "accent")

# OmniVoice's strict vocabulary forbids mixing English and Chinese in the same
# instruct string. We detect language per value and reject mixes at parse time.
# - pitch + style are always English (we control their tokens).
# - accent is free-form; users may write either e.g. "british" or "四川话".

_FRONTMATTER_FENCE_RE = re.compile(r"^---\s*$")
# Speaker-tag line with bracketed attrs. Speaker name is up to 80 chars and must
# not contain ':' '\n' or '['. Match anchors on the original line (no leading WS).
_BRACKET_LINE_RE = re.compile(r"^[^:\n\[]{1,80}\[[^\]]+\]\s*:")
# Plain speaker-tag line (no brackets), used by the body parser.
_SPEAKER_LINE_RE = re.compile(r"^(?P<speaker>[^:\n\[]{1,80})(?:\[(?P<attrs>[^\]]*)\])?:\s?(?P<text>.*)$")


class VoiceDesignError(ValueError):
    """Validation error raised by :func:`parse`.

    ``line`` and ``column`` are 1-based positions in the original input text.
    """

    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"{message} (line {line}, column {column})")
        self.line = line
        self.column = column


@dataclass(slots=True)
class TurnDesign:
    speaker: str
    instructions: Optional[str]


@dataclass(slots=True)
class ParseResult:
    cleaned_text: str
    turn_designs: list[TurnDesign] = field(default_factory=list)
    has_markup: bool = False


def has_markup(text: str) -> bool:
    """Cheap regex-only detection — does not run the full parser."""
    if not text:
        return False
    for line in text.splitlines():
        if _FRONTMATTER_FENCE_RE.match(line):
            return True
        if _BRACKET_LINE_RE.match(line):
            return True
    return False


def parse(text: str) -> ParseResult:
    """Parse frontmatter + body. Raise :class:`VoiceDesignError` on validation failure."""
    if not text:
        return ParseResult(cleaned_text="")

    lines = text.splitlines(keepends=False)
    fm_defaults, fm_speakers, fm_present, body_start = _extract_frontmatter(lines)

    body_lines = lines[body_start:]
    cleaned_body_lines, turn_designs, brackets_seen = _parse_body(
        body_lines,
        body_start,
        fm_defaults,
        fm_speakers,
    )

    # Preserve trailing newline if the original input had one.
    cleaned_text = "\n".join(cleaned_body_lines)
    if text.endswith("\n") and cleaned_text and not cleaned_text.endswith("\n"):
        cleaned_text += "\n"

    return ParseResult(
        cleaned_text=cleaned_text,
        turn_designs=turn_designs,
        has_markup=fm_present or brackets_seen,
    )


# ── Frontmatter ─────────────────────────────────────────────────────────────


def _extract_frontmatter(
    lines: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, str]], bool, int]:
    """Return (defaults, speakers, present, body_start_line_index)."""
    if not lines or not _FRONTMATTER_FENCE_RE.match(lines[0]):
        return {}, {}, False, 0

    end_idx = -1
    for i in range(1, len(lines)):
        if _FRONTMATTER_FENCE_RE.match(lines[i]):
            end_idx = i
            break
    if end_idx == -1:
        raise VoiceDesignError("unterminated frontmatter block", line=1, column=1)

    yaml_text = "\n".join(lines[1:end_idx])
    try:
        data = yaml.safe_load(yaml_text) if yaml_text.strip() else {}
    except yaml.YAMLError as exc:
        raise VoiceDesignError(f"invalid YAML in frontmatter: {exc}", line=2, column=1) from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise VoiceDesignError("frontmatter must be a YAML mapping", line=2, column=1)

    vd = data.get("voice-design")
    if vd is None or vd == {}:
        # Skip past frontmatter; ignore empty body.
        return {}, {}, True, end_idx + 1
    if not isinstance(vd, dict):
        raise VoiceDesignError("'voice-design' must be a mapping", line=2, column=1)

    allowed = {"default", "speakers"}
    extra = set(vd.keys()) - allowed
    if extra:
        raise VoiceDesignError(
            f"unknown key in voice-design: {sorted(extra)[0]}", line=2, column=1
        )

    defaults_raw = vd.get("default") or {}
    speakers_raw = vd.get("speakers") or {}

    if not isinstance(defaults_raw, dict):
        raise VoiceDesignError("'voice-design.default' must be a mapping", line=2, column=1)
    if not isinstance(speakers_raw, dict):
        raise VoiceDesignError("'voice-design.speakers' must be a mapping", line=2, column=1)

    defaults = _validate_attr_map(defaults_raw, where="voice-design.default", line=2)
    speakers: dict[str, dict[str, str]] = {}
    for name, attrs in speakers_raw.items():
        if not isinstance(attrs, dict):
            raise VoiceDesignError(
                f"'voice-design.speakers.{name}' must be a mapping", line=2, column=1
            )
        speakers[str(name)] = _validate_attr_map(
            attrs, where=f"voice-design.speakers.{name}", line=2
        )

    return defaults, speakers, True, end_idx + 1


def _validate_attr_map(raw: dict, where: str, line: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in raw.items():
        key_s = str(key)
        if key_s not in DESIGN_KEYS:
            raise VoiceDesignError(f"unknown key '{key_s}' in {where}", line=line, column=1)
        value_s = str(value)
        _validate_value(key_s, value_s, line=line, column=1)
        out[key_s] = value_s
    return out


def _validate_value(key: str, value: str, line: int, column: int) -> None:
    if key == "pitch" and value not in PITCH_VALUES:
        raise VoiceDesignError(
            f"unknown pitch value '{value}' (allowed: {', '.join(PITCH_VALUES)})",
            line=line, column=column,
        )
    if key == "style" and value not in STYLE_VALUES:
        raise VoiceDesignError(
            f"unknown style value '{value}' (allowed: {', '.join(STYLE_VALUES)})",
            line=line, column=column,
        )
    # accent: free-form


# ── Body ────────────────────────────────────────────────────────────────────


def _parse_body(
    body_lines: list[str],
    body_start_index: int,
    fm_defaults: dict[str, str],
    fm_speakers: dict[str, dict[str, str]],
) -> tuple[list[str], list[TurnDesign], bool]:
    cleaned: list[str] = []
    turns: list[TurnDesign] = []
    brackets_seen = False

    current_speaker: Optional[str] = None
    current_attrs: dict[str, str] | None = None
    current_turn_open = False

    for offset, line in enumerate(body_lines):
        line_no = body_start_index + offset + 1  # 1-based
        stripped = line.strip()

        if not stripped:
            cleaned.append(line)
            current_turn_open = False
            continue

        match = _SPEAKER_LINE_RE.match(line)
        if match and match.group("speaker") and not match.group("speaker").endswith(" "):
            speaker = match.group("speaker").strip()
            attrs_raw = match.group("attrs")
            text_part = match.group("text")

            line_attrs: dict[str, str] = {}
            if attrs_raw is not None:
                brackets_seen = True
                # Column of the '[' character — find it on the original line.
                bracket_col = line.index("[") + 1
                line_attrs = _parse_attr_list(attrs_raw, line=line_no, column=bracket_col)

            merged = _merge_attrs(fm_defaults, fm_speakers.get(speaker, {}), line_attrs)
            instructions = _serialize(merged)
            turns.append(TurnDesign(speaker=speaker, instructions=instructions))

            current_speaker = speaker
            current_attrs = merged
            current_turn_open = True
            cleaned.append(f"{speaker}: {text_part}")
            continue

        # Continuation line: inherits current speaker + attrs.
        cleaned.append(line)
        if current_turn_open and current_speaker is not None:
            # Already accounted for in the open turn; nothing to do.
            _ = current_attrs

    return cleaned, turns, brackets_seen


def _parse_attr_list(raw: str, line: int, column: int) -> dict[str, str]:
    """Parse the inside of ``[ ... ]`` into a validated key→value map."""
    out: dict[str, str] = {}
    parts = [p for p in raw.split(",")]
    for part in parts:
        item = part.strip()
        if not item:
            raise VoiceDesignError("empty attribute in []", line=line, column=column)
        if "=" not in item:
            raise VoiceDesignError(
                f"attribute '{item}' is missing '='", line=line, column=column
            )
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise VoiceDesignError("attribute key is empty", line=line, column=column)
        if key not in DESIGN_KEYS:
            raise VoiceDesignError(
                f"unknown attribute key '{key}'", line=line, column=column
            )
        if key in out:
            raise VoiceDesignError(
                f"duplicate attribute key '{key}'", line=line, column=column
            )
        _validate_value(key, value, line=line, column=column)
        out[key] = value
    return out


def _merge_attrs(
    defaults: dict[str, str],
    speaker_attrs: dict[str, str],
    line_attrs: dict[str, str],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update(defaults)
    merged.update(speaker_attrs)
    merged.update(line_attrs)
    return merged


def _serialize(merged: dict[str, str]) -> Optional[str]:
    if not merged:
        return None
    # Emit OmniVoice-compatible tokens in fixed order: pitch, style, accent.
    # English values get explicit "pitch"/"accent" suffixes; the style value
    # ("whisper") is bare. Chinese accent values are emitted as-is (no suffix).
    # Mixing English and Chinese values in one instruct is rejected, since
    # OmniVoice requires a single-language string per call.
    tokens: list[tuple[str, bool]] = []  # (token, is_ascii)
    if "pitch" in merged:
        raw = merged["pitch"]
        pitch_token = raw.replace("-", " ") + " pitch"  # e.g. "very-low" -> "very low pitch"
        tokens.append((pitch_token, True))
    if "style" in merged:
        tokens.append((merged["style"], True))  # only "whisper" is supported by OmniVoice
    if "accent" in merged:
        raw = merged["accent"]
        is_ascii = raw.isascii()
        if is_ascii:
            tokens.append((f"{raw} accent", True))
        else:
            tokens.append((raw, False))

    en_count = sum(1 for _, a in tokens if a)
    cn_count = sum(1 for _, a in tokens if not a)
    if en_count and cn_count:
        # Caller validation should have prevented this, but be defensive.
        raise VoiceDesignError(
            "voice-design instruct cannot mix English and Chinese values "
            "(e.g. 'pitch=low' + 'accent=四川话'); pick one language per turn",
            line=1, column=1,
        )

    if cn_count:
        # All values are Chinese -> use full-width comma per OmniVoice convention.
        return "，".join(t for t, _ in tokens)
    return ", ".join(t for t, _ in tokens)
