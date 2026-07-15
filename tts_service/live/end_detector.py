"""End-of-speech detector for live recording.

Full implementation of LCS alignment, VAD, trigger state machine, and debounce.
Tests must import and call these functions, never reimplement.

Full spec: .project/specs/spec.md SPEC-008.
"""

from __future__ import annotations

import time
from typing import Optional


def normalize_text(text: str) -> str:
    """Normalize text for alignment comparison.

    - Strip whitespace
    - Convert traditional Chinese to simplified (zhconv, optional)
    - Remove punctuation
    - Lowercase
    """
    import re
    try:
        import zhconv
        text = zhconv.convert(text, "zh-cn")
    except ImportError:
        pass
    text = re.sub(r"[^\w]", "", text)
    return text.strip().lower()


def longest_common_subsequence_length(a: str, b: str) -> int:
    """Compute the length of the longest common subsequence of two strings."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use rolling array for O(min(m,n)) space
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


def compute_alignment_ratio(asr_text: str, target_text: str) -> float:
    """Compute alignment ratio using LCS.

    ratio = len(LCS(normalize(asr), normalize(target))) / max(1, len(normalize(target)))
    """
    norm_asr = normalize_text(asr_text)
    norm_target = normalize_text(target_text)
    if not norm_target:
        return 0.0
    lcs_len = longest_common_subsequence_length(norm_asr, norm_target)
    return lcs_len / len(norm_target)


class EndDetector:
    """Detects when a live speaker has finished speaking.

    Call update_vad() with each audio frame's dBFS.
    Call update_asr() with each ASR partial result.
    Call on_speech_start() / on_speech_end() for lifecycle events.
    """

    def __init__(
        self,
        target_text: str,
        silence_db_threshold: float = -45.0,
        end_near_threshold: float = 0.85,
        end_alignment_threshold: float = 0.98,
        end_silence_ms: int = 300,
        force_end_silence_ms: int = 4000,
        silence_only_end_ms: int = 800,
        debounce_ms: int = 200,
    ):
        self.target = target_text
        self.normalized_target = normalize_text(target_text)
        self.silence_db_threshold = silence_db_threshold
        self.end_near_threshold = end_near_threshold
        self.end_alignment_threshold = end_alignment_threshold
        self.end_silence_ms = end_silence_ms
        self.force_end_silence_ms = force_end_silence_ms
        self.silence_only_end_ms = silence_only_end_ms
        self.debounce_ms = debounce_ms

        self.silence_ms = 0
        self.is_speaking = False
        self.alignment_ratio = 0.0
        self.last_trigger: Optional[str] = None
        self.last_trigger_time: float = 0.0
        self._end_near_triggered = False

    def reset(self) -> None:
        """Reset all state for a new recording segment."""
        self.silence_ms = 0
        self.is_speaking = False
        self.alignment_ratio = 0.0
        self.last_trigger = None
        self.last_trigger_time = 0.0
        self._end_near_triggered = False

    def on_speech_start(self) -> None:
        """Called when VAD detects speech onset."""
        self.is_speaking = True
        self.silence_ms = 0

    def on_speech_end(self) -> None:
        """Called when VAD detects speech offset."""
        self.is_speaking = False

    def update_vad(self, dbfs: float, frame_ms: int = 20) -> None:
        """Update VAD state with a frame's dBFS level."""
        if dbfs < self.silence_db_threshold:
            self.silence_ms += frame_ms
            if self.is_speaking and self.silence_ms > 50:
                self.on_speech_end()
        else:
            if not self.is_speaking:
                self.on_speech_start()
            self.silence_ms = 0

    def update_asr(self, asr_text: str) -> Optional[str]:
        """Update alignment with ASR text and check trigger conditions.

        Returns one of: 'end_near', 'end', 'user_skipped', or None.
        """
        self.alignment_ratio = compute_alignment_ratio(asr_text, self.target)

        now = time.time()
        if self.last_trigger and now - self.last_trigger_time < self.debounce_ms / 1000.0:
            return None

        # Stuck / force end (highest priority)
        if self.silence_ms >= self.force_end_silence_ms:
            self.last_trigger = "user_skipped"
            self.last_trigger_time = now
            return "user_skipped"

        # End trigger: alignment + silence OR long silence alone
        if (self.alignment_ratio >= self.end_alignment_threshold
                and self.silence_ms >= self.end_silence_ms):
            self.last_trigger = "end"
            self.last_trigger_time = now
            return "end"

        if self.silence_ms >= self.silence_only_end_ms:
            self.last_trigger = "end"
            self.last_trigger_time = now
            return "end"

        # End-near trigger for prefetch
        if self.alignment_ratio >= self.end_near_threshold and not self._end_near_triggered:
            self._end_near_triggered = True
            self.last_trigger = "end_near"
            self.last_trigger_time = now
            return "end_near"

        return None
