"""Live session state machine and registry.

Trigger-based state machine: callers pass semantic trigger strings,
not raw target states. This encapsulates transition legality inside
the session, not in the caller.

Full spec: .project/specs/spec.md SPEC-004.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from tts_service.models import PodcastSegment


class LiveState(str, Enum):
    IDLE = "IDLE"
    AI_SPEAKING = "AI_SPEAKING"
    RECORDING = "RECORDING"
    DETECTING = "DETECTING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    ABANDONED = "ABANDONED"
    ERROR = "ERROR"


class Trigger(str, Enum):
    """Semantic triggers that drive state transitions."""
    START_AI = "start_ai"               # IDLE/DETECTING -> AI_SPEAKING
    AI_FINISHED = "ai_finished"          # AI_SPEAKING -> DETECTING
    START_RECORDING = "start_recording"  # AI_SPEAKING/DETECTING -> RECORDING
    RECORDING_DONE = "recording_done"    # RECORDING -> DETECTING
    ALL_DONE = "all_done"                # DETECTING -> FINISHED
    CLIENT_DISCONNECT = "client_disconnect"  # active -> PAUSED
    CLIENT_RECONNECT = "client_reconnect"    # PAUSED -> previous active state
    CANCEL = "cancel"                    # any active -> ABANDONED
    FORCE_STOP = "force_stop"            # any active -> FINISHED
    ERROR = "error"                      # any -> ERROR


class IllegalStateTransition(Exception):
    """Raised when a trigger is not valid in the current state."""


# Trigger -> (set of valid from-states) -> target state
_TRIGGER_TABLE: Dict[Trigger, Dict[frozenset, LiveState]] = {
    Trigger.START_AI: {
        frozenset({LiveState.IDLE, LiveState.DETECTING, LiveState.PAUSED}): LiveState.AI_SPEAKING,
    },
    Trigger.AI_FINISHED: {
        frozenset({LiveState.AI_SPEAKING}): LiveState.DETECTING,
    },
    Trigger.START_RECORDING: {
        frozenset({LiveState.IDLE, LiveState.AI_SPEAKING, LiveState.DETECTING, LiveState.PAUSED}): LiveState.RECORDING,
    },
    Trigger.RECORDING_DONE: {
        frozenset({LiveState.RECORDING}): LiveState.DETECTING,
    },
    Trigger.ALL_DONE: {
        frozenset({LiveState.DETECTING}): LiveState.FINISHED,
    },
    Trigger.CLIENT_DISCONNECT: {
        frozenset({LiveState.AI_SPEAKING, LiveState.RECORDING, LiveState.DETECTING}): LiveState.PAUSED,
    },
    Trigger.CLIENT_RECONNECT: {
        frozenset({LiveState.PAUSED}): LiveState.AI_SPEAKING,  # simplified: resume to AI_SPEAKING
    },
    Trigger.CANCEL: {
        frozenset({LiveState.IDLE, LiveState.AI_SPEAKING, LiveState.RECORDING,
                   LiveState.DETECTING, LiveState.PAUSED}): LiveState.ABANDONED,
    },
    Trigger.FORCE_STOP: {
        frozenset({LiveState.IDLE, LiveState.AI_SPEAKING, LiveState.RECORDING,
                   LiveState.DETECTING, LiveState.PAUSED}): LiveState.FINISHED,
    },
    Trigger.ERROR: {
        frozenset({LiveState.IDLE, LiveState.AI_SPEAKING, LiveState.RECORDING,
                   LiveState.DETECTING, LiveState.PAUSED}): LiveState.ERROR,
    },
}


def _resolve_target(trigger: Trigger, current: LiveState) -> LiveState:
    """Return the target state for a trigger from the current state."""
    rules = _TRIGGER_TABLE.get(trigger)
    if rules is None:
        raise IllegalStateTransition(f"Unknown trigger: {trigger}")
    for valid_from, target in rules.items():
        if current in valid_from:
            return target
    raise IllegalStateTransition(
        f"Trigger {trigger.value} not valid in state {current.value}"
    )


def get_legal_transitions() -> Dict[LiveState, set]:
    """Return the state->set-of-valid-next-states map derived from the trigger table."""
    result: Dict[LiveState, set] = {s: set() for s in LiveState}
    for trigger, rules in _TRIGGER_TABLE.items():
        for valid_from, target in rules.items():
            for src in valid_from:
                result[src].add(target)
    return result


@dataclass
class LiveSession:
    session_id: str
    project_id: str
    cursor: int
    state: LiveState
    segments: List[PodcastSegment]
    audio_buffer: Dict[int, bytes] = field(default_factory=dict)
    alignment_score: float = 0.0
    last_asr_text: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    _pre_pause_state: Optional[LiveState] = None

    def transition(self, trigger: Trigger) -> LiveState:
        """Apply a semantic trigger to advance the state machine.

        Raises IllegalStateTransition if the trigger is not valid
        in the current state.
        """
        if self.state in (LiveState.FINISHED, LiveState.ABANDONED, LiveState.ERROR):
            raise IllegalStateTransition(
                f"Cannot trigger {trigger.value} from terminal state {self.state.value}"
            )
        target = _resolve_target(trigger, self.state)
        if trigger == Trigger.CLIENT_DISCONNECT:
            self._pre_pause_state = self.state
        if trigger == Trigger.CLIENT_RECONNECT and self._pre_pause_state is not None:
            target = self._pre_pause_state
            self._pre_pause_state = None
        self.state = target
        if target in (LiveState.FINISHED, LiveState.ABANDONED, LiveState.ERROR):
            self.finished_at = time.time()
        return target

    def can_accept_command(self) -> bool:
        """Return False if session is in a terminal state."""
        return self.state not in (LiveState.FINISHED, LiveState.ABANDONED, LiveState.ERROR)

    def advance_cursor(self) -> int:
        """Advance cursor to the next segment. Returns the new cursor value."""
        if self.cursor < len(self.segments):
            self.cursor += 1
        return self.cursor


class LiveSessionRegistry:
    """Registry of live sessions."""

    def __init__(self):
        self._sessions: Dict[tuple[str, str], LiveSession] = {}
        self._asr = None

    def start_live_session(
        self,
        project_id: str,
        segments: List[PodcastSegment],
    ) -> LiveSession:
        """Create a new live session in IDLE state."""
        session = LiveSession(
            session_id=uuid.uuid4().hex[:12],
            project_id=project_id,
            cursor=0,
            state=LiveState.IDLE,
            segments=segments,
            started_at=time.time(),
        )
        self._sessions[(project_id, session.session_id)] = session
        return session

    def get(self, project_id: str, session_id: str) -> Optional[LiveSession]:
        return self._sessions.get((project_id, session_id))

    def stop_live_session(self, project_id: str, session_id: str) -> Optional[LiveSession]:
        """Gracefully stop a session -> FINISHED, then remove from registry."""
        session = self.get(project_id, session_id)
        if session is None:
            return None
        if not session.can_accept_command():
            return None
        session.transition(Trigger.FORCE_STOP)
        self._sessions.pop((project_id, session_id), None)
        return session

    def cancel_session(self, project_id: str, session_id: str) -> Optional[LiveSession]:
        """Cancel a session -> ABANDONED."""
        session = self.get(project_id, session_id)
        if session is None:
            return None
        session.transition(Trigger.CANCEL)
        self._sessions.pop((project_id, session_id), None)
        return session

    def transition(self, project_id: str, session_id: str, trigger: Trigger) -> LiveState:
        """Apply a trigger to a session."""
        session = self.get(project_id, session_id)
        if session is None:
            raise KeyError(f"Session {project_id}/{session_id} not found")
        return session.transition(trigger)

    def get_asr(self):
        """Return the shared ASR instance (lazy-init)."""
        if self._asr is None:
            from tts_service.live.asr_engine import EmbeddedASR, ASRConfig
            self._asr = EmbeddedASR(ASRConfig())
        return self._asr
