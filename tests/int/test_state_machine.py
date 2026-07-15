"""Integration tests for ST-LP-004: LiveSession state machine.

All tests call session.transition(Trigger.X) or registry methods.
No test sets session.state directly or hardcodes the transition table.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.live.session import (
    LiveState, LiveSession, LiveSessionRegistry,
    IllegalStateTransition, Trigger, get_legal_transitions,
)
from tts_service.models import PodcastSegment, SegmentSource


def _make_segments(n: int = 3) -> list[PodcastSegment]:
    return [PodcastSegment(index=i, text=f"seg{i}", speaker="A") for i in range(n)]


class LiveStateEnumTests(unittest.TestCase):

    def test_all_states_present(self):
        expected = {"IDLE", "AI_SPEAKING", "RECORDING", "DETECTING",
                     "PAUSED", "FINISHED", "ABANDONED", "ERROR"}
        actual = {s.value for s in LiveState}
        self.assertEqual(actual, expected)


class TriggerTransitionTests(unittest.TestCase):
    """Every test calls transition() and verifies the resulting state."""

    def _make_session(self, state=LiveState.IDLE) -> LiveSession:
        return LiveSession(
            session_id="t1", project_id="p1", cursor=0, state=state,
            segments=_make_segments(), started_at=0.0,
        )

    def test_idle_to_ai_speaking(self):
        s = self._make_session()
        s.transition(Trigger.START_AI)
        self.assertEqual(s.state, LiveState.AI_SPEAKING)

    def test_ai_speaking_to_recording(self):
        """AI_SPEAKING -> RECORDING (next segment is live)."""
        s = self._make_session(LiveState.AI_SPEAKING)
        s.transition(Trigger.START_RECORDING)
        self.assertEqual(s.state, LiveState.RECORDING)

    def test_detecting_to_ai_speaking(self):
        """DETECTING -> AI_SPEAKING (next segment is tts)."""
        s = self._make_session(LiveState.DETECTING)
        s.transition(Trigger.START_AI)
        self.assertEqual(s.state, LiveState.AI_SPEAKING)

    def test_paused_to_recording(self):
        """PAUSED -> RECORDING (resume directly to recording)."""
        s = self._make_session(LiveState.PAUSED)
        s.transition(Trigger.START_RECORDING)
        self.assertEqual(s.state, LiveState.RECORDING)

    def test_ai_speaking_to_detecting(self):
        s = self._make_session(LiveState.AI_SPEAKING)
        s.transition(Trigger.AI_FINISHED)
        self.assertEqual(s.state, LiveState.DETECTING)

    def test_detecting_to_recording(self):
        s = self._make_session(LiveState.DETECTING)
        s.transition(Trigger.START_RECORDING)
        self.assertEqual(s.state, LiveState.RECORDING)

    def test_idle_to_recording(self):
        """First segment is live -> IDLE directly to RECORDING."""
        s = self._make_session(LiveState.IDLE)
        s.transition(Trigger.START_RECORDING)
        self.assertEqual(s.state, LiveState.RECORDING)

    def test_recording_to_detecting(self):
        s = self._make_session(LiveState.RECORDING)
        s.transition(Trigger.RECORDING_DONE)
        self.assertEqual(s.state, LiveState.DETECTING)

    def test_detecting_to_finished(self):
        s = self._make_session(LiveState.DETECTING)
        s.transition(Trigger.ALL_DONE)
        self.assertEqual(s.state, LiveState.FINISHED)
        self.assertIsNotNone(s.finished_at)

    def test_active_to_paused_on_disconnect(self):
        s = self._make_session(LiveState.RECORDING)
        s.transition(Trigger.CLIENT_DISCONNECT)
        self.assertEqual(s.state, LiveState.PAUSED)

    def test_paused_resumes_on_reconnect(self):
        s = self._make_session(LiveState.RECORDING)
        s.transition(Trigger.CLIENT_DISCONNECT)
        s.transition(Trigger.CLIENT_RECONNECT)
        self.assertEqual(s.state, LiveState.RECORDING)

    def test_cancel_from_recording(self):
        s = self._make_session(LiveState.RECORDING)
        s.transition(Trigger.CANCEL)
        self.assertEqual(s.state, LiveState.ABANDONED)

    def test_force_stop_from_recording(self):
        s = self._make_session(LiveState.RECORDING)
        s.transition(Trigger.FORCE_STOP)
        self.assertEqual(s.state, LiveState.FINISHED)

    def test_error_from_any_active(self):
        s = self._make_session(LiveState.AI_SPEAKING)
        s.transition(Trigger.ERROR)
        self.assertEqual(s.state, LiveState.ERROR)


class IllegalTransitionTests(unittest.TestCase):
    """Negative tests: verify IllegalStateTransition is raised."""

    def _make_session(self, state=LiveState.IDLE) -> LiveSession:
        return LiveSession(
            session_id="t1", project_id="p1", cursor=0, state=state,
            segments=_make_segments(), started_at=0.0,
        )

    def test_recording_done_from_idle_raises(self):
        s = self._make_session(LiveState.IDLE)
        with self.assertRaises(IllegalStateTransition):
            s.transition(Trigger.RECORDING_DONE)

    def test_ai_finished_from_recording_raises(self):
        s = self._make_session(LiveState.RECORDING)
        with self.assertRaises(IllegalStateTransition):
            s.transition(Trigger.AI_FINISHED)

    def test_any_trigger_from_finished_raises(self):
        s = self._make_session(LiveState.DETECTING)
        s.transition(Trigger.ALL_DONE)
        with self.assertRaises(IllegalStateTransition):
            s.transition(Trigger.START_AI)

    def test_any_trigger_from_abandoned_raises(self):
        s = self._make_session(LiveState.RECORDING)
        s.transition(Trigger.CANCEL)
        with self.assertRaises(IllegalStateTransition):
            s.transition(Trigger.START_AI)

    def test_any_trigger_from_error_raises(self):
        s = self._make_session(LiveState.RECORDING)
        s.transition(Trigger.ERROR)
        with self.assertRaises(IllegalStateTransition):
            s.transition(Trigger.START_AI)


class LegalTransitionsImportedTests(unittest.TestCase):
    """Import the transition table from production code, don't hardcode."""

    def test_legal_transitions_returned(self):
        table = get_legal_transitions()
        self.assertIsInstance(table, dict)
        self.assertIn(LiveState.IDLE, table)
        self.assertIn(LiveState.AI_SPEAKING, table[LiveState.IDLE])

    def test_terminal_states_have_empty_transitions(self):
        table = get_legal_transitions()
        for terminal in (LiveState.FINISHED, LiveState.ABANDONED, LiveState.ERROR):
            self.assertEqual(table[terminal], set(),
                             f"{terminal} should have no legal transitions")

    def test_every_non_terminal_has_at_least_one_transition(self):
        table = get_legal_transitions()
        for state in (LiveState.IDLE, LiveState.AI_SPEAKING, LiveState.RECORDING,
                      LiveState.DETECTING, LiveState.PAUSED):
            self.assertGreater(len(table[state]), 0,
                               f"{state} should have at least one transition")


class CursorTests(unittest.TestCase):

    def test_cursor_starts_at_zero(self):
        s = LiveSession(session_id="t", project_id="p", cursor=0,
                        state=LiveState.IDLE, segments=_make_segments(5),
                        started_at=0.0)
        self.assertEqual(s.cursor, 0)

    def test_advance_cursor(self):
        s = LiveSession(session_id="t", project_id="p", cursor=0,
                        state=LiveState.IDLE, segments=_make_segments(5),
                        started_at=0.0)
        s.advance_cursor()
        self.assertEqual(s.cursor, 1)
        s.advance_cursor()
        self.assertEqual(s.cursor, 2)

    def test_cursor_clamps_at_segment_count(self):
        s = LiveSession(session_id="t", project_id="p", cursor=4,
                        state=LiveState.IDLE, segments=_make_segments(5),
                        started_at=0.0)
        s.advance_cursor()
        self.assertEqual(s.cursor, 5)
        s.advance_cursor()  # should not go beyond
        self.assertEqual(s.cursor, 5)


class RegistryTests(unittest.TestCase):
    """Tests call start_live_session / stop_live_session / cancel_session / transition."""

    def test_start_creates_session(self):
        reg = LiveSessionRegistry()
        s = reg.start_live_session("p1", _make_segments())
        self.assertEqual(s.state, LiveState.IDLE)
        self.assertIsNotNone(s.session_id)

    def test_get_returns_session(self):
        reg = LiveSessionRegistry()
        s = reg.start_live_session("p1", _make_segments())
        retrieved = reg.get("p1", s.session_id)
        self.assertIs(retrieved, s)

    def test_get_nonexistent_returns_none(self):
        reg = LiveSessionRegistry()
        self.assertIsNone(reg.get("p1", "nope"))

    def test_stop_transitions_to_finished(self):
        reg = LiveSessionRegistry()
        s = reg.start_live_session("p1", _make_segments())
        reg.transition("p1", s.session_id, Trigger.START_AI)
        reg.stop_live_session("p1", s.session_id)
        self.assertEqual(s.state, LiveState.FINISHED)
        self.assertIsNone(reg.get("p1", s.session_id))

    def test_cancel_transitions_to_abandoned(self):
        reg = LiveSessionRegistry()
        s = reg.start_live_session("p1", _make_segments())
        reg.transition("p1", s.session_id, Trigger.START_AI)
        reg.cancel_session("p1", s.session_id)
        self.assertEqual(s.state, LiveState.ABANDONED)
        self.assertIsNone(reg.get("p1", s.session_id))

    def test_registry_transition_applies_trigger(self):
        reg = LiveSessionRegistry()
        s = reg.start_live_session("p1", _make_segments())
        result = reg.transition("p1", s.session_id, Trigger.START_AI)
        self.assertEqual(result, LiveState.AI_SPEAKING)

    def test_terminal_session_rejects_commands(self):
        reg = LiveSessionRegistry()
        s = reg.start_live_session("p1", _make_segments())
        reg.transition("p1", s.session_id, Trigger.START_AI)
        reg.stop_live_session("p1", s.session_id)
        # After stop, session is removed from registry; calling stop again returns None
        self.assertIsNone(reg.stop_live_session("p1", s.session_id))


class CanAcceptCommandTests(unittest.TestCase):

    def _make(self, state):
        return LiveSession(session_id="t", project_id="p", cursor=0,
                           state=state, segments=_make_segments(), started_at=0.0)

    def test_active_states_accept_commands(self):
        for state in (LiveState.IDLE, LiveState.AI_SPEAKING, LiveState.RECORDING,
                      LiveState.DETECTING, LiveState.PAUSED):
            self.assertTrue(self._make(state).can_accept_command(),
                            f"{state} should accept commands")

    def test_terminal_states_reject_commands(self):
        for state in (LiveState.FINISHED, LiveState.ABANDONED, LiveState.ERROR):
            self.assertFalse(self._make(state).can_accept_command(),
                             f"{state} should reject commands")


if __name__ == "__main__":
    unittest.main()