"""End-to-end tests for Live Podcast happy paths.

All tests drive real objects: LiveSessionRegistry, transition(Trigger),
EndDetector, and frame constructors. No test sets session.state directly.
"""

from __future__ import annotations

import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.live.session import (
    LiveState, LiveSessionRegistry, Trigger, IllegalStateTransition,
)
from tts_service.live.end_detector import EndDetector, compute_alignment_ratio
from tts_service.live.ws_protocol import (
    live_state, segment_begin, asr_partial, asr_final, audio_info,
    encode_json_frame,
)
from tts_service.models import PodcastSegment, SegmentSource


def _make_segments(script: list[tuple[str, str]], live_speakers: set[str]) -> list[PodcastSegment]:
    segments = []
    for i, (speaker, text) in enumerate(script):
        source = SegmentSource.LIVE if speaker in live_speakers else SegmentSource.TTS
        segments.append(PodcastSegment(index=i, text=text, speaker=speaker,
                                        source=source, status="pending"))
    return segments


# ── Happy Path 1: Config -> Create -> Start Session ────────────────────────

class E2ESetupAndStartTests(unittest.TestCase):

    def test_setup_creates_session_with_correct_sources(self):
        live_speakers = {"Aaron"}
        segs = _make_segments([
            ("Flora", "大家好"),
            ("Aaron", "我是 Aaron"),
        ], live_speakers)
        self.assertEqual(segs[0].source, SegmentSource.TTS)
        self.assertEqual(segs[1].source, SegmentSource.LIVE)

        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)
        self.assertEqual(session.state, LiveState.IDLE)
        self.assertEqual(session.cursor, 0)

        # First segment is TTS -> transition to AI_SPEAKING
        reg.transition("p1", session.session_id, Trigger.START_AI)
        self.assertEqual(session.state, LiveState.AI_SPEAKING)

    def test_setup_with_live_first_segment(self):
        live_speakers = {"Aaron"}
        segs = _make_segments([("Aaron", "hello")], live_speakers)
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)
        reg.transition("p1", session.session_id, Trigger.START_RECORDING)
        self.assertEqual(session.state, LiveState.RECORDING)


# ── Happy Path 2: AI -> Record -> Detect -> Switch ─────────────────────────

class E2EAIToRecordFlowTests(unittest.TestCase):

    def test_ai_speaking_to_recording_to_detecting_to_ai(self):
        live_speakers = {"Aaron"}
        segs = _make_segments([
            ("Flora", "AI 第一段"),
            ("Aaron", "真人段"),
            ("Flora", "AI 第二段"),
        ], live_speakers)
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)

        # seg 0: TTS -> AI_SPEAKING
        reg.transition("p1", session.session_id, Trigger.START_AI)
        self.assertEqual(session.state, LiveState.AI_SPEAKING)

        # AI finished -> DETECTING
        reg.transition("p1", session.session_id, Trigger.AI_FINISHED)
        self.assertEqual(session.state, LiveState.DETECTING)
        session.advance_cursor()
        self.assertEqual(session.cursor, 1)

        # seg 1: LIVE -> RECORDING
        reg.transition("p1", session.session_id, Trigger.START_RECORDING)
        self.assertEqual(session.state, LiveState.RECORDING)

        # Recording done -> DETECTING
        reg.transition("p1", session.session_id, Trigger.RECORDING_DONE)
        self.assertEqual(session.state, LiveState.DETECTING)
        session.advance_cursor()
        self.assertEqual(session.cursor, 2)

        # seg 2: TTS -> AI_SPEAKING
        reg.transition("p1", session.session_id, Trigger.START_AI)
        self.assertEqual(session.state, LiveState.AI_SPEAKING)

        # Last segment done
        reg.transition("p1", session.session_id, Trigger.AI_FINISHED)
        reg.transition("p1", session.session_id, Trigger.ALL_DONE)
        self.assertEqual(session.state, LiveState.FINISHED)


# ── Happy Path 3: Record + Merge Output ────────────────────────────────────

class E2ERecordMergeTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_mixed_sources(self):
        from tts_service.models import PodcastProject
        from tts_service.podcast_manager import PodcastManager

        project_id = "e2e-merge"
        audio_dir = self.tmpdir / "podcasts" / project_id
        audio_dir.mkdir(parents=True, exist_ok=True)

        def _write_wav(name, sr, dur):
            samples = int(sr * dur)
            t = np.linspace(0, dur, samples, endpoint=False)
            audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
            import soundfile as sf
            sf.write(str(audio_dir / name), audio, sr, format="WAV", subtype="PCM_16")

        _write_wav("seg_0000.wav", 24000, 1.0)
        _write_wav("live_0001.wav", 48000, 1.0)
        _write_wav("seg_0002.wav", 24000, 1.0)

        segs = [
            PodcastSegment(index=0, text="a", speaker="Flora", source=SegmentSource.TTS,
                          status="generated", audio_filename="seg_0000.wav"),
            PodcastSegment(index=1, text="b", speaker="Aaron", source=SegmentSource.LIVE,
                          status="generated", audio_filename="live_0001.wav"),
            PodcastSegment(index=2, text="c", speaker="Flora", source=SegmentSource.TTS,
                          status="generated", audio_filename="seg_0002.wav"),
        ]
        project = PodcastProject(
            id=project_id, title="Test", created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00", segments=segs, gap_seconds=0.5,
        )
        projects_dir = self.tmpdir / "projects"
        projects_dir.mkdir()
        (projects_dir / f"{project_id}.json").write_text(
            project.model_dump_json(indent=2), encoding="utf-8")

        pm = PodcastManager(projects_dir, self.tmpdir)
        filename = pm.merge_project(project_id)
        self.assertTrue((self.tmpdir / filename).exists())


# ── Happy Path 4: Cancel Session ───────────────────────────────────────────

class E2ECancelTests(unittest.TestCase):

    def test_cancel_mid_session(self):
        segs = _make_segments([("Flora", "a"), ("Aaron", "b")], {"Aaron"})
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)
        reg.transition("p1", session.session_id, Trigger.START_AI)
        reg.transition("p1", session.session_id, Trigger.AI_FINISHED)
        reg.transition("p1", session.session_id, Trigger.START_RECORDING)

        # Cancel
        reg.cancel_session("p1", session.session_id)
        self.assertEqual(session.state, LiveState.ABANDONED)
        self.assertIsNone(reg.get("p1", session.session_id))

    def test_cancelled_session_rejects_commands(self):
        segs = _make_segments([("Aaron", "a")], {"Aaron"})
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)
        reg.cancel_session("p1", session.session_id)
        self.assertFalse(session.can_accept_command())


# ── Happy Path 5: Disconnect + Resume ──────────────────────────────────────

class E2EDisconnectResumeTests(unittest.TestCase):

    def test_disconnect_and_reconnect(self):
        segs = _make_segments([("Flora", "a"), ("Aaron", "b")], {"Aaron"})
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)
        reg.transition("p1", session.session_id, Trigger.START_AI)
        reg.transition("p1", session.session_id, Trigger.AI_FINISHED)
        reg.transition("p1", session.session_id, Trigger.START_RECORDING)
        saved_cursor = session.cursor

        # Disconnect
        reg.transition("p1", session.session_id, Trigger.CLIENT_DISCONNECT)
        self.assertEqual(session.state, LiveState.PAUSED)

        # Reconnect -> back to RECORDING
        reg.transition("p1", session.session_id, Trigger.CLIENT_RECONNECT)
        self.assertEqual(session.state, LiveState.RECORDING)
        self.assertEqual(session.cursor, saved_cursor)


# ── Happy Path 6: Redo Live Segment ────────────────────────────────────────

class E2ERedoTests(unittest.TestCase):

    def test_redo_resets_cursor_for_live_segment(self):
        """Redo moves cursor back to a live segment and re-enters RECORDING."""
        segs = _make_segments([
            ("Flora", "a"), ("Aaron", "b"), ("Flora", "c"),
        ], {"Aaron"})
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)

        # Advance past seg 1 (live)
        reg.transition("p1", session.session_id, Trigger.START_AI)
        reg.transition("p1", session.session_id, Trigger.AI_FINISHED)
        session.advance_cursor()  # cursor=1
        reg.transition("p1", session.session_id, Trigger.START_RECORDING)
        reg.transition("p1", session.session_id, Trigger.RECORDING_DONE)
        session.advance_cursor()  # cursor=2

        # Redo seg 1: pause first, then go back to RECORDING via DETECTING
        self.assertEqual(segs[1].source, SegmentSource.LIVE)
        reg.transition("p1", session.session_id, Trigger.CLIENT_DISCONNECT)
        reg.transition("p1", session.session_id, Trigger.CLIENT_RECONNECT)
        # TODO: replace with Trigger.REDO once production redo API exists
        # Move cursor back to the live segment
        session.cursor = 1
        reg.transition("p1", session.session_id, Trigger.START_RECORDING)
        self.assertEqual(session.state, LiveState.RECORDING)


# ── Happy Path 7: EndDetector Drives State Transition ──────────────────────

class E2EEndDetectorDrivenTests(unittest.TestCase):

    def test_end_detector_triggers_recording_done(self):
        """EndDetector detects end-of-speech, session transitions to DETECTING."""
        target = "今天我们聊聊人工智能的发展"
        det = EndDetector(target, end_near_threshold=0.75,
                          end_alignment_threshold=0.95,
                          end_silence_ms=100, debounce_ms=0)

        segs = _make_segments([("Aaron", target)], {"Aaron"})
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)
        reg.transition("p1", session.session_id, Trigger.START_RECORDING)

        # Simulate speech + ASR
        det.update_vad(-20.0, frame_ms=20)
        det.update_asr("今天我们")
        det.update_asr("今天我们聊聊人工智能")  # end_near

        # Full text + silence -> end
        det.update_asr("今天我们聊聊人工智能的发展")
        for _ in range(10):
            det.update_vad(-55.0, frame_ms=20)
        trigger = det.update_asr("今天我们聊聊人工智能的发展")
        self.assertEqual(trigger, "end")

        # EndDetector says "end" -> session transitions
        reg.transition("p1", session.session_id, Trigger.RECORDING_DONE)
        self.assertEqual(session.state, LiveState.DETECTING)


# ── Happy Path 8: Full Multi-Speaker Podcast ──────────────────────────────

class E2EFullPodcastTests(unittest.TestCase):

    def test_full_five_segment_podcast(self):
        live_speakers = {"Aaron"}
        script = [
            ("Flora", "大家好，欢迎收听今天的节目。"),
            ("Aaron", "大家好，我是 Aaron。"),
            ("Flora", "今天我们来聊聊人工智能的发展。"),
            ("Aaron", "这个话题很有意思，我认为 AI 正在改变世界。"),
            ("Flora", "感谢大家的收听，我们下期再见。"),
        ]
        segs = _make_segments(script, live_speakers)
        reg = LiveSessionRegistry()
        session = reg.start_live_session("full", segs)

        # seg 0: Flora TTS
        reg.transition("full", session.session_id, Trigger.START_AI)
        reg.transition("full", session.session_id, Trigger.AI_FINISHED)
        session.advance_cursor()

        # seg 1: Aaron LIVE
        reg.transition("full", session.session_id, Trigger.START_RECORDING)
        reg.transition("full", session.session_id, Trigger.RECORDING_DONE)
        session.advance_cursor()

        # seg 2: Flora TTS
        reg.transition("full", session.session_id, Trigger.START_AI)
        reg.transition("full", session.session_id, Trigger.AI_FINISHED)
        session.advance_cursor()

        # seg 3: Aaron LIVE
        reg.transition("full", session.session_id, Trigger.START_RECORDING)
        reg.transition("full", session.session_id, Trigger.RECORDING_DONE)
        session.advance_cursor()

        # seg 4: Flora TTS
        reg.transition("full", session.session_id, Trigger.START_AI)
        reg.transition("full", session.session_id, Trigger.AI_FINISHED)
        session.advance_cursor()

        # Done
        reg.transition("full", session.session_id, Trigger.ALL_DONE)
        self.assertEqual(session.state, LiveState.FINISHED)
        self.assertEqual(session.cursor, 5)


# ── Happy Path 9: WS Frame Sequence ────────────────────────────────────────

class E2EWSFrameSequenceTests(unittest.TestCase):

    def test_session_drives_expected_state_sequence(self):
        """Drive a real session and collect the state sequence it produces."""
        segs = _make_segments([
            ("Flora", "大家好"), ("Aaron", "我是 Aaron"), ("Flora", "下一段"),
        ], {"Aaron"})
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)

        states = []
        # seg 0: TTS
        states.append(reg.transition("p1", session.session_id, Trigger.START_AI))
        states.append(reg.transition("p1", session.session_id, Trigger.AI_FINISHED))
        session.advance_cursor()
        # seg 1: LIVE
        states.append(reg.transition("p1", session.session_id, Trigger.START_RECORDING))
        states.append(reg.transition("p1", session.session_id, Trigger.RECORDING_DONE))
        session.advance_cursor()
        # seg 2: TTS
        states.append(reg.transition("p1", session.session_id, Trigger.START_AI))
        states.append(reg.transition("p1", session.session_id, Trigger.AI_FINISHED))
        states.append(reg.transition("p1", session.session_id, Trigger.ALL_DONE))

        self.assertEqual(states, [
            LiveState.AI_SPEAKING, LiveState.DETECTING,
            LiveState.RECORDING, LiveState.DETECTING,
            LiveState.AI_SPEAKING, LiveState.DETECTING,
            LiveState.FINISHED,
        ])

    def test_state_frames_constructed_for_each_transition(self):
        """Each state in the session produces a valid live_state() frame."""
        segs = _make_segments([("Flora", "a"), ("Aaron", "b")], {"Aaron"})
        reg = LiveSessionRegistry()
        session = reg.start_live_session("p1", segs)

        state = reg.transition("p1", session.session_id, Trigger.START_AI)
        frame = live_state(state.value)
        self.assertEqual(frame["type"], "state")
        self.assertEqual(frame["state"], "AI_SPEAKING")

    def test_frames_encode_to_json(self):
        """All constructed frames can be JSON-encoded."""
        frames = [
            audio_info(24000),
            live_state("AI_SPEAKING"),
            segment_begin(0, "tts", "Flora", "hello"),
            asr_partial("hi", 100),
        ]
        for f in frames:
            encoded = encode_json_frame(f)
            self.assertIsInstance(encoded, str)


# ── Happy Path 10: Live-only Mode ──────────────────────────────────────────

class E2ELiveOnlyTests(unittest.TestCase):

    def test_live_only_skips_tts_in_merge(self):
        """In live-only mode, TTS segments are not generated and merge
        substitutes silence for them."""
        from tts_service.models import PodcastProject
        from tts_service.podcast_manager import PodcastManager
        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        projects_dir = tmpdir / "projects"
        projects_dir.mkdir()
        outputs_dir = tmpdir / "outputs"
        outputs_dir.mkdir()
        audio_dir = outputs_dir / "podcasts" / "p1"
        audio_dir.mkdir(parents=True)

        # Create only the live segment's WAV (TTS segments have no audio)
        import soundfile as sf
        import numpy as np
        live_samples = np.zeros(24000, dtype=np.float32)  # 1s silence
        sf.write(str(audio_dir / "live_0001.wav"), live_samples, 24000)

        segs = [
            PodcastSegment(index=0, text="tts seg", speaker="Flora",
                          source=SegmentSource.TTS, status="pending"),
            PodcastSegment(index=1, text="live seg", speaker="Aaron",
                          source=SegmentSource.LIVE, status="generated",
                          audio_filename="live_0001.wav"),
            PodcastSegment(index=2, text="tts seg 2", speaker="Flora",
                          source=SegmentSource.TTS, status="pending"),
        ]
        project = PodcastProject(
            id="p1", title="Test", created_at="x", updated_at="x",
            segments=segs, gap_seconds=0.5,
        )
        (projects_dir / "p1.json").write_text(
            project.model_dump_json(indent=2), encoding="utf-8")

        pm = PodcastManager(projects_dir, outputs_dir)
        filename = pm.merge_project("p1")
        merged_path = outputs_dir / filename
        self.assertTrue(merged_path.exists())
        info = sf.info(str(merged_path))
        # At least the 1s live segment
        self.assertGreater(info.duration, 0.5)


if __name__ == "__main__":
    unittest.main()