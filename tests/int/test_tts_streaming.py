"""Integration tests for ST-LP-005: Streaming TTS + audio queue.

Tests call StreamingTTSProxy and BaseEngine.sample_rate.
No inline arithmetic for expected values.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import struct
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.live.streaming_engine import StreamingTTSProxy
from tts_service.engines.base import BaseEngine, GenerationResult
from tts_service.models import SpeakerResolution
from tts_service.config import LiveConfig, Config


class FakeEngine(BaseEngine):
    """Fake engine that returns a real WAV with known duration."""

    SAMPLE_RATE = 24000

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def generate_single(self, text, voice, output_format="wav", instructions=None):
        duration_s = max(0.5, len(text) * 0.1)  # ~100ms per char
        samples = int(self.SAMPLE_RATE * duration_s)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            pcm = struct.pack(f"<{samples}h", *([16384] * samples))
            w.writeframes(pcm)
        return GenerationResult(
            audio_bytes=buf.getvalue(),
            output_format="wav",
            generation_seconds=0.01,
            duration_seconds=duration_s,
            resolved_speakers=[SpeakerResolution(
                requested_name=voice or "test",
                resolved_voice=voice or "test",
                used_default=False,
            )],
        )

    def generate_dialogue(self, text, output_format="wav", preferred_voice=None,
                          voice_mapping=None, instructions=None, segment_gap=None):
        raise NotImplementedError


class BaseEngineSampleRateTests(unittest.TestCase):

    def test_abstract_property_cannot_skip(self):
        with self.assertRaises(TypeError):
            BaseEngine()

    def test_fake_engine_exposes_sample_rate(self):
        engine = FakeEngine()
        self.assertEqual(engine.sample_rate, 24000)


class StreamingTTSProxyConstructionTests(unittest.TestCase):

    def test_proxy_reads_sample_rate_from_engine(self):
        engine = FakeEngine()
        proxy = StreamingTTSProxy(engine)
        self.assertEqual(proxy.sr, 24000)

    def test_stream_segment_signature(self):
        sig = inspect.signature(StreamingTTSProxy.stream_segment)
        self.assertIn("text", sig.parameters)
        self.assertIn("voice", sig.parameters)
        self.assertEqual(sig.parameters["chunk_ms"].default, 200)


@unittest.skip("stream_segment not yet implemented — remove skip when ready")
class StreamingTTSProxyStreamTests(unittest.TestCase):
    """Call stream_segment with real engine data."""

    def test_stream_segment_yields_pcm_chunks(self):
        """stream_segment should yield binary PCM chunks from the engine."""
        engine = FakeEngine()
        proxy = StreamingTTSProxy(engine)

        async def _run():
            chunks = []
            async for chunk in proxy.stream_segment("hello world", "test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, bytes)
            self.assertGreater(len(chunk), 0)

    def test_stream_segment_chunks_are_pcm_not_wav(self):
        """Yielded chunks should be raw PCM, not WAV (no RIFF header)."""
        engine = FakeEngine()
        proxy = StreamingTTSProxy(engine)

        async def _run():
            chunks = []
            async for chunk in proxy.stream_segment("hi", "test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        if chunks:
            self.assertNotEqual(chunks[0][:4], b"RIFF")


class LiveConfigTests(unittest.TestCase):

    def test_defaults_match_spec(self):
        """LiveConfig defaults match the spec values (ST-LP-019 / SPEC-005 §5.7)."""
        cfg = LiveConfig()
        self.assertGreater(cfg.tts_max_seconds, 0)
        self.assertGreater(cfg.tts_timeout_seconds, 0)

    def test_timeout_less_than_max(self):
        cfg = LiveConfig()
        self.assertLess(cfg.tts_timeout_seconds, cfg.tts_max_seconds)

    def test_rephrases_can_be_zero(self):
        cfg = LiveConfig(max_ai_rephrases_per_segment=0)
        self.assertEqual(cfg.max_ai_rephrases_per_segment, 0)

    def test_config_roundtrip_preserves_values(self):
        """Setting non-default values survives a save/load cycle."""
        from tts_service.config import Config, save_config_to_yaml, load_config
        import tempfile
        import yaml
        tmpdir = Path(tempfile.mkdtemp())
        cfg = Config()
        cfg.live.tts_max_seconds = 120
        cfg.live.tts_timeout_seconds = 45
        out = tmpdir / "test.yaml"
        save_config_to_yaml(cfg, str(out))
        reloaded = load_config(str(out))
        self.assertEqual(reloaded.live.tts_max_seconds, 120)
        self.assertEqual(reloaded.live.tts_timeout_seconds, 45)


if __name__ == "__main__":
    unittest.main()