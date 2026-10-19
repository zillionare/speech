"""Tests for MiniMax provider configuration and API payloads."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tts_service.config import Config, load_config, save_config_to_yaml
from tts_service.engines.minimax import MiniMaxEngine
from tts_service.engines.omnivoice import OmniVoiceRemoteEngine
from tts_service.sample_manager import SampleManager


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    samples = np.zeros(2400, dtype=np.float32)
    sf.write(output, samples, 24000, format="WAV", subtype="PCM_16")
    return output.getvalue()


class MiniMaxEngineTests(unittest.TestCase):
    def test_native_request_uses_t2a_v2_and_voice_controls(self):
        config = Config()
        config.model.minimax_api_key = "test-key"
        config.model.minimax_base_url = "https://api.minimaxi.com"
        config.model.minimax_model_id = "speech-2.8-hd"
        config.model.minimax_voice_id = "AaronTest"
        config.model.minimax_speed = 1.2
        config.model.minimax_emotion = "happy"
        config.model.stereo = False
        config.model.spatial_jitter = False
        manager = SampleManager(Path(tempfile.mkdtemp()), "Aaron")
        engine = MiniMaxEngine(config, manager)
        response_body = json.dumps({
            "data": {"audio": _wav_bytes().hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }).encode()

        with patch(
            "tts_service.engines.minimax.urllib.request.urlopen",
            return_value=_Response(response_body),
        ) as request_mock:
            result = engine.generate_single("你好。", None, "wav")

        request = request_mock.call_args.args[0]
        payload = json.loads(request.data.decode())
        self.assertEqual(request.full_url, "https://api.minimaxi.com/v1/t2a_v2")
        self.assertEqual(payload["voice_setting"]["voice_id"], "AaronTest")
        self.assertEqual(payload["voice_setting"]["speed"], 1.2)
        self.assertEqual(payload["voice_setting"]["emotion"], "happy")
        self.assertEqual(result.output_format, "wav")

    def test_compatible_mode_uses_audio_speech_endpoint(self):
        config = Config()
        config.model.minimax_api_mode = "openai"
        config.model.minimax_api_key = "test-key"
        config.model.minimax_base_url = "http://127.0.0.1:4001"
        config.model.minimax_model_id = "minimax-speech-2.8-hd"
        config.model.minimax_voice_id = "alloy"
        config.model.stereo = False
        config.model.spatial_jitter = False
        manager = SampleManager(Path(tempfile.mkdtemp()), "Aaron")
        engine = MiniMaxEngine(config, manager)

        with patch(
            "tts_service.engines.minimax.urllib.request.urlopen",
            return_value=_Response(_wav_bytes()),
        ) as request_mock:
            engine.generate_single("你好。", None, "wav")

        request = request_mock.call_args.args[0]
        payload = json.loads(request.data.decode())
        self.assertEqual(request.full_url, "http://127.0.0.1:4001/v1/audio/speech")
        self.assertEqual(payload["model"], "minimax-speech-2.8-hd")
        self.assertEqual(payload["voice"], "alloy")

    def test_api_keys_are_stored_separately_from_config_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = Config()
            config.model.qwen_api_key = "qwen-secret"
            config.model.minimax_api_key = "minimax-secret"
            save_config_to_yaml(config, str(config_path))

            yaml_text = config_path.read_text(encoding="utf-8")
            self.assertNotIn("qwen-secret", yaml_text)
            self.assertNotIn("minimax-secret", yaml_text)
            loaded = load_config(str(config_path))
            self.assertEqual(loaded.model.qwen_api_key, "qwen-secret")
            self.assertEqual(loaded.model.minimax_api_key, "minimax-secret")

    def test_omnivoice_sends_reference_audio_and_transcript(self):
        config = Config()
        config.model.omnivoice_base_url = "http://127.0.0.1:8002"
        config.model.omnivoice_model_id = "k2-fsa/OmniVoice"
        config.model.stereo = False
        config.model.spatial_jitter = False
        voices_dir = Path(tempfile.mkdtemp())
        (voices_dir / "Aaron.txt").write_text("这是参考文本。", encoding="utf-8")
        sf.write(voices_dir / "Aaron.wav", np.zeros(2400, dtype=np.float32), 24000)
        manager = SampleManager(voices_dir, "Aaron")
        engine = OmniVoiceRemoteEngine(config, manager)

        with patch(
            "tts_service.engines.omnivoice.urllib.request.urlopen",
            return_value=_Response(_wav_bytes()),
        ) as request_mock:
            result = engine.generate_single("你好。", None, "wav")

        request = request_mock.call_args.args[0]
        payload = json.loads(request.data.decode())
        self.assertEqual(request.full_url, "http://127.0.0.1:8002/v1/audio/speech")
        self.assertEqual(payload["model"], "k2-fsa/OmniVoice")
        self.assertEqual(payload["ref_text"], "这是参考文本。")
        self.assertTrue(payload["ref_audio"])
        self.assertEqual(result.output_format, "wav")


if __name__ == "__main__":
    unittest.main()
