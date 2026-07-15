"""Integration tests for ST-LP-001: Live Speaker configuration.

All tests call real config classes and functions.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_service.config import load_config, save_config_to_yaml, Config, LiveConfig
from tts_service.models import PodcastProject, PodcastSegment


def _make_config_yaml(tmpdir: Path, live_speakers: list | None = None) -> Path:
    data = {
        "model": {"model_id": "test-model", "quantize_bits": 8},
        "voices": {
            "base_dir": str(tmpdir / "voices"),
            "default_voice": "Aaron",
            "live_speakers": live_speakers or [],
        },
        "outputs": {"base_dir": str(tmpdir / "outputs")},
        "server": {"host": "0.0.0.0", "port": 8123},
        "pid_file": str(tmpdir / "tts.pid"),
    }
    path = tmpdir / "config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


class ConfigLoadTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_empty_live_speakers(self):
        cfg = load_config(str(_make_config_yaml(self.tmpdir, [])))
        self.assertEqual(cfg.voices.live_speakers, [])

    def test_load_populated_live_speakers(self):
        cfg = load_config(str(_make_config_yaml(self.tmpdir, ["Aaron", "Ben"])))
        self.assertEqual(cfg.voices.live_speakers, ["Aaron", "Ben"])

    def test_load_missing_field_backward_compatible(self):
        data = {
            "model": {"model_id": "test"},
            "voices": {"base_dir": str(self.tmpdir / "v"), "default_voice": "Aaron"},
            "outputs": {"base_dir": str(self.tmpdir / "o")},
            "server": {"host": "0.0.0.0", "port": 8123},
        }
        path = self.tmpdir / "config.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")
        cfg = load_config(str(path))
        self.assertEqual(cfg.voices.live_speakers, [])


class ConfigSaveTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_preserves_live_speakers(self):
        cfg = load_config(str(_make_config_yaml(self.tmpdir, ["Aaron"])))
        cfg.voices.live_speakers = ["Aaron", "Flora"]
        out = self.tmpdir / "out.yaml"
        save_config_to_yaml(cfg, str(out))
        reloaded = load_config(str(out))
        self.assertEqual(reloaded.voices.live_speakers, ["Aaron", "Flora"])

    def test_save_preserves_other_fields(self):
        cfg = load_config(str(_make_config_yaml(self.tmpdir, ["Aaron"])))
        original_model = cfg.model.model_id
        out = self.tmpdir / "out.yaml"
        save_config_to_yaml(cfg, str(out))
        reloaded = load_config(str(out))
        self.assertEqual(reloaded.model.model_id, original_model)
        self.assertEqual(reloaded.live.tts_timeout_seconds, LiveConfig().tts_timeout_seconds)

    def test_save_preserves_live_config(self):
        cfg = load_config(str(_make_config_yaml(self.tmpdir, [])))
        cfg.live.tts_max_seconds = 120
        out = self.tmpdir / "out.yaml"
        save_config_to_yaml(cfg, str(out))
        reloaded = load_config(str(out))
        self.assertEqual(reloaded.live.tts_max_seconds, 120)


class ConfigLiveConfigTests(unittest.TestCase):
    """LiveConfig integration with Config root - config-specific tests.

    ProjectOverrideTests are in test_models.py to avoid duplication.
    """

    def test_defaults(self):
        """Defaults should come from LiveConfig field definitions, not hardcode."""
        cfg = LiveConfig()
        defaults = LiveConfig.model_fields
        self.assertEqual(cfg.tts_max_seconds, defaults["tts_max_seconds"].default)
        self.assertEqual(cfg.tts_timeout_seconds, defaults["tts_timeout_seconds"].default)
        self.assertEqual(cfg.max_ai_rephrases_per_segment, defaults["max_ai_rephrases_per_segment"].default)
    def test_in_root_config(self):
        cfg = Config()
        self.assertIsInstance(cfg.live, LiveConfig)

    def test_timeout_less_than_max(self):
        cfg = LiveConfig()
        self.assertLess(cfg.tts_timeout_seconds, cfg.tts_max_seconds)

    def test_rephrases_can_be_zero(self):
        cfg = LiveConfig(max_ai_rephrases_per_segment=0)
        self.assertEqual(cfg.max_ai_rephrases_per_segment, 0)


if __name__ == "__main__":
    unittest.main()