"""vibevoice-mlx based TTS engine used by the API and UI.

This module re-exports the local engine and factory for backward compatibility.
New code should import from tts_service.engines directly.
"""

from __future__ import annotations

from .engines.base import BaseEngine, GenerationResult, create_engine  # noqa: F401
from .engines.local_vibevoice import LocalVibeVoiceEngine  # noqa: F401

# Backward-compat alias
TTSEngine = LocalVibeVoiceEngine
