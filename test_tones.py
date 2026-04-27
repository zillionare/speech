#!/usr/bin/env python3
"""Batch test tone effectiveness with Qwen3-TTS remote engine.

Run on the deployment host (mini-one):
    cd /Users/openclaw/apps/speech
    .venv/bin/python test_tones.py

Outputs go to outputs/tone_test/ with subfolders per tone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tts_service.config import load_config
from tts_service.sample_manager import SampleManager
from tts_service.engines.qwen_remote import QwenRemoteEngine


BASELINE_TEXT = (
    "你好，这是一个语气测试。我们在验证不同的情感标签对语音生成的影响，"
    "希望找到真正能产生听觉差异的描述词。"
)

TONE_CANDIDATES = [
    "兴奋",
    "压抑",
    "开朗",
    "笑意",
    "温柔",
    "严肃",
    "悲伤",
    "愤怒",
    "惊讶",
    "冷静",
    "活泼",
    "沉稳",
    "轻快",
    "低沉",
    "亲切",
    "冷漠",
    "紧张",
    "慵懒",
    "激昂",
    "俏皮",
]


def main() -> None:
    config = load_config("config.yaml")
    sample_manager = SampleManager(config.voices.expanded_base_dir, config.voices.default_voice)
    engine = QwenRemoteEngine(config, sample_manager)

    out_dir = Path("outputs") / "tone_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (no tone)
    print("[baseline] generating...")
    start = time.perf_counter()
    result = engine.generate_single(text=BASELINE_TEXT, voice=None, output_format="wav")
    baseline_path = out_dir / "baseline.wav"
    baseline_path.write_bytes(result.audio_bytes)
    print(f"[baseline] done in {time.perf_counter() - start:.1f}s -> {baseline_path}")

    manifest: list[dict] = [
        {"tone": "baseline", "file": str(baseline_path), "duration": result.duration_seconds}
    ]

    for tone in TONE_CANDIDATES:
        tagged_text = f"[tone:{tone}]{BASELINE_TEXT}"
        print(f"[{tone}] generating...")
        start = time.perf_counter()
        try:
            result = engine.generate_single(
                text=tagged_text,
                voice=None,
                output_format="wav",
            )
            tone_path = out_dir / f"{tone}.wav"
            tone_path.write_bytes(result.audio_bytes)
            duration = result.duration_seconds
            elapsed = time.perf_counter() - start
            print(f"[{tone}] done in {elapsed:.1f}s -> {tone_path} ({duration:.1f}s)")
            manifest.append({"tone": tone, "file": str(tone_path), "duration": duration})
        except Exception as exc:
            print(f"[{tone}] FAILED: {exc}")
            manifest.append({"tone": tone, "error": str(exc)})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest saved to {manifest_path}")
    print(f"Total: {len(manifest)} files in {out_dir}")


if __name__ == "__main__":
    main()
