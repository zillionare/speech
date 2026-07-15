#!/usr/bin/env python3
"""Generate a single comparison audio with multiple speed segments.

Run on the deployment host (mini-one):
    cd /Users/openclaw/apps/speech
    .venv/bin/python test_speed_comparison.py

Outputs: outputs/speed_test/comparison.wav + report.json
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_service.config import load_config
from tts_service.sample_manager import SampleManager
from tts_service.engines.qwen_remote import QwenRemoteEngine


TEST_TEXT = "大家好，欢迎来到量化好声音播客。今天我们将跟大家聊一聊，如何养龙虾的事儿。"
SPEEDS = [0.5, 1.0, 1.5]
GAP_MS = 500  # silence between segments in ms


def main() -> None:
    config = load_config("config.yaml")
    sample_manager = SampleManager(config.voices.expanded_base_dir, config.voices.default_voice)
    engine = QwenRemoteEngine(config, sample_manager)

    out_dir = Path("outputs") / "speed_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    segments: list[tuple[np.ndarray, int]] = []
    report: list[dict] = []
    sr = 24000

    for speed in SPEEDS:
        tagged = f"[speed:{speed}]{TEST_TEXT}"
        print(f"[speed={speed}] generating...")
        result = engine.generate_single(text=tagged, voice=None, output_format="wav")

        # Measure actual duration
        info = sf.info(io.BytesIO(result.audio_bytes))
        duration = info.duration
        print(f"[speed={speed}] duration={duration:.2f}s  filesize={len(result.audio_bytes)}B")

        report.append({"speed": speed, "duration": duration, "filesize": len(result.audio_bytes)})

        audio, file_sr = sf.read(io.BytesIO(result.audio_bytes))
        if file_sr != sr:
            sr = file_sr
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        segments.append((audio, sr))

    # Concatenate with gaps
    gap_samples = int(sr * (GAP_MS / 1000))
    gap = np.zeros((gap_samples, 1), dtype=np.float32)

    # Ensure all segments are mono for concatenation
    all_mono: list[np.ndarray] = []
    for audio, _ in segments:
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        all_mono.append(audio)
        all_mono.append(gap)

    # Remove trailing gap
    if all_mono:
        all_mono.pop()

    combined = np.concatenate(all_mono, axis=0)
    out_path = out_dir / "comparison.wav"
    sf.write(out_path, combined, sr, format="WAV")
    print(f"\nComparison saved: {out_path} ({len(combined)/sr:.2f}s total)")

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved: {report_path}")

    # Quick analysis
    base_duration = next((r["duration"] for r in report if r["speed"] == 1.0), None)
    if base_duration:
        print("\n--- Analysis ---")
        for r in report:
            expected = base_duration / r["speed"]
            actual = r["duration"]
            delta = actual - expected
            print(f"speed={r['speed']}: expected={expected:.2f}s actual={actual:.2f}s delta={delta:+.2f}s")


if __name__ == "__main__":
    main()
