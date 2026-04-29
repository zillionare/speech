#!/usr/bin/env python3
"""Generate speech with Aaron and Elyn reference voices."""

import base64
import json
import urllib.request
from pathlib import Path

URL = "http://192.168.0.102:8000/v1/audio/speech"
MODEL = "Qwen3-TTS-12Hz-1.7B-Base-8bit"
TEXT = "这件事让 Peter 非常震惊。"
INSTRUCTIONS = "读到'非常'这个词时，语速放慢，语气加重，带有明显的停顿和强调感"
FORMAT = "wav"

VOICES_DIR = Path("/Users/aaronyang/workspace/speech/voices")


def encode_audio(wav_path: Path) -> str:
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate(name: str, wav_path: Path, ref_text: str) -> bytes:
    payload = {
        "model": MODEL,
        "input": TEXT,
        "voice": name,
        "ref_audio": encode_audio(wav_path),
        "ref_text": ref_text,
        "response_format": FORMAT,
        "instructions": INSTRUCTIONS,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def main():
    voices = [
        ("Aaron", VOICES_DIR / "Aaron.wav", (VOICES_DIR / "Aaron.txt").read_text(encoding="utf-8").strip()),
        ("Elyn", VOICES_DIR / "Elyn.wav", (VOICES_DIR / "Elyn.txt").read_text(encoding="utf-8").strip()),
    ]

    for name, wav_path, ref_text in voices:
        output_path = Path(f"output_{name.lower()}.wav")
        print(f"Generating {name} -> {output_path} ...")
        audio_bytes = generate(name, wav_path, ref_text)
        output_path.write_bytes(audio_bytes)
        print(f"Saved {output_path} ({len(audio_bytes)} bytes)")


if __name__ == "__main__":
    main()
