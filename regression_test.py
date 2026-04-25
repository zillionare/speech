#!/usr/bin/env python3
"""Regression test suite for the TTS service.

Run before and after UI changes to ensure no functional regression.
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8123"
ERRORS: list[str] = []


def req_json(method: str, url: str, data: dict | None = None) -> tuple[int, dict]:
    r = urllib.request.Request(url, method=method)
    if data is not None:
        r.add_header("Content-Type", "application/json")
        r.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body}
    except Exception as exc:
        return -1, {"error": str(exc)}


def req_binary(method: str, url: str, data: dict | None = None) -> tuple[int, bytes]:
    r = urllib.request.Request(url, method=method)
    if data is not None:
        r.add_header("Content-Type", "application/json")
        r.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as exc:
        return -1, b""


def make_wav(seed: int = 0) -> bytes:
    sample_rate = 24000
    duration = 2
    samples = []
    freq = 440 + seed * 50
    for i in range(sample_rate * duration):
        samples.append(int(16000 * math.sin(2 * math.pi * freq * i / sample_rate)))
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(samples) * 2))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHI", 1, 1, sample_rate))
    buf.write(struct.pack("<IHH", sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(samples) * 2))
    for s in samples:
        buf.write(struct.pack("<h", s))
    return buf.getvalue()


def upload_voice(speaker: str, wav: bytes, transcript: str = "") -> tuple[int, dict]:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="speaker"\r\n\r\n{speaker}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="transcript"\r\n\r\n{transcript}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio_file"; filename="test.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    r = urllib.request.Request(f"{BASE}/api/voices", data=body)
    r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    except Exception as exc:
        return -1, {"error": str(exc)}


def assert_eq(name: str, actual, expected):
    if actual != expected:
        ERRORS.append(f"{name}: expected {expected!r}, got {actual!r}")


def assert_ok(name: str, status: int, data: dict):
    if status != 200:
        ERRORS.append(f"{name}: HTTP {status} -> {data}")
        return False
    return True


def cleanup_all_voices():
    for _ in range(5):
        _, data = req_json("GET", f"{BASE}/api/voices")
        voices = data.get("voices", [])
        if not voices:
            break
        for v in voices:
            req_json("DELETE", f"{BASE}/api/voices/{v['speaker']}")
        time.sleep(0.2)


def test_health():
    print("[T01] Health check")
    status, data = req_json("GET", f"{BASE}/health")
    if assert_ok("health", status, data):
        print(f"      model={data['model']}, voices={data['voices_count']}, default={data['default_voice']}")


def test_config():
    print("[T02] Config GET/POST")
    s1, cfg = req_json("GET", f"{BASE}/api/config")
    if not assert_ok("config_get", s1, cfg):
        return
    original_steps = cfg["diffusion_steps"]

    payload = {
        "voices_path": "./voices",
        "outputs_path": "./outputs",
        "default_voice": cfg["default_voice"],
        "diffusion_steps": 99,
        "quantize_bits": cfg["quantize_bits"],
        "cfg_scale": 1.3,
        "max_speech_tokens": 200,
        "use_semantic": True,
        "use_coreml_semantic": cfg.get("use_coreml_semantic", False),
        "seed": 42,
    }
    s2, saved = req_json("POST", f"{BASE}/api/config", payload)
    assert_ok("config_post", s2, saved)
    assert_eq("config_persisted", saved.get("diffusion_steps"), 99)

    # restore
    payload["diffusion_steps"] = original_steps
    req_json("POST", f"{BASE}/api/config", payload)


def _ensure_test_voice():
    # delete if exists from previous run
    req_json("DELETE", f"{BASE}/api/voices/Test-Spk")
    time.sleep(0.2)
    s, data = upload_voice("Test-Spk", make_wav(), "test transcript")
    if s != 200:
        ERRORS.append(f"voice_setup: HTTP {s} -> {data}")
    return s == 200


def test_voices_crud():
    print("[T03] Voice CRUD")
    if not _ensure_test_voice():
        return

    # list — Test-Spk must be present
    s, data = req_json("GET", f"{BASE}/api/voices")
    if assert_ok("voice_list", s, data):
        voices = data.get("voices", [])
        spk_names = [v["speaker"] for v in voices]
        assert_eq("voice_list_has_test", "Test-Spk" in spk_names, True)
        test_voice = next((v for v in voices if v["speaker"] == "Test-Spk"), None)
        if test_voice:
            assert_eq("voice_list_cache", test_voice["cache_ready"], False)

    # update transcript
    s, data = req_json("PUT", f"{BASE}/api/voices/Test-Spk/transcript", {"transcript": "updated"})
    if assert_ok("voice_transcript", s, data):
        assert_eq("voice_transcript_value", data.get("transcript"), "updated")

    # warm cache
    s, data = req_json("POST", f"{BASE}/api/voices/Test-Spk/cache")
    if assert_ok("voice_cache", s, data):
        assert_eq("voice_cache_ready", data.get("cache_ready"), True)

    # get audio (binary)
    s, body = req_binary("GET", f"{BASE}/api/voices/Test-Spk/audio")
    assert_eq("voice_audio_status", s, 200)
    assert_eq("voice_audio_size", len(body) > 0, True)

    # delete
    s, data = req_json("DELETE", f"{BASE}/api/voices/Test-Spk")
    assert_ok("voice_delete", s, data)

    # verify deleted
    s, data = req_json("GET", f"{BASE}/api/voices")
    if assert_ok("voice_list_after_delete", s, data):
        spk_names = [v["speaker"] for v in data.get("voices", [])]
        assert_eq("voice_list_no_test", "Test-Spk" not in spk_names, True)


def test_generate():
    print("[T04] Audio generation")
    # ensure at least one voice exists
    _, voices = req_json("GET", f"{BASE}/api/voices")
    if not voices.get("voices"):
        upload_voice("Gen-Spk", make_wav(), "gen")
        req_json("POST", f"{BASE}/api/voices/Gen-Spk/cache")
        time.sleep(0.3)

    s, data = req_json("POST", f"{BASE}/api/generate", {
        "text": "这是一个测试句子。",
        "output_format": "wav",
        "voice": None,
        "voice_mapping": {},
    })
    if assert_ok("generate", s, data):
        assert_eq("generate_duration_positive", data["duration_seconds"] > 0, True)
        assert_eq("generate_has_url", bool(data.get("audio_url")), True)


def test_history_and_outputs():
    print("[T05] History & output pruning")
    s, data = req_json("GET", f"{BASE}/api/generations")
    assert_ok("history_get", s, data)

    s, data = req_json("POST", f"{BASE}/api/outputs/prune", {"keep_count": 0})
    if assert_ok("outputs_prune", s, data):
        assert_eq("prune_deleted_list", isinstance(data.get("deleted"), list), True)
        assert_eq("prune_kept_list", isinstance(data.get("kept"), list), True)


def test_openai_compat():
    print("[T06] OpenAI compatible API")
    s, data = req_json("GET", f"{BASE}/v1/voices")
    assert_ok("openai_voices", s, data)

    s, data = req_json("GET", f"{BASE}/v1/voices/details")
    assert_ok("openai_voice_details", s, data)

    # streaming audio endpoint returns binary
    s, body = req_binary("POST", f"{BASE}/v1/audio/speech", {
        "input": "测试",
        "voice": None,
        "response_format": "wav",
    })
    assert_eq("openai_speech_status", s, 200)
    assert_eq("openai_speech_body", len(body) > 0, True)


def test_errors():
    print("[T07] Error handling")
    # 404 delete non-existent
    s, _ = req_json("DELETE", f"{BASE}/api/voices/__nonexistent")
    assert_eq("error_404_delete", s, 404)

    # 400 delete default voice
    _, cfg = req_json("GET", f"{BASE}/api/config")
    default = cfg.get("default_voice", "")
    s, _ = req_json("DELETE", f"{BASE}/api/voices/{default}")
    assert_eq("error_400_delete_default", s, 400)

    # 422 invalid config save
    r = urllib.request.Request(f"{BASE}/api/config", data=b"not json", method="POST")
    r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r):
            pass
    except urllib.error.HTTPError as e:
        assert_eq("error_422_bad_json", e.code, 422)


def main() -> int:
    print("=" * 50)
    print("REGRESSION TEST SUITE")
    print("=" * 50)

    test_health()
    test_config()
    test_voices_crud()
    test_generate()
    test_history_and_outputs()
    test_openai_compat()
    test_errors()

    print()
    print("=" * 50)
    if ERRORS:
        print(f"FAILED: {len(ERRORS)} error(s)")
        for e in ERRORS:
            print(f"  - {e}")
        return 1
    else:
        print("ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
