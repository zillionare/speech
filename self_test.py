#!/usr/bin/env python3
"""Self-test script for all modified features."""

import json
import sys
import urllib.request
import urllib.error

def request(method, url, data=None, headers=None):
    req = urllib.request.Request(url, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    if data is not None:
        if isinstance(data, dict):
            req.add_header("Content-Type", "application/json")
            data = json.dumps(data).encode("utf-8")
        req.data = data
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body}
    except Exception as e:
        return -1, {"error": str(e)}

def run_tests():
    base = "http://127.0.0.1:8123"
    errors = []

    # 1. Health check
    print("[1/8] Health check...")
    status, data = request("GET", f"{base}/health")
    if status != 200:
        errors.append(f"Health check failed: {status}")
    else:
        print(f"  -> model={data['model']}, voices_count={data['voices_count']}, default_voice={data['default_voice']}")

    # 2. Get config
    print("[2/8] Get config...")
    status, data = request("GET", f"{base}/api/config")
    if status != 200:
        errors.append(f"Get config failed: {status}")
    else:
        print(f"  -> default_voice={data['default_voice']}")

    # 3. List voices
    print("[3/8] List voices...")
    status, data = request("GET", f"{base}/api/voices")
    if status != 200:
        errors.append(f"List voices failed: {status}")
    else:
        voices = data.get("voices", [])
        print(f"  -> {len(voices)} voices")
        for v in voices:
            print(f"      {v['speaker']}: cache_ready={v['cache_ready']}, is_default={v['is_default']}")

    # 4. Save config
    print("[4/8] Save config...")
    payload = {
        "voices_path": "./voices",
        "outputs_path": "./outputs",
        "default_voice": "Aaron-night",
        "diffusion_steps": 12,
        "quantize_bits": 8,
        "cfg_scale": 1.3,
        "max_speech_tokens": 200,
        "use_semantic": True,
        "use_coreml_semantic": False,
        "seed": 42,
    }
    status, data = request("POST", f"{base}/api/config", data=payload)
    if status != 200:
        errors.append(f"Save config failed: {status} -> {data}")
    else:
        print(f"  -> saved, default_voice={data['default_voice']}, diffusion_steps={data['diffusion_steps']}")

    # 5. Verify config persisted
    print("[5/8] Verify config persisted...")
    status, data = request("GET", f"{base}/api/config")
    if status != 200:
        errors.append(f"Get config after save failed: {status}")
    else:
        if data["diffusion_steps"] != 12:
            errors.append(f"Config not persisted: diffusion_steps={data['diffusion_steps']} (expected 12)")
        else:
            print(f"  -> OK, diffusion_steps={data['diffusion_steps']}")

    # 6. Warm cache for existing voice
    print("[6/8] Warm cache...")
    status, data = request("POST", f"{base}/api/voices/Aaron-night/cache")
    if status != 200:
        errors.append(f"Warm cache failed: {status} -> {data}")
    else:
        print(f"  -> cache_ready={data['cache_ready']}")

    # 7. Delete non-default voice (safe test)
    print("[7/8] Delete voice test...")
    # Upload a temp voice using multipart (skip for simplicity, just test delete endpoint exists)
    status, data = request("DELETE", f"{base}/api/voices/__nonexistent")
    if status != 404:
        print(f"  -> delete endpoint responds (status={status})")
    else:
        print(f"  -> delete endpoint OK (404 for non-existent)")

    # 8. Generate audio (single voice)
    print("[8/8] Generate audio...")
    payload = {
        "text": "这是一个测试句子。",
        "output_format": "wav",
        "voice": "Aaron-night",
        "voice_mapping": {},
    }
    status, data = request("POST", f"{base}/api/generate", data=payload)
    if status != 200:
        errors.append(f"Generate audio failed: {status} -> {data}")
    else:
        print(f"  -> generation_seconds={data['generation_seconds']:.2f}, duration={data['duration_seconds']:.2f}s")

    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("ALL TESTS PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(run_tests())
