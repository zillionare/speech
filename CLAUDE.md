# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech Studio — a Chinese TTS and voice-cloning web service with dual-engine support:
- **Primary:** Remote Qwen3-TTS via OpenAI-compatible API (hosted on `192.168.0.102:8000`)
- **Fallback:** Local MLX inference using the VibeVoice model (Apple Silicon only)

The backend is Python (FastAPI), and the frontend is vanilla HTML/CSS/JS served as static files (no Node.js build step). Users can toggle between engines per-generation.

## Common Commands

- **Install dependencies:** `pip install -r requirements.txt`
- **Start server (foreground, auto-reload):** `python run.py start --no-daemon`
- **Start server (background daemon):** `python run.py start --daemon`
- **Stop / restart / status:** `python run.py stop` / `restart` / `status`
- **Unit tests (audio preprocessing):** `python -m unittest tests/test_sample_manager.py`
- **Integration tests (require running server on localhost:8123):**
  - `python self_test.py` — 8-step feature validation
  - `python regression_test.py` — full regression suite
  - `bash test.sh` — cURL smoke test for `/v1/audio/speech`

There is no pytest, Makefile, or centralized test runner. Integration test scripts at the root are run individually as Python scripts.

## High-Level Architecture

### Entry Point and CLI
- `run.py` is the entrypoint. It delegates to `tts_service.cli.main()`, which provides `start`, `stop`, `restart`, and `status` commands.
- The CLI manages a PID file (`/tmp/tts_service.pid`) for daemon mode. Foreground mode runs uvicorn directly with auto-reload enabled by default.

### Server Layer (`tts_service/server.py`)
- FastAPI app created by `create_app(config_path)`. It serves both the REST API and the static web UI.
- Static files are mounted at `/static` from `tts_service/static/`.
- Key endpoint groups:
  - Web UI and management: `/`, `/health`, `/api/config`, `/api/voices`, `/api/generate`, `/api/generations`
  - OpenAI-compatible: `/v1/audio/speech`, `/v1/audio/podcast`, `/v1/voices`
- A `deque` with `maxlen=config.outputs.history_limit` holds recent generation records in memory.

### TTS Engines (`tts_service/engines/`)
- **`QwenRemoteEngine`** (primary): Calls the remote Qwen3-TTS service at `config.model.qwen_base_url`. Sends `ref_audio` (base64) + `ref_text` for voice cloning. ~10x faster than local MLX.
- **`LocalVibeVoiceEngine`** (fallback): Direct Python integration with `vibevoice-mlx` on Apple Silicon via MLX. Loads the model on first generation.
- **Engine factory** (`create_engine()` in `base.py`): Returns the remote engine if `use_remote_qwen=true`, else local. Users can override per-request via the `engine` field.
- **`BaseEngine`** provides shared utilities: text segmentation for long inputs, ffmpeg-based audio concatenation, and post-processing (speed via `atempo`, mono→stereo, spatial jitter via `apulsator`).
- **Dialogue parsing:** Input in `Speaker: text` format is split by speaker. Speaker resolution order: exact local voice match → alias matching → default voice (`config.voices.default_voice`).
- **Number-to-Chinese:** Arabic numerals are converted to Chinese text before TTS. Years (4 digits + "年") and codes/IDs are read digit-by-digit; other numbers are read by value via `cn2an`.

### Voice Library (`tts_service/sample_manager.py`)
- Voices are stored as `{speaker}.wav` + `{speaker}.txt` pairs in `voices/` (configured in `config.yaml`).
- On first use, voice embeddings are encoded and cached to `voices/.cache/{speaker}.safetensors`.
- **Audio preprocessing on upload:** multi-channel downmix to mono, highpass filter (~70 Hz), spectral denoising (noise profile from edge segments), silence trimming, and peak normalization to 0.95.
- `SampleManager` also handles bundled preset voices listed in `config.voices.bundled_chinese_voices`.

### Configuration (`tts_service/config.py`)
- Pydantic models with YAML persistence. `load_config()` reads `config.yaml`; relative paths for `voices.base_dir`, `outputs.base_dir`, and `pid_file` are resolved relative to the config file's directory.
- Runtime config can be updated via `/api/config` and saved back with `save_config_to_yaml()`.

### Vendored Dependency
- `vibevoice-mlx/` is a vendored Python package (with its own `pyproject.toml` and `uv.lock`). It is not installed as an editable package; instead, `tts_engine.py` inserts its path into `sys.path` at import time.

## Important Design Constraints

- **Primary engine is remote Qwen-TTS** — local MLX is a fallback for offline use or when the remote server is unavailable.
- **Apple Silicon only for local engine** — MLX is a hard dependency of the local path, not the remote path.
- **No `mlx-audio` dependency** — this was an explicit architectural decision to avoid the old path.
- **No Node.js frontend build** — the UI in `tts_service/static/` is vanilla HTML/CSS/JS served directly by FastAPI. Do not introduce a bundler.
- **Voice assets are local-first** — the service does not auto-download remote `.safetensors` presets by speaker name. Bundled voices ship as local `.wav` files.
- Transcript `.txt` files are managed alongside `.wav` files; the remote Qwen engine uses them as `ref_text`, while the local MLX engine only uses audio for cloning.
