# VibeVoice MLX Studio Spec

## Goal

Build a new implementation in a dedicated git worktree that uses `vibevoice-mlx` as the generation core, keeps local voices in a managed `voices/` library, and provides a browser UI inspired by VibeVoiceFusion's workflow.

## Product Decisions

1. Generation backend uses `gafiatulin/vibevoice-1.5b-mlx` with `8-bit` quantization by default.
2. The new service does not reuse the old `mlx-audio` path.
3. Voice assets live in `voices/` as `{speaker}.wav` + `{speaker}.txt` pairs.
4. Each voice also gets an internal cache file at `voices/.cache/{speaker}.safetensors` on first use.
5. Dialogue input format is `SpeakerName: text`.
6. Speaker resolution prefers an exact local voice match, then alias matching, then the default voice.
7. Default voice is `zh-Bowen_man`.
8. The UI is built into the FastAPI service as a static web app.

## Functional Requirements

### Generation

1. Users can paste plain narration text and generate a single-speaker result.
2. Users can paste dialogue text in `Aaron: 这是我要说的话` format.
3. Distinct speaker names are mapped to local voices from `voices/`.
4. If a speaker is not found, the service falls back to `zh-Bowen_man`.
5. The API returns generated audio and exposes saved outputs for playback.

### Voice Library

1. Voices are listed from `voices/`.
2. Uploading a voice creates or replaces `{speaker}.wav` and `{speaker}.txt`.
3. Users can edit transcript text for an existing voice.
4. Users can preview and delete voices from the UI.
5. Users can warm the `.safetensors` cache explicitly, but caching also happens automatically during generation.

### UI

1. Left column: voice library management.
2. Right column: generation form and recent outputs.
3. The UI shows which voices were actually resolved for the latest generation.
4. The UI is intentionally local-first and does not depend on a separate Node frontend.

## Non-Goals

1. Reusing the old `tts_service` implementation details.
2. Supporting built-in VibeVoice symbolic voice names as the primary source of truth.
3. Auto-downloading remote `.safetensors` voice presets by speaker name.

## Technical Notes

1. `vibevoice_mlx.e2e_pipeline` is used as the reference implementation for tokenization, voice encoding, semantic encoder loading, and generation.
2. Transcript `.txt` files are managed because the product requires them, even though the current `vibevoice-mlx` inference flow only uses audio for cloning.
3. Bundled preset voices can be shipped as local `.wav` assets and cached to `.safetensors` on demand.
4. The compatibility API keeps `/v1/audio/speech` and `/v1/audio/podcast`, but the primary product surface is the built-in `/` web UI.
5. The implementation must not depend on `mlx-audio`, Python Fire, or a separate Node frontend.
