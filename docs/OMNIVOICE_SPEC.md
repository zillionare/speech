# OmniVoice Integration Spec (v1.1, post-acceptance)

## Goal

Add OmniVoice (k2-fsa) as a third TTS engine alongside Qwen3-TTS and the local VibeVoice. OmniVoice runs in its own minimal HTTP service (`omnivoice-server`) on the .102 box, separate from omlx. A new in-text Voice-Design markup overlays per-turn pitch/style/accent hints on top of normal voice cloning. Qwen3-TTS still hosts on omlx and is untouched.

## Deployment Layout

```
192.168.0.100 (mini-one)            192.168.0.102 (mini-pro)
─────────────────────────           ───────────────────────────
~/apps/speech       (run)           omlx               port 8000  (LLMs + Qwen3-TTS)
~/workspace/speech  (dev)           ~/apps/omnivoice   port 8002  (OmniVoice mini-server)
~/workspace/omnivoice-server (dev → rsync → .102:~/apps/omnivoice)
```

**Directory convention**: develop in `~/workspace/<project>`; deploy (rsync) to `~/apps/<project>`. Never edit `~/apps/...` directly. The rsync direction is workspace → apps; using `apps → workspace` with `--delete` will erase workspace-only artefacts like `docs/`.

## Authoritative Facts

| Item | Value |
|---|---|
| omlx URL (Qwen3-TTS) | `http://192.168.0.102:8000` |
| omnivoice URL | `http://192.168.0.102:8002` |
| omlx API key (for `/v1/models` enumeration only) | `1234` |
| OmniVoice model id on omnivoice-server | `OmniVoice` (HF model: `k2-fsa/OmniVoice`) |
| OmniVoice model id on omlx (deprecated path) | `OmniVoice-bf16` — DO NOT use; omlx's mlx-audio path does not load OmniVoice cleanly. |
| Qwen3-TTS model id on omlx | `Qwen3-TTS-12Hz-1.7B-Base-8bit` |
| HuggingFace mirror | `HF_ENDPOINT=https://hf-mirror.com` (set by `omnivoice-server/run.sh`) |
| ssh access to .102 | user `quantide`, ed25519 key installed; passwordless |

## Engine Architecture

A single `tts_service/engines/omlx_remote.py::OmlxRemoteEngine` class drives BOTH remote engines, parameterized by `model_id` and `base_url`. The engine factory dispatches:

```python
engine: qwen_remote      → OmlxRemoteEngine(model_id=qwen_remote_model,      base_url=omlx_base_url)
engine: omnivoice_remote → OmlxRemoteEngine(model_id=omnivoice_remote_model, base_url=omnivoice_base_url)
engine: local_vibevoice  → LocalVibeVoiceEngine(...)
```

`config.yaml` keys (under `model:`):

```yaml
engine: qwen_remote                                      # default
omlx_base_url: http://192.168.0.102:8000                 # qwen3-tts
omnivoice_base_url: http://192.168.0.102:8002            # omnivoice mini-server
qwen_remote_model: Qwen3-TTS-12Hz-1.7B-Base-8bit
omnivoice_remote_model: OmniVoice
```

Legacy `use_remote_qwen` + `qwen_base_url` keys are migrated at load-time and dropped on round-trip save.

## Voice Design Markup

### Sources

Two layered markup inputs, parsed by `tts_service/voice_design.py`:

1. Optional YAML frontmatter (`---` ... `---` at the top of the input).
2. Optional per-line `Speaker[k=v, k=v]:` brackets on speaker tag lines.

Plus the existing plain `Speaker: text` dialogue format (unchanged).

### Frontmatter schema

```yaml
voice-design:
  default:                # optional, applies to every turn before per-speaker overrides
    pitch: moderate
  speakers:               # optional, per-speaker entries override default
    Aaron:
      pitch: low
      accent: american
    Elyn:
      style: whisper
```

Unknown top-level keys under `voice-design` raise `VoiceDesignError` at parse time.

### Per-line brackets

```
Aaron[pitch=high, style=whisper]: 真的吗？
```

Brackets attach to the speaker tag line; continuation lines inherit. Duplicate keys in the same bracket raise `VoiceDesignError`.

### Vocabulary (must match OmniVoice's `_resolve_instruct` table)

| key | values |
|---|---|
| `pitch` | `very-low`, `low`, `moderate`, `high`, `very-high` |
| `style` | `whisper` |
| `accent` | free-form; OmniVoice's known set: EN — `american`, `australian`, `british`, `canadian`, `chinese`, `indian`, `japanese`, `korean`, `portuguese`, `russian` · CN — `四川话`, `北京话`, `陕西话`, `东北话`, `云南话`, `宁夏话`, `桂林话`, `河南话`, `济南话`, `甘肃话`, `石家庄话`, `贵州话`, `青岛话` |

> `pitch` and `style` are strict enums; unknown values raise `VoiceDesignError`.
> `accent` is **free-form**: the parser does not validate the value list. OmniVoice itself will reject unknown accents at request time with HTTP 500 carrying its full vocabulary in the error body.

### Precedence (low → high)

1. `voice-design.default`
2. `voice-design.speakers.<name>`
3. line-level `[attrs]`

Per-key override at each layer; result is a `{pitch?, style?, accent?}` map per turn.

### Serialization → OmniVoice `instructions`

For each turn:

| key & value | emitted token |
|---|---|
| `pitch: very-low` | `very low pitch` |
| `pitch: low` | `low pitch` |
| `pitch: moderate` | `moderate pitch` |
| `pitch: high` | `high pitch` |
| `pitch: very-high` | `very high pitch` |
| `style: whisper` | `whisper` |
| `accent: <ascii>` | `<ascii> accent` (e.g. `british accent`) |
| `accent: <non-ascii>` | `<non-ascii>` (e.g. `四川话`) — no suffix |

Token order is fixed: `pitch → style → accent`. Missing keys are skipped.

**Language separation**: a single `instructions` string must be either all-English or all-Chinese (OmniVoice requirement, enforced by `_resolve_instruct`). The serializer detects mixed-language values and raises `VoiceDesignError` immediately; the API surfaces this as HTTP 400 from `/api/generate`.

- All-English tokens → joined with half-width `, `
- All-Chinese tokens → joined with full-width `，`

Examples:

```
{pitch=low, accent=american}          → "low pitch, american accent"
{pitch=high, style=whisper}           → "high pitch, whisper"
{accent=四川话}                       → "四川话"
{pitch=low, accent=四川话}            → VoiceDesignError (mixed)
```

### Turn semantics

A **turn** = one OmniVoice generation call. Boundaries:
- starts at `Speaker[attrs]?: text`
- ends at the next blank line OR next speaker tag line
- continuation lines (no speaker tag, no blank) extend the turn
- to shift attrs mid-speaker, re-issue the speaker tag with new brackets

If a turn exceeds `max_segment_chars`, the existing segmentation splits it into chunks; **every chunk shares the same `instructions` value** (attrs travel with the turn, not the chunk).

## Markup-on-Qwen3 Behaviour

If `voice_design.has_markup(text)` returns True and the resolved engine is not `omnivoice_remote`, `/api/generate`, `/api/generate/stream`, `/v1/audio/speech`, `/v1/audio/podcast` all raise:

```
HTTP 400
{"detail": "Input contains voice-design markup which is only supported by the OmniVoice engine. Switch the engine to omnivoice_remote, or remove the markup."}
```

`has_markup` is a cheap regex scan (`^---\s*$` + `^[^:\n\[]{1,80}\[[^\]]+\]\s*:`); no full parse.

## omnivoice-server (the mini-server)

Lives at `~/workspace/omnivoice-server/` (dev) → `quantide@192.168.0.102:~/apps/omnivoice/` (deploy).

```
omnivoice_server.py    # single-file FastAPI app
requirements.txt       # torch>=2.8 / torchaudio / omnivoice / fastapi / uvicorn / soundfile
run.sh                 # exports HF_ENDPOINT=hf-mirror, starts uvicorn on 0.0.0.0:8002
README.md              # full deploy walkthrough
```

Endpoints:
- `GET /health` → `{status, model_loaded, model_id, sample_rate}`
- `GET /v1/models` → `{data: [{id: "OmniVoice", ...}]}`
- `POST /v1/audio/speech` → OpenAI-shaped synth. Body: `{model, input, voice?, ref_audio? (base64 wav), ref_text?, instructions?, response_format?}`. Returns raw audio bytes at 24 kHz.

Implementation notes:
- Lazy model load via `OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=mps|cuda|cpu, dtype=bfloat16|float32)`.
- Single asyncio lock serializes calls (OmniVoice is per-process single-model).
- `ref_audio` is decoded to a temp file because the omnivoice Python API takes a path.
- `instructions` (if present) is passed verbatim to `model.generate(instruct=...)`.
- First-ever request downloads ~2 GB from `hf-mirror.com`. Subsequent loads use HF cache (`~/.cache/huggingface/`).

## End-to-End Acceptance (all PASS in v1.1)

| ID | What | Result |
|---|---|---|
| A1 | OmniVoice single-speaker (`Aaron`, no markup) | ✅ 6.44 s wav |
| A2 | OmniVoice two-speaker dialogue (`Aaron`/`Elyn`, no markup) via `/api/generate` | ✅ 5.81 s wav, both voices resolved |
| A3 | OmniVoice with frontmatter + line-level `[pitch=high]` override | ✅ 8.56 s wav; captured `instructions` strings exactly match: `'low pitch, american accent'` / `'whisper'` / `'high pitch, american accent'` |
| A4 | Qwen3 + markup → HTTP 400 | ✅ both frontmatter and bracket-only variants reject correctly |

Unit tests: 18/18 in `tests/test_voice_design.py` (15 original + 3 new for the corrected vocabulary).

## What Changed from the v1.0 Spec (decided during acceptance)

The original `_resolve_instruct` constraint from OmniVoice was discovered the hard way when A3 returned HTTP 500. The following changes are now baked into the parser and tests:

1. `style` vocabulary shrunk from `{whisper, excited, calm, serious, gentle, dramatic}` to **just `whisper`**. OmniVoice's vocabulary doesn't include the others. `excited`/`dramatic`/etc. now raise `VoiceDesignError` at parse time.
2. `pitch` value `medium` renamed to `moderate` (OmniVoice's token is `moderate pitch`).
3. Serialization is **no longer "flat comma-joined value-only"** (the user's original instinct). It now emits OmniVoice's exact tokens with `pitch`/`accent` suffixes. Style stays bare.
4. Language mixing (English `pitch=low` + Chinese `accent=四川话` in the same instruct) is now an explicit parse error rather than silent passthrough.
5. Chinese-only instructs use full-width `，`; English-only uses half-width `, `.

The user-facing change: **podcast scripts can no longer write `excited`/`dramatic` etc. as style values**. There is no OmniVoice equivalent for them. If you need expressive emphasis, you have to record it into the reference audio (`ref_audio`) and lean on cloning.

## Out of Scope (not addressed in v1.1)

- Inline (mid-turn) voice-design tags like `<whisper>...</whisper>`.
- Whisper auto-transcribe for missing `ref_text`.
- Voice-design create-new-voice workflow (Design → save as ref).
- Cross-engine mixed dialogue (Qwen3 + OmniVoice in one podcast).
- Auto-translation between English and Chinese instruct vocabularies.
- Cleanup of the stale top-level duplicates at `~/apps/speech/{base.py, qwen_remote.py, engines/}`.
- Better PID cleanup in `run.py restart` (current implementation can leave orphan processes).
