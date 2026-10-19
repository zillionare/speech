"""Shared audio post-processing helpers."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from .engines.base import _FFMPEG_PATH


DEFAULT_TARGET_LUFS = -18.0
DEFAULT_TRUE_PEAK_DBTP = -1.0
DEFAULT_LOUDNESS_RANGE_LU = 7.0


def normalize_wav_loudness(
    path: Path,
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    true_peak_dbtp: float = DEFAULT_TRUE_PEAK_DBTP,
    loudness_range_lu: float = DEFAULT_LOUDNESS_RANGE_LU,
) -> None:
    """Normalize a WAV in place using EBU R128 loudness measurement."""
    path = Path(path)
    info = sf.info(path)
    if info.frames == 0:
        return

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-normalized-",
        suffix=".wav",
        dir=path.parent,
        delete=False,
    ) as temporary:
        output = Path(temporary.name)

    audio_filter = (
        f"loudnorm=I={target_lufs}:TP={true_peak_dbtp}:"
        f"LRA={loudness_range_lu}"
    )
    command = [
        _FFMPEG_PATH,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-af",
        audio_filter,
        "-ar",
        str(info.samplerate),
        "-ac",
        str(info.channels),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not produce normalized audio")
        output.replace(path)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"loudness normalization failed: {detail}") from exc
    finally:
        output.unlink(missing_ok=True)
