"""Audio preprocessing: convert any audio file to 16 kHz mono WAV.

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

TARGET_SR = 16_000

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
    ".aiff", ".aif", ".mp4", ".wma",
}


def _is_target_wav(path: Path) -> bool:
    """Return True if path is already a 16 kHz mono WAV."""
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate() == TARGET_SR and wf.getnchannels() == 1
    except wave.Error:
        return False


def preprocess_file(
    src: str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert src to a 16 kHz mono WAV in out_dir.

    If src is already a 16 kHz mono WAV it is copied only when overwrite=True;
    otherwise the original path is returned as-is (no copy).

    Returns the path to the output WAV.
    """
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / (src.stem + ".wav")

    if _is_target_wav(src) and not overwrite:
        return src

    if dest.exists() and not overwrite:
        return dest

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src}:\n{result.stderr.decode(errors='replace')}"
        )

    return dest


def preprocess_folder(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    recursive: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Preprocess all audio files in src_dir, writing 16 kHz mono WAVs to out_dir.

    Returns a list of output WAV paths in the same order as the inputs.
    """
    src_dir = Path(src_dir)
    pattern = "**/*" if recursive else "*"
    candidates = sorted(
        p for p in src_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No audio files found in: {src_dir}")

    return [preprocess_file(p, out_dir, overwrite=overwrite) for p in candidates]
