from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd

from .opensmile_runner import run_smileextract


def _guess_csv_delimiter(path: Path) -> str:
    with path.open("r", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t ")
        return dialect.delimiter
    except Exception:
        return ","


def extract_features(
    wav_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str],
    output_csv: str | os.PathLike[str] | None = None,
    smileextract_path: str | os.PathLike[str] | None = None,
    opensmile_home: str | os.PathLike[str] | None = None,
    input_option: str = "-I",
    output_option: str = "-csvoutput",
    extra_args: tuple[str, ...] = (),
    cwd: str | os.PathLike[str] | None = None,
    frame_step_s: float = 0.001,
    add_time_column: bool = True,
    time_column: str = "time_s",
    delimiter: str | Literal["auto"] = "auto",
) -> pd.DataFrame:
    """Extract frame-level acoustic features from a WAV using openSMILE.

    This function runs SMILExtract and reads the resulting CSV into a DataFrame.

    Args:
        wav_path: Path to an input WAV file.
        config_path: Path to an openSMILE config (`.conf`) that produces frame-level output.
        output_csv: Optional path to write the CSV. If None, uses a temporary file.
        smileextract_path: Optional explicit SMILExtract path.
        opensmile_home: Optional openSMILE repo/install root. Used to locate SMILExtract.
        input_option/output_option: Command-line flags used by the config for input/output.
        frame_step_s: Hop size in seconds (used only to synthesize a time column if needed).
        add_time_column: If True and no obvious time column exists, add `time_column`.
        time_column: Name for the synthesized time column.
        delimiter: CSV delimiter or 'auto' to sniff.

    Returns:
        DataFrame with one row per frame.
    """

    wav_path = Path(wav_path).expanduser()
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    config_path = Path(config_path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"openSMILE config not found: {config_path}")

    if output_csv is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="vao_")
        output_path = Path(tmpdir.name) / "opensmile.csv"
    else:
        tmpdir = None
        output_path = Path(output_csv).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    run_smileextract(
        input_wav=wav_path,
        config_path=config_path,
        output_path=output_path,
        smileextract_path=smileextract_path,
        opensmile_home=opensmile_home,
        input_option=input_option,
        output_option=output_option,
        extra_args=extra_args,
        cwd=cwd,
    )

    delim = _guess_csv_delimiter(output_path) if delimiter == "auto" else delimiter
    df = pd.read_csv(output_path, sep=delim)

    if add_time_column:
        likely_time_cols = {"time", "Time", "timestamp", "frameTime", "frame_time", "t"}
        if not any(col in likely_time_cols for col in df.columns):
            df.insert(0, time_column, df.index.to_numpy() * float(frame_step_s))

    if tmpdir is not None:
        tmpdir.cleanup()

    return df


def extract_features_folder(
    wav_dir: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    combined_csv: str | os.PathLike[str] | None = None,
    smileextract_path: str | os.PathLike[str] | None = None,
    opensmile_home: str | os.PathLike[str] | None = None,
    input_option: str = "-I",
    output_option: str = "-csvoutput",
    extra_args: tuple[str, ...] = (),
    cwd: str | os.PathLike[str] | None = None,
    frame_step_s: float = 0.001,
    add_time_column: bool = True,
    time_column: str = "time_s",
    delimiter: str | Literal["auto"] = "auto",
    recording_column: str = "recording",
) -> pd.DataFrame:
    """Extract features for all WAV files in a folder.

    Writes one CSV per recording into `output_dir` and also writes a single
    combined CSV (defaults to `<output_dir>/combined.csv`).

    The returned DataFrame is the concatenation of all recordings and includes
    a `recording_column` so you can group rows by source file.
    """

    wav_dir = Path(wav_dir).expanduser()
    if not wav_dir.is_dir():
        raise NotADirectoryError(f"WAV directory not found: {wav_dir}")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted([p for p in wav_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in: {wav_dir}")

    frames: list[pd.DataFrame] = []

    for wav_path in wav_paths:
        per_file_csv = out_dir / f"{wav_path.stem}.csv"
        df = extract_features(
            wav_path,
            config_path=config_path,
            output_csv=per_file_csv,
            smileextract_path=smileextract_path,
            opensmile_home=opensmile_home,
            input_option=input_option,
            output_option=output_option,
            extra_args=extra_args,
            cwd=cwd,
            frame_step_s=frame_step_s,
            add_time_column=add_time_column,
            time_column=time_column,
            delimiter=delimiter,
        )

        df.insert(0, recording_column, wav_path.name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    combined_path = Path(combined_csv).expanduser() if combined_csv is not None else (out_dir / "combined.csv")
    combined.to_csv(combined_path, index=False)

    return combined
