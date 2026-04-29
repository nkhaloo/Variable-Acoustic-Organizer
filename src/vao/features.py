from __future__ import annotations

import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd

from .opensmile_runner import run_smileextract


def _nanify_egemaps_placeholder_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Convert openSMILE placeholder 0s to NaN for eGeMAPS-style outputs.

    In the shipped GeMAPS/eGeMAPS configs, several voicing-dependent LLD streams
    emit 0.0 for frames where the value is undefined (e.g., unvoiced frames).
    This makes it hard to distinguish a true numeric 0 from "not computed".

    We apply a conservative transformation:
    - Detect an "unvoiced/undefined" frame mask from a small set of key columns.
    - For *_sma3nz and *_sma3nz_de columns, replace 0.0 with NaN *only* on those
      unvoiced/undefined frames.
    - For formant frequency/bandwidth columns, also replace 0.0 with NaN (0 Hz
      and 0 bandwidth are not physically meaningful and are always placeholders).

    This function is a no-op for non-eGeMAPS-style outputs.
    """

    if df.empty:
        return df

    # Heuristic: only run on eGeMAPS-style outputs.
    if not any("_sma3nz" in c for c in df.columns):
        return df
    if not any(c.startswith("F0semitoneFrom27.5Hz") for c in df.columns):
        return df

    # Replace placeholder zeros for all "nz" outputs.
    #
    # In the shipped eGeMAPS pipeline, 0.0 is frequently used to represent
    # "undefined" for voicing-dependent features (notably F0 and related).
    # Some other streams (e.g., formants) may not drop to 0 due to the
    # noZeroSma smoother behaviour, so a row-level "all keys are zero" mask is
    # not reliable.
    nz_base_cols = [c for c in df.columns if c.endswith("_sma3nz")]
    for col in nz_base_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] == 0, col] = pd.NA

    # Propagate missingness into delta columns (if base is missing, delta is too).
    nz_delta_cols = [c for c in df.columns if c.endswith("_sma3nz_de")]
    for delta_col in nz_delta_cols:
        base_col = delta_col[: -len("_de")]
        if base_col in df.columns and pd.api.types.is_numeric_dtype(df[delta_col]):
            df.loc[df[base_col].isna(), delta_col] = pd.NA

    # Additionally, formant frequency/bandwidth = 0 is always a placeholder.
    always_missing_prefixes = (
        "F1frequency",
        "F2frequency",
        "F3frequency",
        "F4frequency",
        "F5frequency",
        "F1bandwidth",
        "F2bandwidth",
        "F3bandwidth",
        "F4bandwidth",
        "F5bandwidth",
    )
    for col in df.columns:
        if col.startswith(always_missing_prefixes) and pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] == 0, col] = pd.NA

    return df


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

    df = _nanify_egemaps_placeholder_zeros(df)

    if add_time_column:
        likely_time_cols = {"time", "Time", "timestamp", "frameTime", "frame_time", "t"}
        if not any(col in likely_time_cols for col in df.columns):
            df.insert(0, time_column, df.index.to_numpy() * float(frame_step_s))

    # If the caller requested a persistent output CSV, rewrite it after
    # post-processing so the on-disk CSV matches the returned DataFrame.
    if output_csv is not None:
        df.to_csv(output_path, index=False, sep=delim, na_rep="NaN")

    if tmpdir is not None:
        tmpdir.cleanup()

    return df


def _extract_one(args: tuple) -> pd.DataFrame:
    """Worker function for parallel extraction. Must be module-level for multiprocessing."""
    (wav_path, config_path, per_file_csv, smileextract_path, opensmile_home,
     input_option, output_option, extra_args, cwd, frame_step_s,
     add_time_column, time_column, delimiter, recording_column, recording_value) = args

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
    df.insert(0, recording_column, recording_value)
    return df


def extract_features_folder(
    wav_dir: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    combined_csv: str | os.PathLike[str] | None = None,
    write_combined_csv: bool = False,
    write_per_file_csvs: bool = False,
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
    workers: int | None = None,
    recursive: bool = False,
) -> pd.DataFrame:
    """Extract features for all WAV files in a folder.

    Returns a combined DataFrame with one row per frame across all recordings.
    Nothing is written to disk by default — pass `write_combined_csv=True` or
    `write_per_file_csvs=True` to save output.

    Args:
        workers: Number of parallel processes. Defaults to all available CPU cores.
            Set to 1 to disable parallelism (useful for debugging).
        write_combined_csv: If True, write a combined CSV to `output_dir/combined.csv`
            (or `combined_csv` if provided).
        write_per_file_csvs: If True, write one CSV per recording into `output_dir`.
        recursive: If True, search for WAV files in all subdirectories.
    """

    wav_dir = Path(wav_dir).expanduser()
    if not wav_dir.is_dir():
        raise NotADirectoryError(f"WAV directory not found: {wav_dir}")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        wav_paths = sorted([p for p in wav_dir.rglob("*.wav") if p.is_file()])
    else:
        wav_paths = sorted([p for p in wav_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in: {wav_dir}")

    n_workers = workers if workers is not None else (os.cpu_count() or 1)

    def _per_file_csv(wav_path: Path) -> Path | None:
        if not write_per_file_csvs:
            return None
        if recursive:
            # Flatten subdirectory structure into filename to avoid collisions.
            # e.g. TRAIN/DR1/FCJF0/SA1.WAV.wav → TRAIN_DR1_FCJF0_SA1.WAV.csv
            rel = wav_path.relative_to(wav_dir)
            flat = "_".join(rel.with_suffix("").parts) + ".csv"
            return out_dir / flat
        return out_dir / f"{wav_path.stem}.csv"

    def _recording_value(wav_path: Path) -> str:
        if recursive:
            return str(wav_path.relative_to(wav_dir))
        return wav_path.name

    args_list = [
        (
            wav_path,
            Path(config_path),
            _per_file_csv(wav_path),
            smileextract_path,
            opensmile_home,
            input_option,
            output_option,
            extra_args,
            cwd,
            frame_step_s,
            add_time_column,
            time_column,
            delimiter,
            recording_column,
            _recording_value(wav_path),
        )
        for wav_path in wav_paths
    ]

    if n_workers == 1:
        frames = [_extract_one(args) for args in args_list]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            frames = list(executor.map(_extract_one, args_list))

    combined = pd.concat(frames, ignore_index=True)

    if write_combined_csv or combined_csv is not None:
        combined_path = Path(combined_csv).expanduser() if combined_csv is not None else (out_dir / "combined.csv")
        combined.to_csv(combined_path, index=False, na_rep="NaN")

    return combined
