from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import extract_features_folder
from .opensmile_presets import get_preset


DEFAULT_PRESET = "egemapsv02_lld_25ms_10ms"


def vao_extract(
    wav_dir: str | Path,
    *,
    opensmile_home: str | Path | None = None,
    opensmile_default: str | Path | None = None,
    output_dir: str | Path | None = None,
    combined_csv: str | Path | None = None,
    preset: str = DEFAULT_PRESET,
    frame_step_s: float = 0.010,
    write_per_file_csvs: bool = False,
) -> pd.DataFrame:
    """One-call batch extraction for a folder of WAV files.

    This is the minimal user-facing API:
    - User supplies a directory of WAV files
    - User supplies their local openSMILE root folder (where `config/` lives)
    - VAO runs the eGeMAPSv02 frame-level preset (25 ms window, 10 ms hop)
    - Returns a single combined DataFrame and also writes `combined.csv`

    Args:
        wav_dir: Folder containing `.wav` files.
        opensmile_home: Path to the openSMILE repo/install root.
        opensmile_default: Alias for `opensmile_home` (kept for user ergonomics).
        output_dir: Folder to write per-file CSVs + combined CSV.
            Defaults to `<wav_dir>/vao_output`.
        combined_csv: Optional explicit path for the combined CSV.
            Defaults to `<output_dir>/combined.csv`.
        preset: Preset name to use. Defaults to `egemapsv02_lld_25ms_10ms`.
        frame_step_s: Hop size in seconds (should match the wrapper config).
        write_per_file_csvs: If True, also write one CSV per recording.

    Returns:
        Combined DataFrame with a `recording` column.
    """

    if opensmile_home is None and opensmile_default is None:
        raise ValueError("Missing openSMILE path. Pass `opensmile_home=...` (or `opensmile_default=...`).")
    if opensmile_home is not None and opensmile_default is not None:
        raise ValueError("Pass only one of `opensmile_home` or `opensmile_default`.")

    resolved_opensmile_home = opensmile_home if opensmile_home is not None else opensmile_default
    assert resolved_opensmile_home is not None

    wav_dir_path = Path(wav_dir).expanduser()
    opensmile_home_path = Path(resolved_opensmile_home).expanduser()

    if output_dir is None:
        output_dir_path = wav_dir_path / "vao_output"
    else:
        output_dir_path = Path(output_dir).expanduser()

    preset_obj = get_preset(preset, opensmile_home=opensmile_home_path)

    return extract_features_folder(
        wav_dir_path,
        config_path=preset_obj.config_path,
        output_dir=output_dir_path,
        combined_csv=combined_csv,
        write_per_file_csvs=write_per_file_csvs,
        opensmile_home=opensmile_home_path,
        output_option="-csvoutput",
        extra_args=preset_obj.extra_args,
        frame_step_s=frame_step_s,
    )
