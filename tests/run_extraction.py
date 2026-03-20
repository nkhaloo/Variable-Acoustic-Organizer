#!/usr/bin/env python3
"""Run VAO feature extraction on a folder of WAVs.

This is a reference script showing how to execute VAO against an openSMILE
SMILExtract build.

Example:
    python tests/run_extraction.py --wav-dir tests/wav --output-dir tests/output

Notes:
- This script assumes you have built openSMILE and can point to it via
    a local openSMILE folder (repo root).
- By default, this script uses the hardcoded `OPENSMILE_HOME_DEFAULT` below.
    Users should edit that one line to match their computer.
- By default, this script uses a VAO preset (eGeMAPSv02 at frame-level).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vao import extract_features_folder
from vao.opensmile_presets import get_preset, list_presets


DEFAULT_PRESET = "egemapsv02_lld_25ms_1ms"

# Edit this to match where openSMILE lives on your machine.
OPENSMILE_HOME_DEFAULT = Path("/Users/noahkhaloo/Desktop/opensmile")


def _default_opensmile_home() -> Path | None:
    return OPENSMILE_HOME_DEFAULT if OPENSMILE_HOME_DEFAULT.is_dir() else None


def _resolve_config_path(*, opensmile_home: Path, config: Path | None) -> Path:
    """Resolve config path.

    If `config` is None, use a default path relative to `opensmile_home`.
    If `config` is relative, interpret it relative to `opensmile_home`.
    If `config` is absolute, use it as-is.
    """

    if config is None:
        raise ValueError("config is None; use preset resolution")
    config = config.expanduser()
    if config.is_absolute():
        return config.resolve()
    return (opensmile_home / config).resolve()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Extract openSMILE features for a folder of WAV files")
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=repo_root / "tests" / "wav",
        help="Folder containing .wav files (default: tests/wav)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "tests" / "output",
        help="Folder to write per-file CSVs + combined.csv (default: tests/output)",
    )
    parser.add_argument(
        "--opensmile-home",
        type=Path,
        default=None,
        help=(
            "Path to openSMILE repo/install root. If omitted, uses OPENSMILE_HOME env var, "
            "otherwise falls back to OPENSMILE_HOME_DEFAULT inside this script." 
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        help=(
            "Name of a VAO preset config. Defaults to a maximal frame-level set. "
            f"Supported: {', '.join(list_presets())}"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to openSMILE .conf file. If relative, it is resolved relative to OPENSMILE_HOME. "
            "If provided, this overrides --preset."
        ),
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="Path for the combined CSV (default: <output-dir>/combined.csv)",
    )

    args = parser.parse_args()

    opensmile_home = args.opensmile_home or (
        Path(os.environ["OPENSMILE_HOME"]).expanduser() if os.environ.get("OPENSMILE_HOME") else None
    )
    opensmile_home = opensmile_home or _default_opensmile_home()

    if opensmile_home is None:
        raise SystemExit(
            "Missing openSMILE location. Edit OPENSMILE_HOME_DEFAULT in tests/run_extraction.py, "
            "or set OPENSMILE_HOME, or pass --opensmile-home." 
        )

    opensmile_home = opensmile_home.expanduser().resolve()

    extra_args: tuple[str, ...] = ()
    if args.config is not None:
        config_path = _resolve_config_path(opensmile_home=opensmile_home, config=args.config)
    else:
        preset = get_preset(args.preset, opensmile_home=opensmile_home)
        config_path = preset.config_path
        extra_args = preset.extra_args

    combined = extract_features_folder(
        args.wav_dir,
        config_path=config_path,
        output_dir=args.output_dir,
        combined_csv=args.combined_csv,
        opensmile_home=opensmile_home,
        output_option="-csvoutput",
        extra_args=extra_args,
        frame_step_s=0.001,
    )

    print(f"Processed {combined['recording'].nunique()} recordings")
    print(f"Combined rows: {len(combined)}")
    print(f"Wrote per-file CSVs to: {args.output_dir}")
    if args.combined_csv is not None:
        print(f"Wrote combined CSV to: {args.combined_csv}")
    else:
        print(f"Wrote combined CSV to: {Path(args.output_dir) / 'combined.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
