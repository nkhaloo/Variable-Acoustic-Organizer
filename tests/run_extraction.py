#!/usr/bin/env python3
"""Minimal example: one-call extraction.

Edit the two paths below and run:
    python tests/run_extraction.py
"""

from __future__ import annotations

from pathlib import Path

from vao import vao_extract


WAV_DIR = Path("tests/wav")
OUTPUT_DIR = Path("tests/output")
OPENSMILE_DEFAULT = Path("/Users/noahkhaloo/Desktop/opensmile")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = vao_extract(
        WAV_DIR,
        opensmile_default=OPENSMILE_DEFAULT,
        output_dir=OUTPUT_DIR,
    )

    print(df.head())
    print(f"Processed {df['recording'].nunique()} recordings")
    print(f"Combined rows: {len(df)}")

    out_csv = OUTPUT_DIR / "combined.csv"
    df.to_csv(out_csv, index=False, na_rep="NaN")
    print(f"Wrote combined CSV to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
