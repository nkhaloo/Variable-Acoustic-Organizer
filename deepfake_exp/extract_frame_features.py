"""Extract VAO frame-level acoustic features for ASVspoof5 Track 1.

Reads the metadata Parquet produced by extract_metadata.py, processes audio
files in chunks using vao_extract (no gating, no normalization), and writes
chunked Parquet output.  Resumable: existing chunks are skipped unless
--overwrite is set.

Example usage on the SSH machine:
    python deepfake_exp/extract_frame_features.py \
        --metadata output/asvspoof5_track1_metadata.parquet \
        --output-dir output/frame_features \
        --opensmile-home /path/to/opensmile \
        --split train \
        --chunk-size 1000
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd
from vao import vao_extract


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metadata", required=True, help="Path to asvspoof5_track1_metadata.parquet")
    p.add_argument("--output-dir", required=True, help="Directory to write output Parquet chunks")
    p.add_argument("--opensmile-home", required=True, help="Path to openSMILE root (where config/ lives)")
    p.add_argument("--split", default="all", choices=["train", "dev", "eval", "all"])
    p.add_argument("--max-files", type=int, default=None, help="Cap number of files (for debugging)")
    p.add_argument("--chunk-size", type=int, default=1000, help="Utterances per output Parquet file")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk files")
    p.add_argument("--csv", action="store_true", help="Save as CSV instead of Parquet (useful for debugging)")
    return p.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("extract_frame_features")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(output_dir / "extract.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def chunk_filename(split: str, chunk_idx: int, csv: bool = False) -> str:
    ext = "csv" if csv else "parquet"
    return f"{split}_part_{chunk_idx:03d}.{ext}"


def append_failures(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "flac_file_name", "audio_path", "error", "traceback"],
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_dir)
    fail_log_path = output_dir / "failed_files.tsv"

    log.info("Reading metadata from %s", args.metadata)
    metadata = pd.read_parquet(args.metadata)
    log.info("Metadata shape: %s", metadata.shape)

    if args.split != "all":
        metadata = metadata[metadata["split"] == args.split].copy()

    if "audio_exists" in metadata.columns:
        n_missing = (~metadata["audio_exists"].astype(bool)).sum()
        if n_missing:
            log.warning("Dropping %d rows with audio_exists=False", n_missing)
        metadata = metadata[metadata["audio_exists"].astype(bool)].copy()

    if args.max_files is not None:
        metadata = metadata.head(args.max_files).copy()
        log.info("--max-files %d: capped to %d rows", args.max_files, len(metadata))

    log.info("Total utterances to process: %d", len(metadata))

    opensmile_home = Path(args.opensmile_home)
    meta_cols = [c for c in metadata.columns if c != "audio_exists"]

    for split in sorted(metadata["split"].unique()):
        split_df = metadata[metadata["split"] == split].reset_index(drop=True)
        n = len(split_df)
        n_chunks = (n + args.chunk_size - 1) // args.chunk_size

        log.info("Split '%s': %d utterances → %d chunks", split, n, n_chunks)

        for chunk_idx in range(n_chunks):
            chunk_path = output_dir / chunk_filename(split, chunk_idx, csv=args.csv)

            if chunk_path.exists() and not args.overwrite:
                log.info("  Chunk %s exists, skipping", chunk_path.name)
                continue

            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, n)
            chunk_meta = split_df.iloc[start:end]

            log.info("  Chunk %d/%d: utterances %d–%d", chunk_idx + 1, n_chunks, start, end - 1)

            try:
                with tempfile.TemporaryDirectory(prefix="vao_chunk_") as tmp_str:
                    tmp_dir = Path(tmp_str)
                    for _, row in chunk_meta.iterrows():
                        src = Path(row["audio_path"])
                        (tmp_dir / src.name).symlink_to(src)

                    df = vao_extract(
                        tmp_dir,
                        opensmile_default=opensmile_home,
                        apply_gate=False,
                        normalize=False,
                    )

                # Same join as practice_extract.py: strip .wav suffix to match flac_file_name
                df["flac_file_name"] = df["recording"].str.rsplit(".", n=1).str[0]
                df = df.merge(chunk_meta[meta_cols], on="flac_file_name", how="left")

                if args.csv:
                    df.to_csv(chunk_path, index=False, na_rep="NaN")
                else:
                    df.to_parquet(chunk_path, index=False, compression="zstd")
                log.info("  Wrote %d frames to %s", len(df), chunk_path.name)
                del df

            except Exception as exc:
                log.error("  Chunk %d FAILED: %s", chunk_idx, exc)
                failures = [
                    {
                        "split": row["split"],
                        "flac_file_name": row["flac_file_name"],
                        "audio_path": row["audio_path"],
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    for _, row in chunk_meta.iterrows()
                ]
                append_failures(fail_log_path, failures)

    if fail_log_path.exists():
        log.warning("Some chunks failed — see %s", fail_log_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
