#!/usr/bin/env python3
"""Extract VAO frame-level acoustic features for ASVspoof5 Track 1.

Reads the metadata Parquet produced by extract_metadata.py, processes audio
files one at a time (FLAC → 16 kHz WAV → openSMILE → gate), and writes
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

from vao.audio_preprocess import preprocess_file
from vao.features import extract_features
from vao.opensmile_presets import get_preset

METADATA_COLS = [
    "split",
    "speaker_id",
    "flac_file_name",
    "audio_path",
    "gender",
    "codec",
    "codec_q",
    "codec_seed",
    "attack_tag",
    "attack_label",
    "key",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metadata",
        required=True,
        help="Path to asvspoof5_track1_metadata.parquet",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output Parquet chunks",
    )
    p.add_argument(
        "--opensmile-home",
        required=True,
        help="Path to openSMILE root (where config/ lives)",
    )
    p.add_argument(
        "--split",
        default="all",
        choices=["train", "dev", "eval", "all"],
        help="Which split to process (default: all)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most N files total; useful for debugging",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of utterances per output Parquet file (default: 1000)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing chunk files instead of skipping them",
    )
    p.add_argument(
        "--preset",
        default="egemapsv02_lld_25ms_10ms",
        help="VAO openSMILE preset name (default: egemapsv02_lld_25ms_10ms)",
    )
    p.add_argument(
        "--no-gate",
        action="store_true",
        help="Skip segment_class (obstruent/sonorant/silence) labeling",
    )
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


def process_one(
    row: pd.Series,
    preset_obj,
    opensmile_home: Path,
    tmp_dir: Path,
    apply_gate: bool,
) -> pd.DataFrame:
    """FLAC → WAV → features → optional gate → annotated DataFrame."""
    audio_path = Path(row["audio_path"])

    wav_path = preprocess_file(audio_path, tmp_dir)

    df = extract_features(
        wav_path,
        config_path=preset_obj.config_path,
        opensmile_home=opensmile_home,
        output_option="-csvoutput",
        extra_args=preset_obj.extra_args,
        frame_step_s=0.010,
    )

    if apply_gate:
        from vao.gate.classifier import apply_gate as _apply_gate
        df = _apply_gate(df)

    df.insert(0, "frame_idx", range(len(df)))

    # Prepend metadata columns in order (insert at 0 reverses, so iterate reversed).
    for col in reversed(METADATA_COLS):
        df.insert(0, col, row[col])

    return df


def chunk_filename(split: str, chunk_idx: int) -> str:
    return f"{split}_part_{chunk_idx:03d}.parquet"


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

    # Drop rows where audio is confirmed missing.
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
    preset_obj = get_preset(args.preset, opensmile_home=opensmile_home)
    apply_gate = not args.no_gate

    splits_to_process = sorted(metadata["split"].unique())

    for split in splits_to_process:
        split_df = metadata[metadata["split"] == split].reset_index(drop=True)
        n = len(split_df)
        chunk_size = args.chunk_size
        n_chunks = (n + chunk_size - 1) // chunk_size

        log.info(
            "Split '%s': %d utterances → %d chunks of up to %d",
            split, n, n_chunks, chunk_size,
        )

        for chunk_idx in range(n_chunks):
            chunk_path = output_dir / chunk_filename(split, chunk_idx)

            if chunk_path.exists() and not args.overwrite:
                log.info("  Chunk %s exists, skipping", chunk_path.name)
                continue

            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n)
            chunk_rows = split_df.iloc[start:end]

            log.info(
                "  Chunk %d/%d (%s): utterances %d–%d",
                chunk_idx + 1, n_chunks, split, start, end - 1,
            )

            frames: list[pd.DataFrame] = []
            chunk_failures: list[dict] = []

            # One shared temp dir per chunk; cleaned up when the block exits.
            with tempfile.TemporaryDirectory(prefix="vao_pp_") as tmp_str:
                tmp_dir = Path(tmp_str)

                for i, (_, row) in enumerate(chunk_rows.iterrows()):
                    try:
                        utt_df = process_one(
                            row, preset_obj, opensmile_home, tmp_dir, apply_gate
                        )
                        frames.append(utt_df)
                    except Exception as exc:
                        log.error("  FAILED %s: %s", row.get("audio_path", "?"), exc)
                        chunk_failures.append(
                            {
                                "split": row.get("split", ""),
                                "flac_file_name": row.get("flac_file_name", ""),
                                "audio_path": row.get("audio_path", ""),
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )

                    if (i + 1) % 100 == 0:
                        log.info(
                            "    %d/%d utterances done in chunk %d",
                            i + 1, end - start, chunk_idx + 1,
                        )

            append_failures(fail_log_path, chunk_failures)

            if frames:
                chunk_df = pd.concat(frames, ignore_index=True)
                chunk_df.to_parquet(chunk_path, index=False)
                log.info(
                    "  Wrote %d frames (%d utterances) to %s",
                    len(chunk_df), len(frames), chunk_path.name,
                )
                del chunk_df
            else:
                log.warning("  No frames produced for chunk %d — skipping write", chunk_idx)

            del frames

    if fail_log_path.exists():
        log.warning("Some files failed — see %s", fail_log_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
