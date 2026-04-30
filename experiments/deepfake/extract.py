# Runs vao_extract on the Release in the Wild corpus (train/val/test splits)
# and saves each split as a Parquet file with a real/fake label column.
# Run this once — outputs are used by train.py for classifier training.

from pathlib import Path
import pandas as pd
from vao import vao_extract

CORPUS_DIR = Path("/Users/noahkhaloo/Desktop/release_in_the_wild")
OUTPUT_DIR = Path("/Users/noahkhaloo/Desktop/release_in_the_wild_features")
OPENSMILE_HOME = Path("/Users/noahkhaloo/Desktop/opensmile")

SPLITS = ["train", "val", "test"]


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    # recording column is like "real/filename.wav" or "fake/filename.wav"
    # since we ran vao_extract with the split folder as root
    df["label"] = df["recording"].apply(
        lambda r: "real" if r.startswith("real/") else "fake"
    )
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_dir = CORPUS_DIR / split
        if not split_dir.is_dir():
            print(f"Skipping {split} (folder not found)")
            continue

        print(f"\nExtracting {split}...")
        df = vao_extract(
            split_dir,
            opensmile_default=OPENSMILE_HOME,
            output_dir=OUTPUT_DIR / f"{split}_tmp",
            recursive=True,
            apply_gate=True,
        )

        df = add_label(df)

        out_path = OUTPUT_DIR / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {len(df):,} frames, {df['recording'].nunique()} recordings → {out_path}")

    print("\nDone.")
