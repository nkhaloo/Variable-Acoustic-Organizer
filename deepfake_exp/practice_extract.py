"""Sample 1 file from each split, run vao_extract, save to CSV."""

import tempfile
from pathlib import Path

import pandas as pd
from vao import vao_extract

METADATA = Path(__file__).parent / "output" / "asvspoof5_track1_metadata.parquet"
OPENSMILE_HOME = Path("/Users/noahkhaloo/Desktop/opensmile")
OUT_CSV = Path(__file__).parent / "output_test.csv"

meta = pd.read_parquet(METADATA)
sample = meta.groupby("split", group_keys=False).apply(lambda g: g.sample(1)).reset_index(drop=True)

with tempfile.TemporaryDirectory(prefix="vao_test_") as tmp:
    tmp_dir = Path(tmp)
    for _, row in sample.iterrows():
        src = Path(row["audio_path"])
        (tmp_dir / src.name).symlink_to(src)

    df = vao_extract(tmp_dir, opensmile_default=OPENSMILE_HOME, apply_gate=True, normalize=False)

df.to_csv(OUT_CSV, index=False, na_rep="NaN")
print(f"Saved {len(df):,} frames to {OUT_CSV}")
