"""Extract VAO frame-level features for all ASVspoof5 Track 1 audio, save to Parquet."""

import tempfile
from pathlib import Path

import pandas as pd
from vao import vao_extract

METADATA = Path(__file__).parent / "output" / "asvspoof5_track1_metadata.parquet"
OPENSMILE_HOME = Path("/home/nkhaloo/Desktop/opensmile")
OUT_PARQUET = Path(__file__).parent / "output" / "frame_features.parquet"

meta = pd.read_parquet(METADATA)
meta = meta.groupby("split", group_keys=False).apply(lambda g: g.sample(1)).reset_index(drop=True)

with tempfile.TemporaryDirectory(prefix="vao_extract_") as tmp:
    tmp_dir = Path(tmp)
    for _, row in meta.iterrows():
        src = Path(row["audio_path"])
        (tmp_dir / src.name).symlink_to(src)

    df = vao_extract(tmp_dir, opensmile_default=OPENSMILE_HOME, apply_gate=False, normalize=False)

df["flac_file_name"] = df["recording"].str.rsplit(".", n=1).str[0]
df = df.merge(meta.drop(columns=["audio_exists"], errors="ignore"), on="flac_file_name", how="left")

out = Path(__file__).parent / "practice_output.csv"
df.to_csv(out, index=False, na_rep="NaN")
print(f"Saved {len(df):,} frames to {out}")
