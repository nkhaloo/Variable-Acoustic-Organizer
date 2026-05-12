"""Extract VAO frame-level features for all ASVspoof5 Track 1 audio, save to Parquet."""
import sys; print("Starting...", flush=True); sys.stdout.flush()

import tempfile
from pathlib import Path

import pandas as pd
from vao import vao_extract

METADATA = Path(__file__).parent / "output" / "asvspoof5_track1_metadata.parquet"
OPENSMILE_HOME = Path("/home/nkhaloo/Desktop/opensmile")
OUT_PARQUET = Path(__file__).parent / "output" / "frame_features.parquet"

print("Loading metadata...")
meta = pd.read_parquet(METADATA)
print(f"  {len(meta):,} utterances loaded")

with tempfile.TemporaryDirectory(prefix="vao_extract_") as tmp:
    tmp_dir = Path(tmp)
    print("Symlinking audio files...")
    for _, row in meta.iterrows():
        src = Path(row["audio_path"])
        (tmp_dir / src.name).symlink_to(src)
    print(f"  {len(meta):,} files ready")

    print("Running vao_extract (this will take a while)...")
    df = vao_extract(tmp_dir, opensmile_default=OPENSMILE_HOME, apply_gate=False, normalize=False, preprocess=False)
    print(f"  Extraction done: {len(df):,} frames")

print("Merging metadata...")
df["flac_file_name"] = df["recording"].str.rsplit(".", n=1).str[0]
df = df.merge(meta.drop(columns=["audio_exists"], errors="ignore"), on="flac_file_name", how="left")

print("Saving to Parquet...")
df.to_parquet(OUT_PARQUET, index=False, compression="zstd")
print(f"Done. Saved {len(df):,} frames to {OUT_PARQUET}")
