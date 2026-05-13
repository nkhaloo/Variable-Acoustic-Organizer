print("Starting...", flush=True)

import tempfile
from pathlib import Path

import pandas as pd
from vao import vao_extract

METADATA    = Path(__file__).parent / "output" / "asvspoof5_track1_metadata.parquet"
OPENSMILE_HOME = Path("/home/nkhaloo/Desktop/opensmile")
OUT_DIR     = Path(__file__).parent / "output" / "frame_features"
CHUNK_SIZE  = 1000

OUT_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(Path.home() / "tmp")
Path(tempfile.tempdir).mkdir(exist_ok=True)

print("Loading metadata...")
meta = pd.read_parquet(METADATA)
meta = meta[meta["audio_exists"].astype(bool)].reset_index(drop=True)
meta_cols = [c for c in meta.columns if c != "audio_exists"]
print(f"  {len(meta):,} utterances")

for split in sorted(meta["split"].unique()):
    split_df = meta[meta["split"] == split].reset_index(drop=True)
    n = len(split_df)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\nSplit '{split}': {n:,} utterances → {n_chunks} chunks")

    for chunk_idx in range(n_chunks):
        chunk_path = OUT_DIR / f"{split}_part_{chunk_idx:03d}.parquet"

        if chunk_path.exists():
            print(f"  Chunk {chunk_path.name} exists, skipping")
            continue

        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        chunk_meta = split_df.iloc[start:end]
        print(f"  Chunk {chunk_idx + 1}/{n_chunks}: files {start}–{end - 1}", flush=True)

        try:
            with tempfile.TemporaryDirectory(prefix="vao_chunk_") as tmp:
                tmp_dir = Path(tmp)
                for _, row in chunk_meta.iterrows():
                    src = Path(row["audio_path"])
                    (tmp_dir / src.name).symlink_to(src)

                df = vao_extract(tmp_dir, opensmile_default=OPENSMILE_HOME, apply_gate=False, normalize=False)

            df["flac_file_name"] = df["recording"].str.rsplit(".", n=1).str[0]
            df = df.merge(chunk_meta[meta_cols], on="flac_file_name", how="left")
            df.to_parquet(chunk_path, index=False, compression="zstd")
            print(f"    Saved {len(df):,} frames to {chunk_path.name}", flush=True)
            del df

        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            with (OUT_DIR / "failed.log").open("a") as f:
                f.write(f"{split} chunk {chunk_idx}: {exc}\n")

print("\nDone.")
