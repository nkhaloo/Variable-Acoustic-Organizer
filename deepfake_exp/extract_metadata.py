from pathlib import Path
import pandas as pd

BASE_DIR = Path.home() / "Desktop" / "asvspoof5"
PROTOCOL_DIR = BASE_DIR / "protocol"
AUDIO_DIR = BASE_DIR / "audio"
REPO_DIR = Path.home() / "Desktop" / "Variable-Acoustic-Organizer"
OUT_DIR = REPO_DIR / "deepfake_exp" 
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "speaker_id",
    "flac_file_name",
    "gender",
    "codec",
    "codec_q",
    "codec_seed",
    "attack_tag",
    "attack_label",
    "key",
    "tmp",
]

def load_protocol(split_name: str, protocol_file: str, audio_subdir: str) -> pd.DataFrame:
    """Load one ASVspoof Track 1 protocol file and attach audio paths."""
    protocol_path = PROTOCOL_DIR / protocol_file
    audio_root = AUDIO_DIR / audio_subdir

    df = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        names=COLUMNS,
        header=None,
        dtype=str,
    )

    df.insert(0, "split", split_name)

    df["audio_path"] = df["flac_file_name"].apply(
        lambda x: str(audio_root / f"{x}.flac")
    )

    df = df[
        [
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
    ]

    return df


train = load_protocol(
    split_name="train",
    protocol_file="ASVspoof5.train.tsv",
    audio_subdir="flac_T",
)

dev = load_protocol(
    split_name="dev",
    protocol_file="ASVspoof5.dev.track_1.tsv",
    audio_subdir="flac_D",
)

eval_ = load_protocol(
    split_name="eval",
    protocol_file="ASVspoof5.eval.track_1.tsv",
    audio_subdir="flac_E_eval",
)

metadata = pd.concat([train, dev, eval_], ignore_index=True)

# Check whether all referenced audio files exist.
metadata["audio_exists"] = metadata["audio_path"].apply(lambda p: Path(p).exists())

print("Rows by split:")
print(metadata["split"].value_counts())

print("\nLabels:")
print(metadata["key"].value_counts())

print("\nCodec distribution by split:")
print(pd.crosstab(metadata["split"], metadata["codec"]))

print("\nMissing audio files:")
missing = metadata.loc[~metadata["audio_exists"]]
print(len(missing))

if len(missing) > 0:
    print(missing.head(20))

# Save metadata.
metadata_path = OUT_DIR / "output" / "asvspoof5_track1_metadata.parquet"
metadata_path.parent.mkdir(parents=True, exist_ok=True)
metadata.to_parquet(metadata_path, index=False)

print(f"\nSaved metadata to: {metadata_path}")