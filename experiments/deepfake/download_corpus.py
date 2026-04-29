# Downloads a balanced subset of Release in the Wild (~3hrs real + ~3hrs fake)
# Respects the existing train/val/test split from the dataset.
# Requires: pip install kaggle, with ~/.kaggle/kaggle.json configured

import random
import subprocess
from pathlib import Path

from kaggle import api

DATASET = "bhaveshkumars/release-in-the-wild"
TARGET_HOURS_PER_CLASS = 3.0
TARGET_BYTES_PER_CLASS = int(TARGET_HOURS_PER_CLASS * 3600 * 16000 * 2)  # 16kHz, 16-bit mono
OUTPUT_DIR = Path("/Users/noahkhaloo/Desktop/release_in_the_wild")
SEED = 42

random.seed(SEED)

SPLITS = ["train", "val", "test"]
SPLIT_WEIGHTS = {"train": 0.70, "val": 0.15, "test": 0.15}


def list_all_files() -> list[dict]:
    """Fetch all WAV file metadata via Kaggle SDK with pagination."""
    files = []
    page_token = None
    while True:
        result = api.dataset_list_files(DATASET, page_token=page_token, page_size=20)
        for f in result.files:
            if f.name.endswith(".wav"):
                files.append({"name": f.name, "size": f.total_bytes})
        page_token = result.next_page_token
        if not page_token:
            break
    return files


def sample_split(files: list[dict], split: str, label: str, target_bytes: int) -> list[dict]:
    """Randomly sample files from one split/class until target duration is hit."""
    subset = [f for f in files if f"/{split}/{label}/" in f["name"]]
    random.shuffle(subset)
    selected = []
    total = 0
    for f in subset:
        if total >= target_bytes:
            break
        selected.append(f)
        total += f["size"]
    return selected


def download_file(name: str) -> None:
    rel = Path(name).relative_to("release_in_the_wild")
    dest = OUTPUT_DIR / rel.parent
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", DATASET, "--file", name,
         "--path", str(dest), "--unzip"],
        check=True,
    )


if __name__ == "__main__":
    print("Fetching file list (may take a moment due to pagination)...")
    all_files = list_all_files()
    print(f"  Total WAV files found: {len(all_files)}")

    selected = []
    for split in SPLITS:
        target = int(TARGET_BYTES_PER_CLASS * SPLIT_WEIGHTS[split])
        for label in ["real", "fake"]:
            subset = sample_split(all_files, split, label, target)
            hours = sum(f["size"] for f in subset) / (3600 * 16000 * 2)
            print(f"  {split}/{label}: {len(subset)} files, ~{hours:.1f} hours")
            selected.extend(subset)

    total_hours = sum(f["size"] for f in selected) / (3600 * 16000 * 2)
    print(f"\nTotal: {len(selected)} files, ~{total_hours:.1f} hours")
    print(f"Downloading to {OUTPUT_DIR}...")

    for i, f in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {f['name']}")
        download_file(f["name"])

    print("\nDone.")
