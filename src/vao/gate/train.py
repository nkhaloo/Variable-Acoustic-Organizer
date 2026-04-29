"""Train the obstruent/sonorant/silence gate classifier on TIMIT.

Usage:
    python -m vao.gate.train \\
        --timit /path/to/archive/data \\
        --train-csv /path/to/timit_train.csv \\
        --test-csv /path/to/timit_test.csv \\
        --out src/vao/gate/model.joblib

Requires: scikit-learn, joblib, pandas, numpy
"""

#trains model on dataset from extract_timit.py, then tests it and reports results 

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

try:
    from .phoneme_map import timit_label
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from vao.gate.phoneme_map import timit_label

SAMPLE_RATE = 16_000
FEATURES = [
    "HNRdBACF_sma3nz",
    "voicingFinalUnclipped",
    "pcm_fftMag_spectralEntropy_sma",
    "pcm_fftMag_spectralHarmonicity_sma",
    "Loudness_sma3",
    "spectralFlux_sma3",
    "alphaRatio_sma3",
]

TIME_COLS = {"time_s", "frameTime", "frame_time", "timestamp", "time", "Time"}


def parse_phn(phn_path: Path) -> list[tuple[float, float, str]]:
    """Parse a TIMIT .PHN file into (start_s, end_s, phoneme) tuples."""
    segments = []
    for line in phn_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        segments.append((int(parts[0]) / SAMPLE_RATE, int(parts[1]) / SAMPLE_RATE, parts[2]))
    return segments


def recording_to_phn(recording: str, timit_data_root: Path, split: str) -> Path | None:
    """Reconstruct the PHN path from a recording column value.

    recording is a relative path like DR1/FCJF0/SA1.WAV.wav (relative to split root)
    Maps to: timit_data_root/TRAIN/DR1/FCJF0/SA1.PHN
    """
    rel = Path(recording)
    utt_stem = Path(rel.stem).stem  # SA1.WAV.wav → SA1.WAV → SA1
    phn_path = timit_data_root / split / rel.parent / f"{utt_stem}.PHN"
    return phn_path if phn_path.is_file() else None


def label_frames(time_s: np.ndarray, segments: list[tuple[float, float, str]]) -> list[str | None]:
    """Assign a class label to each frame based on its timestamp."""
    labels = []
    seg_idx = 0
    for t in time_s:
        while seg_idx < len(segments) - 1 and t >= segments[seg_idx][1]:
            seg_idx += 1
        start, end, phoneme = segments[seg_idx]
        labels.append(timit_label(phoneme) if start <= t < end else None)
    return labels


def build_dataset(timit_data_root: Path, csv_path: Path, split: str) -> pd.DataFrame:
    """Join VAO features with TIMIT phoneme labels from a combined CSV."""
    df = pd.read_csv(csv_path)

    time_col = next((c for c in df.columns if c in TIME_COLS), None)
    if "recording" not in df.columns or time_col is None:
        raise ValueError(f"Combined CSV must have 'recording' and a time column: {csv_path}")

    results = []
    skipped = 0
    for recording, group in df.groupby("recording"):
        phn_path = recording_to_phn(str(recording), timit_data_root, split)
        if phn_path is None:
            skipped += 1
            continue
        segments = parse_phn(phn_path)
        group = group.copy()
        group["label"] = label_frames(group[time_col].to_numpy(), segments)
        results.append(group)

    if skipped:
        print(f"  Skipped {skipped} recordings (no matching PHN file)")
    if not results:
        raise RuntimeError("No labeled frames produced. Check that recording paths match TIMIT structure.")

    return pd.concat(results, ignore_index=True)


def prepare(df: pd.DataFrame, available_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Drop unlabeled/NaN rows and return (X, y)."""
    df = df.dropna(subset=["label"] + available_features)
    return df[available_features].to_numpy(dtype=float), df["label"].to_numpy()


def train(
    timit_data_root: Path,
    train_csv: Path,
    test_csv: Path,
    out_path: Path,
) -> None:
    print("Building training set...")
    train_df = build_dataset(timit_data_root, train_csv, "TRAIN")

    print("Building test set...")
    test_df = build_dataset(timit_data_root, test_csv, "TEST")

    available_features = [f for f in FEATURES if f in train_df.columns]
    missing = set(FEATURES) - set(available_features)
    if missing:
        print(f"Warning: features not found in VAO output, skipping: {missing}")

    X_train, y_train = prepare(train_df, available_features)
    X_test, y_test = prepare(test_df, available_features)

    print(f"Train: {len(X_train):,} frames — {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test:  {len(X_test):,} frames — {pd.Series(y_test).value_counts().to_dict()}")

    model = Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)),
    ])

    # GradientBoostingClassifier has no class_weight parameter, so we compute
    # sample weights manually. Each frame is weighted inversely proportional to
    # its class frequency — obstruent and silence frames get upweighted so the
    # model can't just default to predicting sonorant for everything.
    class_counts = pd.Series(y_train).value_counts()
    # square root dampens the correction — less aggressive than full inverse frequency
    sample_weight = np.array([1.0 / np.sqrt(class_counts[label]) for label in y_train])

    print("\nTraining...")
    model.fit(X_train, y_train, clf__sample_weight=sample_weight)

    print("\nEvaluation on TIMIT test split:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    results_df = pd.DataFrame(report).transpose().round(3)
    results_path = out_path.parent / "model_results.csv"
    results_df.to_csv(results_path)
    print(f"Results saved to {results_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": available_features}, out_path)
    print(f"Model saved to {out_path}")


TIMIT_DATA_DIR = Path("/Users/noahkhaloo/Desktop/archive/data")
TRAIN_CSV = Path("/Users/noahkhaloo/Desktop/timit_vao_output/train.csv")
TEST_CSV = Path("/Users/noahkhaloo/Desktop/timit_vao_output/test.csv")
MODEL_OUT = Path(__file__).resolve().parent / "model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VAO gate classifier on TIMIT.")
    parser.add_argument("--timit", default=TIMIT_DATA_DIR, type=Path)
    parser.add_argument("--train-csv", default=TRAIN_CSV, type=Path)
    parser.add_argument("--test-csv", default=TEST_CSV, type=Path)
    parser.add_argument("--out", default=MODEL_OUT, type=Path)
    args = parser.parse_args()

    train(args.timit, args.train_csv, args.test_csv, args.out)


if __name__ == "__main__":
    main()


# precision = of all the frames the model called 'X', it was right X% of the time 
# recall = of all the X frames in the dataset, the model only found X% of them

# high precision but low recall = when the model called something 'X' it was right most of the time, but it missed alot of frames 
