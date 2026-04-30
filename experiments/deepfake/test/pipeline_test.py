# Pipeline test: 5-fold cross validation with file-level grouping.
# All frames from the same recording stay in the same fold.
# Compares: frame+gate, frame only, file-level averaged.

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold

from vao import vao_extract

CORPUS_DIR = Path("/Users/noahkhaloo/Desktop/release_in_the_wild")
OPENSMILE_HOME = Path("/Users/noahkhaloo/Desktop/opensmile")
N_PER_CLASS = 200
N_FOLDS = 5
SEED = 42

NON_FEATURE_COLS = {
    "recording", "name", "frameTime", "frame_time", "time_s",
    "timestamp", "label", "segment_class",
}

random.seed(SEED)


def sample_files(folder: Path, n: int) -> list[Path]:
    files = list(folder.glob("*.wav"))
    return random.sample(files, min(n, len(files)))


def make_wav_dir(wav_paths: list[Path], wav_dir: Path) -> None:
    wav_dir.mkdir(parents=True, exist_ok=True)
    for p in wav_paths:
        link = wav_dir / p.name
        if not link.exists():
            link.symlink_to(p)


def extract_all(files: dict[str, list[Path]], tmp: Path,
                use_gate: bool, frame_level: bool) -> pd.DataFrame:
    dfs = []
    for label, paths in files.items():
        wav_dir = tmp / label / "wavs"
        make_wav_dir(paths, wav_dir)
        df = vao_extract(
            wav_dir,
            opensmile_default=OPENSMILE_HOME,
            output_dir=tmp / label / "vao_out",
            apply_gate=use_gate,
            mask_features=use_gate,
            frame_level=frame_level,
        )
        df["label"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def run_cv(name: str, df: pd.DataFrame, use_gate: bool, frame_level: bool) -> dict:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    if frame_level and use_gate:
        df = df[df["segment_class"] != "silence"].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS
                    and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].to_numpy(dtype=float)
    y = (df["label"] == "fake").astype(int).to_numpy()

    # Groups: one group per recording so files never split across folds
    if frame_level:
        groups = df["recording"].to_numpy()
    else:
        groups = df["recording"].to_numpy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    file_aucs, file_accs = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        model = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[te])[:, 1]

        if frame_level:
            fold_df = df.iloc[te].copy()
            fold_df["prob"] = probs
            file_probs = fold_df.groupby("recording")["prob"].mean()
            file_labels = (fold_df.groupby("recording")["label"].first() == "fake").astype(int)
        else:
            file_probs = pd.Series(probs, index=df.iloc[te]["recording"])
            file_labels = (df.iloc[te]["label"] == "fake").astype(int)
            file_labels.index = df.iloc[te]["recording"]

        file_auc = roc_auc_score(file_labels, file_probs)
        file_acc = accuracy_score(file_labels, (file_probs >= 0.5).astype(int))
        file_aucs.append(file_auc)
        file_accs.append(file_acc)
        print(f"  Fold {fold}: File AUC={file_auc:.3f}  File accuracy={file_acc:.3f}")

    mean_auc = np.mean(file_aucs)
    mean_acc = np.mean(file_accs)
    print(f"  Mean: File AUC={mean_auc:.3f} ± {np.std(file_aucs):.3f}  "
          f"File accuracy={mean_acc:.3f} ± {np.std(file_accs):.3f}")

    return {"File AUC": mean_auc, "File accuracy": mean_acc}


if __name__ == "__main__":
    print("Sampling files...")
    all_files = {
        "real": sample_files(CORPUS_DIR / "train" / "real", N_PER_CLASS),
        "fake": sample_files(CORPUS_DIR / "train" / "fake", N_PER_CLASS),
    }
    print(f"  {len(all_files['real'])} real, {len(all_files['fake'])} fake")

    tmp_base = Path("/tmp/vao_deepfake_cv")

    print("\nExtracting frame-level with gate...")
    df_gate = extract_all(all_files, tmp_base / "gate", use_gate=True, frame_level=True)

    print("Extracting frame-level without gate...")
    df_nogate = extract_all(all_files, tmp_base / "nogate", use_gate=False, frame_level=True)

    print("Extracting file-level averaged...")
    df_file = extract_all(all_files, tmp_base / "file", use_gate=False, frame_level=False)

    r1 = run_cv("Frame-level with gate",    df_gate,   use_gate=True,  frame_level=True)
    r2 = run_cv("Frame-level without gate", df_nogate, use_gate=False, frame_level=True)
    r3 = run_cv("File-level averaged",      df_file,   use_gate=False, frame_level=False)

    print("\n" + "="*60)
    print(f"  SUMMARY ({N_FOLDS}-fold CV, file-level groups, N={N_PER_CLASS}/class)")
    print("="*60)
    print(f"{'Metric':<18} {'Frame+Gate':>12} {'Frame':>10} {'File-level':>12}")
    print("-"*54)
    for metric in ["File AUC", "File accuracy"]:
        print(f"{metric:<18} {r1[metric]:>12.3f} {r2[metric]:>10.3f} {r3[metric]:>12.3f}")
