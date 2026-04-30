# Trains a binary real/fake deepfake classifier on Release in the Wild features.
# Reads Parquet files produced by extract.py.
# Uses HistGradientBoostingClassifier (200 trees, handles NaN natively).
# 5-fold file-level cross validation on train split, final eval on val and test.

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

FEATURES_DIR = Path("/Users/noahkhaloo/Desktop/release_in_the_wild_features")
MODEL_OUT = Path(__file__).parent / "deepfake_model.joblib"

# Columns that are not acoustic features
NON_FEATURE_COLS = {
    "recording", "name", "frameTime", "frame_time", "time_s",
    "timestamp", "label", "segment_class",
}


def load_split(name: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run extract.py first.")
    df = pd.read_parquet(path)
    # Drop silence frames — all features are NaN, no information
    return df[df["segment_class"] != "silence"].reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == "fake").astype(int).to_numpy()


def file_level_auc(df: pd.DataFrame, probs: np.ndarray) -> float:
    """Aggregate frame probabilities to file level and compute AUC."""
    df = df.copy()
    df["prob_fake"] = probs
    file_probs = df.groupby("recording")["prob_fake"].mean()
    file_labels = df.groupby("recording")["label"].first().map({"real": 0, "fake": 1})
    return roc_auc_score(file_labels, file_probs)


def file_level_accuracy(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5) -> float:
    df = df.copy()
    df["prob_fake"] = probs
    file_probs = df.groupby("recording")["prob_fake"].mean()
    file_preds = (file_probs >= threshold).astype(int)
    file_labels = df.groupby("recording")["label"].first().map({"real": 0, "fake": 1})
    return accuracy_score(file_labels, file_preds)


if __name__ == "__main__":
    print("Loading data...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    feature_cols = get_feature_cols(train_df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train frames: {len(train_df):,} ({train_df['recording'].nunique()} recordings)")
    print(f"  Val frames:   {len(val_df):,} ({val_df['recording'].nunique()} recordings)")
    print(f"  Test frames:  {len(test_df):,} ({test_df['recording'].nunique()} recordings)")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = encode_labels(train_df)
    groups = train_df["recording"].to_numpy()

    # 5-fold file-level cross validation
    print("\n5-fold file-level cross validation on train split...")
    gkf = GroupKFold(n_splits=5)
    cv_frame_aucs, cv_file_aucs = [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups), 1):
        fold_model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        fold_model.fit(X_train[tr_idx], y_train[tr_idx])
        fold_probs = fold_model.predict_proba(X_train[val_idx])[:, 1]
        fold_df = train_df.iloc[val_idx]

        frame_auc = roc_auc_score(y_train[val_idx], fold_probs)
        file_auc = file_level_auc(fold_df, fold_probs)
        cv_frame_aucs.append(frame_auc)
        cv_file_aucs.append(file_auc)
        print(f"  Fold {fold}: frame AUC={frame_auc:.3f}  file AUC={file_auc:.3f}")

    print(f"  Mean frame AUC: {np.mean(cv_frame_aucs):.3f} ± {np.std(cv_frame_aucs):.3f}")
    print(f"  Mean file AUC:  {np.mean(cv_file_aucs):.3f} ± {np.std(cv_file_aucs):.3f}")

    # Train final model on all train data
    print("\nTraining final model on full train split...")
    model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # Val evaluation
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = encode_labels(val_df)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    print("\nVal set results:")
    print(f"  Frame AUC:      {roc_auc_score(y_val, val_probs):.3f}")
    print(f"  Frame accuracy: {accuracy_score(y_val, val_preds):.3f}")
    print(f"  File AUC:       {file_level_auc(val_df, val_probs):.3f}")
    print(f"  File accuracy:  {file_level_accuracy(val_df, val_probs):.3f}")
    print(classification_report(y_val, val_preds, target_names=["real", "fake"]))

    # Test evaluation
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = encode_labels(test_df)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    print("\nTest set results:")
    print(f"  Frame AUC:      {roc_auc_score(y_test, test_probs):.3f}")
    print(f"  Frame accuracy: {accuracy_score(y_test, test_preds):.3f}")
    print(f"  File AUC:       {file_level_auc(test_df, test_probs):.3f}")
    print(f"  File accuracy:  {file_level_accuracy(test_df, test_probs):.3f}")
    print(classification_report(y_test, test_preds, target_names=["real", "fake"]))

    joblib.dump({"model": model, "features": feature_cols}, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")
