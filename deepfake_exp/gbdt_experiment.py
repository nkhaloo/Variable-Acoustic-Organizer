"""GBDT spoof detection experiment.

Samples 250 utterances from train and 250 from eval, extracts VAO frame-level
features (no gate, no normalization), trains a GradientBoostingClassifier on
train frames, evaluates on eval frames, and reports accuracy, AUC, and top 15
feature importances.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from vao import vao_extract

METADATA       = Path(__file__).parent / "output" / "asvspoof5_track1_metadata.parquet"
OPENSMILE_HOME = Path("/home/nkhaloo/Desktop/opensmile")
N_SAMPLES      = 250


def extract(rows: pd.DataFrame, tmp_dir: Path) -> pd.DataFrame:
    for _, row in rows.iterrows():
        src = Path(row["audio_path"])
        (tmp_dir / src.name).symlink_to(src)

    df = vao_extract(tmp_dir, opensmile_default=OPENSMILE_HOME, apply_gate=False, normalize=False)
    df["flac_file_name"] = df["recording"].str.rsplit(".", n=1).str[0]
    df = df.merge(rows.drop(columns=["audio_exists"], errors="ignore"), on="flac_file_name", how="left")
    return df


# ── Load metadata ──────────────────────────────────────────────────────────────
print("Loading metadata...")
meta = pd.read_parquet(METADATA)
meta = meta[meta["audio_exists"].astype(bool)]

train_sample = meta[meta["split"] == "train"].sample(N_SAMPLES, random_state=42).reset_index(drop=True)
eval_sample  = meta[meta["split"] == "eval"].sample(N_SAMPLES, random_state=42).reset_index(drop=True)
print(f"  Train: {N_SAMPLES} utterances | Eval: {N_SAMPLES} utterances")

# ── Extract features ───────────────────────────────────────────────────────────
tempfile.tempdir = str(Path.home() / "tmp")
Path(tempfile.tempdir).mkdir(exist_ok=True)

print("Extracting train features...")
with tempfile.TemporaryDirectory(prefix="vao_train_") as tmp:
    train_df = extract(train_sample, Path(tmp))
print(f"  {len(train_df):,} frames")

print("Extracting eval features...")
with tempfile.TemporaryDirectory(prefix="vao_eval_") as tmp:
    eval_df = extract(eval_sample, Path(tmp))
print(f"  {len(eval_df):,} frames")

# ── Prepare features and labels ────────────────────────────────────────────────
meta_cols = ["recording", "flac_file_name", "split", "speaker_id", "audio_path",
             "gender", "codec", "codec_q", "codec_seed", "attack_tag", "attack_label",
             "key", "name", "frameTime"]
feature_cols = [c for c in train_df.columns if c not in meta_cols and train_df[c].dtype != object]

le = LabelEncoder()
X_train = train_df[feature_cols].fillna(0)
y_train = le.fit_transform(train_df["key"])

X_eval = eval_df[feature_cols].fillna(0)
y_eval = le.transform(eval_df["key"])

print(f"\nFeature columns: {len(feature_cols)}")
print(f"Train frames: {len(X_train):,} | Eval frames: {len(X_eval):,}")
print(f"Classes: {le.classes_}")

# ── Train GBDT ─────────────────────────────────────────────────────────────────
print("\nTraining LightGBM (default hyperparameters)...")
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
print("  Done.")

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred  = clf.predict(X_eval)
y_proba = clf.predict_proba(X_eval)[:, 1]

accuracy = accuracy_score(y_eval, y_pred)
auc      = roc_auc_score(y_eval, y_proba)
print(f"\nAccuracy : {accuracy:.4f}")
print(f"AUC      : {auc:.4f}")

# ── Feature importances ────────────────────────────────────────────────────────
importances = pd.Series(clf.feature_importances_, index=feature_cols)
top15 = importances.sort_values(ascending=False).head(15)

print("\nTop 15 feature importances:")
print(top15.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
top15.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top 15 Feature Importances — GBDT Spoof Detection")
ax.set_xlabel("Importance")
plt.tight_layout()
out_plot = Path(__file__).parent / "feature_importances.png"
fig.savefig(out_plot, dpi=150)
print(f"\nPlot saved to {out_plot}")
