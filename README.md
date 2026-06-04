# Variable Acoustic Organizer (VAO)

VAO is a Python package for frame-level acoustic feature extraction and acoustic segment classification. It wraps openSMILE's `SMILExtract` binary to produce per-frame eGeMAPSv02 features, then optionally classifies each frame as **silence**, **obstruent**, or **sonorant** using a trained gate model.

## What it does

**Feature extraction** — VAO runs openSMILE's eGeMAPSv02 LLD preset (25 ms Hann window, 10 ms hop) on every audio file in a folder, yielding ~88 acoustic features per frame: loudness, spectral balance (alpha ratio, Hammarberg index, spectral slope), MFCCs 1–4, F0, jitter, shimmer, HNR, formant frequencies and bandwidths (F1–F5), spectral shape (roll-off, centroid, entropy, variance, skewness, kurtosis), and their delta coefficients. Placeholder zeros emitted by openSMILE for undefined frames are converted to `NaN` automatically.

**Acoustic gate** — A `HistGradientBoostingClassifier` trained on TIMIT labels every 10 ms frame as `silence`, `obstruent`, or `sonorant` using 48 spectral/energy/voicing features. The gate is pre-trained and ships inside the package as `gate/model.joblib`. Training used sqrt-inverse-frequency sample weighting to handle TIMIT's class imbalance.

**Optional post-processing**
- `mask_features=True` — NaN out acoustically invalid features per class: silence removes all features; obstruents remove F0, jitter, shimmer, H1-H2, H1-A3, and formant features (+ deltas); sonorants keep everything.
- `normalize=True` — per-recording z-score normalization of all acoustic columns.
- `smooth_gate=True` — temporal smoothing that removes short isolated segment runs (default: ≥30 ms for obstruent/sonorant, ≥100 ms for silence).
- `frame_level=False` — aggregate frames to utterance level (mean + std per feature), the traditional openSMILE functional approach.

## Installation

```bash
pip install git+https://github.com/nkhaloo/Variable-Acoustic-Organizer.git
```

or

```bash
git clone https://github.com/nkhaloo/Variable-Acoustic-Organizer.git
cd Variable-Acoustic-Organizer
pip install -e .
```

Requires openSMILE (the repo/install root is passed at call time) and `ffmpeg` on `PATH` for audio preprocessing.

## Usage

```python
from vao import vao_extract

df = vao_extract(
    "/path/to/audio/",
    opensmile_home="/path/to/opensmile",
    apply_gate=True,
    mask_features=True,
    normalize=True,
)
df.to_csv("features.csv", index=False, na_rep="NaN")
```

`vao_extract` returns a DataFrame with one row per 10 ms frame across all recordings. Each row has a `recording` column and, if `apply_gate=True`, a `segment_class` column (`silence`/`obstruent`/`sonorant`).

## Gate model

The gate is trained on TIMIT (6,300 utterances, ~462k labeled frames). TIMIT phoneme symbols are mapped to three classes:

| Class | Phonemes |
|---|---|
| silence | `h#`, `pau`, `epi`, stop closures (`bcl`, `dcl`, …) |
| obstruent | stops, affricates, fricatives (`b`, `d`, `g`, `p`, `t`, `k`, `s`, `sh`, `f`, `v`, …) |
| sonorant | nasals, glides, semivowels, vowels (`m`, `n`, `l`, `r`, `w`, `y`, `iy`, `ah`, …) |

To retrain the gate:
```bash
python -m vao.gate.train \
    --timit /path/to/timit/data \
    --train-csv /path/to/timit_train.csv \
    --test-csv /path/to/timit_test.csv \
    --out src/vao/gate/model.joblib
```

## Deepfake detection experiment

`deepfake_exp/gbdt_experiment.py` tests whether raw VAO frame-level features carry enough signal for spoof detection on **ASVspoof5 Track 1**. The setup:

- 1,000 utterances sampled from each split (500 spoof / 500 bonafide, balanced)
- VAO extraction with `apply_gate=True`, `normalize=False`
- Frames are aggregated to utterance level by taking the mean of each feature
- A `LGBMClassifier` (LightGBM, default hyperparameters) is trained on mean-aggregated utterance features and evaluated on the held-out eval split

The experiment is a baseline sanity check — it treats the gate output and raw eGeMAPSv02 means as utterance-level descriptors without any temporal modeling. Its purpose is to confirm that the feature space is informative before more complex frame-level models are applied.
