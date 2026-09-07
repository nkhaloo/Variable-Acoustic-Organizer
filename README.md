# Variable Acoustic Organizer (VAO)

VAO gates speech into **silence**, **obstruent**, and **sonorant** directly from acoustics—no transcript or forced alignment required. A pre-trained classifier uses openSMILE features to label each 10 ms frame, so you can select sound classes and analyze their acoustic features.

## Installation

```bash
pip install git+https://github.com/nkhaloo/Variable-Acoustic-Organizer.git
```

### Download openSMILE and its configurations

Install the build tools and `ffmpeg` first. On **macOS**, with [Homebrew](https://brew.sh/) installed:

```bash
xcode-select --install  # If Command Line Tools are not already installed
brew install git cmake ffmpeg
```

On **Ubuntu/Debian**:

```bash
sudo apt update
sudo apt install git build-essential cmake ffmpeg
```

Then download the [official openSMILE repository](https://github.com/audeering/opensmile#quick-start) and compile its executable:

```bash
git clone https://github.com/audeering/opensmile.git ~/opensmile
cd ~/opensmile
bash build.sh
./build/progsrc/smilextract/SMILExtract -h
```

The last command checks that the executable runs. VAO finds it at `~/opensmile/build/progsrc/smilextract/SMILExtract` when you pass `opensmile_home="~/opensmile"`.

The download also includes the `config/` files VAO needs. VAO bundles its own frame-level configuration; no manual configuration edits are needed.

## Usage

```python
from vao import vao_extract

df = vao_extract(
    "/path/to/audio/",
    opensmile_home="~/opensmile",
    apply_gate=True,
)
df.to_csv("gated_features.csv", index=False, na_rep="NaN")
```

Each row contains a recording ID, frame time, acoustic features, and a `segment_class` label. Optional settings: `mask_features=True` masks features unsuitable for each class, `smooth_gate=True` smooths short label runs, and `normalize=True` normalizes features per recording.

Example output (selected rows and columns, with masking and normalization enabled; values rounded and recording renamed). `frameTime` is in seconds; `NaN` marks masked or undefined features.

| recording | frameTime | Loudness_sma3 | mfcc1_sma3 | segment_class |
|---|---:|---:|---:|---|
| test.wav | 0.00 | NaN | NaN | silence |
| test.wav | 0.47 | -1.289 | -1.463 | obstruent |
| test.wav | 0.52 | -0.799 | -2.399 | sonorant |

## Gate performance

The bundled gradient-boosting classifier was trained on TIMIT. Saved results on 278,474 test frames: **92.4% accuracy**, **81.3% macro F1**.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Silence | 62.6% | 76.6% | 68.9% |
| Obstruent | 76.5% | 81.0% | 78.7% |
| Sonorant | 97.4% | 95.2% | 96.3% |

Sonorants account for most test frames; macro F1 gives each class equal weight.
