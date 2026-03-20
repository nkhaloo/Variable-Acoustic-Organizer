# Variable Acoustic Organizer (VAO)

VAO is a Python package for:
- extracting frame-level acoustic features with openSMILE (via `SMILExtract`)
- (next) applying rule-based logic to assign sound-class labels per millisecond

## Quickstart

### 1) Ensure openSMILE is available

You built openSMILE from source. Point VAO at it using one of:
- `OPENSMILE_HOME` (recommended): the openSMILE repo root, e.g. `/Users/noahkhaloo/Desktop/opensmile`
- `OPENSMILE_SMILEEXTRACT`: full path to the `SMILExtract` binary

Example:

```bash
export OPENSMILE_HOME="$HOME/Desktop/opensmile"
```

### 2) Install VAO (editable)

From this repo root:

```bash
python -m pip install -e .
```

### 3) Extract features to a DataFrame

```python
from vao import extract_features

df = extract_features(
	"path/to/audio.wav",
	config_path="/Users/noahkhaloo/Desktop/opensmile/config/your_config.conf",
)
print(df.head())
```

### 4) Batch extract a folder + create one combined CSV

```python
from vao import extract_features_folder

combined = extract_features_folder(
	"tests/wav",
	config_path="/Users/noahkhaloo/Desktop/opensmile/config/vao/MFCC12_E_D_A_25ms_1ms.conf",
	output_dir="tests/output",
	opensmile_home="/Users/noahkhaloo/Desktop/opensmile",
)
print(combined.shape)
```

Notes:
- To get 1 ms frame shift and 25 ms windows, your openSMILE config must set `frameStep=0.001` and `frameSize=0.025`.

