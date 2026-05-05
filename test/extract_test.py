from pathlib import Path
from vao import vao_extract

WAV_DIR = Path(__file__).parent / "wav"
OPENSMILE_HOME = Path("/Users/noahkhaloo/Desktop/opensmile")

df = vao_extract(
    WAV_DIR,
    opensmile_default=OPENSMILE_HOME,
    apply_gate=True,
    mask_features=True,
    normalize=True,
)

df.to_csv(Path(__file__).parent / "output.csv", index=False, na_rep="NaN")
