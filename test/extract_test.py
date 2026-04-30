from pathlib import Path
from vao import vao_extract

WAV_DIR = Path(__file__).parent / "wav"
OUTPUT_DIR = Path(__file__).parent
OPENSMILE_HOME = Path("/Users/noahkhaloo/Desktop/opensmile")

if __name__ == "__main__":
    df = vao_extract(
        WAV_DIR,
        opensmile_default=OPENSMILE_HOME,
        output_dir=OUTPUT_DIR,
        apply_gate=True,
        mask_features=True,
        normalize=True,
    )

    df.to_csv(OUTPUT_DIR / "output.csv", index=False, na_rep="NaN")
    print(df[["recording", "segment_class"]].value_counts())
    print(f"\n{len(df):,} frames extracted")
