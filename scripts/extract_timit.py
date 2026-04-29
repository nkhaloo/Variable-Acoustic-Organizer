# extracts TIMIT train and test splits separately, saves two combined CSVs

from pathlib import Path
from vao import vao_extract

TIMIT_DATA_DIR = Path("/Users/noahkhaloo/Desktop/archive/data")
TIMIT_OUTPUT_DIR = Path("/Users/noahkhaloo/Desktop/timit_vao_output")
OPENSMILE_HOME = Path("/Users/noahkhaloo/Desktop/opensmile")

if __name__ == "__main__":
    TIMIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting TRAIN split...")
    train_df = vao_extract(
        TIMIT_DATA_DIR / "TRAIN",
        opensmile_default=OPENSMILE_HOME,
        output_dir=TIMIT_OUTPUT_DIR,
        recursive=True,
        apply_gate=False,
    )
    train_df.to_csv(TIMIT_OUTPUT_DIR / "train.csv", index=False, na_rep="NaN")
    print(f"  {train_df['recording'].nunique()} recordings, {len(train_df):,} frames")

    print("Extracting TEST split...")
    test_df = vao_extract(
        TIMIT_DATA_DIR / "TEST",
        opensmile_default=OPENSMILE_HOME,
        output_dir=TIMIT_OUTPUT_DIR,
        recursive=True,
        apply_gate=False,
    )
    test_df.to_csv(TIMIT_OUTPUT_DIR / "test.csv", index=False, na_rep="NaN")
    print(f"  {test_df['recording'].nunique()} recordings, {len(test_df):,} frames")

    print(f"\nDone. CSVs written to: {TIMIT_OUTPUT_DIR}")
