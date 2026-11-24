
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# Original dataset: never modify these files
INPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\data\raw\HuggingFaceFull\machine_learning"
)

# Store cleaned copies separately
OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\cleaned_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.parquet"))

if len(files) != 8:
    raise ValueError(
        f"Expected 8 source files, found {len(files)}"
    )

total_original = 0
total_cleaned = 0
total_removed = 0

label_counts = Counter()

print("=" * 65)
print("CIC-IDS2017 DATA CLEANING")
print("=" * 65)

for file in files:

    print(f"\nProcessing: {file.name}")

    df = pd.read_parquet(file)

    original_rows = len(df)

    # Check for missing values across all columns
    missing_mask = df.isna().any(axis=1)

    # Check all numeric feature columns for NaN and infinity
    numeric = df.select_dtypes(include=[np.number])

    finite_mask = np.isfinite(
        numeric.to_numpy(dtype=np.float64)
    ).all(axis=1)

    # Keep only rows with complete, finite values
    valid_mask = (~missing_mask).to_numpy() & finite_mask

    cleaned = df.loc[valid_mask].copy()

    removed_rows = original_rows - len(cleaned)

    # Preserve labels at this stage
    counts = cleaned["Label"].astype(str).value_counts()

    label_counts.update({
        str(label): int(count)
        for label, count in counts.items()
    })

    # Save one cleaned file for every original file
    output_file = OUTPUT_DIR / file.name

    cleaned.to_parquet(
        output_file,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    total_original += original_rows
    total_cleaned += len(cleaned)
    total_removed += removed_rows

    print("Original rows:", f"{original_rows:,}")
    print("Removed rows:", f"{removed_rows:,}")
    print("Cleaned rows:", f"{len(cleaned):,}")
    print("Saved:", output_file.name)

    del df, cleaned, numeric

print("\n" + "=" * 65)
print("CLEANING SUMMARY")
print("=" * 65)

print("Original rows:", f"{total_original:,}")
print("Removed rows:", f"{total_removed:,}")
print("Cleaned rows:", f"{total_cleaned:,}")

print("\nCleaned label distribution:")

for label, count in label_counts.most_common():
    print(f"{label}: {count:,}")

print("\nExpected paper values:")
print("Original: 2,830,743")
print("Removed:  2,867")
print("Cleaned:  2,827,876")

print("\nCleaning completed.")
print("Original source files remain unchanged.")
print("Output directory:", OUTPUT_DIR)
