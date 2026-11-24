
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"E:\IoMT_FL_IDS\data\raw\HuggingFaceFull\machine_learning"
)

files = sorted(DATA_DIR.glob("*.parquet"))

if len(files) != 8:
    raise ValueError(
        f"Expected 8 Parquet files, found {len(files)}"
    )

total_rows = 0
total_missing_cells = 0
total_infinite_cells = 0

rows_with_missing = 0
rows_with_infinite = 0
rows_with_any_invalid = 0
rows_with_both = 0

hash_parts = []

print("=" * 65)
print("ROW-LEVEL DATA QUALITY AND DUPLICATE AUDIT")
print("=" * 65)

for file in files:

    print(f"\nProcessing: {file.name}")

    df = pd.read_parquet(file)

    numeric = df.select_dtypes(include=[np.number])

    # Identify missing cells and rows
    missing_mask = df.isna().any(axis=1).to_numpy()

    # Identify infinite numeric cells and rows
    inf_array = np.isinf(numeric.to_numpy())
    infinite_mask = inf_array.any(axis=1)

    # Combine invalid-row conditions
    invalid_mask = missing_mask | infinite_mask

    n_rows = len(df)
    n_missing_cells = int(df.isna().sum().sum())
    n_infinite_cells = int(inf_array.sum())

    n_missing_rows = int(missing_mask.sum())
    n_infinite_rows = int(infinite_mask.sum())
    n_invalid_rows = int(invalid_mask.sum())
    n_both = int((missing_mask & infinite_mask).sum())

    total_rows += n_rows
    total_missing_cells += n_missing_cells
    total_infinite_cells += n_infinite_cells

    rows_with_missing += n_missing_rows
    rows_with_infinite += n_infinite_rows
    rows_with_any_invalid += n_invalid_rows
    rows_with_both += n_both

    # Calculate fingerprints for full-row duplicate detection
    row_hashes = pd.util.hash_pandas_object(
        df,
        index=False
    ).to_numpy(dtype=np.uint64)

    hash_parts.append(row_hashes)

    print("Total rows:", f"{n_rows:,}")
    print("Rows with missing values:", f"{n_missing_rows:,}")
    print("Rows with infinite values:", f"{n_infinite_rows:,}")
    print("Rows with either problem:", f"{n_invalid_rows:,}")
    print("Rows with both problems:", f"{n_both:,}")

    del df, numeric, inf_array, row_hashes

# Check duplicates across all eight files
all_hashes = np.concatenate(hash_parts)

unique_hash_count = len(np.unique(all_hashes))

duplicate_count = total_rows - unique_hash_count

# Summary
print("\n" + "=" * 65)
print("OVERALL AUDIT RESULTS")
print("=" * 65)

print("Total records:", f"{total_rows:,}")
print("Missing cells:", f"{total_missing_cells:,}")
print("Infinite numeric cells:", f"{total_infinite_cells:,}")

print("Rows with missing values:", f"{rows_with_missing:,}")
print("Rows with infinite values:", f"{rows_with_infinite:,}")
print("Rows with either problem:", f"{rows_with_any_invalid:,}")
print("Rows with both problems:", f"{rows_with_both:,}")

print("Exact-row duplicate candidates:", f"{duplicate_count:,}")

print(
    "Remaining after removing invalid rows:",
    f"{total_rows - rows_with_any_invalid:,}"
)

print(
    "Paper's reported cleaned records:",
    "2,827,876"
)

print(
    "Paper's reported removed records:",
    "2,867"
)

print("\nAudit completed. No original records were modified.")
