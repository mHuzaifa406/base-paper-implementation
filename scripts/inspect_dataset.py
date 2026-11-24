from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(r"E:\IoMT_FL_IDS\data\raw\HuggingFaceFull\machine_learning")
files = sorted(DATA_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")

total_rows = 0
total_missing = 0
total_infinite = 0
all_labels = {}

print("=" * 60)
print("CIC-IDS2017 DATA QUALITY AUDIT")
print("=" * 60)

for file in files:
    print(f"\nInspecting: {file.name}")

    # Read metadata first
    metadata = pq.read_metadata(file)
    total_rows += metadata.num_rows

    # Load one file at a time
    df = pd.read_parquet(file)

    if "Label" not in df.columns:
        raise ValueError(f"Label column missing in {file.name}")

    numeric = df.select_dtypes(include=[np.number])

    missing = int(df.isna().sum().sum())
    infinite = int(np.isinf(numeric.to_numpy()).sum())

    total_missing += missing
    total_infinite += infinite

    print("Rows:", f"{len(df):,}")
    print("Total columns:", len(df.columns))
    print("Features excluding Label:", len(df.columns) - 1)
    print("Missing cells:", f"{missing:,}")
    print("Infinite numeric values:", f"{infinite:,}")
    print("Destination Port present:",
          "Destination Port" in df.columns)

    print("Labels in this file:")
    counts = df["Label"].astype(str).value_counts()

    for label, count in counts.items():
        all_labels[label] = all_labels.get(label, 0) + int(count)
        print(f"  {label}: {count:,}")

    del df

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)
print("Files audited:", len(files))
print("Total rows:", f"{total_rows:,}")
print("Total missing cells:", f"{total_missing:,}")
print("Total infinite values:", f"{total_infinite:,}")
print("Unique labels:", len(all_labels))

print("\nCombined label distribution:")
for label, count in sorted(
    all_labels.items(), key=lambda item: item[1], reverse=True
):
    print(f"{label}: {count:,}")

print("\nAudit completed. Original files were not modified.")