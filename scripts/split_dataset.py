
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


# Input and output directories
INPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\mapped_v1"
)

OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\split_v1"
)

TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# Reproducibility settings
TEST_FRACTION = 0.20
RANDOM_SEED = 42

CLASSES = [
    "Benign",
    "Bot",
    "DoS",
    "Infiltration",
    "PortScan",
    "Web Attack",
    "Brute Force",
]


files = sorted(INPUT_DIR.glob("*.parquet"))

if len(files) != 8:
    raise ValueError(
        f"Expected 8 mapped files, found {len(files)}"
    )

print("=" * 65)
print("STRATIFIED TRAIN/TEST SPLITTING")
print("=" * 65)


# Read labels only to determine stratified split indices
label_parts = []
file_offsets = []

offset = 0

for file in files:

    labels = pd.read_parquet(
        file,
        columns=["Label_7class"]
    )["Label_7class"].astype(str)

    codes = pd.Categorical(
        labels,
        categories=CLASSES
    ).codes.astype(np.int8)

    if (codes < 0).any():
        raise ValueError(
            f"Unknown class found in {file.name}"
        )

    file_offsets.append((offset, offset + len(codes), file))
    label_parts.append(codes)

    offset += len(codes)


y = np.concatenate(label_parts)
total_rows = len(y)

# Boolean masks for all records
test_mask = np.zeros(total_rows, dtype=bool)

rng = np.random.default_rng(RANDOM_SEED)

print("\nCreating stratified split...")

for class_id, class_name in enumerate(CLASSES):

    indices = np.flatnonzero(y == class_id)

    if len(indices) < 2:
        raise ValueError(
            f"Not enough records for class {class_name}"
        )

    rng.shuffle(indices)

    # Floor of 20% for each class, following
    # the approximate per-class proportions in Table 1
    test_count = int(np.floor(
        len(indices) * TEST_FRACTION
    ))

    if test_count == 0:
        raise ValueError(
            f"Test set would be empty for {class_name}"
        )

    test_indices = indices[:test_count]

    test_mask[test_indices] = True

    print(
        f"{class_name}: Total={len(indices):,} | "
        f"Train={len(indices)-test_count:,} | "
        f"Test={test_count:,}"
    )


# Write the split files one source file at a time
train_total = 0
test_total = 0

train_counts = Counter()
test_counts = Counter()

print("\nSaving training and testing files...")

for start, end, file in file_offsets:

    df = pd.read_parquet(file)

    local_test_mask = test_mask[start:end]
    local_train_mask = ~local_test_mask

    train_df = df.loc[local_train_mask].copy()
    test_df = df.loc[local_test_mask].copy()

    train_path = TRAIN_DIR / file.name
    test_path = TEST_DIR / file.name

    train_df.to_parquet(
        train_path,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    test_df.to_parquet(
        test_path,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    train_total += len(train_df)
    test_total += len(test_df)

    train_counts.update(
        train_df["Label_7class"].value_counts().to_dict()
    )

    test_counts.update(
        test_df["Label_7class"].value_counts().to_dict()
    )

    print(
        f"{file.name}: "
        f"Train={len(train_df):,}, "
        f"Test={len(test_df):,}"
    )

    del df, train_df, test_df


# Save experimental configuration
metadata = {
    "total_records": total_rows,
    "training_records": train_total,
    "testing_records": test_total,
    "test_fraction": TEST_FRACTION,
    "random_seed": RANDOM_SEED,
    "classes": CLASSES,
    "method": "Stratified split within each class",
    "test_count_rule": "floor(class_count * 0.20)",
    "class_training_counts": dict(train_counts),
    "class_testing_counts": dict(test_counts),
}

with open(
    OUTPUT_DIR / "split_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=4)


print("\n" + "=" * 65)
print("SPLITTING SUMMARY")
print("=" * 65)

print("Total records:", f"{total_rows:,}")
print("Training records:", f"{train_total:,}")
print("Testing records:", f"{test_total:,}")

print("\nTraining class distribution:")

for label in CLASSES:
    print(f"{label}: {train_counts[label]:,}")

print("\nTesting class distribution:")

for label in CLASSES:
    print(f"{label}: {test_counts[label]:,}")

print("\nTotal consistency:",
      train_total + test_total == total_rows)

print("Saved metadata:", OUTPUT_DIR / "split_metadata.json")
print("Splitting completed successfully.")
