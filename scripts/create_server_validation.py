
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

OUTPUT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v2"
)

NUM_CLIENTS = 10
VALIDATION_FRACTION = 0.05
RANDOM_SEED = 42

LABEL_COLUMN = "Label_id"

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]


# ============================================================
# SAFETY CHECKS
# ============================================================

if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise FileExistsError(
        f"Output folder is not empty: {OUTPUT_ROOT}"
    )

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(SOURCE_ROOT)


# ============================================================
# DETERMINE FEATURE COLUMNS
# ============================================================

first_files = sorted(
    (SOURCE_ROOT / "client_01").glob("*.parquet")
)

if len(first_files) != 8:
    raise ValueError(
        f"Expected eight files for Client 1, "
        f"found {len(first_files)}"
    )

first_df = pd.read_parquet(first_files[0])

FEATURE_COLUMNS = [
    c for c in first_df.columns
    if c not in LABEL_COLUMNS
]

if len(FEATURE_COLUMNS) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(FEATURE_COLUMNS)}"
    )

del first_df


# ============================================================
# LOAD ALL TEN CLIENTS
# ============================================================

print("=" * 70)
print("CREATING TRUSTED SERVER VALIDATION SET")
print("=" * 70)

client_frames = []

original_client_counts = {}

for client_id in range(1, NUM_CLIENTS + 1):

    print(f"\nLoading client {client_id}/{NUM_CLIENTS}")

    client_dir = (
        SOURCE_ROOT / f"client_{client_id:02d}"
    )

    files = sorted(client_dir.glob("*.parquet"))

    if len(files) != 8:
        raise ValueError(
            f"Client {client_id}: expected 8 files, "
            f"found {len(files)}"
        )

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    df = pd.concat(
        frames,
        ignore_index=True
    )

    del frames

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in client {client_id}: {missing}"
        )

    # Keep only model features and numeric class labels.
    df = df[required_columns].copy()

    df["__client_id"] = client_id

    original_client_counts[str(client_id)] = len(df)

    client_frames.append(df)

    print(
        "Records:",
        f"{len(df):,}"
    )

    del df


# ============================================================
# COMBINE DATA AND CHECK LABELS
# ============================================================

all_data = pd.concat(
    client_frames,
    ignore_index=True
)

del client_frames

original_total = len(all_data)

print("\nTotal records:", f"{original_total:,}")

class_counts = (
    all_data[LABEL_COLUMN]
    .value_counts()
    .sort_index()
)

print("\nOriginal class distribution:")
print(class_counts.to_string())


if not set(class_counts.index).issubset(set(range(7))):
    raise ValueError(
        "Unexpected class IDs detected."
    )


# ============================================================
# STRATIFIED SERVER VALIDATION SPLIT
# ============================================================

print("\nCreating the 5% validation split...")

training_data, validation_data = train_test_split(
    all_data,
    test_size=VALIDATION_FRACTION,
    random_state=RANDOM_SEED,
    stratify=all_data[LABEL_COLUMN]
)

del all_data

# Keep client ID only in the remaining training records.
validation_data = validation_data.drop(
    columns=["__client_id"]
).reset_index(drop=True)

training_data = training_data.reset_index(drop=True)


# ============================================================
# SAVE NEW CLIENT DATA
# ============================================================

for client_id in range(1, NUM_CLIENTS + 1):

    client_data = training_data[
        training_data["__client_id"] == client_id
    ].drop(
        columns=["__client_id"]
    ).reset_index(drop=True)

    client_dir = (
        OUTPUT_ROOT / f"client_{client_id:02d}"
    )

    client_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        client_dir / "client_data.parquet"
    )

    client_data.to_parquet(
        output_file,
        index=False,
        compression="snappy"
    )

    print(
        f"Saved Client {client_id}: "
        f"{len(client_data):,} records"
    )

    del client_data


# ============================================================
# SAVE TRUSTED VALIDATION DATA
# ============================================================

validation_dir = OUTPUT_ROOT / "server_validation"

validation_dir.mkdir(
    parents=True,
    exist_ok=True
)

validation_file = (
    validation_dir / "validation.parquet"
)

validation_data.to_parquet(
    validation_file,
    index=False,
    compression="snappy"
)

print(
    "\nServer validation records:",
    f"{len(validation_data):,}"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "source": str(SOURCE_ROOT),
    "output": str(OUTPUT_ROOT),
    "number_of_clients": NUM_CLIENTS,
    "validation_fraction": VALIDATION_FRACTION,
    "random_seed": RANDOM_SEED,
    "original_total_records": original_total,
    "remaining_training_records": len(training_data),
    "validation_records": len(validation_data),
    "original_client_records": original_client_counts,
    "validation_class_distribution": {
        str(k): int(v)
        for k, v in validation_data[LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .items()
    },
    "test_data_modified": False
}

metadata_file = OUTPUT_ROOT / "metadata.json"

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# FINAL CHECK
# ============================================================

assert (
    sum(original_client_counts.values())
    == original_total
)

assert (
    len(training_data) + len(validation_data)
    == original_total
)

assert len(validation_data) > 0

print("\n" + "=" * 70)
print("VALIDATION DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print("Original records:", f"{original_total:,}")
print("Training records:", f"{len(training_data):,}")
print("Validation records:", f"{len(validation_data):,}")

print("\nOutput:", OUTPUT_ROOT)
print("Original federated_v1 remains unchanged.")
print("Existing test dataset remains unchanged.")
