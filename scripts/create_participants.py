
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


# Normalized training data only
INPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\normalized_v1\train"
)

# Output directory for ten participants
OUTPUT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

NUMBER_OF_CLIENTS = 10
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "Benign",
    1: "Bot",
    2: "DoS",
    3: "Infiltration",
    4: "PortScan",
    5: "Web Attack",
    6: "Brute Force",
}


# Check source files
files = sorted(INPUT_DIR.glob("*.parquet"))

if len(files) != 8:
    raise ValueError(
        f"Expected 8 training files, found {len(files)}"
    )

# Protect against accidental overwriting
if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise FileExistsError(
        f"Output folder is not empty: {OUTPUT_ROOT}. "
        "Use a new output folder or inspect the existing files."
    )


print("=" * 65)
print("FEDERATED PARTICIPANT DATA DISTRIBUTION")
print("=" * 65)


# -------------------------------------------------
# STEP 1: Read training labels only
# -------------------------------------------------

label_parts = []
file_offsets = []

offset = 0

for file in files:

    labels = pd.read_parquet(
        file,
        columns=["Label_id"]
    )["Label_id"].to_numpy(dtype=np.int8)

    if not np.isin(labels, list(CLASS_NAMES)).all():
        raise ValueError(
            f"Invalid class IDs found in {file.name}"
        )

    label_parts.append(labels)

    file_offsets.append(
        (offset, offset + len(labels), file)
    )

    offset += len(labels)


all_labels = np.concatenate(label_parts)

TOTAL_ROWS = len(all_labels)

print("Total training records:", f"{TOTAL_ROWS:,}")
print("Number of participants:", NUMBER_OF_CLIENTS)


# -------------------------------------------------
# STEP 2: Assign records to participants
# -------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

participant_ids = np.full(
    TOTAL_ROWS,
    -1,
    dtype=np.int8
)

print("\nCreating stratified IID-style partitions...")

for class_id, class_name in CLASS_NAMES.items():

    indices = np.flatnonzero(
        all_labels == class_id
    )

    rng.shuffle(indices)

    # Distribute each class approximately equally
    # across all ten participants
    partitions = np.array_split(
        indices,
        NUMBER_OF_CLIENTS
    )

    for client_id, client_indices in enumerate(partitions):

        participant_ids[client_indices] = client_id

    print(
        f"{class_name}: {len(indices):,} records distributed"
    )


if np.any(participant_ids < 0):
    raise RuntimeError(
        "Some records were not assigned to participants."
    )


# -------------------------------------------------
# STEP 3: Create participant directories
# -------------------------------------------------

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

client_dirs = []

for client_id in range(NUMBER_OF_CLIENTS):

    client_dir = OUTPUT_ROOT / (
        f"client_{client_id + 1:02d}"
    )

    client_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    client_dirs.append(client_dir)


client_totals = [0] * NUMBER_OF_CLIENTS

client_class_counts = [
    Counter() for _ in range(NUMBER_OF_CLIENTS)
]


# -------------------------------------------------
# STEP 4: Save participant training records
# -------------------------------------------------

print("\nSaving participant datasets...")

for start, end, file in file_offsets:

    df = pd.read_parquet(file)

    assigned_ids = participant_ids[start:end]

    if len(df) != len(assigned_ids):
        raise RuntimeError(
            f"Row alignment error in {file.name}"
        )

    for client_id in range(NUMBER_OF_CLIENTS):

        mask = assigned_ids == client_id

        if not mask.any():
            continue

        client_df = df.loc[mask].copy()

        output_file = (
            client_dirs[client_id] / file.name
        )

        client_df.to_parquet(
            output_file,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        client_totals[client_id] += len(client_df)

        counts = client_df["Label_id"].value_counts()

        for label_id, count in counts.items():

            class_name = CLASS_NAMES[int(label_id)]

            client_class_counts[client_id][class_name] += int(count)

        del client_df

    print("Processed:", file.name)

    del df


# -------------------------------------------------
# STEP 5: Verify participant distribution
# -------------------------------------------------

print("\n" + "=" * 65)
print("PARTICIPANT DISTRIBUTION SUMMARY")
print("=" * 65)

metadata = {
    "number_of_participants": NUMBER_OF_CLIENTS,
    "total_training_records": TOTAL_ROWS,
    "partition_method": "Stratified IID-style distribution",
    "random_seed": RANDOM_SEED,
    "class_mapping": CLASS_NAMES,
    "participants": {}
}

for i in range(NUMBER_OF_CLIENTS):

    client_name = f"client_{i + 1:02d}"

    print(f"\n{client_name}")
    print("Total records:", f"{client_totals[i]:,}")

    for class_name in CLASS_NAMES.values():

        count = client_class_counts[i][class_name]

        print(f"  {class_name}: {count:,}")

    metadata["participants"][client_name] = {
        "total_records": client_totals[i],
        "class_distribution": dict(
            client_class_counts[i]
        )
    }


# Save partition metadata
metadata_path = OUTPUT_ROOT / "participant_metadata.json"

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(metadata, f, indent=4)


# Final integrity checks
print("\n" + "=" * 65)
print("FINAL INTEGRITY CHECK")
print("=" * 65)

print("Expected training records:", f"{TOTAL_ROWS:,}")
print("Assigned records:", f"{sum(client_totals):,}")
print(
    "All records assigned:",
    sum(client_totals) == TOTAL_ROWS
)

print(
    "Number of participants:",
    len(client_totals)
)

print(
    "Largest client:",
    f"{max(client_totals):,}"
)

print(
    "Smallest client:",
    f"{min(client_totals):,}"
)

print("\nMetadata saved:", metadata_path)
print("Participant data saved:", OUTPUT_ROOT)
print("\nParticipant distribution completed.")
