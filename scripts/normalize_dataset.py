
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Input directories
TRAIN_INPUT = Path(
    r"E:\IoMT_FL_IDS\data\processed\split_v1\train"
)

TEST_INPUT = Path(
    r"E:\IoMT_FL_IDS\data\processed\split_v1\test"
)

# Output directories
OUTPUT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\normalized_v1"
)

TRAIN_OUTPUT = OUTPUT_ROOT / "train"
TEST_OUTPUT = OUTPUT_ROOT / "test"

TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT.mkdir(parents=True, exist_ok=True)


# Fixed class-to-integer mapping
CLASS_TO_ID = {
    "Benign": 0,
    "Bot": 1,
    "DoS": 2,
    "Infiltration": 3,
    "PortScan": 4,
    "Web Attack": 5,
    "Brute Force": 6,
}

LABEL_COLUMNS = [
    "Label",
    "Label_7class"
]


def get_files(directory):

    files = sorted(directory.glob("*.parquet"))

    if len(files) != 8:
        raise ValueError(
            f"Expected 8 files in {directory}, "
            f"found {len(files)}"
        )

    return files


train_files = get_files(TRAIN_INPUT)
test_files = get_files(TEST_INPUT)

if {f.name for f in train_files} != {
    f.name for f in test_files
}:
    raise ValueError(
        "Training and testing file names do not match."
    )


# -------------------------------------------------
# STEP 1: Identify numerical features
# -------------------------------------------------

first_df = pd.read_parquet(train_files[0])

feature_columns = [
    col for col in first_df.columns
    if col not in LABEL_COLUMNS
]

if len(feature_columns) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(feature_columns)}"
    )

non_numeric = [
    col for col in feature_columns
    if not pd.api.types.is_numeric_dtype(first_df[col])
]

if non_numeric:
    raise TypeError(
        f"Non-numeric features found: {non_numeric}"
    )

del first_df

print("=" * 65)
print("FEATURE NORMALIZATION")
print("=" * 65)

print("Features:", len(feature_columns))
print("Training files:", len(train_files))
print("Testing files:", len(test_files))


# -------------------------------------------------
# STEP 2: Fit StandardScaler using training data
# -------------------------------------------------

scaler = StandardScaler()

fit_rows = 0

print("\nFitting scaler on training data only...")

for file in train_files:

    print("Fitting:", file.name)

    df = pd.read_parquet(file)

    if not all(col in df.columns for col in feature_columns):
        raise ValueError("Feature schema mismatch.")

    X = df[feature_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite values found in {file.name}"
        )

    # Incremental fitting avoids combining all files in RAM
    scaler.partial_fit(X)

    fit_rows += len(df)

    del df, X


print("\nScaler fitting completed.")
print("Rows used for fitting:", f"{fit_rows:,}")


# Save fitted scaler for future use
scaler_path = OUTPUT_ROOT / "standard_scaler.joblib"

joblib.dump(scaler, scaler_path, compress=3)

print("Scaler saved:", scaler_path)


# -------------------------------------------------
# STEP 3: Transform training and testing data
# -------------------------------------------------

def transform_split(input_files, output_dir, split_name):

    total_rows = 0
    class_counts = {}

    print(f"\nNormalizing {split_name} data...")

    for file in input_files:

        print("Processing:", file.name)

        df = pd.read_parquet(file)

        X = df[feature_columns].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Invalid feature values in {file.name}"
            )

        # Apply the same fitted scaler
        Z = scaler.transform(X).astype(np.float32)

        if not np.isfinite(Z).all():
            raise ValueError(
                f"Invalid normalized values in {file.name}"
            )

        # Store normalized features as float32
        normalized_features = pd.DataFrame(
            Z,
            columns=feature_columns,
            index=df.index
        )

        # Preserve original and mapped labels
        labels = df[LABEL_COLUMNS].copy()

        labels["Label_id"] = (
            labels["Label_7class"]
            .map(CLASS_TO_ID)
        )

        if labels["Label_id"].isna().any():
            raise ValueError(
                f"Unknown class in {file.name}"
            )

        labels["Label_id"] = labels["Label_id"].astype(
            "int8"
        )

        # Combine normalized features and labels
        output_df = pd.concat(
            [normalized_features, labels],
            axis=1
        )

        output_file = output_dir / file.name

        output_df.to_parquet(
            output_file,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        total_rows += len(output_df)

        counts = labels["Label_7class"].value_counts()

        for label, count in counts.items():
            class_counts[label] = (
                class_counts.get(label, 0) + int(count)
            )

        print("Saved:", output_file.name)
        print("Rows:", f"{len(output_df):,}")

        del df, X, Z, normalized_features, labels, output_df

    return total_rows, class_counts


# Transform both splits using the training-fitted scaler
train_rows, train_counts = transform_split(
    train_files,
    TRAIN_OUTPUT,
    "TRAINING"
)

test_rows, test_counts = transform_split(
    test_files,
    TEST_OUTPUT,
    "TESTING"
)


# -------------------------------------------------
# STEP 4: Save experiment metadata
# -------------------------------------------------

metadata = {
    "normalization": "StandardScaler",
    "scaler_fitted_on": "Training data only",
    "scaler_file": "standard_scaler.joblib",
    "features": feature_columns,
    "number_of_features": len(feature_columns),
    "classes": CLASS_TO_ID,
    "training_records": train_rows,
    "testing_records": test_rows,
    "total_records": train_rows + test_rows,
    "training_class_counts": train_counts,
    "testing_class_counts": test_counts,
    "numeric_storage_dtype": "float32",
}

metadata_path = OUTPUT_ROOT / "normalization_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)


# -------------------------------------------------
# STEP 5: Final summary
# -------------------------------------------------

print("\n" + "=" * 65)
print("NORMALIZATION SUMMARY")
print("=" * 65)

print("Number of features:", len(feature_columns))
print("Training records:", f"{train_rows:,}")
print("Testing records:", f"{test_rows:,}")
print("Total records:", f"{train_rows + test_rows:,}")

print("Training scaler fit records:", f"{fit_rows:,}")

print("Expected training records: 2,262,303")
print("Expected testing records: 565,573")
print("Expected total records: 2,827,876")

print("\nTraining and testing files saved successfully.")
print("Original split files were not modified.")
print("Scaler and metadata saved successfully.")
