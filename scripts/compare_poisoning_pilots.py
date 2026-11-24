
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from models import IntrusionDetectionMLP


# ------------------------------------------------
# PATHS
# ------------------------------------------------

TEST_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\normalized_v1\test"
)

BENIGN_CHECKPOINT = Path(
    r"E:\IoMT_FL_IDS\results\fedavg_pilot\fedavg_pilot_round1.pt"
)

POISON_CHECKPOINT = Path(
    r"E:\IoMT_FL_IDS\results\fedavg_poison_pilot_r20\poisoned_global_round1.pt"
)

OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\pilot_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

SEED = 42
TEST_SAMPLE_PER_FILE = 2500
BATCH_SIZE = 512

CLASS_NAMES = [
    "Benign",
    "Bot",
    "DoS",
    "Infiltration",
    "PortScan",
    "Web Attack",
    "Brute Force"
]

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]

torch.set_num_threads(4)


# ------------------------------------------------
# LOAD IDENTICAL TEST SAMPLE
# ------------------------------------------------

def load_test_data():

    files = sorted(TEST_DIR.glob("*.parquet"))

    if len(files) != 8:
        raise ValueError(
            f"Expected 8 test files, found {len(files)}"
        )

    frames = []

    first = pd.read_parquet(files[0])

    features = [
        c for c in first.columns
        if c not in LABEL_COLUMNS
    ]

    del first

    for i, file in enumerate(files):

        df = pd.read_parquet(file)

        if len(df) > TEST_SAMPLE_PER_FILE:

            df = df.sample(
                n=TEST_SAMPLE_PER_FILE,
                random_state=SEED + i
            )

        frames.append(df)

    test_df = pd.concat(
        frames,
        ignore_index=True
    )

    X = test_df[features].to_numpy(
        dtype=np.float32
    )

    y = test_df["Label_id"].to_numpy(
        dtype=np.int64
    )

    del frames, test_df

    return X, y


# ------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------

def load_model(checkpoint):

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    )

    try:
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        state = torch.load(
            checkpoint,
            map_location="cpu"
        )

    model.load_state_dict(state)
    model.eval()

    return model


# ------------------------------------------------
# PREDICTIONS
# ------------------------------------------------

def predict(model, X):

    predictions = []

    model.eval()

    with torch.inference_mode():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            )

            output = model(batch)

            predictions.extend(
                output.argmax(dim=1).numpy()
            )

    return np.asarray(predictions)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    print("=" * 70)
    print("BENIGN VS POISONED MODEL COMPARISON")
    print("=" * 70)

    X, y_true = load_test_data()

    print("Common test records:", f"{len(y_true):,}")

    benign_model = load_model(
        BENIGN_CHECKPOINT
    )

    poisoned_model = load_model(
        POISON_CHECKPOINT
    )

    print("\nGenerating benign predictions...")

    benign_predictions = predict(
        benign_model,
        X
    )

    print("Generating poisoned predictions...")

    poisoned_predictions = predict(
        poisoned_model,
        X
    )

    # Calculate accuracy
    benign_accuracy = accuracy_score(
        y_true,
        benign_predictions
    ) * 100

    poisoned_accuracy = accuracy_score(
        y_true,
        poisoned_predictions
    ) * 100

    # Count changed predictions
    changed_predictions = int(
        np.sum(
            benign_predictions != poisoned_predictions
        )
    )

    # Compare model parameters
    benign_state = benign_model.state_dict()
    poisoned_state = poisoned_model.state_dict()

    squared_differences = []

    for key in benign_state:

        difference = (
            poisoned_state[key].float()
            - benign_state[key].float()
        )

        squared_differences.append(
            torch.sum(difference ** 2).item()
        )

    parameter_difference = float(
        np.sqrt(sum(squared_differences))
    )

    # Classification reports
    benign_report = classification_report(
        y_true,
        benign_predictions,
        labels=range(7),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    poisoned_report = classification_report(
        y_true,
        poisoned_predictions,
        labels=range(7),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    benign_cm = confusion_matrix(
        y_true,
        benign_predictions,
        labels=range(7)
    )

    poisoned_cm = confusion_matrix(
        y_true,
        poisoned_predictions,
        labels=range(7)
    )

    # ------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Benign accuracy: {benign_accuracy:.4f}%"
    )

    print(
        f"Poisoned accuracy: {poisoned_accuracy:.4f}%"
    )

    print(
        f"Accuracy difference: "
        f"{poisoned_accuracy - benign_accuracy:+.4f} "
        "percentage points"
    )

    print(
        "Changed predictions:",
        f"{changed_predictions:,}"
    )

    print(
        "Parameter difference (L2 norm):",
        f"{parameter_difference:.10f}"
    )

    print("\n" + "=" * 70)
    print("PER-CLASS COMPARISON")
    print("=" * 70)

    for class_name in CLASS_NAMES:

        b = benign_report[class_name]
        p = poisoned_report[class_name]

        print(f"\n{class_name}")

        print(
            f"Benign F1: {b['f1-score'] * 100:.4f}%"
        )

        print(
            f"Poisoned F1: {p['f1-score'] * 100:.4f}%"
        )

        print(
            f"F1 difference: "
            f"{(p['f1-score'] - b['f1-score']) * 100:+.4f} "
            "percentage points"
        )

    # Save confusion matrices
    pd.DataFrame(
        benign_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(
        OUTPUT_DIR / "benign_confusion_matrix.csv"
    )

    pd.DataFrame(
        poisoned_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(
        OUTPUT_DIR / "poisoned_confusion_matrix.csv"
    )

    # Save detailed reports
    results = {
        "test_records": len(y_true),
        "benign_accuracy_percent": float(benign_accuracy),
        "poisoned_accuracy_percent": float(poisoned_accuracy),
        "accuracy_difference_percentage_points": float(
            poisoned_accuracy - benign_accuracy
        ),
        "changed_predictions": changed_predictions,
        "parameter_difference_l2": parameter_difference,
        "benign_report": benign_report,
        "poisoned_report": poisoned_report
    }

    with open(
        OUTPUT_DIR / "comparison_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(results, f, indent=4)

    print("\nComparison files saved to:")
    print(OUTPUT_DIR)

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()
