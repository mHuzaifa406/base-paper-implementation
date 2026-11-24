from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from models import IntrusionDetectionMLP


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"E:\IoMT_FL_IDS")

TEST_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "normalized_v1" / "test"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT / "results" / "round11_defense_evaluation_checkpoint_v1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "final_test_evaluation_v1"
)

SCENARIOS = [
    "benign_defense",
    "poisoned_no_defense",
    "poisoned_defense",
]

CLASS_NAMES = [
    "Benign",
    "Bot",
    "DoS",
    "Infiltration",
    "PortScan",
    "Web Attack",
    "Brute Force",
]

NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 2048
NUM_THREADS = 4
EXPECTED_FEATURES = 78
EXPECTED_TEST_FILES = 8


torch.set_num_threads(NUM_THREADS)


# ============================================================
# SAFETY CHECKS
# ============================================================

def validate_paths_and_output():
    if not TEST_ROOT.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_ROOT}")

    if not CHECKPOINT_ROOT.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT_ROOT}"
        )

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_ROOT}\n"
            "Choose a new OUTPUT_ROOT; existing results will not be overwritten."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD AND VERIFY CHECKPOINTS
# ============================================================

def load_checkpoints():
    print("=" * 80)
    print("FINAL HELD-OUT TEST-SET EVALUATION")
    print("=" * 80)
    print("\nLoading saved global-model checkpoints...")

    models = {}
    reference_features = None
    checkpoint_metadata = {}

    for scenario in SCENARIOS:
        checkpoint_path = (
            CHECKPOINT_ROOT / scenario / "final_global_model.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {scenario}: {checkpoint_path}"
            )

        if checkpoint_path.stat().st_size == 0:
            raise ValueError(f"Checkpoint is empty: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Compatibility fallback for older PyTorch versions.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        required_keys = {
            "state_dict",
            "model_config",
            "feature_columns",
            "label_column",
            "class_names",
        }

        missing = required_keys.difference(checkpoint.keys())
        if missing:
            raise ValueError(
                f"Checkpoint {scenario} is missing fields: {sorted(missing)}"
            )

        feature_columns = list(checkpoint["feature_columns"])

        if len(feature_columns) != EXPECTED_FEATURES:
            raise ValueError(
                f"{scenario}: expected {EXPECTED_FEATURES} feature names, "
                f"found {len(feature_columns)}"
            )

        if reference_features is None:
            reference_features = feature_columns
        elif feature_columns != reference_features:
            raise ValueError(
                f"Feature-column order differs in checkpoint {scenario}."
            )

        if checkpoint["label_column"] != "Label_id":
            raise ValueError(
                f"Unexpected label column in {scenario}: "
                f"{checkpoint['label_column']}"
            )

        if list(checkpoint["class_names"]) != CLASS_NAMES:
            raise ValueError(
                f"Class-name mapping differs in checkpoint {scenario}."
            )

        model_config = checkpoint["model_config"]
        input_features = int(model_config["input_features"])
        num_classes = int(model_config["num_classes"])

        if input_features != EXPECTED_FEATURES or num_classes != NUM_CLASSES:
            raise ValueError(
                f"Unexpected model configuration for {scenario}: "
                f"{model_config}"
            )

        model = IntrusionDetectionMLP(
            input_features=input_features,
            num_classes=num_classes,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()

        models[scenario] = model
        checkpoint_metadata[scenario] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "scenario": checkpoint.get("scenario", scenario),
            "training_config": checkpoint.get("training_config", {}),
            "defense_config": checkpoint.get("defense_config", {}),
            "final_validation_metrics": checkpoint.get(
                "final_validation_metrics", {}
            ),
        }

        print(
            f"Loaded {scenario}: "
            f"{checkpoint_path.stat().st_size:,} bytes"
        )

        del checkpoint

    return models, reference_features, checkpoint_metadata


# ============================================================
# DISCOVER TEST DATA
# ============================================================

def discover_test_files(feature_columns):
    files = sorted(TEST_ROOT.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet test files found in {TEST_ROOT}"
        )

    if len(files) != EXPECTED_TEST_FILES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_FILES} test Parquet files, "
            f"found {len(files)}. Inspect the directory before proceeding."
        )

    required_columns = feature_columns + ["Label_id"]

    print("\nDiscovered test files:")
    for file in files:
        print(" -", file.name)

    # Check schema before running inference.
    for file in files:
        columns = pd.read_parquet(file).columns.tolist()
        missing = [column for column in required_columns if column not in columns]
        if missing:
            raise ValueError(
                f"Test file {file.name} is missing columns: {missing}"
            )

    return files


# ============================================================
# EVALUATE THE THREE MODELS ON THE SAME HELD-OUT TEST SET
# ============================================================

def evaluate_all_models(models, feature_columns, test_files):
    predictions = {scenario: [] for scenario in SCENARIOS}
    targets_all = []
    total_records = 0
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    print("\nEvaluating models on the untouched test dataset...")
    print("Batch size:", BATCH_SIZE)

    for file_index, file_path in enumerate(test_files, start=1):
        print(
            f"\nReading test file {file_index}/{len(test_files)}: "
            f"{file_path.name}"
        )

        df = pd.read_parquet(file_path)

        X = df[feature_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        y = df["Label_id"].to_numpy(
            dtype=np.int64,
            copy=True,
        )

        del df

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite feature values found in {file_path.name}"
            )

        if not np.isin(y, np.arange(NUM_CLASSES)).all():
            raise ValueError(
                f"Unexpected class IDs found in {file_path.name}"
            )

        class_counts += np.bincount(y, minlength=NUM_CLASSES)
        targets_all.append(y.copy())

        for start in range(0, len(y), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(y))
            batch_X = torch.from_numpy(X[start:end])

            for scenario in SCENARIOS:
                model = models[scenario]
                with torch.inference_mode():
                    logits = model(batch_X)
                    batch_predictions = logits.argmax(dim=1).cpu().numpy()
                predictions[scenario].append(batch_predictions)

        total_records += len(y)
        print(f"Records processed: {total_records:,}")

        del X, y
        gc.collect()

    y_true = np.concatenate(targets_all)

    if len(y_true) != total_records:
        raise RuntimeError("Test target count mismatch.")

    if class_counts.sum() != total_records:
        raise RuntimeError("Test class-count total mismatch.")

    if not np.any(y_true != 0):
        raise ValueError(
            "Test data has no non-Benign records; ASR cannot be computed."
        )

    all_reports = {}
    summary_rows = []

    for scenario in SCENARIOS:
        y_pred = np.concatenate(predictions[scenario])

        if len(y_pred) != len(y_true):
            raise RuntimeError(
                f"Prediction count mismatch for {scenario}."
            )

        accuracy = accuracy_score(y_true, y_pred)

        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        )

        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(range(NUM_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        )

        malicious_mask = y_true != 0
        malicious_count = int(malicious_mask.sum())
        malicious_as_benign = int(
            np.sum(y_pred[malicious_mask] == 0)
        )
        attack_success_rate = (
            malicious_as_benign / malicious_count
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        scenario_dir = OUTPUT_ROOT / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        matrix_df = pd.DataFrame(
            matrix,
            index=[f"Actual_{name}" for name in CLASS_NAMES],
            columns=[f"Predicted_{name}" for name in CLASS_NAMES],
        )
        matrix_df.to_csv(scenario_dir / "confusion_matrix.csv")

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(scenario_dir / "classification_report.csv")

        with open(
            scenario_dir / "classification_report.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(report, handle, indent=4)

        # Save compact per-model evaluation metadata.
        model_metrics = {
            "scenario": scenario,
            "test_records": int(total_records),
            "test_accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "attack_success_rate": float(attack_success_rate),
            "malicious_test_records": malicious_count,
            "malicious_records_predicted_benign": malicious_as_benign,
        }

        with open(
            scenario_dir / "test_metrics.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(model_metrics, handle, indent=4)

        all_reports[scenario] = report
        summary_rows.append(model_metrics)

        print("\n" + "-" * 70)
        print("TEST RESULTS:", scenario.upper())
        print("-" * 70)
        print(f"Accuracy:          {accuracy * 100:.4f}%")
        print(f"Macro precision:   {macro_precision:.6f}")
        print(f"Macro recall:      {macro_recall:.6f}")
        print(f"Macro F1:          {macro_f1:.6f}")
        print(f"Weighted F1:       {weighted_f1:.6f}")
        print(f"Attack success:    {attack_success_rate * 100:.4f}%")
        print(
            "Malicious as Benign: "
            f"{malicious_as_benign:,}/{malicious_count:,}"
        )
        print("Confusion matrix saved:", scenario_dir / "confusion_matrix.csv")

        del y_pred

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ROOT / "test_set_comparison.csv", index=False)

    class_distribution = pd.DataFrame({
        "class_id": list(range(NUM_CLASSES)),
        "class_name": CLASS_NAMES,
        "test_records": class_counts.tolist(),
        "test_percentage": (
            class_counts / total_records * 100
        ).tolist(),
    })
    class_distribution.to_csv(
        OUTPUT_ROOT / "test_class_distribution.csv",
        index=False,
    )

    return summary_df, total_records, class_counts


# ============================================================
# MAIN
# ============================================================

def main():
    validate_paths_and_output()

    models, feature_columns, checkpoint_metadata = load_checkpoints()
    test_files = discover_test_files(feature_columns)

    summary_df, total_records, class_counts = evaluate_all_models(
        models,
        feature_columns,
        test_files,
    )

    metadata = {
        "evaluation": "final held-out test-set evaluation",
        "test_directory": str(TEST_ROOT),
        "checkpoint_directory": str(CHECKPOINT_ROOT),
        "test_records": int(total_records),
        "test_files": [file.name for file in test_files],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "class_names": CLASS_NAMES,
        "class_distribution": {
            CLASS_NAMES[i]: int(class_counts[i])
            for i in range(NUM_CLASSES)
        },
        "batch_size": BATCH_SIZE,
        "device": "CPU",
        "test_data_used_for_training_or_threshold_selection": False,
        "checkpoints": checkpoint_metadata,
        "results": summary_df.to_dict(orient="records"),
    }

    with open(
        OUTPUT_ROOT / "evaluation_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=4)

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST-SET COMPARISON")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nTest records evaluated:", f"{total_records:,}")
    print("Results saved to:", OUTPUT_ROOT)
    print("Final test evaluation completed.")


if __name__ == "__main__":
    main()
