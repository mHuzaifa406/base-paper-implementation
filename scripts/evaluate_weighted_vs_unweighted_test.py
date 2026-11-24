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
TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "normalized_v1" / "test"

MODEL_GROUPS = {
    "unweighted": PROJECT_ROOT / "results" / "round11_defense_evaluation_checkpoint_v1",
    "class_weighted": PROJECT_ROOT / "results" / "round11_defense_classweighted_v1",
}

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
EXPECTED_FEATURES = 78
EXPECTED_TEST_FILES = 8
BATCH_SIZE = 2048
NUM_THREADS = 4

OUTPUT_ROOT = PROJECT_ROOT / "results" / "final_test_weighted_comparison_v1"

torch.set_num_threads(NUM_THREADS)


# ============================================================
# VALIDATION
# ============================================================
def validate_paths():
    if not TEST_ROOT.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_ROOT}")

    for group, checkpoint_root in MODEL_GROUPS.items():
        if not checkpoint_root.exists():
            raise FileNotFoundError(
                f"Checkpoint directory for {group} not found: {checkpoint_root}"
            )

        for scenario in SCENARIOS:
            checkpoint_path = (
                checkpoint_root / scenario / "final_global_model.pt"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing {group}/{scenario} checkpoint: {checkpoint_path}"
                )
            if checkpoint_path.stat().st_size == 0:
                raise ValueError(f"Empty checkpoint: {checkpoint_path}")

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT_ROOT}. "
            "Existing results will not be overwritten."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CHECKPOINTS
# ============================================================
def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions without weights_only.
        return torch.load(path, map_location="cpu")


def load_models():
    print("=" * 85)
    print("HELD-OUT TEST: CLASS-WEIGHTED VS UNWEIGHTED MODELS")
    print("=" * 85)

    models = {}
    feature_columns_reference = None
    checkpoint_info = {}

    for group, checkpoint_root in MODEL_GROUPS.items():
        for scenario in SCENARIOS:
            checkpoint_path = (
                checkpoint_root / scenario / "final_global_model.pt"
            )

            checkpoint = safe_torch_load(checkpoint_path)

            required = {
                "state_dict",
                "model_config",
                "feature_columns",
                "label_column",
                "class_names",
            }
            missing = required.difference(checkpoint.keys())
            if missing:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} missing keys: {sorted(missing)}"
                )

            feature_columns = list(checkpoint["feature_columns"])
            if len(feature_columns) != EXPECTED_FEATURES:
                raise ValueError(
                    f"{group}/{scenario}: expected {EXPECTED_FEATURES} features, "
                    f"found {len(feature_columns)}"
                )

            if feature_columns_reference is None:
                feature_columns_reference = feature_columns
            elif feature_columns != feature_columns_reference:
                raise ValueError(
                    f"Feature order mismatch in {group}/{scenario} checkpoint"
                )

            if checkpoint["label_column"] != "Label_id":
                raise ValueError(
                    f"Unexpected label column in {group}/{scenario}"
                )

            if list(checkpoint["class_names"]) != CLASS_NAMES:
                raise ValueError(
                    f"Class mapping mismatch in {group}/{scenario}"
                )

            config = checkpoint["model_config"]
            input_features = int(config["input_features"])
            num_classes = int(config["num_classes"])
            if input_features != EXPECTED_FEATURES or num_classes != NUM_CLASSES:
                raise ValueError(
                    f"Unexpected model config in {group}/{scenario}: {config}"
                )

            model = IntrusionDetectionMLP(
                input_features=input_features,
                num_classes=num_classes,
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()

            key = f"{group}__{scenario}"
            models[key] = model

            checkpoint_info[key] = {
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "training_config": checkpoint.get("training_config", {}),
                "defense_config": checkpoint.get("defense_config", {}),
                "final_validation_metrics": checkpoint.get(
                    "final_validation_metrics", {}
                ),
            }

            print(
                f"Loaded {key}: {checkpoint_path.stat().st_size:,} bytes"
            )
            del checkpoint

    return models, feature_columns_reference, checkpoint_info


# ============================================================
# TEST FILE DISCOVERY
# ============================================================
def discover_test_files(feature_columns):
    files = sorted(TEST_ROOT.glob("*.parquet"))
    if len(files) != EXPECTED_TEST_FILES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_FILES} test Parquet files, found {len(files)}"
        )

    required_columns = feature_columns + ["Label_id"]
    print("\nTest files:")
    for path in files:
        print(" -", path.name)
        columns = pd.read_parquet(path).columns.tolist()
        missing = [col for col in required_columns if col not in columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")

    return files


# ============================================================
# EVALUATION
# ============================================================
def evaluate_models(models, feature_columns, test_files):
    model_keys = list(models.keys())
    prediction_parts = {key: [] for key in model_keys}
    target_parts = []
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_records = 0

    print("\nEvaluating all six checkpoints against the same untouched test records...")

    for file_index, file_path in enumerate(test_files, start=1):
        print(
            f"\nReading test file {file_index}/{len(test_files)}: {file_path.name}"
        )
        frame = pd.read_parquet(file_path)
        X = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
        y = frame["Label_id"].to_numpy(dtype=np.int64, copy=True)
        del frame

        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite feature values in {file_path.name}")
        if not np.isin(y, np.arange(NUM_CLASSES)).all():
            raise ValueError(f"Unexpected class IDs in {file_path.name}")

        class_counts += np.bincount(y, minlength=NUM_CLASSES)
        target_parts.append(y.copy())

        for start in range(0, len(y), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(y))
            batch = torch.from_numpy(X[start:end])

            for key in model_keys:
                with torch.inference_mode():
                    logits = models[key](batch)
                    pred = logits.argmax(dim=1).cpu().numpy()
                prediction_parts[key].append(pred)

        total_records += len(y)
        print(f"Records processed: {total_records:,}")
        del X, y
        gc.collect()

    y_true = np.concatenate(target_parts)
    if len(y_true) != total_records or int(class_counts.sum()) != total_records:
        raise RuntimeError("Test record count validation failed")

    malicious_mask = y_true != 0
    malicious_count = int(malicious_mask.sum())
    if malicious_count == 0:
        raise ValueError("No malicious test records; ASR cannot be computed")

    summary_rows = []
    report_metadata = {}

    for key in model_keys:
        y_pred = np.concatenate(prediction_parts[key])
        if len(y_pred) != len(y_true):
            raise RuntimeError(f"Prediction count mismatch for {key}")

        accuracy = accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            average="weighted",
            zero_division=0,
        )

        malicious_as_benign = int(np.sum(y_pred[malicious_mask] == 0))
        asr = malicious_as_benign / malicious_count

        matrix = confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_CLASSES))
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        group, scenario = key.split("__", 1)
        model_dir = OUTPUT_ROOT / group / scenario
        model_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            matrix,
            index=[f"Actual_{name}" for name in CLASS_NAMES],
            columns=[f"Predicted_{name}" for name in CLASS_NAMES],
        ).to_csv(model_dir / "confusion_matrix.csv")

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(model_dir / "classification_report.csv")
        with open(model_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        metrics = {
            "model_group": group,
            "scenario": scenario,
            "test_records": int(total_records),
            "accuracy": float(accuracy),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "attack_success_rate": float(asr),
            "malicious_test_records": malicious_count,
            "malicious_records_predicted_benign": malicious_as_benign,
        }
        summary_rows.append(metrics)
        report_metadata[key] = metrics

        print("\n" + "-" * 75)
        print(f"TEST RESULTS: {group.upper()} / {scenario.upper()}")
        print("-" * 75)
        print(f"Accuracy:          {accuracy * 100:.4f}%")
        print(f"Macro precision:   {macro_p:.6f}")
        print(f"Macro recall:      {macro_r:.6f}")
        print(f"Macro F1:          {macro_f1:.6f}")
        print(f"Weighted F1:       {weighted_f1:.6f}")
        print(f"Attack success:    {asr * 100:.4f}%")
        print(
            f"Malicious as Benign: {malicious_as_benign:,}/{malicious_count:,}"
        )
        print("Classification report:", model_dir / "classification_report.csv")
        print("Confusion matrix:", model_dir / "confusion_matrix.csv")
        del y_pred

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_ROOT / "six_model_test_comparison.csv", index=False)

    # Paired comparison: class-weighted minus unweighted, same scenario.
    comparison_rows = []
    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "attack_success_rate",
        "malicious_records_predicted_benign",
    ]

    for scenario in SCENARIOS:
        unweighted = summary[
            (summary["model_group"] == "unweighted")
            & (summary["scenario"] == scenario)
        ].iloc[0]
        weighted = summary[
            (summary["model_group"] == "class_weighted")
            & (summary["scenario"] == scenario)
        ].iloc[0]

        row = {"scenario": scenario}
        for metric in metric_columns:
            row[f"unweighted_{metric}"] = unweighted[metric]
            row[f"class_weighted_{metric}"] = weighted[metric]
            row[f"weighted_minus_unweighted_{metric}"] = (
                weighted[metric] - unweighted[metric]
            )
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        OUTPUT_ROOT / "paired_weighting_comparison.csv",
        index=False,
    )

    distribution = pd.DataFrame({
        "class_id": list(range(NUM_CLASSES)),
        "class_name": CLASS_NAMES,
        "test_records": class_counts.tolist(),
        "test_percentage": (class_counts / total_records * 100).tolist(),
    })
    distribution.to_csv(
        OUTPUT_ROOT / "test_class_distribution.csv",
        index=False,
    )

    return summary, comparison, distribution, report_metadata, total_records


# ============================================================
# MAIN
# ============================================================
def main():
    validate_paths()
    models, feature_columns, checkpoint_info = load_models()
    test_files = discover_test_files(feature_columns)

    summary, comparison, distribution, report_metadata, total_records = (
        evaluate_models(models, feature_columns, test_files)
    )

    metadata = {
        "evaluation": "paired held-out test comparison: unweighted vs class-weighted",
        "test_directory": str(TEST_ROOT),
        "test_records": int(total_records),
        "test_files": [path.name for path in test_files],
        "feature_count": len(feature_columns),
        "class_names": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "device": "CPU",
        "test_data_used_for_training_or_threshold_selection": False,
        "checkpoints": checkpoint_info,
        "metrics": report_metadata,
        "interpretation_note": (
            "Class weighting is an additional implementation experiment, not a method "
            "specified by the source paper. Results are measured on the same held-out "
            "test records for all six checkpoints."
        ),
    }
    with open(OUTPUT_ROOT / "evaluation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 85)
    print("FINAL SIX-MODEL HELD-OUT TEST COMPARISON")
    print("=" * 85)
    print(summary.to_string(index=False))

    print("\n" + "=" * 85)
    print("PAIRED CLASS-WEIGHTED MINUS UNWEIGHTED COMPARISON")
    print("=" * 85)
    print(comparison.to_string(index=False))

    print("\nTest records evaluated:", f"{total_records:,}")
    print("Results saved:", OUTPUT_ROOT)
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
