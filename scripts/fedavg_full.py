
from pathlib import Path
from copy import deepcopy
from collections import Counter
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from models import IntrusionDetectionMLP


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLIENT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

TEST_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\normalized_v1\test"
)

OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\fedavg_full_v1"
)

NUM_CLIENTS = 10
NUM_ROUNDS = 10

LOCAL_EPOCHS = 1
BATCH_SIZE = 512
LEARNING_RATE = 0.01

RANDOM_SEED = 42
NUM_THREADS = 4

DEVICE = torch.device("cpu")

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

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(NUM_THREADS)


# ============================================================
# 2. PREPARE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_DIR}. "
        "Please use a new output directory to avoid overwriting "
        "previous experiment results."
    )


# ============================================================
# 3. IDENTIFY FEATURE COLUMNS
# ============================================================

client_01_files = sorted(
    (CLIENT_ROOT / "client_01").glob("*.parquet")
)

if len(client_01_files) != 8:
    raise ValueError(
        "Client 01 must contain eight training files."
    )

first_df = pd.read_parquet(client_01_files[0])

FEATURE_COLUMNS = [
    col for col in first_df.columns
    if col not in LABEL_COLUMNS
]

if len(FEATURE_COLUMNS) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(FEATURE_COLUMNS)}"
    )

del first_df


# ============================================================
# 4. LOAD ONE CLIENT'S COMPLETE DATASET
# ============================================================

def load_client_data(client_id):

    client_dir = CLIENT_ROOT / f"client_{client_id:02d}"

    files = sorted(client_dir.glob("*.parquet"))

    if len(files) != 8:
        raise ValueError(
            f"Expected eight files for client {client_id}."
        )

    frames = []

    for file in files:

        df = pd.read_parquet(file)

        if len(df.columns) != 81:
            raise ValueError(
                f"Unexpected number of columns in {file.name}"
            )

        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    del frames

    X = data[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True
    )

    y = data["Label_id"].to_numpy(
        dtype=np.int64,
        copy=True
    )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite features in client {client_id}"
        )

    if not np.isin(y, range(7)).all():
        raise ValueError(
            f"Invalid labels in client {client_id}"
        )

    del data

    return (
        torch.from_numpy(X),
        torch.from_numpy(y)
    )


# ============================================================
# 5. TRAIN A LOCAL PARTICIPANT MODEL
# ============================================================

def train_local_model(global_state, client_id, round_id):

    X, y = load_client_data(client_id)

    dataset = TensorDataset(X, y)

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED + round_id * 100 + client_id
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    local_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    ).to(DEVICE)

    local_model.load_state_dict(global_state)

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    local_model.train()

    total_loss = 0.0
    total_samples = 0

    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * len(batch_y)
            )

            total_samples += len(batch_y)

    average_loss = total_loss / total_samples

    updated_state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }

    sample_count = len(y)

    del X, y, dataset, loader, local_model, optimizer

    return updated_state, sample_count, average_loss


# ============================================================
# 6. FEDERATED AVERAGING
# ============================================================

def fedavg(client_updates):

    total_samples = sum(
        item["samples"]
        for item in client_updates
    )

    if total_samples <= 0:
        raise ValueError("No client training samples found.")

    global_state = {}

    for key in client_updates[0]["state"]:

        aggregated = torch.zeros_like(
            client_updates[0]["state"][key]
        )

        for item in client_updates:

            weight = item["samples"] / total_samples

            aggregated += (
                item["state"][key] * weight
            )

        global_state[key] = aggregated

    return global_state


# ============================================================
# 7. EVALUATE ON THE COMPLETE TEST DATASET
# ============================================================

def evaluate_model(model):

    test_files = sorted(
        TEST_ROOT.glob("*.parquet")
    )

    if len(test_files) != 8:
        raise ValueError(
            f"Expected eight test files, found {len(test_files)}"
        )

    confusion_matrix = np.zeros(
        (7, 7),
        dtype=np.int64
    )

    total_samples = 0

    model.eval()

    with torch.inference_mode():

        for file in test_files:

            df = pd.read_parquet(file)

            X = df[FEATURE_COLUMNS].to_numpy(
                dtype=np.float32,
                copy=True
            )

            y = df["Label_id"].to_numpy(
                dtype=np.int64,
                copy=True
            )

            dataset = TensorDataset(
                torch.from_numpy(X),
                torch.from_numpy(y)
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )

            for batch_X, batch_y in loader:

                outputs = model(batch_X)

                predictions = outputs.argmax(dim=1)

                actual = batch_y.numpy()
                predicted = predictions.numpy()

                encoded = actual * 7 + predicted

                counts = np.bincount(
                    encoded,
                    minlength=49
                )

                confusion_matrix += counts.reshape(7, 7)

            total_samples += len(df)

            del df, X, y, dataset, loader

    # Calculate performance metrics
    true_positive = np.diag(confusion_matrix)

    support = confusion_matrix.sum(axis=1)
    predicted_count = confusion_matrix.sum(axis=0)

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(7, dtype=float),
        where=predicted_count != 0
    )

    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(7, dtype=float),
        where=support != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(7, dtype=float),
        where=(precision + recall) != 0
    )

    accuracy = (
        true_positive.sum() / total_samples
    ) * 100

    metrics = {
        "accuracy_percent": float(accuracy),
        "macro_precision_percent": float(precision.mean() * 100),
        "macro_recall_percent": float(recall.mean() * 100),
        "macro_f1_percent": float(f1.mean() * 100),
        "test_records": int(total_samples),
        "per_class": {}
    }

    for index, class_name in enumerate(CLASS_NAMES):

        metrics["per_class"][class_name] = {
            "precision_percent": float(precision[index] * 100),
            "recall_percent": float(recall[index] * 100),
            "f1_percent": float(f1[index] * 100),
            "support": int(support[index])
        }

    return metrics, confusion_matrix


# ============================================================
# 8. SAVE EXPERIMENT HISTORY
# ============================================================

history = []

def save_history():

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(history, file, indent=4)

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "round_metrics.csv",
        index=False
    )


# ============================================================
# 9. MAIN FEDERATED LEARNING EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("FULL-SCALE FEDERATED LEARNING - FEDAVG")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Participants:", NUM_CLIENTS)
    print("Communication rounds:", NUM_ROUNDS)
    print("Local epochs:", LOCAL_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Number of features:", len(FEATURE_COLUMNS))

    # Initialize central global model
    global_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    ).to(DEVICE)

    global_state = deepcopy(
        global_model.state_dict()
    )

    # Evaluate initial global model
    print("\nEvaluating initial global model...")

    initial_metrics, _ = evaluate_model(global_model)

    print(
        "Initial full-test accuracy:",
        f"{initial_metrics['accuracy_percent']:.4f}%"
    )

    for round_id in range(1, NUM_ROUNDS + 1):

        print("\n" + "=" * 70)
        print(f"COMMUNICATION ROUND {round_id}/{NUM_ROUNDS}")
        print("=" * 70)

        client_updates = []

        # Train all ten participants
        for client_id in range(1, NUM_CLIENTS + 1):

            print(
                f"\nTraining participant "
                f"{client_id}/{NUM_CLIENTS}..."
            )

            state, samples, loss = train_local_model(
                global_state,
                client_id,
                round_id
            )

            client_updates.append({
                "state": state,
                "samples": samples
            })

            print("Training samples:", f"{samples:,}")
            print("Local training loss:", f"{loss:.6f}")

            del state

        total_local_samples = sum(
            item["samples"]
            for item in client_updates
        )

        # Aggregate the local models
        print("\nCentral server: performing FedAvg...")

        global_state = fedavg(client_updates)

        global_model.load_state_dict(
            global_state
        )

        del client_updates

        # Evaluate on all 565,573 test records
        print("Evaluating global model on the complete test set...")

        metrics, confusion_matrix = evaluate_model(
            global_model
        )

        round_result = {
            "round": round_id,
            "total_local_samples": int(total_local_samples),
            "accuracy_percent": metrics["accuracy_percent"],
            "macro_precision_percent": metrics["macro_precision_percent"],
            "macro_recall_percent": metrics["macro_recall_percent"],
            "macro_f1_percent": metrics["macro_f1_percent"],
            "test_records": metrics["test_records"],
            "per_class": metrics["per_class"]
        }

        history.append(round_result)

        # Save model checkpoint after each round
        checkpoint_path = (
            OUTPUT_DIR /
            f"global_model_round_{round_id:02d}.pt"
        )

        torch.save(
            global_model.state_dict(),
            checkpoint_path
        )

        # Save the confusion matrix from the latest round
        cm_df = pd.DataFrame(
            confusion_matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        )

        cm_df.to_csv(
            OUTPUT_DIR / "confusion_matrix_latest.csv"
        )

        save_history()

        print("\nROUND RESULTS")
        print("-" * 45)

        print(
            "Training records:",
            f"{total_local_samples:,}"
        )

        print(
            "Test records:",
            f"{metrics['test_records']:,}"
        )

        print(
            "Accuracy:",
            f"{metrics['accuracy_percent']:.4f}%"
        )

        print(
            "Macro precision:",
            f"{metrics['macro_precision_percent']:.4f}%"
        )

        print(
            "Macro recall:",
            f"{metrics['macro_recall_percent']:.4f}%"
        )

        print(
            "Macro F1:",
            f"{metrics['macro_f1_percent']:.4f}%"
        )

        print("Checkpoint saved:", checkpoint_path)

    # Save final detailed results
    final_metrics = history[-1]

    with open(
        OUTPUT_DIR / "final_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(final_metrics, file, indent=4)

    print("\n" + "=" * 70)
    print("FULL FEDAVG EXPERIMENT COMPLETED")
    print("=" * 70)

    print("Communication rounds:", NUM_ROUNDS)
    print("Participants:", NUM_CLIENTS)

    print(
        "Final accuracy:",
        f"{final_metrics['accuracy_percent']:.4f}%"
    )

    print(
        "Final macro F1:",
        f"{final_metrics['macro_f1_percent']:.4f}%"
    )

    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
