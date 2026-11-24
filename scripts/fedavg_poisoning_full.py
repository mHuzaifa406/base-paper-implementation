
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
# 1. EXPERIMENT CONFIGURATION
# ============================================================

CLIENT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

TEST_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\normalized_v1\test"
)

OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\fedavg_poisoning_full_v1"
)

NUM_CLIENTS = 10
MALICIOUS_CLIENT = 10
NUM_ROUNDS = 10

POISONING_RATIO = 0.20
TARGET_LABEL = 0

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
# 2. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_DIR}. "
        "Use a different directory rather than overwrite results."
    )


# ============================================================
# 3. FEATURE SCHEMA
# ============================================================

client_files = sorted(
    (CLIENT_ROOT / "client_01").glob("*.parquet")
)

if len(client_files) != 8:
    raise ValueError("Expected 8 files for Client 01.")

first_df = pd.read_parquet(client_files[0])

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
# 4. LOAD A CLIENT'S FULL TRAINING DATA
# ============================================================

def load_client_data(client_id):

    client_dir = (
        CLIENT_ROOT / f"client_{client_id:02d}"
    )

    files = sorted(client_dir.glob("*.parquet"))

    if len(files) != 8:
        raise ValueError(
            f"Expected 8 files for Client {client_id}"
        )

    frames = []

    for file in files:
        frames.append(pd.read_parquet(file))

    df = pd.concat(
        frames,
        ignore_index=True
    )

    del frames

    original_labels = df["Label_id"].to_numpy(
        dtype=np.int64,
        copy=True
    )

    poisoned_count = 0
    eligible_count = 0

    # Apply the attack only to Client 10
    if client_id == MALICIOUS_CLIENT:

        eligible_indices = np.flatnonzero(
            original_labels != TARGET_LABEL
        )

        eligible_count = len(eligible_indices)

        poisoned_count = int(
            np.floor(
                eligible_count * POISONING_RATIO
            )
        )

        rng = np.random.default_rng(
            RANDOM_SEED + client_id
        )

        selected_indices = rng.choice(
            eligible_indices,
            size=poisoned_count,
            replace=False
        )

        # Change labels only.
        # Features remain unchanged.
        df.loc[
            selected_indices,
            "Label_id"
        ] = TARGET_LABEL

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True
    )

    y = df["Label_id"].to_numpy(
        dtype=np.int64,
        copy=True
    )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Invalid numeric features in Client {client_id}"
        )

    if not np.isin(y, range(7)).all():
        raise ValueError(
            f"Invalid class labels in Client {client_id}"
        )

    sample_count = len(y)

    del df, original_labels

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        sample_count,
        poisoned_count,
        eligible_count
    )


# ============================================================
# 5. LOCAL MODEL TRAINING
# ============================================================

def train_local_model(global_state, client_id, round_id):

    X, y, samples, poisoned, eligible = (
        load_client_data(client_id)
    )

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
    total_seen = 0

    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in loader:

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

            total_seen += len(batch_y)

    average_loss = total_loss / total_seen

    local_state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }

    del X, y, dataset, loader, local_model, optimizer

    return {
        "state": local_state,
        "samples": samples,
        "loss": average_loss,
        "poisoned": poisoned,
        "eligible": eligible
    }


# ============================================================
# 6. FEDAVG AGGREGATION
# ============================================================

def fedavg(client_updates):

    total_samples = sum(
        update["samples"]
        for update in client_updates
    )

    if total_samples == 0:
        raise ValueError("No client training samples.")

    global_state = {}

    for key in client_updates[0]["state"]:

        aggregated = torch.zeros_like(
            client_updates[0]["state"][key]
        )

        for update in client_updates:

            weight = (
                update["samples"] / total_samples
            )

            aggregated += (
                update["state"][key] * weight
            )

        global_state[key] = aggregated

    return global_state


# ============================================================
# 7. FULL TEST-SET EVALUATION
# ============================================================

def evaluate_model(model):

    test_files = sorted(
        TEST_ROOT.glob("*.parquet")
    )

    if len(test_files) != 8:
        raise ValueError("Expected 8 test files.")

    confusion = np.zeros(
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

            loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(X),
                    torch.from_numpy(y)
                ),
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            for batch_X, batch_y in loader:

                outputs = model(batch_X)

                predictions = outputs.argmax(dim=1)

                encoded = (
                    batch_y.numpy() * 7
                    + predictions.numpy()
                )

                confusion += np.bincount(
                    encoded,
                    minlength=49
                ).reshape(7, 7)

            total_samples += len(df)

            del df, X, y, loader

    tp = np.diag(confusion)

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    precision = np.divide(
        tp,
        predicted,
        out=np.zeros(7, dtype=float),
        where=predicted != 0
    )

    recall = np.divide(
        tp,
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

    return {
        "accuracy_percent": float(
            tp.sum() / total_samples * 100
        ),
        "macro_precision_percent": float(
            precision.mean() * 100
        ),
        "macro_recall_percent": float(
            recall.mean() * 100
        ),
        "macro_f1_percent": float(
            f1.mean() * 100
        ),
        "test_records": int(total_samples),
        "per_class": {
            name: {
                "precision_percent": float(
                    precision[i] * 100
                ),
                "recall_percent": float(
                    recall[i] * 100
                ),
                "f1_percent": float(
                    f1[i] * 100
                ),
                "support": int(support[i])
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion.tolist()
    }


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("FULL-SCALE FEDAVG WITH LABEL-FLIPPING ATTACK")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Clients:", NUM_CLIENTS)
    print("Malicious client:", MALICIOUS_CLIENT)
    print("Poisoning ratio:", POISONING_RATIO)
    print("Communication rounds:", NUM_ROUNDS)

    global_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    ).to(DEVICE)

    global_state = deepcopy(
        global_model.state_dict()
    )

    history = []

    for round_id in range(1, NUM_ROUNDS + 1):

        print("\n" + "=" * 70)
        print(
            f"POISONED COMMUNICATION ROUND "
            f"{round_id}/{NUM_ROUNDS}"
        )
        print("=" * 70)

        client_updates = []
        total_poisoned = 0
        client_losses = {}

        for client_id in range(1, NUM_CLIENTS + 1):

            print(
                f"\nTraining Client {client_id}/{NUM_CLIENTS}"
            )

            update = train_local_model(
                global_state,
                client_id,
                round_id
            )

            client_updates.append(update)

            client_losses[str(client_id)] = update["loss"]

            if client_id == MALICIOUS_CLIENT:

                total_poisoned = update["poisoned"]

                print(
                    "Eligible malicious records:",
                    f"{update['eligible']:,}"
                )

                print(
                    "Flipped labels:",
                    f"{update['poisoned']:,}"
                )

            print(
                "Training records:",
                f"{update['samples']:,}"
            )

            print(
                "Local loss:",
                f"{update['loss']:.6f}"
            )

        print("\nCentral server: FedAvg aggregation...")

        global_state = fedavg(client_updates)

        global_model.load_state_dict(global_state)

        total_training = sum(
            update["samples"]
            for update in client_updates
        )

        del client_updates

        print("Evaluating on complete test set...")

        metrics = evaluate_model(global_model)

        result = {
            "round": round_id,
            "participants": NUM_CLIENTS,
            "malicious_client": MALICIOUS_CLIENT,
            "eligible_malicious_records": (
                update["eligible"]
            ),
            "poisoned_records": total_poisoned,
            "training_records": total_training,
            "client_losses": client_losses,
            "accuracy_percent": metrics["accuracy_percent"],
            "macro_precision_percent": (
                metrics["macro_precision_percent"]
            ),
            "macro_recall_percent": (
                metrics["macro_recall_percent"]
            ),
            "macro_f1_percent": metrics["macro_f1_percent"],
            "test_records": metrics["test_records"],
            "per_class": metrics["per_class"]
        }

        history.append(result)

        checkpoint = (
            OUTPUT_DIR /
            f"poisoned_global_round_{round_id:02d}.pt"
        )

        torch.save(
            global_model.state_dict(),
            checkpoint
        )

        pd.DataFrame(
            metrics["confusion_matrix"],
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        ).to_csv(
            OUTPUT_DIR / "confusion_matrix_latest.csv"
        )

        with open(
            OUTPUT_DIR / "training_history.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(history, f, indent=4)

        pd.DataFrame([
            {
                key: value
                for key, value in row.items()
                if key not in ["client_losses", "per_class"]
            }
            for row in history
        ]).to_csv(
            OUTPUT_DIR / "round_metrics.csv",
            index=False
        )

        print("\nROUND SUMMARY")
        print("Training records:", f"{total_training:,}")
        print("Test records:", f"{metrics['test_records']:,}")
        print(
            "Flipped records:",
            f"{total_poisoned:,}"
        )

        print(
            "Accuracy:",
            f"{metrics['accuracy_percent']:.4f}%"
        )

        print(
            "Macro F1:",
            f"{metrics['macro_f1_percent']:.4f}%"
        )

        print("Checkpoint saved:", checkpoint)

    final_result = history[-1]

    with open(
        OUTPUT_DIR / "final_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(final_result, f, indent=4)

    print("\n" + "=" * 70)
    print("FULL POISONING EXPERIMENT COMPLETED")
    print("=" * 70)

    print("Communication rounds:", NUM_ROUNDS)
    print("Participants:", NUM_CLIENTS)
    print("Malicious client:", MALICIOUS_CLIENT)

    print(
        "Final accuracy:",
        f"{final_result['accuracy_percent']:.4f}%"
    )

    print(
        "Final macro F1:",
        f"{final_result['macro_f1_percent']:.4f}%"
    )

    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
