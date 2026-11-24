
from pathlib import Path
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from models import IntrusionDetectionMLP


# ==================================================
# CONFIGURATION
# ==================================================

CLIENT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

TEST_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\normalized_v1\test"
)

RESULT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\fedavg_poison_pilot_r20"
)

NUM_CLIENTS = 10
MALICIOUS_CLIENT = 10

POISONING_RATIO = 0.20
TARGET_LABEL = 0

SAMPLE_PER_CLIENT = 5000
TEST_SAMPLE_PER_FILE = 2500

BATCH_SIZE = 512
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

RANDOM_SEED = 42
DEVICE = torch.device("cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(4)

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]

CLASS_NAMES = [
    "Benign",
    "Bot",
    "DoS",
    "Infiltration",
    "PortScan",
    "Web Attack",
    "Brute Force"
]

if RESULT_DIR.exists() and any(RESULT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {RESULT_DIR}"
    )

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD FEATURES
# ==================================================

first_file = sorted(
    (CLIENT_ROOT / "client_01").glob("*.parquet")
)[0]

first_df = pd.read_parquet(first_file)

FEATURE_COLUMNS = [
    c for c in first_df.columns
    if c not in LABEL_COLUMNS
]

if len(FEATURE_COLUMNS) != 78:
    raise ValueError("Expected exactly 78 features.")

del first_df


# ==================================================
# LOAD CLIENT DATA AND APPLY LABEL FLIPPING
# ==================================================

def load_client_sample(client_id):

    client_dir = CLIENT_ROOT / f"client_{client_id:02d}"

    files = sorted(client_dir.glob("*.parquet"))

    if len(files) != 8:
        raise ValueError(
            f"Expected eight files for client {client_id}"
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

    original_labels = df["Label_id"].to_numpy(
        dtype=np.int64,
        copy=True
    )

    poisoned_indices = np.array([], dtype=np.int64)

    # Only Client 10 is malicious
    if client_id == MALICIOUS_CLIENT:

        malicious_indices = np.flatnonzero(
            original_labels != TARGET_LABEL
        )

        number_to_flip = int(
            len(malicious_indices) * POISONING_RATIO
        )

        rng = np.random.default_rng(
            RANDOM_SEED + client_id
        )

        poisoned_indices = rng.choice(
            malicious_indices,
            size=number_to_flip,
            replace=False
        )

        # Change labels only, never traffic features
        df.loc[poisoned_indices, "Label_id"] = TARGET_LABEL

        print(
            f"Client {client_id}: "
            f"{number_to_flip:,} of "
            f"{len(malicious_indices):,} malicious "
            f"labels modified."
        )

    # Use the same sample-selection procedure
    # as the original FedAvg pilot
    if len(df) > SAMPLE_PER_CLIENT:

        df = df.sample(
            n=SAMPLE_PER_CLIENT,
            random_state=RANDOM_SEED + client_id
        )

    poisoned_in_sample = int(
        df.index.isin(poisoned_indices).sum()
    )

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True
    )

    y = df["Label_id"].to_numpy(
        dtype=np.int64,
        copy=True
    )

    if not np.isfinite(X).all():
        raise ValueError("Non-finite input features detected.")

    if not np.isin(y, range(7)).all():
        raise ValueError("Invalid class labels detected.")

    # Verify that the feature data was not changed
    del df

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        poisoned_in_sample
    )


# ==================================================
# LOCAL TRAINING
# ==================================================

def train_local_model(global_state, client_id):

    X, y, poisoned_count = load_client_sample(
        client_id
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    ).to(DEVICE)

    model.load_state_dict(global_state)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    total_loss = 0.0
    total_samples = 0

    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

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

    local_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    sample_count = len(y)

    del X, y, dataset, loader, model, optimizer

    return (
        local_state,
        sample_count,
        average_loss,
        poisoned_count
    )


# ==================================================
# FEDAVG AGGREGATION
# ==================================================

def fedavg(client_updates):

    total_samples = sum(
        item["samples"]
        for item in client_updates
    )

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


# ==================================================
# TEST DATA
# ==================================================

def load_test_sample():

    files = sorted(TEST_ROOT.glob("*.parquet"))

    frames = []

    for i, file in enumerate(files):

        df = pd.read_parquet(file)

        if len(df) > TEST_SAMPLE_PER_FILE:

            df = df.sample(
                n=TEST_SAMPLE_PER_FILE,
                random_state=RANDOM_SEED + i
            )

        frames.append(df)

    test_df = pd.concat(
        frames,
        ignore_index=True
    )

    X = torch.from_numpy(
        test_df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )
    )

    y = torch.from_numpy(
        test_df["Label_id"].to_numpy(
            dtype=np.int64
        )
    )

    del frames, test_df

    return X, y


def evaluate_model(model, X, y):

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    correct = 0
    total = 0

    model.eval()

    with torch.inference_mode():

        for batch_X, batch_y in loader:

            outputs = model(batch_X)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == batch_y
            ).sum().item()

            total += len(batch_y)

    return 100 * correct / total


# ==================================================
# MAIN EXPERIMENT
# ==================================================

def main():

    print("=" * 65)
    print("LABEL-FLIPPING POISONING PILOT")
    print("=" * 65)

    print("Malicious participant:", MALICIOUS_CLIENT)
    print("Poisoning ratio:", POISONING_RATIO * 100, "%")
    print("Target label: Benign")
    print("Device:", DEVICE)

    X_test, y_test = load_test_sample()

    global_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    ).to(DEVICE)

    global_state = deepcopy(
        global_model.state_dict()
    )

    initial_accuracy = evaluate_model(
        global_model,
        X_test,
        y_test
    )

    print(
        "\nInitial accuracy:",
        f"{initial_accuracy:.4f}%"
    )

    client_updates = []
    total_poisoned_in_sample = 0

    for client_id in range(1, NUM_CLIENTS + 1):

        print(
            f"\nTraining client {client_id}/{NUM_CLIENTS}"
        )

        state, samples, loss, poisoned = train_local_model(
            global_state,
            client_id
        )

        client_updates.append({
            "state": state,
            "samples": samples
        })

        total_poisoned_in_sample += poisoned

        print("Training records:", samples)
        print("Local loss:", f"{loss:.6f}")
        print("Poisoned sampled labels:", poisoned)

    print("\nCentral server: applying FedAvg...")

    global_state = fedavg(client_updates)

    global_model.load_state_dict(global_state)

    final_accuracy = evaluate_model(
        global_model,
        X_test,
        y_test
    )

    checkpoint = RESULT_DIR / "poisoned_global_round1.pt"

    torch.save(
        global_model.state_dict(),
        checkpoint
    )

    results = {
        "experiment": "FedAvg label-flipping pilot",
        "communication_rounds": 1,
        "number_of_clients": NUM_CLIENTS,
        "malicious_client": MALICIOUS_CLIENT,
        "poisoning_ratio": POISONING_RATIO,
        "target_label": "Benign",
        "samples_per_client": SAMPLE_PER_CLIENT,
        "poisoned_labels_in_training_sample": total_poisoned_in_sample,
        "test_sample_count": len(y_test),
        "initial_accuracy_percent": initial_accuracy,
        "poisoned_accuracy_percent": final_accuracy,
        "accuracy_change_percentage_points": (
            final_accuracy - initial_accuracy
        ),
        "device": str(DEVICE)
    }

    with open(
        RESULT_DIR / "poisoning_pilot_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(results, f, indent=4)

    print("\n" + "=" * 65)
    print("POISONING PILOT SUMMARY")
    print("=" * 65)

    print("Participants:", NUM_CLIENTS)
    print("Malicious participant:", MALICIOUS_CLIENT)

    print(
        "Poisoned labels in sampled training data:",
        total_poisoned_in_sample
    )

    print(
        "Initial accuracy:",
        f"{initial_accuracy:.4f}%"
    )

    print(
        "Accuracy after poisoning:",
        f"{final_accuracy:.4f}%"
    )

    print(
        "Accuracy change:",
        f"{final_accuracy - initial_accuracy:+.4f}",
        "percentage points"
    )

    print("Checkpoint:", checkpoint)
    print("Results:", RESULT_DIR)

    print("\nPoisoning pilot completed.")


if __name__ == "__main__":
    main()
