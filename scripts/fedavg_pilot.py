
from pathlib import Path
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from models import IntrusionDetectionMLP


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

CLIENT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

TEST_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\normalized_v1\test"
)

RESULT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\fedavg_pilot"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 10
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


# ------------------------------------------------
# FEATURE CONFIGURATION
# ------------------------------------------------

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


def get_feature_columns(df):

    columns = [
        c for c in df.columns
        if c not in LABEL_COLUMNS
    ]

    if len(columns) != 78:
        raise ValueError(
            f"Expected 78 features, found {len(columns)}"
        )

    return columns


# ------------------------------------------------
# LOAD A RANDOM SAMPLE FROM ONE CLIENT
# ------------------------------------------------

def load_client_sample(client_id, feature_columns):

    client_dir = (
        CLIENT_ROOT / f"client_{client_id:02d}"
    )

    files = sorted(client_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No training files found for client {client_id}"
        )

    frames = []

    for file in files:
        df = pd.read_parquet(file)
        frames.append(df)

    full_df = pd.concat(
        frames,
        ignore_index=True
    )

    if len(full_df) > SAMPLE_PER_CLIENT:

        full_df = full_df.sample(
            n=SAMPLE_PER_CLIENT,
            random_state=RANDOM_SEED + client_id
        )

    X = full_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y = full_df["Label_id"].to_numpy(
        dtype=np.int64
    )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite features in client {client_id}"
        )

    del frames, full_df

    return (
        torch.from_numpy(X),
        torch.from_numpy(y)
    )


# ------------------------------------------------
# LOAD A SMALL TEST SAMPLE
# ------------------------------------------------

def load_test_sample(feature_columns):

    files = sorted(TEST_ROOT.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            "No test Parquet files found."
        )

    frames = []

    for file_index, file in enumerate(files):

        df = pd.read_parquet(file)

        if len(df) > TEST_SAMPLE_PER_FILE:

            df = df.sample(
                n=TEST_SAMPLE_PER_FILE,
                random_state=RANDOM_SEED + file_index
            )

        frames.append(df)

    test_df = pd.concat(
        frames,
        ignore_index=True
    )

    X = test_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y = test_df["Label_id"].to_numpy(
        dtype=np.int64
    )

    del frames, test_df

    return (
        torch.from_numpy(X),
        torch.from_numpy(y)
    )


# ------------------------------------------------
# LOCAL CLIENT TRAINING
# ------------------------------------------------

def train_local_model(global_state, client_id, feature_columns):

    X, y = load_client_sample(
        client_id,
        feature_columns
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
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

    sample_count = len(X)

    del X, y, dataset, loader, local_model

    return updated_state, sample_count, average_loss


# ------------------------------------------------
# FEDERATED AVERAGING
# ------------------------------------------------

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


# ------------------------------------------------
# GLOBAL MODEL EVALUATION
# ------------------------------------------------

def evaluate_model(model, X, y):

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch_X, batch_y in loader:

            outputs = model(batch_X)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == batch_y
            ).sum().item()

            total += len(batch_y)

    return 100.0 * correct / total


# ------------------------------------------------
# MAIN FEDERATED LEARNING PILOT
# ------------------------------------------------

def main():

    print("=" * 65)
    print("FEDERATED LEARNING - FEDAVG PILOT")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Participants:", NUM_CLIENTS)
    print("Samples per participant:", SAMPLE_PER_CLIENT)
    print("Local epochs:", LOCAL_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    # Get feature schema
    first_client = CLIENT_ROOT / "client_01"

    first_file = sorted(
        first_client.glob("*.parquet")
    )[0]

    first_df = pd.read_parquet(first_file)

    feature_columns = get_feature_columns(first_df)

    del first_df

    # Initialize central global model
    global_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    ).to(DEVICE)

    global_state = deepcopy(
        global_model.state_dict()
    )

    # Prepare small test sample
    X_test, y_test = load_test_sample(
        feature_columns
    )

    initial_accuracy = evaluate_model(
        global_model,
        X_test,
        y_test
    )

    print("\nInitial untrained accuracy:",
          f"{initial_accuracy:.4f}%")

    client_updates = []

    print("\nStarting communication round 1...")

    # Train each participant
    for client_id in range(1, NUM_CLIENTS + 1):

        print(
            f"\nTraining client {client_id}/{NUM_CLIENTS}"
        )

        state, samples, loss = train_local_model(
            global_state,
            client_id,
            feature_columns
        )

        client_updates.append({
            "state": state,
            "samples": samples
        })

        print("Training samples:", f"{samples:,}")
        print("Local loss:", f"{loss:.6f}")

    # Central server aggregates client updates
    print("\nPerforming FedAvg aggregation...")

    updated_global_state = fedavg(
        client_updates
    )

    global_model.load_state_dict(
        updated_global_state
    )

    # Evaluate the aggregated model
    final_accuracy = evaluate_model(
        global_model,
        X_test,
        y_test
    )

    # Save pilot model
    checkpoint_path = (
        RESULT_DIR / "fedavg_pilot_round1.pt"
    )

    torch.save(
        global_model.state_dict(),
        checkpoint_path
    )

    results = {
        "experiment": "FedAvg pilot",
        "communication_rounds": 1,
        "participants": NUM_CLIENTS,
        "samples_per_participant": SAMPLE_PER_CLIENT,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "test_sample_count": len(y_test),
        "initial_accuracy_percent": initial_accuracy,
        "aggregated_accuracy_percent": final_accuracy,
        "device": str(DEVICE)
    }

    results_path = RESULT_DIR / "pilot_results.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 65)
    print("FEDAVG PILOT SUMMARY")
    print("=" * 65)

    print("Participants trained:", len(client_updates))
    print(
        "Total local training samples:",
        sum(x["samples"] for x in client_updates)
    )

    print("Initial accuracy:",
          f"{initial_accuracy:.4f}%")

    print("Accuracy after aggregation:",
          f"{final_accuracy:.4f}%")

    print("Checkpoint:", checkpoint_path)
    print("Results:", results_path)

    print("\nFedAvg pilot completed successfully.")


if __name__ == "__main__":
    main()
