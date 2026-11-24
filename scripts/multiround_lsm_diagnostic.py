
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans

from models import IntrusionDetectionMLP


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v2"
)

VALIDATION_FILE = (
    DATA_ROOT / "server_validation" / "validation.parquet"
)

OUTPUT_ROOT = Path(
    r"E:\IoMT_FL_IDS\results\multiround_lsm_diagnostic_v1"
)

NUM_CLIENTS = 10
NUM_ROUNDS = 10

MALICIOUS_CLIENT = 10
POISONING_RATIO = 0.20
TARGET_LABEL = 0

BATCH_SIZE = 512
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

RANDOM_SEED = 42
NUM_THREADS = 4

SCENARIOS = [
    "benign",
    "poisoned"
]

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]

torch.set_num_threads(NUM_THREADS)


# ============================================================
# SAFETY CHECKS
# ============================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(DATA_ROOT)

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(VALIDATION_FILE)

if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise FileExistsError(
        f"Output folder is not empty: {OUTPUT_ROOT}"
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CLEAN VALIDATION DATA
# ============================================================

print("=" * 75)
print("MULTI-ROUND FEDERATED LSM DIAGNOSTIC")
print("=" * 75)

print("\nLoading clean server validation data...")

validation_df = pd.read_parquet(
    VALIDATION_FILE
)

FEATURE_COLUMNS = [
    c for c in validation_df.columns
    if c not in LABEL_COLUMNS
]

if len(FEATURE_COLUMNS) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(FEATURE_COLUMNS)}"
    )

X_validation = validation_df[
    FEATURE_COLUMNS
].to_numpy(
    dtype=np.float32,
    copy=True
)

y_validation = validation_df[
    "Label_id"
].to_numpy(
    dtype=np.int64,
    copy=True
)

del validation_df

if not np.isfinite(X_validation).all():
    raise ValueError(
        "Validation data contains invalid features."
    )

validation_dataset = TensorDataset(
    torch.from_numpy(X_validation),
    torch.from_numpy(y_validation)
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Validation records:",
    f"{len(y_validation):,}"
)


# ============================================================
# LOAD ALL TEN CLIENT DATASETS
# ============================================================

print("\nLoading ten client datasets...")

clients = {}

for client_id in range(1, NUM_CLIENTS + 1):

    client_file = (
        DATA_ROOT
        / f"client_{client_id:02d}"
        / "client_data.parquet"
    )

    if not client_file.exists():
        raise FileNotFoundError(client_file)

    df = pd.read_parquet(client_file)

    required_columns = (
        FEATURE_COLUMNS + ["Label_id"]
    )

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Client {client_id} missing columns: {missing}"
        )

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True
    )

    y = df["Label_id"].to_numpy(
        dtype=np.int64,
        copy=True
    )

    del df

    if not np.isfinite(X).all():
        raise ValueError(
            f"Client {client_id} has invalid features."
        )

    if not np.isin(y, np.arange(7)).all():
        raise ValueError(
            f"Client {client_id} has unexpected labels."
        )

    clients[client_id] = {
        "X": torch.from_numpy(X),
        "y_clean": torch.from_numpy(y),
        "samples": len(y)
    }

    print(
        f"Client {client_id:02d}: "
        f"{len(y):,} records loaded"
    )

    del X, y


# ============================================================
# CREATE FIXED POISONED LABELS
# The same records remain poisoned throughout training.
# ============================================================

print("\nPreparing attack configuration...")

poisoned_labels = clients[
    MALICIOUS_CLIENT
]["y_clean"].clone()

original_labels = poisoned_labels.numpy().copy()

eligible_indices = np.flatnonzero(
    original_labels != TARGET_LABEL
)

poisoned_count = int(
    np.floor(
        len(eligible_indices) * POISONING_RATIO
    )
)

poison_rng = np.random.default_rng(
    RANDOM_SEED + MALICIOUS_CLIENT
)

selected_indices = poison_rng.choice(
    eligible_indices,
    size=poisoned_count,
    replace=False
)

poisoned_labels[selected_indices] = TARGET_LABEL

print(
    "Client 10 eligible records:",
    f"{len(eligible_indices):,}"
)

print(
    "Client 10 poisoned records:",
    f"{poisoned_count:,}"
)

del original_labels, eligible_indices, selected_indices


# ============================================================
# CLEAN VALIDATION EVALUATION
# ============================================================

criterion_sum = nn.CrossEntropyLoss(
    reduction="sum"
)


def evaluate_model(model):

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for batch_X, batch_y in validation_loader:

            outputs = model(batch_X)

            loss = criterion_sum(
                outputs,
                batch_y
            )

            loss_sum += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == batch_y
            ).sum().item()

            total += len(batch_y)

    return (
        loss_sum / total,
        correct / total
    )


# ============================================================
# LSM AND MANHATTAN DISTANCE
# ============================================================

def calculate_lsm(results, loss_column):

    sample_sizes = np.array(
        [r["samples"] for r in results],
        dtype=np.float64
    )

    losses = np.array(
        [r[loss_column] for r in results],
        dtype=np.float64
    )

    if not np.isfinite(losses).all():
        raise ValueError("Non-finite losses found.")

    if np.any(losses <= 0):
        raise ValueError("Losses must be positive.")

    # Equation 4
    inverse_losses = 1.0 / losses

    scores = (
        (sample_sizes / sample_sizes.sum())
        * (inverse_losses / inverse_losses.sum())
    )

    # Equation 5
    distances = np.abs(
        scores[:, None] - scores[None, :]
    ).sum(axis=1)

    kmeans = KMeans(
        n_clusters=2,
        random_state=RANDOM_SEED,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        distances.reshape(-1, 1)
    )

    centers = kmeans.cluster_centers_.flatten()

    suspicious_cluster = int(
        np.argmax(centers)
    )

    suspicious_clients = [
        results[i]["client_id"]
        for i in range(NUM_CLIENTS)
        if clusters[i] == suspicious_cluster
    ]

    return {
        "scores": scores,
        "distances": distances,
        "clusters": clusters,
        "centers": centers.tolist(),
        "suspicious_clients": suspicious_clients
    }


# ============================================================
# FEDAVG AGGREGATION
# ============================================================

def aggregate_models(local_states, sample_sizes):

    total_samples = sum(sample_sizes)

    weights = [
        n / total_samples
        for n in sample_sizes
    ]

    new_global_state = {}

    for key in local_states[0]:

        averaged_tensor = torch.zeros_like(
            local_states[0][key]
        )

        for i in range(NUM_CLIENTS):

            averaged_tensor += (
                local_states[i][key] * weights[i]
            )

        new_global_state[key] = averaged_tensor

    return new_global_state


# ============================================================
# RUN ONE FEDERATED SCENARIO
# ============================================================

def run_scenario(scenario):

    print("\n\n" + "=" * 75)
    print("STARTING SCENARIO:", scenario.upper())
    print("=" * 75)

    # Same initial model for both scenarios.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    initial_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    )

    global_state = {
        key: value.detach().clone()
        for key, value in initial_model.state_dict().items()
    }

    del initial_model

    scenario_rows = []
    round_rows = []

    for round_number in range(1, NUM_ROUNDS + 1):

        print("\n" + "-" * 75)
        print(
            f"{scenario.upper()} - ROUND "
            f"{round_number}/{NUM_ROUNDS}"
        )
        print("-" * 75)

        local_states = []
        sample_sizes = []
        round_results = []

        # ----------------------------------------------------
        # PARTICIPANT LOCAL TRAINING
        # ----------------------------------------------------

        for client_id in range(1, NUM_CLIENTS + 1):

            client = clients[client_id]

            X = client["X"]

            if (
                scenario == "poisoned"
                and client_id == MALICIOUS_CLIENT
            ):
                y = poisoned_labels
            else:
                y = client["y_clean"]

            dataset = TensorDataset(X, y)

            client_seed = (
                RANDOM_SEED
                + round_number * 1000
                + client_id
            )

            generator = torch.Generator()

            generator.manual_seed(client_seed)

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                generator=generator
            )

            # Same client/round random seed in both scenarios.
            torch.manual_seed(client_seed)

            local_model = IntrusionDetectionMLP(
                input_features=78,
                num_classes=7
            )

            local_model.load_state_dict(
                global_state
            )

            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=LEARNING_RATE
            )

            criterion = nn.CrossEntropyLoss()

            local_model.train()

            online_loss_sum = 0.0
            online_samples = 0

            for epoch in range(LOCAL_EPOCHS):

                for batch_X, batch_y in loader:

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    outputs = local_model(batch_X)

                    loss = criterion(
                        outputs,
                        batch_y
                    )

                    loss.backward()
                    optimizer.step()

                    online_loss_sum += (
                        loss.item() * len(batch_y)
                    )

                    online_samples += len(batch_y)

            online_loss = (
                online_loss_sum / online_samples
            )

            # ------------------------------------------------
            # EVALUATE LOCAL MODEL ON COMMON CLEAN VALIDATION
            # ------------------------------------------------

            validation_loss, validation_accuracy = (
                evaluate_model(local_model)
            )

            samples = client["samples"]

            round_results.append({
                "scenario": scenario,
                "round": round_number,
                "client_id": client_id,
                "samples": samples,
                "online_training_loss": online_loss,
                "clean_validation_loss": validation_loss,
                "clean_validation_accuracy": validation_accuracy,
                "poisoned_records": (
                    poisoned_count
                    if (
                        scenario == "poisoned"
                        and client_id == MALICIOUS_CLIENT
                    )
                    else 0
                )
            })

            local_states.append({
                key: value.detach().clone()
                for key, value in local_model.state_dict().items()
            })

            sample_sizes.append(samples)

            print(
                f"Client {client_id:02d}: "
                f"Online loss={online_loss:.6f}, "
                f"Clean validation loss={validation_loss:.6f}"
            )

            del (
                dataset,
                loader,
                local_model,
                optimizer
            )

        # ----------------------------------------------------
        # CALCULATE BOTH LSM VARIANTS
        # ----------------------------------------------------

        online_detection = calculate_lsm(
            round_results,
            "online_training_loss"
        )

        validation_detection = calculate_lsm(
            round_results,
            "clean_validation_loss"
        )

        for i, row in enumerate(round_results):

            row["online_lsm_score"] = float(
                online_detection["scores"][i]
            )

            row["online_manhattan_distance"] = float(
                online_detection["distances"][i]
            )

            row["online_cluster"] = int(
                online_detection["clusters"][i]
            )

            row["validation_lsm_score"] = float(
                validation_detection["scores"][i]
            )

            row["validation_manhattan_distance"] = float(
                validation_detection["distances"][i]
            )

            row["validation_cluster"] = int(
                validation_detection["clusters"][i]
            )

            row["suspicious_by_online_loss"] = (
                row["client_id"]
                in online_detection["suspicious_clients"]
            )

            row["suspicious_by_validation_loss"] = (
                row["client_id"]
                in validation_detection["suspicious_clients"]
            )

        scenario_rows.extend(round_results)

        # ----------------------------------------------------
        # FEDAVG: AGGREGATE ALL TEN CLIENTS
        # No defensive exclusion in this diagnostic.
        # ----------------------------------------------------

        global_state = aggregate_models(
            local_states,
            sample_sizes
        )

        # ----------------------------------------------------
        # EVALUATE UPDATED GLOBAL MODEL
        # ----------------------------------------------------

        global_model = IntrusionDetectionMLP(
            input_features=78,
            num_classes=7
        )

        global_model.load_state_dict(global_state)

        global_val_loss, global_val_accuracy = (
            evaluate_model(global_model)
        )

        del global_model, local_states

        # ----------------------------------------------------
        # ROUND SUMMARY
        # ----------------------------------------------------

        round_summary = {
            "scenario": scenario,
            "round": round_number,
            "online_loss_suspicious_clients": (
                online_detection["suspicious_clients"]
            ),
            "validation_loss_suspicious_clients": (
                validation_detection["suspicious_clients"]
            ),
            "online_cluster_centers": (
                online_detection["centers"]
            ),
            "validation_cluster_centers": (
                validation_detection["centers"]
            ),
            "global_clean_validation_loss": global_val_loss,
            "global_clean_validation_accuracy": (
                global_val_accuracy
            )
        }

        round_rows.append(round_summary)

        print("\nROUND DETECTION SUMMARY")

        print(
            "Online-loss suspicious clients:",
            online_detection["suspicious_clients"]
        )

        print(
            "Clean-validation-loss suspicious clients:",
            validation_detection["suspicious_clients"]
        )

        print(
            "Global validation accuracy:",
            f"{global_val_accuracy * 100:.4f}%"
        )

        print(
            "Global validation loss:",
            f"{global_val_loss:.8f}"
        )

        gc.collect()

    # --------------------------------------------------------
    # SAVE SCENARIO RESULTS
    # --------------------------------------------------------

    scenario_dir = OUTPUT_ROOT / scenario

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(scenario_rows).to_csv(
        scenario_dir / "client_round_metrics.csv",
        index=False
    )

    pd.DataFrame(round_rows).to_csv(
        scenario_dir / "round_summary.csv",
        index=False
    )

    metadata = {
        "scenario": scenario,
        "rounds": NUM_ROUNDS,
        "clients": NUM_CLIENTS,
        "fedavg": True,
        "defense_enabled": False,
        "malicious_client": (
            MALICIOUS_CLIENT
            if scenario == "poisoned"
            else None
        ),
        "poisoning_ratio_of_non_benign_records": (
            POISONING_RATIO
            if scenario == "poisoned"
            else 0.0
        ),
        "poisoned_records": (
            poisoned_count
            if scenario == "poisoned"
            else 0
        ),
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "validation_records": len(y_validation),
        "round_summaries": round_rows
    }

    with open(
        scenario_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\nSaved scenario:", scenario_dir)

    return round_rows


# ============================================================
# RUN BENIGN AND POISONED SCENARIOS
# ============================================================

all_summaries = {}

for scenario in SCENARIOS:

    all_summaries[scenario] = run_scenario(
        scenario
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 75)
print("FINAL TEN-ROUND COMPARISON")
print("=" * 75)

for scenario in SCENARIOS:

    print("\nSCENARIO:", scenario.upper())

    for result in all_summaries[scenario]:

        print(
            f"Round {result['round']:02d}: "
            f"Online LSM flags="
            f"{result['online_loss_suspicious_clients']} | "
            f"Validation LSM flags="
            f"{result['validation_loss_suspicious_clients']} | "
            f"Global validation accuracy="
            f"{result['global_clean_validation_accuracy'] * 100:.4f}%"
        )

print("\nSaved all results:", OUTPUT_ROOT)
print("\nMulti-round diagnostic completed.")
