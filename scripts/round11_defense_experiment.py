
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

from models import IntrusionDetectionMLP


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v2"
)

VALIDATION_FILE = (
    DATA_ROOT / "server_validation" / "validation.parquet"
)

OUTPUT_ROOT = Path(
    r"E:\IoMT_FL_IDS\results\round11_defense_evaluation_v2"
)

NUM_CLIENTS = 10
NUM_ROUNDS = 20
WARMUP_ROUNDS = 10

# Independent experiment configuration
RANDOM_SEED = 49
MALICIOUS_CLIENT = 5

POISONING_RATIO = 0.20
TARGET_LABEL = 0

# Candidate threshold selected before this experiment
RATIO_THRESHOLD = 3.5

BATCH_SIZE = 512
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

torch.set_num_threads(4)

SCENARIOS = [
    {
        "name": "benign_defense",
        "poisoned": False,
        "defense": True
    },
    {
        "name": "poisoned_no_defense",
        "poisoned": True,
        "defense": False
    },
    {
        "name": "poisoned_defense",
        "poisoned": True,
        "defense": True
    }
]

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]


# ============================================================
# SAFETY CHECKS
# ============================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(DATA_ROOT)

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(VALIDATION_FILE)

if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_ROOT}"
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD SERVER VALIDATION DATA
# ============================================================

print("=" * 80)
print("ROUND-11 FEDERATED DEFENSE EXPERIMENT")
print("=" * 80)

print("\nLoading clean validation data...")

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
        "Validation features contain invalid values."
    )

X_validation_tensor = torch.from_numpy(
    X_validation
)

y_validation_tensor = torch.from_numpy(
    y_validation
)

validation_dataset = TensorDataset(
    X_validation_tensor,
    y_validation_tensor
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
# LOAD TEN CLIENTS
# ============================================================

print("\nLoading client datasets...")

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
            f"Invalid features in Client {client_id}"
        )

    if not np.isin(y, np.arange(7)).all():
        raise ValueError(
            f"Unexpected labels in Client {client_id}"
        )

    clients[client_id] = {
        "X": torch.from_numpy(X),
        "y_clean": torch.from_numpy(y),
        "samples": len(y)
    }

    print(
        f"Client {client_id:02d}: "
        f"{len(y):,} records"
    )

    del X, y


# ============================================================
# PREPARE FIXED LABEL-FLIPPING ATTACK
# ============================================================

attacker_labels = clients[
    MALICIOUS_CLIENT
]["y_clean"].clone()

attacker_original = (
    attacker_labels.numpy().copy()
)

eligible_indices = np.flatnonzero(
    attacker_original != TARGET_LABEL
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

attacker_labels[
    torch.from_numpy(selected_indices)
] = TARGET_LABEL

print("\nATTACK CONFIGURATION")
print("Attacker:", MALICIOUS_CLIENT)
print("Eligible records:", f"{len(eligible_indices):,}")
print("Poisoned records:", f"{poisoned_count:,}")
print(
    "Poisoning percentage of all client records:",
    f"{poisoned_count / len(attacker_original) * 100:.4f}%"
)

del attacker_original, eligible_indices, selected_indices


# ============================================================
# EVALUATE A GLOBAL MODEL
# ============================================================

def evaluate_global_model(model):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    with torch.inference_mode():

        for batch_X, batch_y in validation_loader:

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item()

            correct += (
                predictions == batch_y
            ).sum().item()

            total += len(batch_y)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                batch_y.cpu().numpy()
            )

    accuracy = correct / total

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )     # Attack success rate:
    # Percentage of originally malicious traffic
    # incorrectly predicted as Benign (class 0).

    targets_array = np.asarray(all_targets)
    predictions_array = np.asarray(all_predictions)

    malicious_mask = targets_array != 0

    malicious_count = int(
        malicious_mask.sum()
    )

    if malicious_count > 0:

        attack_success_rate = float(
            np.mean(
                predictions_array[malicious_mask] == 0
            )
        )

    else:

        attack_success_rate = 0.0


    return {
    "validation_loss": total_loss / total,
    "validation_accuracy": accuracy,
    "validation_macro_f1": macro_f1,
    "attack_success_rate": attack_success_rate,
    "malicious_validation_records": malicious_count
}


# ============================================================
# LSM + MANHATTAN + K-MEANS
# ============================================================

def calculate_detection(client_results):

    sample_sizes = np.array(
        [r["samples"] for r in client_results],
        dtype=np.float64
    )

    losses = np.array(
        [r["online_training_loss"] for r in client_results],
        dtype=np.float64
    )

    if not np.isfinite(losses).all():
        raise ValueError(
            "Non-finite client losses."
        )

    if np.any(losses <= 0):
        raise ValueError(
            "Client losses must be positive."
        )

    total_samples = sample_sizes.sum()

    # Equation 4: Loss-Based Scoring Mechanism
    inverse_losses = 1.0 / losses

    scores = (
        (sample_sizes / total_samples)
        * (inverse_losses / inverse_losses.sum())
    )

    # Equation 5: Manhattan similarity/distance
    distances = np.abs(
        scores[:, None] - scores[None, :]
    ).sum(axis=1)

    # K-Means clustering
    kmeans = KMeans(
        n_clusters=2,
        random_state=RANDOM_SEED,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        distances.reshape(-1, 1)
    )

    centers = kmeans.cluster_centers_.flatten()

    low_center = float(np.min(centers))
    high_center = float(np.max(centers))

    if low_center <= 1e-15:

        ratio = (
            float("inf")
            if high_center > 1e-15
            else 1.0
        )

    else:

        ratio = high_center / low_center

    suspicious_cluster = int(
        np.argmax(centers)
    )

    suspicious_clients = [
        client_results[i]["client_id"]
        for i in range(NUM_CLIENTS)
        if clusters[i] == suspicious_cluster
    ]

    # Attach diagnostic metrics.
    for i, result in enumerate(client_results):

        result["lsm_score"] = float(scores[i])

        result["manhattan_distance"] = float(
            distances[i]
        )

        result["cluster"] = int(clusters[i])

        result["kmeans_suspicious"] = (
            result["client_id"] in suspicious_clients
        )

    return {
        "centers": centers.tolist(),
        "ratio": float(ratio),
        "suspicious_clients": suspicious_clients,
        "scores": scores,
        "distances": distances
    }


# ============================================================
# FEDAVG AGGREGATION
# ============================================================

def aggregate_models(local_states, sample_sizes):

    if len(local_states) == 0:
        raise ValueError(
            "No participants available for aggregation."
        )

    if len(local_states) != len(sample_sizes):
        raise ValueError(
            "Model count and sample count do not match."
        )

    total_samples = sum(sample_sizes)

    if total_samples <= 0:
        raise ValueError(
            "No training samples available."
        )

    weights = [
        n / total_samples
        for n in sample_sizes
    ]

    aggregated_state = {}

    for key in local_states[0]:

        aggregate_tensor = torch.zeros_like(
            local_states[0][key]
        )

        for i in range(len(local_states)):

            aggregate_tensor += (
                local_states[i][key] * weights[i]
            )

        aggregated_state[key] = aggregate_tensor

    return aggregated_state


# ============================================================
# INITIALIZE THE SAME GLOBAL MODEL FOR EACH SCENARIO
# ============================================================

def initialize_global_model():

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    )

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def run_scenario(settings):

    scenario_name = settings["name"]
    poisoned = settings["poisoned"]
    defense_enabled = settings["defense"]

    print("\n\n" + "=" * 80)
    print("SCENARIO:", scenario_name.upper())
    print("=" * 80)

    global_state = initialize_global_model()

    client_round_rows = []
    round_rows = []

    for round_number in range(1, NUM_ROUNDS + 1):

        print("\n" + "-" * 80)
        print(
            f"{scenario_name} | "
            f"ROUND {round_number}/{NUM_ROUNDS}"
        )
        print("-" * 80)

        local_states = []
        sample_sizes = []
        client_results = []

        # ----------------------------------------------------
        # LOCAL TRAINING
        # ----------------------------------------------------

        for client_id in range(1, NUM_CLIENTS + 1):

            client = clients[client_id]

            X = client["X"]

            if (
                poisoned
                and client_id == MALICIOUS_CLIENT
            ):

                y = attacker_labels

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

            loss_sum = 0.0
            trained_samples = 0

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

                    loss_sum += (
                        loss.item() * len(batch_y)
                    )

                    trained_samples += len(batch_y)

            online_loss = (
                loss_sum / trained_samples
            )

            samples = client["samples"]

            client_results.append({
                "client_id": client_id,
                "samples": samples,
                "online_training_loss": online_loss,
                "poisoned_records": (
                    poisoned_count
                    if (
                        poisoned
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
                f"loss={online_loss:.6f}"
            )

            del (
                dataset,
                loader,
                local_model,
                optimizer
            )

        # ----------------------------------------------------
        # CENTRAL SERVER DETECTION
        # ----------------------------------------------------

        detection = calculate_detection(
            client_results
        )

        raw_suspicious = (
            detection["suspicious_clients"]
        )

        cluster_ratio = detection["ratio"]

        # Paper-style warm-up: first ten rounds have no
        # exclusion. Candidate defense activates from round 11.
        detection_active = (
            defense_enabled
            and round_number > WARMUP_ROUNDS
        )

        gate_triggered = (
            detection_active
            and cluster_ratio >= RATIO_THRESHOLD
        )

        if gate_triggered:

            excluded_clients = list(raw_suspicious)

        else:

            excluded_clients = []

        accepted_indices = [
            i
            for i, result in enumerate(client_results)
            if result["client_id"] not in excluded_clients
        ]

        if len(accepted_indices) < 2:
            raise RuntimeError(
                "Fewer than two participants remain. "
                "Aggregation cancelled for safety."
            )

        accepted_states = [
            local_states[i]
            for i in accepted_indices
        ]

        accepted_sizes = [
            sample_sizes[i]
            for i in accepted_indices
        ]

        # ----------------------------------------------------
        # FEDAVG WITH ACCEPTED PARTICIPANTS
        # ----------------------------------------------------

        global_state = aggregate_models(
            accepted_states,
            accepted_sizes
        )

        # ----------------------------------------------------
        # GLOBAL MODEL EVALUATION
        # ----------------------------------------------------

        global_model = IntrusionDetectionMLP(
            input_features=78,
            num_classes=7
        )

        global_model.load_state_dict(
            global_state
        )

        global_metrics = evaluate_global_model(
            global_model
        )

        del global_model

        # ----------------------------------------------------
        # CLIENT-LEVEL RECORDS
        # ----------------------------------------------------

        for result in client_results:

            client_id = result["client_id"]

            result["scenario"] = scenario_name
            result["round"] = round_number

            result["ratio_threshold"] = RATIO_THRESHOLD
            result["cluster_center_ratio"] = cluster_ratio

            result["detection_active"] = detection_active
            result["gate_triggered"] = gate_triggered

            result["excluded_from_aggregation"] = (
                client_id in excluded_clients
            )

            result["accepted_for_aggregation"] = (
                client_id not in excluded_clients
            )

            client_round_rows.append(result)

        # ----------------------------------------------------
        # ROUND SUMMARY
        # ----------------------------------------------------

        round_summary = {
		"attack_success_rate": (
    global_metrics["attack_success_rate"]
),
"malicious_validation_records": (
    global_metrics["malicious_validation_records"]
),
            "scenario": scenario_name,
            "round": round_number,
            "poisoned": poisoned,
            "defense_enabled": defense_enabled,
            "detection_active": detection_active,
            "kmeans_suspicious_clients": raw_suspicious,
            "cluster_centers": detection["centers"],
            "cluster_center_ratio": cluster_ratio,
            "ratio_threshold": RATIO_THRESHOLD,
            "gate_triggered": gate_triggered,
            "excluded_clients": excluded_clients,
            "accepted_clients": [
                client_results[i]["client_id"]
                for i in accepted_indices
            ],
            "global_validation_loss": (
                global_metrics["validation_loss"]
            ),
            "global_validation_accuracy": (
                global_metrics["validation_accuracy"]
            ),
            "global_validation_macro_f1": (
                global_metrics["validation_macro_f1"]
            )
        }

        round_rows.append(round_summary)

        # ----------------------------------------------------
        # PRINT ROUND RESULTS
        # ----------------------------------------------------

        print("\nCENTRAL SERVER DECISION")

        print(
            "K-Means suspicious clients:",
            raw_suspicious
        )

        print(
            "Cluster-center ratio:",
            f"{cluster_ratio:.6f}"
        )

        print(
            "Detection active:",
            detection_active
        )

        print(
            "Gate triggered:",
            gate_triggered
        )

        print(
            "Excluded clients:",
            excluded_clients
        )

        print(
            "Accepted clients:",
            round_summary["accepted_clients"]
        )

        print("\nGLOBAL VALIDATION RESULTS")

        print(
            "Accuracy:",
            f"{global_metrics['validation_accuracy'] * 100:.4f}%"
        )

        print(
            "Macro F1:",
            f"{global_metrics['validation_macro_f1']:.6f}"
        )
	print(
    	"Attack success rate:",
    	f"{global_metrics['attack_success_rate'] * 100:.4f}%"
	)

        print(
            "Validation loss:",
            f"{global_metrics['validation_loss']:.8f}"
        )

        gc.collect()

    # --------------------------------------------------------
    # SAVE SCENARIO RESULTS
    # --------------------------------------------------------

    scenario_dir = (
        OUTPUT_ROOT / scenario_name
    )

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(
        client_round_rows
    ).to_csv(
        scenario_dir / "client_round_metrics.csv",
        index=False
    )

    pd.DataFrame(
        round_rows
    ).to_csv(
        scenario_dir / "round_summary.csv",
        index=False
    )

    metadata = {
        "experiment": scenario_name,
        "random_seed": RANDOM_SEED,
        "number_of_clients": NUM_CLIENTS,
        "number_of_rounds": NUM_ROUNDS,
        "warmup_rounds": WARMUP_ROUNDS,
        "detection_starts_from_round": WARMUP_ROUNDS + 1,
        "malicious_client": (
            MALICIOUS_CLIENT if poisoned else None
        ),
        "poisoning_ratio_of_non_benign_records": (
            POISONING_RATIO if poisoned else 0.0
        ),
        "poisoned_records": (
            poisoned_count if poisoned else 0
        ),
        "ratio_threshold": RATIO_THRESHOLD,
        "threshold_is_an_experimental_refinement": True,
        "defense_enabled": defense_enabled,
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "validation_records": len(y_validation),
        "round_results": round_rows
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

    print("\nSaved scenario results:", scenario_dir)

    return round_rows


# ============================================================
# RUN THE THREE SCENARIOS
# ============================================================

all_results = {}

for settings in SCENARIOS:

    all_results[settings["name"]] = run_scenario(
        settings
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 80)
print("FINAL DEFENSE EXPERIMENT COMPARISON")
print("=" * 80)

for scenario_name, rows in all_results.items():

    final_round = rows[-1]

    print("\nScenario:", scenario_name)

    print(
        "Round 20 validation accuracy:",
        f"{final_round['global_validation_accuracy'] * 100:.4f}%"
    )

    print(
        "Round 20 macro F1:",
        f"{final_round['global_validation_macro_f1']:.6f}"
    )

    print(
        "Round 20 validation loss:",
        f"{final_round['global_validation_loss']:.8f}"
    )

    print(
        "Round 20 excluded clients:",
        final_round["excluded_clients"]
    )

# Save combined comparison
summary_rows = []

for scenario_name, rows in all_results.items():

    for row in rows:

        summary_rows.append(row)

summary_file = (
    OUTPUT_ROOT / "combined_round_summary.csv"
)

pd.DataFrame(summary_rows).to_csv(
    summary_file,
    index=False
)

print("\nCombined summary saved:", summary_file)
print("All experiment results:", OUTPUT_ROOT)
print("\nRound-11 defense experiment completed.")
