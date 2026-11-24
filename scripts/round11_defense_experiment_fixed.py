from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

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

# A new directory is used so previous experiment results remain safe.
OUTPUT_ROOT = Path(
    r"E:\IoMT_FL_IDS\results\round11_defense_evaluation_asr_v1"
)

NUM_CLIENTS = 10
NUM_ROUNDS = 20
WARMUP_ROUNDS = 10

# Exactly ONE malicious participant in poisoned scenarios.
RANDOM_SEED = 49
MALICIOUS_CLIENT = 5
POISONING_RATIO = 0.20
TARGET_LABEL = 0  # Benign

# Candidate abstention threshold selected in earlier experiments.
# This is an experimental refinement, not a parameter specified by the paper.
RATIO_THRESHOLD = 3.5

BATCH_SIZE = 512
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1
NUM_THREADS = 4

SCENARIOS = [
    {
        "name": "benign_defense",
        "poisoned": False,
        "defense": True,
    },
    {
        "name": "poisoned_no_defense",
        "poisoned": True,
        "defense": False,
    },
    {
        "name": "poisoned_defense",
        "poisoned": True,
        "defense": True,
    },
]

LABEL_COLUMNS = ["Label", "Label_7class", "Label_id"]


torch.set_num_threads(NUM_THREADS)


# ============================================================
# SAFETY CHECKS
# ============================================================

def check_paths_and_output_directory():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Federated data directory not found: {DATA_ROOT}")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_FILE}")

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_ROOT}\n"
            "Choose a new OUTPUT_ROOT before running; existing results will not be overwritten."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CLEAN SERVER VALIDATION DATA
# ============================================================

def load_validation_data():
    print("=" * 80)
    print("ROUND-11 FEDERATED DEFENSE EXPERIMENT WITH ASR")
    print("=" * 80)
    print("\nLoading clean server validation data...")

    validation_df = pd.read_parquet(VALIDATION_FILE)

    feature_columns = [
        column for column in validation_df.columns
        if column not in LABEL_COLUMNS
    ]

    if len(feature_columns) != 78:
        raise ValueError(
            f"Expected 78 feature columns, found {len(feature_columns)}"
        )

    if "Label_id" not in validation_df.columns:
        raise ValueError("Validation data must contain Label_id.")

    X_validation = validation_df[feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    y_validation = validation_df["Label_id"].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    del validation_df

    if not np.isfinite(X_validation).all():
        raise ValueError("Validation features contain NaN or infinite values.")

    if not np.isin(y_validation, np.arange(7)).all():
        raise ValueError("Validation labels contain unexpected class IDs.")

    if not np.any(y_validation != TARGET_LABEL):
        raise ValueError("Validation data contains no malicious records for ASR.")

    validation_dataset = TensorDataset(
        torch.from_numpy(X_validation),
        torch.from_numpy(y_validation),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Validation records: {len(y_validation):,}")
    print(f"Validation features: {len(feature_columns)}")
    print(
        "Malicious validation records:",
        f"{int(np.sum(y_validation != TARGET_LABEL)):,}",
    )

    return feature_columns, X_validation, y_validation, validation_loader


# ============================================================
# LOAD TEN CLIENT DATASETS
# ============================================================

def load_clients(feature_columns):
    print("\nLoading ten client datasets...")
    clients = {}
    required_columns = feature_columns + ["Label_id"]

    for client_id in range(1, NUM_CLIENTS + 1):
        client_file = (
            DATA_ROOT
            / f"client_{client_id:02d}"
            / "client_data.parquet"
        )

        if not client_file.exists():
            raise FileNotFoundError(client_file)

        df = pd.read_parquet(client_file)

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Client {client_id} is missing columns: {missing_columns}"
            )

        X = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        y = df["Label_id"].to_numpy(dtype=np.int64, copy=True)
        del df

        if not np.isfinite(X).all():
            raise ValueError(
                f"Client {client_id} contains NaN or infinite feature values."
            )

        if not np.isin(y, np.arange(7)).all():
            raise ValueError(
                f"Client {client_id} contains unexpected class IDs."
            )

        if len(y) == 0:
            raise ValueError(f"Client {client_id} has no records.")

        clients[client_id] = {
            "X": torch.from_numpy(X),
            "y_clean": torch.from_numpy(y),
            "samples": int(len(y)),
        }

        print(f"Client {client_id:02d}: {len(y):,} records")
        del X, y

    return clients


# ============================================================
# PREPARE ONE FIXED LABEL-FLIPPING ATTACK
# ============================================================

def prepare_attacker_labels(clients):
    if not 1 <= MALICIOUS_CLIENT <= NUM_CLIENTS:
        raise ValueError("MALICIOUS_CLIENT must be between 1 and NUM_CLIENTS.")

    attacker_labels = clients[MALICIOUS_CLIENT]["y_clean"].clone()
    original_labels = attacker_labels.cpu().numpy().copy()

    eligible_indices = np.flatnonzero(original_labels != TARGET_LABEL)
    poisoned_count = int(np.floor(len(eligible_indices) * POISONING_RATIO))

    rng = np.random.default_rng(RANDOM_SEED + MALICIOUS_CLIENT)
    selected_indices = rng.choice(
        eligible_indices,
        size=poisoned_count,
        replace=False,
    )

    selected_tensor = torch.from_numpy(selected_indices.astype(np.int64))
    attacker_labels[selected_tensor] = TARGET_LABEL

    actual_fraction = poisoned_count / len(original_labels)

    print("\nATTACK CONFIGURATION")
    print(f"Malicious client: {MALICIOUS_CLIENT}")
    print(f"Client records: {len(original_labels):,}")
    print(f"Eligible non-Benign records: {len(eligible_indices):,}")
    print(f"Poisoned records: {poisoned_count:,}")
    print(
        "Poisoned percentage of the complete client dataset: "
        f"{actual_fraction * 100:.4f}%"
    )

    del original_labels, eligible_indices, selected_indices, selected_tensor
    return attacker_labels, poisoned_count


# ============================================================
# GLOBAL MODEL EVALUATION
# ============================================================

def evaluate_global_model(model, validation_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.inference_mode():
        for batch_X, batch_y in validation_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            predictions = outputs.argmax(dim=1)

            batch_count = int(len(batch_y))
            total_loss += float(loss.item())
            correct += int((predictions == batch_y).sum().item())
            total += batch_count

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    if total == 0:
        raise ValueError("Validation loader returned no records.")

    targets_array = np.asarray(all_targets, dtype=np.int64)
    predictions_array = np.asarray(all_predictions, dtype=np.int64)

    accuracy = correct / total
    macro_f1 = f1_score(
        targets_array,
        predictions_array,
        average="macro",
        zero_division=0,
    )

    # Attack success rate (ASR): fraction of originally malicious
    # validation records (any class other than Benign) predicted as Benign.
    malicious_mask = targets_array != TARGET_LABEL
    malicious_count = int(malicious_mask.sum())

    if malicious_count > 0:
        attack_success_rate = float(
            np.mean(predictions_array[malicious_mask] == TARGET_LABEL)
        )
    else:
        attack_success_rate = 0.0

    return {
        "validation_loss": total_loss / total,
        "validation_accuracy": accuracy,
        "validation_macro_f1": float(macro_f1),
        "attack_success_rate": attack_success_rate,
        "malicious_validation_records": malicious_count,
    }


# ============================================================
# LSM, MANHATTAN DISTANCE, AND K-MEANS
# ============================================================

def calculate_detection(client_results):
    sample_sizes = np.asarray(
        [item["samples"] for item in client_results],
        dtype=np.float64,
    )
    losses = np.asarray(
        [item["online_training_loss"] for item in client_results],
        dtype=np.float64,
    )

    if not np.isfinite(losses).all() or np.any(losses <= 0):
        raise ValueError("Client training losses must be finite and positive.")

    total_samples = float(sample_sizes.sum())
    inverse_losses = 1.0 / losses

    # Equation 4: Loss-Based Scoring Mechanism.
    scores = (
        (sample_sizes / total_samples)
        * (inverse_losses / inverse_losses.sum())
    )

    # Equation 5: sum of absolute score differences to all other clients.
    distances = np.abs(scores[:, None] - scores[None, :]).sum(axis=1)

    kmeans = KMeans(
        n_clusters=2,
        random_state=RANDOM_SEED,
        n_init=10,
    )
    clusters = kmeans.fit_predict(distances.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()

    low_center = float(np.min(centers))
    high_center = float(np.max(centers))

    if low_center <= 1e-15:
        ratio = float("inf") if high_center > 1e-15 else 1.0
    else:
        ratio = high_center / low_center

    suspicious_cluster = int(np.argmax(centers))
    suspicious_clients = [
        client_results[index]["client_id"]
        for index in range(NUM_CLIENTS)
        if int(clusters[index]) == suspicious_cluster
    ]

    for index, item in enumerate(client_results):
        item["lsm_score"] = float(scores[index])
        item["manhattan_distance"] = float(distances[index])
        item["kmeans_cluster"] = int(clusters[index])
        item["kmeans_suspicious"] = (
            item["client_id"] in suspicious_clients
        )

    return {
        "cluster_centers": [float(value) for value in centers],
        "cluster_center_ratio": float(ratio),
        "kmeans_suspicious_clients": suspicious_clients,
        "scores": scores,
        "distances": distances,
    }


# ============================================================
# FEDAVG AGGREGATION
# ============================================================

def aggregate_models(local_states, sample_sizes):
    if not local_states:
        raise ValueError("No participant models were supplied to FedAvg.")

    if len(local_states) != len(sample_sizes):
        raise ValueError("Local model count does not match sample-size count.")

    total_samples = int(sum(sample_sizes))
    if total_samples <= 0:
        raise ValueError("Total aggregation sample count must be positive.")

    weights = [size / total_samples for size in sample_sizes]
    aggregated_state = {}

    for key in local_states[0]:
        aggregate_tensor = torch.zeros_like(local_states[0][key])
        for index, state in enumerate(local_states):
            aggregate_tensor += state[key] * weights[index]
        aggregated_state[key] = aggregate_tensor

    return aggregated_state


# ============================================================
# INITIALIZE GLOBAL MODEL
# ============================================================

def initialize_global_state():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7,
    )
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    del model
    return state


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def run_scenario(
    settings,
    clients,
    attacker_labels,
    poisoned_count,
    validation_loader,
    validation_record_count,
):
    scenario_name = settings["name"]
    poisoned = settings["poisoned"]
    defense_enabled = settings["defense"]

    print("\n\n" + "=" * 80)
    print(f"SCENARIO: {scenario_name.upper()}")
    print("=" * 80)

    global_state = initialize_global_state()
    client_round_rows = []
    round_rows = []

    for round_number in range(1, NUM_ROUNDS + 1):
        print("\n" + "-" * 80)
        print(f"{scenario_name} | ROUND {round_number}/{NUM_ROUNDS}")
        print("-" * 80)

        local_states = []
        sample_sizes = []
        client_results = []

        # ----------------------------------------------------
        # LOCAL TRAINING BY ALL TEN PARTICIPANTS
        # ----------------------------------------------------
        for client_id in range(1, NUM_CLIENTS + 1):
            client = clients[client_id]
            X = client["X"]

            if poisoned and client_id == MALICIOUS_CLIENT:
                y = attacker_labels
            else:
                y = client["y_clean"]

            dataset = TensorDataset(X, y)
            client_seed = RANDOM_SEED + round_number * 1000 + client_id
            generator = torch.Generator()
            generator.manual_seed(client_seed)

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                generator=generator,
            )

            # Reset the local RNG so corresponding client/round
            # training uses the same stochastic seed across scenarios.
            torch.manual_seed(client_seed)

            local_model = IntrusionDetectionMLP(
                input_features=78,
                num_classes=7,
            )
            local_model.load_state_dict(global_state)

            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=LEARNING_RATE,
            )
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            loss_sum = 0.0
            trained_samples = 0

            for _epoch in range(LOCAL_EPOCHS):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    outputs = local_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    batch_count = int(len(batch_y))
                    loss_sum += float(loss.item()) * batch_count
                    trained_samples += batch_count

            if trained_samples == 0:
                raise RuntimeError(f"Client {client_id} performed no local training.")

            online_loss = loss_sum / trained_samples
            sample_count = int(client["samples"])

            client_results.append({
                "scenario": scenario_name,
                "round": round_number,
                "client_id": client_id,
                "samples": sample_count,
                "online_training_loss": online_loss,
                "poisoned_records": (
                    poisoned_count
                    if poisoned and client_id == MALICIOUS_CLIENT
                    else 0
                ),
            })

            local_states.append({
                key: value.detach().clone()
                for key, value in local_model.state_dict().items()
            })
            sample_sizes.append(sample_count)

            print(
                f"Client {client_id:02d}: "
                f"loss={online_loss:.6f}"
            )

            del dataset, loader, local_model, optimizer

        # ----------------------------------------------------
        # CENTRAL-SERVER DETECTION
        # ----------------------------------------------------
        detection = calculate_detection(client_results)
        raw_suspicious = detection["kmeans_suspicious_clients"]
        ratio = detection["cluster_center_ratio"]

        # Paper-inspired warm-up: enable the protection stage after
        # the first ten training rounds. The ratio gate is our refinement.
        detection_active = defense_enabled and round_number > WARMUP_ROUNDS
        gate_triggered = (
            detection_active and ratio >= RATIO_THRESHOLD
        )

        excluded_clients = list(raw_suspicious) if gate_triggered else []

        accepted_indices = [
            index
            for index, item in enumerate(client_results)
            if item["client_id"] not in excluded_clients
        ]

        if len(accepted_indices) < 2:
            raise RuntimeError(
                "Fewer than two participants remain after filtering; "
                "FedAvg aggregation was stopped."
            )

        accepted_states = [local_states[index] for index in accepted_indices]
        accepted_sizes = [sample_sizes[index] for index in accepted_indices]

        # ----------------------------------------------------
        # FEDAVG OF ACCEPTED PARTICIPANTS
        # ----------------------------------------------------
        global_state = aggregate_models(accepted_states, accepted_sizes)

        # ----------------------------------------------------
        # EVALUATE UPDATED GLOBAL MODEL
        # ----------------------------------------------------
        global_model = IntrusionDetectionMLP(
            input_features=78,
            num_classes=7,
        )
        global_model.load_state_dict(global_state)

        global_metrics = evaluate_global_model(
            global_model,
            validation_loader,
        )
        del global_model

        # ----------------------------------------------------
        # SAVE CLIENT-LEVEL METRICS
        # ----------------------------------------------------
        for item in client_results:
            client_id = item["client_id"]
            item["ratio_threshold"] = RATIO_THRESHOLD
            item["cluster_center_ratio"] = ratio
            item["detection_active"] = detection_active
            item["gate_triggered"] = gate_triggered
            item["excluded_from_aggregation"] = (
                client_id in excluded_clients
            )
            item["accepted_for_aggregation"] = (
                client_id not in excluded_clients
            )
            item["kmeans_suspicious_clients"] = str(raw_suspicious)
            item["excluded_clients"] = str(excluded_clients)
            client_round_rows.append(item)

        # ----------------------------------------------------
        # ROUND SUMMARY
        # ----------------------------------------------------
        round_summary = {
            "scenario": scenario_name,
            "round": round_number,
            "poisoned_scenario": poisoned,
            "defense_enabled": defense_enabled,
            "detection_active": detection_active,
            "kmeans_suspicious_clients": raw_suspicious,
            "cluster_centers": detection["cluster_centers"],
            "cluster_center_ratio": ratio,
            "ratio_threshold": RATIO_THRESHOLD,
            "gate_triggered": gate_triggered,
            "excluded_clients": excluded_clients,
            "accepted_clients": [
                client_results[index]["client_id"]
                for index in accepted_indices
            ],
            "global_validation_loss": global_metrics["validation_loss"],
            "global_validation_accuracy": global_metrics["validation_accuracy"],
            "global_validation_macro_f1": global_metrics["validation_macro_f1"],
            "attack_success_rate": global_metrics["attack_success_rate"],
            "malicious_validation_records": global_metrics[
                "malicious_validation_records"
            ],
            "validation_records": validation_record_count,
        }
        round_rows.append(round_summary)

        # ----------------------------------------------------
        # PRINT ROUND RESULTS
        # ----------------------------------------------------
        print("\nCENTRAL SERVER DECISION")
        print("K-Means suspicious clients:", raw_suspicious)
        print("Cluster-center ratio:", f"{ratio:.6f}")
        print("Detection active:", detection_active)
        print("Gate triggered:", gate_triggered)
        print("Excluded clients:", excluded_clients)
        print("Accepted clients:", round_summary["accepted_clients"])

        print("\nGLOBAL VALIDATION RESULTS")
        print(
            "Accuracy:",
            f"{global_metrics['validation_accuracy'] * 100:.4f}%",
        )
        print(
            "Macro F1:",
            f"{global_metrics['validation_macro_f1']:.6f}",
        )
        print(
            "Attack success rate:",
            f"{global_metrics['attack_success_rate'] * 100:.4f}%",
        )
        print(
            "Validation loss:",
            f"{global_metrics['validation_loss']:.8f}",
        )

        gc.collect()

    # --------------------------------------------------------
    # SAVE SCENARIO RESULTS
    # --------------------------------------------------------
    scenario_dir = OUTPUT_ROOT / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    client_metrics_file = scenario_dir / "client_round_metrics.csv"
    round_summary_file = scenario_dir / "round_summary.csv"

    pd.DataFrame(client_round_rows).to_csv(
        client_metrics_file,
        index=False,
    )
    pd.DataFrame(round_rows).to_csv(
        round_summary_file,
        index=False,
    )

    metadata = {
        "experiment": scenario_name,
        "random_seed": RANDOM_SEED,
        "number_of_clients": NUM_CLIENTS,
        "number_of_rounds": NUM_ROUNDS,
        "warmup_rounds": WARMUP_ROUNDS,
        "detection_starts_from_round": WARMUP_ROUNDS + 1,
        "malicious_client": MALICIOUS_CLIENT if poisoned else None,
        "poisoning_ratio_of_non_benign_records": (
            POISONING_RATIO if poisoned else 0.0
        ),
        "poisoned_records": poisoned_count if poisoned else 0,
        "ratio_threshold": RATIO_THRESHOLD,
        "threshold_is_an_experimental_refinement": True,
        "defense_enabled": defense_enabled,
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "validation_records": validation_record_count,
        "asr_definition": (
            "Fraction of clean validation records with true class != 0 "
            "that are predicted as class 0 (Benign)."
        ),
        "round_results": round_rows,
    }

    with open(scenario_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nSaved scenario results:", scenario_dir)
    return round_rows


# ============================================================
# MAIN
# ============================================================

def main():
    check_paths_and_output_directory()

    feature_columns, X_validation, y_validation, validation_loader = (
        load_validation_data()
    )

    clients = load_clients(feature_columns)
    attacker_labels, poisoned_count = prepare_attacker_labels(clients)

    all_results = {}

    for settings in SCENARIOS:
        all_results[settings["name"]] = run_scenario(
            settings=settings,
            clients=clients,
            attacker_labels=attacker_labels,
            poisoned_count=poisoned_count,
            validation_loader=validation_loader,
            validation_record_count=len(y_validation),
        )

    # --------------------------------------------------------
    # FINAL COMPARISON: ROUND 20
    # --------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("FINAL DEFENSE EXPERIMENT COMPARISON")
    print("=" * 80)

    final_comparison = []

    for scenario_name, rows in all_results.items():
        final_round = rows[-1]

        print("\nScenario:", scenario_name)
        print(
            "Round 20 validation accuracy:",
            f"{final_round['global_validation_accuracy'] * 100:.4f}%",
        )
        print(
            "Round 20 macro F1:",
            f"{final_round['global_validation_macro_f1']:.6f}",
        )
        print(
            "Round 20 attack success rate:",
            f"{final_round['attack_success_rate'] * 100:.4f}%",
        )
        print(
            "Round 20 validation loss:",
            f"{final_round['global_validation_loss']:.8f}",
        )
        print(
            "Round 20 excluded clients:",
            final_round["excluded_clients"],
        )

        final_comparison.append({
            "scenario": scenario_name,
            "round": final_round["round"],
            "global_validation_accuracy": final_round[
                "global_validation_accuracy"
            ],
            "global_validation_macro_f1": final_round[
                "global_validation_macro_f1"
            ],
            "attack_success_rate": final_round["attack_success_rate"],
            "global_validation_loss": final_round[
                "global_validation_loss"
            ],
            "excluded_clients": str(final_round["excluded_clients"]),
        })

    # Save all per-round summaries and final comparison.
    combined_rows = []
    for scenario_rows in all_results.values():
        combined_rows.extend(scenario_rows)

    combined_summary_file = OUTPUT_ROOT / "combined_round_summary.csv"
    final_comparison_file = OUTPUT_ROOT / "final_comparison.csv"

    pd.DataFrame(combined_rows).to_csv(
        combined_summary_file,
        index=False,
    )
    pd.DataFrame(final_comparison).to_csv(
        final_comparison_file,
        index=False,
    )

    run_metadata = {
        "random_seed": RANDOM_SEED,
        "clients": NUM_CLIENTS,
        "rounds": NUM_ROUNDS,
        "warmup_rounds": WARMUP_ROUNDS,
        "malicious_client_in_poisoned_scenarios": MALICIOUS_CLIENT,
        "poisoning_ratio_of_non_benign_records": POISONING_RATIO,
        "ratio_threshold": RATIO_THRESHOLD,
        "validation_records": len(y_validation),
        "validation_file": str(VALIDATION_FILE),
        "output_directory": str(OUTPUT_ROOT),
        "scenarios": SCENARIOS,
        "code_note": (
            "LSM/Manhattan/K-Means pipeline with an experimental "
            "cluster-centre-ratio abstention gate."
        ),
    }

    with open(OUTPUT_ROOT / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=4)

    print("\nCombined summary saved:", combined_summary_file)
    print("Final comparison saved:", final_comparison_file)
    print("All experiment results:", OUTPUT_ROOT)
    print("\nRound-11 defense experiment with ASR completed.")

    # Release large objects before interpreter exit.
    del clients, attacker_labels, validation_loader
    del X_validation, y_validation
    gc.collect()


if __name__ == "__main__":
    main()
