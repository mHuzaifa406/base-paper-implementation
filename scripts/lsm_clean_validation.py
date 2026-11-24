
from pathlib import Path
import json

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
    r"E:\IoMT_FL_IDS\results\lsm_clean_validation_v1"
)

NUM_CLIENTS = 10

MALICIOUS_CLIENT = 10
POISONING_RATIO = 0.20
TARGET_LABEL = 0

BATCH_SIZE = 512
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

RANDOM_SEED = 42

SCENARIOS = [
    "benign",
    "poisoned"
]

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]

torch.set_num_threads(4)


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
# LOAD SERVER VALIDATION DATA
# ============================================================

print("=" * 75)
print("LSM WITH COMMON CLEAN VALIDATION DATA")
print("=" * 75)

print("\nLoading server validation data...")

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

if not np.isin(y_validation, np.arange(7)).all():
    raise ValueError(
        "Unexpected validation class IDs."
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

print(
    "Validation features:",
    len(FEATURE_COLUMNS)
)


# ============================================================
# DETERMINE COMMON MODEL INITIALIZATION
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

global_model = IntrusionDetectionMLP(
    input_features=78,
    num_classes=7
)

global_state = {
    key: value.detach().clone()
    for key, value in global_model.state_dict().items()
}

del global_model


# ============================================================
# MODEL EVALUATION ON CLEAN SERVER VALIDATION DATA
# ============================================================

def evaluate_on_clean_validation(model):

    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for batch_X, batch_y in validation_loader:

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == batch_y
            ).sum().item()

            total += len(batch_y)

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# CLIENT TRAINING
# ============================================================

def train_client(client_id, scenario):

    client_dir = (
        DATA_ROOT / f"client_{client_id:02d}"
    )

    client_file = (
        client_dir / "client_data.parquet"
    )

    if not client_file.exists():
        raise FileNotFoundError(client_file)

    df = pd.read_parquet(client_file)

    required_columns = FEATURE_COLUMNS + ["Label_id"]

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

    original_labels = df[
        "Label_id"
    ].to_numpy(
        dtype=np.int64,
        copy=True
    )

    del df

    if not np.isfinite(X).all():
        raise ValueError(
            f"Client {client_id} contains invalid features."
        )

    if not np.isin(original_labels, np.arange(7)).all():
        raise ValueError(
            f"Client {client_id} has unexpected labels."
        )

    samples = len(original_labels)

    # Make a separate label copy for local training.
    training_labels = original_labels.copy()

    eligible_indices = np.flatnonzero(
        original_labels != TARGET_LABEL
    )

    eligible_count = len(eligible_indices)
    poisoned_count = 0

    # --------------------------------------------------------
    # LABEL-FLIPPING ATTACK
    # --------------------------------------------------------

    if (
        scenario == "poisoned"
        and client_id == MALICIOUS_CLIENT
    ):

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

        # Change labels only. Features remain untouched.
        training_labels[selected_indices] = (
            TARGET_LABEL
        )

    X_tensor = torch.from_numpy(X)

    y_tensor = torch.from_numpy(
        training_labels
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED + client_id
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    # --------------------------------------------------------
    # LOCAL MODEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COMMON CLEAN VALIDATION
    # --------------------------------------------------------

    validation_loss, validation_accuracy = (
        evaluate_on_clean_validation(local_model)
    )

    result = {
        "client_id": client_id,
        "samples": samples,
        "eligible_non_benign_records": eligible_count,
        "poisoned_records": poisoned_count,
        "online_training_loss": online_loss,
        "clean_validation_loss": validation_loss,
        "clean_validation_accuracy": validation_accuracy
    }

    del (
        X,
        X_tensor,
        y_tensor,
        original_labels,
        training_labels,
        dataset,
        loader,
        local_model,
        optimizer
    )

    return result


# ============================================================
# EQUATION 4: LOSS-BASED SCORING
# ============================================================

def calculate_lsm(results):

    sample_sizes = np.array(
        [r["samples"] for r in results],
        dtype=np.float64
    )

    losses = np.array(
        [r["clean_validation_loss"] for r in results],
        dtype=np.float64
    )

    if np.any(losses <= 0):
        raise ValueError(
            "Validation losses must be positive."
        )

    if not np.isfinite(losses).all():
        raise ValueError(
            "Validation losses must be finite."
        )

    total_samples = sample_sizes.sum()

    inverse_losses = 1.0 / losses

    scores = (
        (sample_sizes / total_samples)
        * (inverse_losses / inverse_losses.sum())
    )

    # --------------------------------------------------------
    # EQUATION 5: MANHATTAN SIMILARITY
    # --------------------------------------------------------

    distances = np.abs(
        scores[:, None] - scores[None, :]
    ).sum(axis=1)

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

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

    # Attach LSM results to each client record.
    for i, result in enumerate(results):

        result["lsm_score"] = float(scores[i])

        result["manhattan_similarity"] = float(
            distances[i]
        )

        result["cluster"] = int(clusters[i])

        result["detection"] = (
            "SUSPICIOUS"
            if result["client_id"] in suspicious_clients
            else "NORMAL"
        )

    return (
        results,
        centers,
        suspicious_clients,
        suspicious_cluster,
        total_samples
    )


# ============================================================
# RUN BOTH SCENARIOS
# ============================================================

summary = {}

for scenario in SCENARIOS:

    print("\n\n" + "=" * 75)
    print("SCENARIO:", scenario.upper())
    print("=" * 75)

    # Reset model and random seeds for fair comparison.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Restore exactly the same global model weights.
    client_results = []

    for client_id in range(1, NUM_CLIENTS + 1):

        print(
            f"\nTraining client {client_id}/{NUM_CLIENTS}"
        )

        result = train_client(
            client_id,
            scenario
        )

        client_results.append(result)

        print(
            "Training samples:",
            f"{result['samples']:,}"
        )

        print(
            "Online training loss:",
            f"{result['online_training_loss']:.8f}"
        )

        print(
            "Clean validation loss:",
            f"{result['clean_validation_loss']:.8f}"
        )

        print(
            "Poisoned records:",
            result["poisoned_records"]
        )

    (
        client_results,
        centers,
        suspicious_clients,
        suspicious_cluster,
        total_samples
    ) = calculate_lsm(client_results)

    # --------------------------------------------------------
    # PRINT CLIENT RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("LSM RESULTS:", scenario.upper())
    print("-" * 75)

    for result in client_results:

        print(
            f"\nClient {result['client_id']:02d}"
        )

        print(
            "Clean validation loss:",
            f"{result['clean_validation_loss']:.10f}"
        )

        print(
            "LSM score:",
            f"{result['lsm_score']:.12f}"
        )

        print(
            "Manhattan similarity:",
            f"{result['manhattan_similarity']:.12f}"
        )

        print(
            "Cluster:",
            result["cluster"]
        )

        print(
            "Detection:",
            result["detection"]
        )

    malicious_detected = (
        MALICIOUS_CLIENT in suspicious_clients
        if scenario == "poisoned"
        else None
    )

    print("\n" + "=" * 75)
    print("FINAL DETECTION:", scenario.upper())
    print("=" * 75)

    print(
        "Suspicious clients:",
        suspicious_clients
    )

    print(
        "K-Means centers:",
        centers.tolist()
    )

    if scenario == "poisoned":

        print(
            "Malicious client detected:",
            malicious_detected
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    scenario_dir = OUTPUT_ROOT / scenario

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        client_results
    )

    results_df.to_csv(
        scenario_dir / "lsm_scores.csv",
        index=False
    )

    metadata = {
        "scenario": scenario,
        "method": (
            "LSM and Manhattan similarity using "
            "common clean server validation loss"
        ),
        "implementation_variant": True,
        "number_of_clients": NUM_CLIENTS,
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
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "validation_records": len(y_validation),
        "total_training_samples": int(total_samples),
        "cluster_centers": centers.tolist(),
        "suspicious_clients": suspicious_clients,
        "malicious_client_detected": malicious_detected,
        "client_results": client_results
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

    summary[scenario] = {
        "suspicious_clients": suspicious_clients,
        "cluster_centers": centers.tolist(),
        "malicious_client_detected": malicious_detected
    }


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 75)
print("BENIGN VS POISONED COMPARISON")
print("=" * 75)

for scenario, result in summary.items():

    print("\nScenario:", scenario.upper())
    print(
        "Suspicious clients:",
        result["suspicious_clients"]
    )
    print(
        "Cluster centers:",
        result["cluster_centers"]
    )

    if scenario == "poisoned":
        print(
            "Attacker detected:",
            result["malicious_client_detected"]
        )

print("\nSaved results:", OUTPUT_ROOT)
print("Both scenarios completed.")
