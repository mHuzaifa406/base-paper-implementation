
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

CLIENT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\lsm_diagnostic_postloss_benign_v1"
)

NUM_CLIENTS = 10
MALICIOUS_CLIENT = None

POISONING_RATIO = 0.0
TARGET_LABEL = 0

BATCH_SIZE = 512
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(4)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output folder is not empty: {OUTPUT_DIR}"
    )

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

first_file = sorted(
    (CLIENT_ROOT / "client_01").glob("*.parquet")
)[0]

first_df = pd.read_parquet(first_file)

FEATURE_COLUMNS = [
    c for c in first_df.columns
    if c not in LABEL_COLUMNS
]

if len(FEATURE_COLUMNS) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(FEATURE_COLUMNS)}"
    )

del first_df


# ============================================================
# INITIALIZE COMMON GLOBAL MODEL
# ============================================================

global_model = IntrusionDetectionMLP(
    input_features=78,
    num_classes=7
)

global_state = {
    key: value.detach().clone()
    for key, value in global_model.state_dict().items()
}


# ============================================================
# CLIENT LOCAL TRAINING AND LOSS COLLECTION
# ============================================================

def collect_client_information(client_id):

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

    eligible_count = 0
    poisoned_count = 0

    # Apply the same attack to Client 10
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

        # Change labels, never features
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
            f"Invalid features for client {client_id}"
        )

    del df, original_labels

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
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

    local_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    )

    local_model.load_state_dict(global_state)

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    local_model.train()

    
    # ---------------------------------------------
    # A. Online training loss
    # ---------------------------------------------

    online_loss_sum = 0.0
    online_samples = 0

    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            outputs = local_model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            online_loss_sum += (
                loss.item() * len(batch_y)
            )

            online_samples += len(batch_y)

    online_average_loss = (
        online_loss_sum / online_samples
    )

    # ---------------------------------------------
    # B. Final local-model loss
    # Evaluate after all parameter updates
    # ---------------------------------------------

    local_model.eval()

    final_loss_sum = 0.0
    final_samples = 0

    with torch.inference_mode():

        for batch_X, batch_y in loader:

            outputs = local_model(batch_X)

            final_loss = criterion(
                outputs,
                batch_y
            )

            final_loss_sum += (
                final_loss.item() * len(batch_y)
            )

            final_samples += len(batch_y)

    average_loss = final_loss_sum / final_samples


    result = {
        "client_id": client_id,
        "samples": final_samples,
        "loss": average_loss,
	"online_training_loss": online_average_loss,
        "eligible_malicious_records": eligible_count,
        "poisoned_records": poisoned_count
    }

    del (
        X, y, dataset, loader,
        local_model, optimizer
    )

    return result


# ============================================================
# COLLECT ALL TEN CLIENTS
# ============================================================

print("=" * 70)
print("LSM BENIGN CONTROL DIAGNOSTIC")
print("=" * 70)

client_results = []

for client_id in range(1, NUM_CLIENTS + 1):

    print(
        f"\nTraining client {client_id}/{NUM_CLIENTS}"
    )

    result = collect_client_information(client_id)

    client_results.append(result)

    print(
        "Training samples:",
        f"{result['samples']:,}"
    )

    print(
        "Local training loss:",
        f"{result['loss']:.8f}"
    )

    print(
        "Poisoned records:",
        result["poisoned_records"]
    )


# ============================================================
# EQUATION 4: LOSS-BASED SCORING MECHANISM
# ============================================================

sample_sizes = np.array(
    [r["samples"] for r in client_results],
    dtype=np.float64
)

losses = np.array(
    [r["loss"] for r in client_results],
    dtype=np.float64
)

total_samples = sample_sizes.sum()

if np.any(losses <= 0) or not np.isfinite(losses).all():
    raise ValueError(
        "Local losses must be finite and positive."
    )

inverse_losses = 1.0 / losses

scores = (
    (sample_sizes / total_samples)
    * (inverse_losses / inverse_losses.sum())
)


# ============================================================
# EQUATION 5: MANHATTAN SIMILARITY
# ============================================================

distances = np.abs(
    scores[:, None] - scores[None, :]
).sum(axis=1)


# ============================================================
# K-MEANS OUTLIER DETECTION
# ============================================================

similarity_data = distances.reshape(-1, 1)

kmeans = KMeans(
    n_clusters=2,
    random_state=RANDOM_SEED,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    similarity_data
)

cluster_centers = kmeans.cluster_centers_.flatten()

# The cluster with the higher average distance
# is treated as the suspicious cluster.
anomalous_cluster = int(
    np.argmax(cluster_centers)
)

suspected_clients = [
    client_results[i]["client_id"]
    for i in range(NUM_CLIENTS)
    if cluster_labels[i] == anomalous_cluster
]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LSM SCORES AND MANHATTAN SIMILARITY")
print("=" * 70)

for i, result in enumerate(client_results):

    result["score"] = float(scores[i])
    result["manhattan_similarity"] = float(distances[i])
    result["cluster"] = int(cluster_labels[i])

    status = (
        "SUSPICIOUS"
        if result["client_id"] in suspected_clients
        else "NORMAL"
    )

    print(
        f"\nClient {result['client_id']:02d}"
    )

    print(
        "Loss:",
        f"{result['loss']:.10f}"
    )

    print(
        "Sample count:",
        f"{result['samples']:,}"
    )

    print(
        "LSM score:",
        f"{scores[i]:.12f}"
    )

    print(
        "Manhattan similarity:",
        f"{distances[i]:.12f}"
    )

    print("K-Means cluster:", cluster_labels[i])
    print("Detection:", status)


print("\n" + "=" * 70)
print("FINAL DETECTION")
print("=" * 70)

print("Malicious client:", MALICIOUS_CLIENT)
print("Suspicious clients:", suspected_clients)

print(
    "Malicious client detected:",
    MALICIOUS_CLIENT in suspected_clients
)

print("K-Means cluster centers:", cluster_centers.tolist())


# ============================================================
# SAVE DIAGNOSTIC RESULTS
# ============================================================

df_results = pd.DataFrame(client_results)

df_results.to_csv(
    OUTPUT_DIR / "lsm_scores.csv",
    index=False
)

metadata = {
    "method": "Paper Equations 4 and 5 with K-Means",
    "number_of_clients": NUM_CLIENTS,
    "malicious_client": MALICIOUS_CLIENT,
    "poisoning_ratio_of_non_benign_records": POISONING_RATIO,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "total_samples": int(total_samples),
    "cluster_centers": cluster_centers.tolist(),
    "anomalous_cluster": anomalous_cluster,
    "suspected_clients": suspected_clients,
    "malicious_client_detected": (
        MALICIOUS_CLIENT in suspected_clients
    ),
    "client_results": client_results
}

with open(
    OUTPUT_DIR / "lsm_diagnostic.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(metadata, f, indent=4)

print("\nSaved results:", OUTPUT_DIR)
print("LSM diagnostic completed.")
