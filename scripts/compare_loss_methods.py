
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ROOT = Path(r"E:\IoMT_FL_IDS\results")

EXPERIMENTS = {
    "POISONED": ROOT / "lsm_diagnostic_postloss_v1" / "lsm_scores.csv",
    "BENIGN": ROOT / "lsm_diagnostic_postloss_benign_v1" / "lsm_scores.csv",
}


def evaluate_lsm(df, loss_column, experiment_name):

    sample_sizes = df["samples"].to_numpy(dtype=np.float64)
    losses = df[loss_column].to_numpy(dtype=np.float64)

    # Equation 4: Loss-Based Scoring Mechanism
    total_samples = sample_sizes.sum()
    inverse_losses = 1.0 / losses

    scores = (
        (sample_sizes / total_samples)
        * (inverse_losses / inverse_losses.sum())
    )

    # Equation 5: Manhattan similarity
    distances = np.abs(
        scores[:, None] - scores[None, :]
    ).sum(axis=1)

    # K-Means clustering
    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        distances.reshape(-1, 1)
    )

    centers = kmeans.cluster_centers_.flatten()

    suspicious_cluster = int(np.argmax(centers))

    suspicious_clients = [
        int(df.iloc[i]["client_id"])
        for i in range(len(df))
        if clusters[i] == suspicious_cluster
    ]

    result = pd.DataFrame({
        "Client": df["client_id"],
        "Loss": losses,
        "LSM_Score": scores,
        "Manhattan_Distance": distances,
        "Cluster": clusters,
        "Detection": [
            "SUSPICIOUS" if c == suspicious_cluster else "NORMAL"
            for c in clusters
        ],
        "Poisoned_Records": df["poisoned_records"]
    })

    print("\n" + "=" * 75)
    print(experiment_name, "-", loss_column)
    print("=" * 75)

    print(result.to_string(index=False))

    print("\nK-Means centers:", centers.tolist())
    print("Suspicious clients:", suspicious_clients)

    return suspicious_clients


for experiment_name, file_path in EXPERIMENTS.items():

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    data = pd.read_csv(file_path)

    print("\n\n############################################")
    print("EXPERIMENT:", experiment_name)
    print("############################################")

    online_suspicious = evaluate_lsm(
        data,
        "online_training_loss",
        experiment_name
    )

    final_suspicious = evaluate_lsm(
        data,
        "loss",
        experiment_name
    )

    print("\nLOSS METHOD COMPARISON")
    print("Online loss flagged:", online_suspicious)
    print("Post-training loss flagged:", final_suspicious)

print("\nAll comparisons completed.")
