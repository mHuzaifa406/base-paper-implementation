
from pathlib import Path
import ast

import pandas as pd


ROOT = Path(r"E:\IoMT_FL_IDS\results")

RUNS = [
    {
        "name": "seed42_client10",
        "folder": "multiround_lsm_diagnostic_v1",
        "attacker": 10
    },
    {
        "name": "seed43_client04",
        "folder": "multiround_lsm_seed43_client04_v1",
        "attacker": 4
    },
    {
        "name": "seed44_client07",
        "folder": "multiround_lsm_seed44_client07_v1",
        "attacker": 7
    }
]

THRESHOLDS = [
    3.0, 3.25, 3.5, 3.6, 3.7, 3.8,
    3.9, 4.0, 4.25, 4.5, 4.75, 5.0
]


def parse_list(value):
    if isinstance(value, list):
        return value
    return ast.literal_eval(str(value))


observations = []

for run in RUNS:

    for scenario in ["benign", "poisoned"]:

        file_path = (
            ROOT / run["folder"]
            / scenario / "round_summary.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():

            centers = parse_list(
                row["online_cluster_centers"]
            )

            if len(centers) != 2:
                raise ValueError(
                    f"Unexpected centers: {centers}"
                )

            low = min(centers)
            high = max(centers)

            if low <= 0:
                raise ValueError(
                    f"Invalid cluster centers: {centers}"
                )

            ratio = high / low

            suspicious = parse_list(
                row["online_loss_suspicious_clients"]
            )

            attacker = (
                run["attacker"]
                if scenario == "poisoned"
                else None
            )

            observations.append({
                "run": run["name"],
                "scenario": scenario,
                "round": int(row["round"]),
                "attacker": attacker,
                "ratio": ratio,
                "suspicious_clients": suspicious,
                "attacker_in_suspicious": (
                    attacker in suspicious
                    if attacker is not None
                    else False
                )
            })


data = pd.DataFrame(observations)

benign = data[data["scenario"] == "benign"]
poisoned = data[data["scenario"] == "poisoned"]

print("=" * 80)
print("POOLED RATIO RANGES")
print("=" * 80)

print(
    "Maximum benign ratio:",
    f"{benign['ratio'].max():.6f}"
)

print(
    "Minimum poisoned ratio:",
    f"{poisoned['ratio'].min():.6f}"
)

print("\nPER-RUN RATIO RANGES")

for run_name in data["run"].unique():

    part = data[data["run"] == run_name]

    b = part[part["scenario"] == "benign"]
    p = part[part["scenario"] == "poisoned"]

    print(
        f"{run_name}: "
        f"benign {b['ratio'].min():.4f}–{b['ratio'].max():.4f}; "
        f"poisoned {p['ratio'].min():.4f}–{p['ratio'].max():.4f}"
    )


# ============================================================
# TEST THRESHOLDS
# ============================================================

results = []

for threshold in THRESHOLDS:

    # Benign run: any alert is a false positive.
    benign_alerts = (
        benign["ratio"] >= threshold
    )

    fp = int(benign_alerts.sum())
    tn = len(benign) - fp

    # Poisoned run: alert must also identify the
    # actual malicious participant.
    poison_alerts = (
        (poisoned["ratio"] >= threshold)
        & poisoned["attacker_in_suspicious"]
    )

    tp = int(poison_alerts.sum())
    fn = len(poisoned) - tp

    fpr = fp / len(benign)
    tpr = tp / len(poisoned)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / (len(benign) + len(poisoned))
    )

    results.append({
        "threshold": threshold,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "detection_rate": tpr,
        "false_positive_rate": fpr,
        "precision": precision,
        "accuracy": accuracy
    })


results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("THRESHOLD PERFORMANCE")
print("=" * 100)

print(
    results_df.round(4).to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

observations.to_pandas() if False else None

data.to_csv(
    ROOT / "ratio_threshold_observations_v1.csv",
    index=False
)

results_df.to_csv(
    ROOT / "ratio_threshold_performance_v1.csv",
    index=False
)

print("\nSaved:")
print(ROOT / "ratio_threshold_observations_v1.csv")
print(ROOT / "ratio_threshold_performance_v1.csv")

print("\nThreshold analysis completed.")
