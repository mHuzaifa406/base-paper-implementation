
from pathlib import Path
import ast
import pandas as pd


ROOT = Path(r"E:\IoMT_FL_IDS\results")

RUNS = [
    ("seed42_client10", "multiround_lsm_diagnostic_v1", 10),
    ("seed43_client04", "multiround_lsm_seed43_client04_v1", 4),
    ("seed44_client07", "multiround_lsm_seed44_client07_v1", 7),
    ("seed45_client01", "multiround_lsm_seed45_client01_v1", 1),
    ("seed46_client06", "multiround_lsm_seed46_client06_v1", 6),
    ("seed47_client08", "multiround_lsm_seed47_client08_v1", 8),
    ("seed48_client03", "multiround_lsm_seed48_client03_v1", 3),
]

ROUND_TO_CHECK = 10


def parse_list(value):
    if isinstance(value, list):
        return value
    return ast.literal_eval(str(value))


rows = []

print("=" * 85)
print("ROUND 10 DETECTION ANALYSIS")
print("=" * 85)

for name, folder, attacker in RUNS:

    print("\n" + "-" * 85)
    print(name)
    print("-" * 85)

    for scenario in ["benign", "poisoned"]:

        file_path = (
            ROOT / folder / scenario / "round_summary.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path)

        selected = df[
            df["round"] == ROUND_TO_CHECK
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Expected one round-10 row in {file_path}"
            )

        result = selected.iloc[0]

        suspicious = parse_list(
            result["online_loss_suspicious_clients"]
        )

        centers = parse_list(
            result["online_cluster_centers"]
        )

        ratio = max(centers) / min(centers)

        if scenario == "benign":

            detected = False
            other_flags = len(suspicious)

        else:

            detected = attacker in suspicious

            other_flags = sum(
                client != attacker
                for client in suspicious
            )

        row = {
            "experiment": name,
            "scenario": scenario,
            "attacker": attacker if scenario == "poisoned" else None,
            "round": ROUND_TO_CHECK,
            "suspicious_clients": str(suspicious),
            "cluster_center_ratio": ratio,
            "attacker_detected": detected,
            "number_of_other_flags": other_flags,
            "global_validation_accuracy": (
                result["global_clean_validation_accuracy"]
            )
        }

        rows.append(row)

        print("Scenario:", scenario)
        print("Suspicious clients:", suspicious)
        print("Cluster-center ratio:", f"{ratio:.6f}")

        if scenario == "poisoned":
            print("Actual attacker:", attacker)
            print("Attacker detected:", detected)

        else:
            print("Honest clients incorrectly flagged:", len(suspicious))


results = pd.DataFrame(rows)

print("\n" + "=" * 85)
print("ROUND 10 SUMMARY")
print("=" * 85)

print(results.to_string(index=False))

benign = results[
    results["scenario"] == "benign"
]

poisoned = results[
    results["scenario"] == "poisoned"
]

print("\nBENIGN RUNS")
print("Experiments with no false positives:",
      int((benign["number_of_other_flags"] == 0).sum()),
      "/", len(benign))

print("Total honest-client flags:",
      int(benign["number_of_other_flags"].sum()))

print("\nPOISONED RUNS")
print("Attackers detected:",
      int(poisoned["attacker_detected"].sum()),
      "/", len(poisoned))

print("Additional honest-client flags:",
      int(poisoned["number_of_other_flags"].sum()))

output = ROOT / "round10_detection_analysis_v1.csv"

results.to_csv(output, index=False)

print("\nSaved:", output)
print("\nAnalysis completed.")
