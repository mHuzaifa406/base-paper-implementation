
from pathlib import Path
import ast

import pandas as pd


ROOT = Path(
    r"E:\IoMT_FL_IDS\results\multiround_lsm_diagnostic_v1"
)

SCENARIOS = ["benign", "poisoned"]


def read_centers(value):

    if isinstance(value, str):
        return ast.literal_eval(value)

    return value


all_data = {}

print("=" * 85)
print("K-MEANS CLUSTER SEPARATION ANALYSIS")
print("=" * 85)


for scenario in SCENARIOS:

    folder = ROOT / scenario

    metrics = pd.read_csv(
        folder / "client_round_metrics.csv"
    )

    summaries = pd.read_csv(
        folder / "round_summary.csv"
    )

    rows = []

    for round_number in range(1, 11):

        group = metrics[
            metrics["round"] == round_number
        ].copy()

        summary = summaries[
            summaries["round"] == round_number
        ].iloc[0]

        # Sort Manhattan distances in descending order.
        group = group.sort_values(
            "online_manhattan_distance",
            ascending=False
        )

        top = group.iloc[0]
        second = group.iloc[1]

        centers = read_centers(
            summary["online_cluster_centers"]
        )

        low_center = min(centers)
        high_center = max(centers)

        center_ratio = (
            high_center / low_center
            if low_center > 0
            else float("inf")
        )

        top_ratio = (
            top["online_manhattan_distance"]
            / second["online_manhattan_distance"]
        )

        flags = group[
            group["suspicious_by_online_loss"] == True
        ]["client_id"].tolist()

        if scenario == "poisoned":

            attacker = group[
                group["client_id"] == 10
            ].iloc[0]

            honest = group[
                group["client_id"] != 10
            ]

            max_honest_distance = (
                honest["online_manhattan_distance"].max()
            )

            attacker_distance = (
                attacker["online_manhattan_distance"]
            )

        else:

            attacker_distance = None
            max_honest_distance = None

        rows.append({
            "round": round_number,
            "top_client": int(top["client_id"]),
            "top_distance": top["online_manhattan_distance"],
            "second_client": int(second["client_id"]),
            "second_distance": second["online_manhattan_distance"],
            "top_to_second_ratio": top_ratio,
            "cluster_center_ratio": center_ratio,
            "number_flagged": len(flags),
            "flagged_clients": str(flags),
            "attacker_distance": attacker_distance,
            "maximum_honest_distance": max_honest_distance
        })

    result = pd.DataFrame(rows)

    all_data[scenario] = result

    print("\n" + "=" * 85)
    print("SCENARIO:", scenario.upper())
    print("=" * 85)

    print(
        result[
            [
                "round",
                "top_client",
                "top_to_second_ratio",
                "cluster_center_ratio",
                "number_flagged",
                "flagged_clients"
            ]
        ].round(4).to_string(index=False)
    )


# ============================================================
# CROSS-SCENARIO SEPARATION
# ============================================================

benign = all_data["benign"]
poisoned = all_data["poisoned"]

max_benign_distance = (
    benign["top_distance"].max()
)

min_attacker_distance = (
    poisoned["attacker_distance"].min()
)

max_poisoned_honest_distance = (
    poisoned["maximum_honest_distance"].max()
)

print("\n" + "=" * 85)
print("CROSS-SCENARIO DISTANCE ANALYSIS")
print("=" * 85)

print(
    "Maximum benign participant distance:",
    f"{max_benign_distance:.10f}"
)

print(
    "Maximum honest participant distance in poisoned scenario:",
    f"{max_poisoned_honest_distance:.10f}"
)

print(
    "Minimum Client 10 distance in poisoned scenario:",
    f"{min_attacker_distance:.10f}"
)

lower_bound = max(
    max_benign_distance,
    max_poisoned_honest_distance
)

upper_bound = min_attacker_distance

if lower_bound < upper_bound:

    print(
        "\nA potential absolute threshold interval exists:"
    )

    print(f"Lower bound: {lower_bound:.10f}")
    print(f"Upper bound: {upper_bound:.10f}")

else:

    print(
        "\nNo single absolute distance threshold "
        "separates all observed conditions."
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

output_file = ROOT / "cluster_separation_analysis.csv"

combined = pd.concat(
    [
        all_data["benign"].assign(scenario="benign"),
        all_data["poisoned"].assign(scenario="poisoned")
    ],
    ignore_index=True
)

combined.to_csv(
    output_file,
    index=False
)

print("\nAnalysis saved:", output_file)
print("\nAnalysis completed.")

