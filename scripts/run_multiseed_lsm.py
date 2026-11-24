
from pathlib import Path
import ast
import subprocess
import sys
from collections import Counter

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_SCRIPT = Path(
    r"E:\IoMT_FL_IDS\scripts\multiround_lsm_diagnostic.py"
)

RESULTS_ROOT = Path(
    r"E:\IoMT_FL_IDS\results"
)


# ============================================================
# NEW INDEPENDENT EXPERIMENTS
# ============================================================

CASES = [
    {"seed": 45, "attacker": 1},
    {"seed": 46, "attacker": 6},
    {"seed": 47, "attacker": 8},
    {"seed": 48, "attacker": 3}
]


# ============================================================
# HELPERS
# ============================================================

def parse_list(value):

    if isinstance(value, list):
        return value

    return ast.literal_eval(str(value))


def ratio_from_centers(value):

    centers = parse_list(value)

    if len(centers) != 2:
        raise ValueError(
            f"Expected two K-Means centers, got {centers}"
        )

    low = min(centers)
    high = max(centers)

    if low <= 0:
        raise ValueError(
            f"Invalid K-Means center: {centers}"
        )

    return high / low


def analyze_case(output_dir, attacker):

    benign_file = (
        output_dir
        / "benign"
        / "round_summary.csv"
    )

    poisoned_file = (
        output_dir
        / "poisoned"
        / "round_summary.csv"
    )

    if not benign_file.exists():
        raise FileNotFoundError(benign_file)

    if not poisoned_file.exists():
        raise FileNotFoundError(poisoned_file)

    benign = pd.read_csv(benign_file)
    poisoned = pd.read_csv(poisoned_file)

    benign_ratios = []
    poisoned_ratios = []

    benign_flag_frequency = Counter()

    attacker_detected_rounds = 0
    poisoned_other_flags = 0

    for _, row in benign.iterrows():

        ratio = ratio_from_centers(
            row["online_cluster_centers"]
        )

        benign_ratios.append(ratio)

        flagged = parse_list(
            row["online_loss_suspicious_clients"]
        )

        benign_flag_frequency.update(flagged)

    for _, row in poisoned.iterrows():

        ratio = ratio_from_centers(
            row["online_cluster_centers"]
        )

        poisoned_ratios.append(ratio)

        flagged = parse_list(
            row["online_loss_suspicious_clients"]
        )

        if attacker in flagged:
            attacker_detected_rounds += 1

        poisoned_other_flags += sum(
            1 for client in flagged
            if client != attacker
        )

    benign_min = min(benign_ratios)
    benign_max = max(benign_ratios)

    poisoned_min = min(poisoned_ratios)
    poisoned_max = max(poisoned_ratios)

    separation = benign_max < poisoned_min

    return {
        "benign_min_ratio": benign_min,
        "benign_max_ratio": benign_max,
        "poisoned_min_ratio": poisoned_min,
        "poisoned_max_ratio": poisoned_max,
        "attacker_detected_rounds": attacker_detected_rounds,
        "other_poisoned_flags": poisoned_other_flags,
        "benign_flag_frequency": dict(
            benign_flag_frequency
        ),
        "ratio_ranges_separate": separation
    }


# ============================================================
# CHECK ORIGINAL SCRIPT
# ============================================================

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(BASE_SCRIPT)

original_code = BASE_SCRIPT.read_text(
    encoding="utf-8-sig"
)

old_root = (
    r"E:\IoMT_FL_IDS\results\multiround_lsm_diagnostic_v1"
)

required_replacements = {
    "MALICIOUS_CLIENT = 10": 1,
    "RANDOM_SEED = 42": 1,
    old_root: 1
}

for old_text, expected_count in required_replacements.items():

    actual_count = original_code.count(old_text)

    if actual_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} occurrence(s) of "
            f"{old_text!r}, found {actual_count}. "
            "No experiment started."
        )


# ============================================================
# EXECUTE NEW EXPERIMENTS
# ============================================================

all_results = []

for case in CASES:

    seed = case["seed"]
    attacker = case["attacker"]

    run_name = (
        f"multiround_lsm_seed{seed}_client{attacker:02d}_v1"
    )

    output_dir = RESULTS_ROOT / run_name

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Results already exist: {output_dir}"
        )

    modified_code = original_code.replace(
        "MALICIOUS_CLIENT = 10",
        f"MALICIOUS_CLIENT = {attacker}",
        1
    )

    modified_code = modified_code.replace(
        "RANDOM_SEED = 42",
        f"RANDOM_SEED = {seed}",
        1
    )

    modified_code = modified_code.replace(
        old_root,
        str(output_dir),
        1
    )

    temp_script = BASE_SCRIPT.with_name(
        f"_temporary_seed{seed}_client{attacker}.py"
    )

    log_file = output_dir.with_name(
        output_dir.name + "_execution.log"
    )

    print("\n" + "=" * 75)
    print("STARTING NEW INDEPENDENT EXPERIMENT")
    print("=" * 75)

    print("Random seed:", seed)
    print("Malicious client:", attacker)
    print("Results directory:", output_dir)

    temp_script.write_text(
        modified_code,
        encoding="utf-8"
    )

    try:

        completed = subprocess.run(
            [
                sys.executable,
                str(temp_script)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        log_file.write_text(
            completed.stdout,
            encoding="utf-8"
        )

        if completed.returncode != 0:

            print("\nEXPERIMENT FAILED")

            print(
                "Last output lines:\n",
                "\n".join(
                    completed.stdout.splitlines()[-35:]
                )
            )

            print("\nFull log:", log_file)

            raise RuntimeError(
                f"Experiment failed for seed {seed}"
            )

    finally:

        if temp_script.exists():
            temp_script.unlink()

    result = analyze_case(
        output_dir,
        attacker
    )

    result["seed"] = seed
    result["attacker"] = attacker
    result["output_directory"] = str(output_dir)

    all_results.append(result)

    print("\nEXPERIMENT RESULTS")
    print("-" * 75)

    print(
        "Benign cluster ratio range:",
        f"{result['benign_min_ratio']:.4f}",
        "to",
        f"{result['benign_max_ratio']:.4f}"
    )

    print(
        "Poisoned cluster ratio range:",
        f"{result['poisoned_min_ratio']:.4f}",
        "to",
        f"{result['poisoned_max_ratio']:.4f}"
    )

    print(
        "Malicious client:",
        attacker
    )

    print(
        "Attacker detected:",
        f"{result['attacker_detected_rounds']}/10 rounds"
    )

    print(
        "Other clients flagged in poisoned rounds:",
        result["other_poisoned_flags"]
    )

    print(
        "Benign client flag frequency:",
        result["benign_flag_frequency"]
    )

    print(
        "Ratio ranges separated:",
        result["ratio_ranges_separate"]
    )

    print("Full output log:", log_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(all_results)

summary_file = (
    RESULTS_ROOT / "multiseed_lsm_holdout_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 75)
print("FINAL MULTI-SEED EXPERIMENT SUMMARY")
print("=" * 75)

print(
    summary_df[
        [
            "seed",
            "attacker",
            "benign_min_ratio",
            "benign_max_ratio",
            "poisoned_min_ratio",
            "poisoned_max_ratio",
            "attacker_detected_rounds",
            "other_poisoned_flags",
            "ratio_ranges_separate"
        ]
    ].to_string(index=False)
)

print("\nSummary saved:", summary_file)
print("\nMulti-seed experiments completed.")
