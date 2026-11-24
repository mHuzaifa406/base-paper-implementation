
from pathlib import Path
import pandas as pd

POISONED_FILE = Path(
    r"E:\IoMT_FL_IDS\results\lsm_diagnostic_v1\lsm_scores.csv"
)

BENIGN_FILE = Path(
    r"E:\IoMT_FL_IDS\results\lsm_benign_control_v1\lsm_scores.csv"
)

poisoned = pd.read_csv(POISONED_FILE)
benign = pd.read_csv(BENIGN_FILE)

columns = [
    "client_id",
    "loss",
    "samples",
    "score",
    "manhattan_similarity",
    "cluster"
]

poisoned = poisoned[columns].copy()
benign = benign[columns].copy()

poisoned = poisoned.rename(
    columns={
        col: f"poisoned_{col}"
        for col in columns if col != "client_id"
    }
)

benign = benign.rename(
    columns={
        col: f"benign_{col}"
        for col in columns if col != "client_id"
    }
)

comparison = benign.merge(
    poisoned,
    on="client_id",
    validate="one_to_one"
)

comparison["loss_change"] = (
    comparison["poisoned_loss"]
    - comparison["benign_loss"]
)

comparison["score_change"] = (
    comparison["poisoned_score"]
    - comparison["benign_score"]
)

comparison["distance_change"] = (
    comparison["poisoned_manhattan_similarity"]
    - comparison["benign_manhattan_similarity"]
)

print("=" * 110)
print("BENIGN VS POISONED LSM COMPARISON")
print("=" * 110)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:.10f}")

print(comparison.to_string(index=False))

print("\n" + "=" * 110)
print("PARTICIPANT 4 VS PARTICIPANT 10")
print("=" * 110)

selected = comparison[
    comparison["client_id"].isin([4, 10])
]

print(selected.to_string(index=False))

print("\n" + "=" * 110)
print("COMPARISON COMPLETED")
print("=" * 110)
