
from pathlib import Path
import json
import pandas as pd

RESULT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\fedavg_full_v1"
)

# ------------------------------------------------
# 1. Inspect performance over ten rounds
# ------------------------------------------------

history_file = RESULT_DIR / "training_history.json"

with open(history_file, "r", encoding="utf-8") as f:
    history = json.load(f)

history_df = pd.DataFrame(history)

print("=" * 75)
print("FEDAVG PERFORMANCE ACROSS COMMUNICATION ROUNDS")
print("=" * 75)

columns = [
    "round",
    "accuracy_percent",
    "macro_precision_percent",
    "macro_recall_percent",
    "macro_f1_percent"
]

print(history_df[columns].to_string(index=False))

# ------------------------------------------------
# 2. Inspect per-class performance
# ------------------------------------------------

metrics_file = RESULT_DIR / "final_metrics.json"

with open(metrics_file, "r", encoding="utf-8") as f:
    final_metrics = json.load(f)

print("\n" + "=" * 75)
print("FINAL PER-CLASS PERFORMANCE")
print("=" * 75)

for class_name, values in final_metrics["per_class"].items():

    print(f"\n{class_name}")
    print(
        "  Precision:",
        f"{values['precision_percent']:.4f}%"
    )
    print(
        "  Recall:",
        f"{values['recall_percent']:.4f}%"
    )
    print(
        "  F1-score:",
        f"{values['f1_percent']:.4f}%"
    )
    print(
        "  Test support:",
        f"{values['support']:,}"
    )

# ------------------------------------------------
# 3. Inspect confusion matrix
# ------------------------------------------------

matrix_file = RESULT_DIR / "confusion_matrix_latest.csv"

confusion_matrix = pd.read_csv(
    matrix_file,
    index_col=0
)

print("\n" + "=" * 75)
print("FINAL CONFUSION MATRIX")
print("=" * 75)

print(confusion_matrix.to_string())

print("\n" + "=" * 75)
print("RESULTS INSPECTION COMPLETED")
print("=" * 75)
