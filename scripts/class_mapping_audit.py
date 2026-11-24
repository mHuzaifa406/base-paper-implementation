
from pathlib import Path
from collections import Counter

import pandas as pd

DATA_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\cleaned_v1"
)

files = sorted(DATA_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError("No cleaned Parquet files found.")

def map_label(value):
    label = str(value).strip()
    normalized = label.lower()

    if normalized == "benign":
        return "Benign"

    if normalized == "bot":
        return "Bot"

    if normalized in {
        "dos hulk",
        "ddos",
        "dos goldeneye",
        "dos slowloris",
        "dos slowhttptest",
        "heartbleed",
    }:
        return "DoS"

    if normalized == "infiltration":
        return "Infiltration"

    if normalized == "portscan":
        return "PortScan"

    if normalized.startswith("web attack"):
        return "Web Attack"

    if normalized in {"ftp-patator", "ssh-patator"}:
        return "Brute Force"

    return None


class_counts = Counter()
original_label_counts = Counter()
unknown_labels = Counter()

total_rows = 0

print("=" * 65)
print("SEVEN-CLASS LABEL MAPPING AUDIT")
print("=" * 65)

for file in files:

    print(f"\nProcessing: {file.name}")

    df = pd.read_parquet(file, columns=["Label"])

    labels = df["Label"].astype(str)
    mapped = labels.map(map_label)

    total_rows += len(df)

    original_label_counts.update(labels.value_counts().to_dict())

    class_counts.update(
        mapped.dropna().value_counts().to_dict()
    )

    unknown = labels[mapped.isna()].value_counts()

    unknown_labels.update(unknown.to_dict())

    print("Rows:", f"{len(df):,}")
    print("Mapped rows:", f"{mapped.notna().sum():,}")
    print("Unmapped rows:", f"{mapped.isna().sum():,}")

    del df

print("\n" + "=" * 65)
print("ORIGINAL LABEL DISTRIBUTION")
print("=" * 65)

for label, count in original_label_counts.most_common():
    print(f"{label}: {count:,}")

print("\n" + "=" * 65)
print("FINAL SEVEN-CLASS DISTRIBUTION")
print("=" * 65)

for label, count in class_counts.most_common():
    print(f"{label}: {count:,}")

expected = {
    "Benign": 2270678,
    "Bot": 1929,
    "DoS": 380523,
    "Infiltration": 36,
    "PortScan": 158780,
    "Web Attack": 2166,
    "Brute Force": 13764,
}

print("\n" + "=" * 65)
print("COMPARISON WITH BASE PAPER")
print("=" * 65)

for label, expected_count in expected.items():

    actual_count = class_counts.get(label, 0)

    status = "MATCH" if actual_count == expected_count else "DIFFERENT"

    print(
        f"{label}: Paper={expected_count:,} | "
        f"Your data={actual_count:,} | {status}"
    )

print("\nTotal mapped records:", f"{sum(class_counts.values()):,}")
print("Total original cleaned records:", f"{total_rows:,}")

if unknown_labels:
    print("\nUNMAPPED LABELS:")
    for label, count in unknown_labels.items():
        print(f"{label}: {count:,}")

else:
    print("\nAll original labels were mapped successfully.")

print("\nAudit completed. No files were modified.")
