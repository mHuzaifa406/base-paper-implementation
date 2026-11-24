
from pathlib import Path
from collections import Counter

import pandas as pd

INPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\cleaned_v1"
)

OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\mapped_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    raise ValueError(f"Unknown traffic label: {value!r}")


files = sorted(INPUT_DIR.glob("*.parquet"))

if len(files) != 8:
    raise ValueError(
        f"Expected 8 files, found {len(files)}"
    )

total_rows = 0
class_counts = Counter()

print("=" * 60)
print("CREATING SEVEN-CLASS DATASET")
print("=" * 60)

for file in files:

    print(f"\nProcessing: {file.name}")

    df = pd.read_parquet(file)

    df["Label_7class"] = df["Label"].map(map_label)

    if df["Label_7class"].isna().any():
        raise ValueError("Some labels were not mapped.")

    counts = df["Label_7class"].value_counts()

    class_counts.update(counts.to_dict())

    output_file = OUTPUT_DIR / file.name

    df.to_parquet(
        output_file,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    total_rows += len(df)

    print("Rows:", f"{len(df):,}")
    print("Saved:", output_file.name)

    del df

print("\n" + "=" * 60)
print("SEVEN-CLASS DATASET SUMMARY")
print("=" * 60)

for label, count in class_counts.most_common():
    print(f"{label}: {count:,}")

print("\nTotal records:", f"{total_rows:,}")
print("Number of classes:", len(class_counts))
print("Mapped files:", len(files))
print("Output directory:", OUTPUT_DIR)

print("\nSeven-class dataset created successfully.")
