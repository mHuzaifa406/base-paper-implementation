
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from models import IntrusionDetectionMLP


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = 10

CLIENT_ROOT = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1"
)

OUTPUT_DIR = Path(
    r"E:\IoMT_FL_IDS\results\poisoning_loss_sweep_v1"
)

TARGET_LABEL = 0

BATCH_SIZE = 512
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

RANDOM_SEED = 42

# Fraction of eligible non-Benign records to flip.
# 0.20 means 20% of non-Benign records, not 20% of
# the complete client dataset.
POISONING_RATIOS = [
    0.0, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9
]

torch.set_num_threads(4)


# ============================================================
# CHECK OUTPUT DIRECTORY
# ============================================================

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output folder is not empty: {OUTPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CLIENT 10 DATA
# ============================================================

client_dir = CLIENT_ROOT / f"client_{CLIENT_ID:02d}"

files = sorted(client_dir.glob("*.parquet"))

if len(files) != 8:
    raise ValueError(
        f"Expected 8 Parquet files, found {len(files)}"
    )

print("=" * 70)
print("POISONING STRENGTH EXPERIMENT")
print("=" * 70)

print("\nLoading Client 10 data...")

frames = [
    pd.read_parquet(file)
    for file in files
]

df = pd.concat(
    frames,
    ignore_index=True
)

del frames

LABEL_COLUMNS = [
    "Label",
    "Label_7class",
    "Label_id"
]

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in LABEL_COLUMNS
]

if len(FEATURE_COLUMNS) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(FEATURE_COLUMNS)}"
    )

X = df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32,
    copy=True
)

original_labels = df["Label_id"].to_numpy(
    dtype=np.int64,
    copy=True
)

del df

if not np.isfinite(X).all():
    raise ValueError(
        "Dataset contains non-finite feature values."
    )

if not np.isin(original_labels, np.arange(7)).all():
    raise ValueError(
        "Unexpected class IDs found."
    )

total_records = len(original_labels)

eligible_indices = np.flatnonzero(
    original_labels != TARGET_LABEL
)

eligible_count = len(eligible_indices)

print("Total client records:", f"{total_records:,}")
print("Eligible non-Benign records:", f"{eligible_count:,}")


# ============================================================
# FIXED GLOBAL MODEL
# Every poisoning ratio starts from the same weights.
# ============================================================

torch.manual_seed(RANDOM_SEED)

global_model = IntrusionDetectionMLP(
    input_features=78,
    num_classes=7
)

global_state = {
    key: value.detach().clone()
    for key, value in global_model.state_dict().items()
}

del global_model

X_tensor = torch.from_numpy(X)

original_labels_tensor = torch.from_numpy(
    original_labels
)


# ============================================================
# FIXED NESTED ATTACK SELECTION
# Higher poisoning ratios include the selections made
# at lower ratios.
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED + CLIENT_ID
)

poisoning_order = rng.permutation(
    eligible_indices
)


# ============================================================
# RUN EXPERIMENTS
# ============================================================

criterion = nn.CrossEntropyLoss()

results = []

for ratio in POISONING_RATIOS:

    print("\n" + "-" * 70)
    print(
        f"Testing {ratio * 100:.0f}% "
        "of non-Benign records"
    )
    print("-" * 70)

    poisoned_labels = original_labels.copy()

    poisoned_count = int(
        np.floor(eligible_count * ratio)
    )

    selected_indices = poisoning_order[
        :poisoned_count
    ]

    poisoned_labels[selected_indices] = TARGET_LABEL

    poisoned_labels_tensor = torch.from_numpy(
        poisoned_labels
    )

    dataset = TensorDataset(
        X_tensor,
        poisoned_labels_tensor
    )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED + CLIENT_ID
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    # Reset RNG so dropout uses the same random sequence
    # for each poisoning ratio.
    torch.manual_seed(
        RANDOM_SEED + 1000
    )

    local_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7
    )

    local_model.load_state_dict(
        global_state
    )

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # A. ONLINE TRAINING LOSS
    # --------------------------------------------------------

    local_model.train()

    online_loss_sum = 0.0
    online_samples = 0

    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in loader:

            optimizer.zero_grad(set_to_none=True)

            outputs = local_model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()
            optimizer.step()

            online_loss_sum += (
                loss.item() * len(batch_y)
            )

            online_samples += len(batch_y)

    online_loss = (
        online_loss_sum / online_samples
    )

    # --------------------------------------------------------
    # B. POST-TRAINING LOSS
    # Evaluate against poisoned and original labels.
    # --------------------------------------------------------

    local_model.eval()

    poisoned_loss_sum = 0.0
    original_loss_sum = 0.0
    evaluated_samples = 0

    with torch.inference_mode():

        for start in range(0, total_records, BATCH_SIZE):

            end = min(
                start + BATCH_SIZE,
                total_records
            )

            batch_X = X_tensor[start:end]

            batch_poisoned_y = (
                poisoned_labels_tensor[start:end]
            )

            batch_original_y = (
                original_labels_tensor[start:end]
            )

            outputs = local_model(batch_X)

            poisoned_loss = criterion(
                outputs,
                batch_poisoned_y
            )

            original_loss = criterion(
                outputs,
                batch_original_y
            )

            batch_size = len(batch_X)

            poisoned_loss_sum += (
                poisoned_loss.item() * batch_size
            )

            original_loss_sum += (
                original_loss.item() * batch_size
            )

            evaluated_samples += batch_size

    post_poisoned_loss = (
        poisoned_loss_sum / evaluated_samples
    )

    post_original_loss = (
        original_loss_sum / evaluated_samples
    )

    actual_total_percentage = (
        poisoned_count / total_records
    ) * 100

    result = {
        "poisoning_ratio_non_benign_percent": ratio * 100,
        "total_client_records": total_records,
        "eligible_non_benign_records": eligible_count,
        "poisoned_records": poisoned_count,
        "actual_percentage_of_all_client_records": (
            actual_total_percentage
        ),
        "online_training_loss": online_loss,
        "post_training_loss_poisoned_labels": (
            post_poisoned_loss
        ),
        "post_training_loss_original_labels": (
            post_original_loss
        )
    }

    results.append(result)

    print("Poisoned records:", f"{poisoned_count:,}")
    print(
        "Percentage of all client records:",
        f"{actual_total_percentage:.3f}%"
    )
    print("Online training loss:", f"{online_loss:.8f}")
    print(
        "Post-training loss (poisoned labels):",
        f"{post_poisoned_loss:.8f}"
    )
    print(
        "Post-training loss (original labels):",
        f"{post_original_loss:.8f}"
    )

    del (
        dataset,
        loader,
        local_model,
        optimizer,
        poisoned_labels_tensor,
        poisoned_labels
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "poisoning_loss_sweep.csv"

results_df.to_csv(
    csv_path,
    index=False
)

print("\n" + "=" * 70)
print("FINAL POISONING LOSS COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nSaved results:", OUTPUT_DIR)
print("CSV file:", csv_path)
print("\nExperiment completed successfully.")
