
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import IntrusionDetectionMLP


DATA_DIR = Path(
    r"E:\IoMT_FL_IDS\data\processed\federated_v1\client_01"
)

torch.manual_seed(42)

# Select one participant file
files = sorted(DATA_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError(
        "No participant training files found."
    )

df = pd.read_parquet(files[0])

label_columns = [
    "Label",
    "Label_7class",
    "Label_id"
]

feature_columns = [
    col for col in df.columns
    if col not in label_columns
]

if len(feature_columns) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(feature_columns)}"
    )

# Use a small batch for the initial test
sample = df.head(512).copy()

X = torch.tensor(
    sample[feature_columns].to_numpy(
        dtype=np.float32
    )
)

y = torch.tensor(
    sample["Label_id"].to_numpy(
        dtype=np.int64
    )
)

del df, sample

# Initialize the model
model = IntrusionDetectionMLP(
    input_features=78,
    num_classes=7
)

criterion = nn.CrossEntropyLoss()

# Forward pass
outputs = model(X)

# Calculate training loss
loss = criterion(outputs, y)

# Backward pass
model.zero_grad()
loss.backward()

print("=" * 50)
print("MODEL TEST RESULTS")
print("=" * 50)

print("Input shape:", tuple(X.shape))
print("Output shape:", tuple(outputs.shape))
print("Number of classes:", outputs.shape[1])
print("Loss:", float(loss.item()))
print("CPU execution:", next(model.parameters()).device)
print("CUDA available:", torch.cuda.is_available())

print(
    "All outputs finite:",
    bool(torch.isfinite(outputs).all())
)

print(
    "Backward pass successful:",
    all(
        p.grad is not None
        for p in model.parameters()
    )
)

print("\nModel test completed successfully.")
