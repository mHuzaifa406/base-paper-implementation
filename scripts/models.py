
import torch
import torch.nn as nn


class IntrusionDetectionMLP(nn.Module):

    def __init__(
        self,
        input_features=78,
        num_classes=7
    ):
        super().__init__()

        self.network = nn.Sequential(

            # Input: 78 features
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Seven output classes
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    model = IntrusionDetectionMLP()

    print("Model architecture:")
    print(model)

    print(
        "\nTotal trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    print("Device: CPU")
