"""
Baseline CNN Model
==================
A lightweight 3-block convolutional neural network designed for
direct comparison against transfer learning approaches.

Architecture rationale:
  - 3 conv blocks progressively increase channel depth (32→64→128)
  - BatchNorm after each conv for training stability
  - MaxPool for spatial reduction
  - Dropout (0.3) to prevent overfitting on small dataset
  - ~250K parameters — trainable on CPU in reasonable time

This serves as the lower-bound baseline to demonstrate the value
of transfer learning on this task.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Lightweight CNN for 64×64 image classification.

    Args:
        num_classes: Number of output classes (default: 10)
        channels: Channel sizes for each conv block
        dropout: Dropout probability before final FC
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        # Block 1: 64×64 → 32×32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: 32×32 → 16×16
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: 16×16 → 8×8
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Adaptive pooling → fixed-size feature vector regardless of input
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[2], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for Conv layers, Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
