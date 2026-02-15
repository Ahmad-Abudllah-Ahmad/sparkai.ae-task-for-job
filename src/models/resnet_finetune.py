"""
ResNet18 Fine-Tuning Model
===========================
Transfer learning from ImageNet-pretrained ResNet18.

Justification for transfer learning:
  1. EuroSAT images (satellite RGB) share low/mid-level visual features
     (edges, textures, color distributions) with ImageNet natural images.
  2. ResNet18 is compute-efficient (~11M params) — suitable for laptop/CPU.
  3. Fine-tuning strategy: freeze backbone initially, then progressively
     unfreeze last N layers after warmup to prevent catastrophic forgetting.

This approach consistently outperforms training from scratch on datasets
of this size (~27K images) while being compute-friendly.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class ResNetFineTune(nn.Module):
    """
    ResNet18 with transfer learning for satellite image classification.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet-pretrained weights
        freeze_backbone: Whether to freeze backbone initially
        unfreeze_layers: Number of final layers to unfreeze after warmup
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 2,
    ):
        super().__init__()
        self.unfreeze_layers = unfreeze_layers

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace final FC layer for our task
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        logger.info(
            f"ResNet18 initialized | Pretrained: {pretrained} | "
            f"Frozen: {freeze_backbone} | "
            f"Trainable params: {self.count_parameters():,}"
        )

    def _freeze_backbone(self) -> None:
        """Freeze all backbone layers except the final FC."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

        logger.info("Backbone frozen — only FC layer is trainable.")

    def unfreeze_last_layers(self) -> None:
        """
        Progressively unfreeze the last N residual layers.
        Called after warmup epochs to allow fine-tuning deeper features.
        """
        # ResNet18 layers: conv1, bn1, layer1, layer2, layer3, layer4, fc
        layer_names = ["layer4", "layer3", "layer2", "layer1"]
        layers_to_unfreeze = layer_names[:self.unfreeze_layers]

        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        logger.info(
            f"Unfroze layers: {layers_to_unfreeze} | "
            f"Trainable params: {self.count_parameters():,}"
        )

    def reset_bn_stats(self) -> None:
        """
        Reset BatchNorm running statistics.

        When fine-tuning across domains (ImageNet → satellite), the pretrained
        BN running_mean/running_var reflect ImageNet statistics, which can
        cause severe prediction errors during eval mode. Resetting forces
        BN to recompute statistics from the new data distribution.
        """
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()
        logger.info("BatchNorm running statistics reset for domain adaptation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
