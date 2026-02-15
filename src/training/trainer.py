"""
Model Trainer
=============
Production-quality training loop with:
  - Per-epoch train/val loss and accuracy logging
  - Learning rate scheduling
  - Early stopping with model checkpointing
  - Progressive unfreezing for transfer learning
  - Class-weighted loss for imbalanced datasets
  - Device-agnostic (CPU / CUDA / MPS)
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.utils import EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """
    Configurable model trainer with full training lifecycle management.

    Args:
        model: PyTorch model to train
        config: Full configuration dictionary
        device: Compute device
        class_weights: Optional tensor of class weights for loss function
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_cfg = config["training"]

        # Loss function with optional class weighting
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device)
            )
            logger.info("Using weighted CrossEntropyLoss for class imbalance")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Early stopping
        es_cfg = self.train_cfg.get("early_stopping", {})
        checkpoint_dir = self.train_cfg.get("checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_type = config["model"]["type"]
        checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_type}.pth")

        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 5),
            min_delta=es_cfg.get("min_delta", 0.001),
            checkpoint_path=checkpoint_path,
        ) if es_cfg.get("enabled", True) else None

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        opt_cfg = self.train_cfg["optimizer"]
        opt_type = opt_cfg.get("type", "adam").lower()

        # Only optimize parameters that require gradients
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_type == "adam":
            return Adam(
                params,
                lr=opt_cfg["learning_rate"],
                weight_decay=opt_cfg.get("weight_decay", 0),
            )
        elif opt_type == "sgd":
            return SGD(
                params,
                lr=opt_cfg["learning_rate"],
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=opt_cfg.get("weight_decay", 0),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _build_scheduler(self):
        """Build learning rate scheduler from config."""
        sch_cfg = self.train_cfg.get("scheduler", {})
        sch_type = sch_cfg.get("type", "step").lower()

        if sch_type == "step":
            return StepLR(
                self.optimizer,
                step_size=sch_cfg.get("step_size", 7),
                gamma=sch_cfg.get("gamma", 0.1),
            )
        elif sch_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_cfg["epochs"],
            )
        else:
            return None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history dictionary
        """
        epochs = self.train_cfg["epochs"]
        model_type = self.config["model"]["type"]
        warmup_epochs = self.config["model"].get("resnet", {}).get("warmup_epochs", 3)

        logger.info("=" * 60)
        logger.info(f"TRAINING: {model_type.upper()} model")
        logger.info(f"Epochs: {epochs} | Batch size: {self.train_cfg['batch_size']}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info("=" * 60)

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Progressive unfreezing for ResNet
            if (model_type == "resnet" and epoch == warmup_epochs + 1
                    and hasattr(self.model, "unfreeze_last_layers")):
                logger.info(f"\n--- Unfreezing backbone layers at epoch {epoch} ---")
                self.model.unfreeze_last_layers()
                # Rebuild optimizer with all trainable params
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()

            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader, epoch, epochs)
            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Log epoch summary
            logger.info(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"  * New best validation accuracy: {best_val_acc:.2f}%")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model, epoch):
                    logger.info(
                        f"\nEarly stopping triggered at epoch {epoch}. "
                        f"Best epoch: {self.early_stopping.best_epoch}"
                    )
                    break

        total_time = time.time() - start_time
        logger.info(f"\nTraining complete in {total_time:.1f}s")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

        # Save final training history
        self._save_history()

        return self.history

    def _train_epoch(
        self, loader: DataLoader, epoch: int, total_epochs: int
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            loader,
            desc=f"Train [{epoch}/{total_epochs}]",
            leave=False,
            ncols=100,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Note: For transfer learning with domain shift (ImageNet → satellite),
        we keep BatchNorm in train mode (use batch statistics) instead of
        eval mode (ImageNet running statistics). This prevents the known
        issue of BN running stats misalignment causing validation collapse.
        """
        self.model.eval()

        # Fix: Keep BatchNorm layers in train mode for domain-shifted data
        # This uses batch statistics instead of stale ImageNet running stats
        model_type = self.config["model"]["type"]
        if model_type == "resnet":
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def _save_history(self) -> None:
        """Save training history to checkpoint directory."""
        import json
        history_path = os.path.join(
            self.train_cfg.get("checkpoint_dir", "./checkpoints"),
            "training_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
