"""
Training Utilities
==================
Seed setting, device detection, and early stopping.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(config_device: str = "auto") -> torch.device:
    """
    Auto-detect best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Args:
        config_device: 'auto', 'cpu', 'cuda', or 'mps'

    Returns:
        torch.device
    """
    if config_device != "auto":
        device = torch.device(config_device)
        logger.info(f"Using configured device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {gpu_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement
    is observed for `patience` consecutive epochs.

    Args:
        patience: Number of epochs with no improvement to wait
        min_delta: Minimum change to qualify as an improvement
        checkpoint_path: Path to save best model checkpoint
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        checkpoint_path: str = "./checkpoints/best_model.pth",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(
        self, val_loss: float, model: torch.nn.Module, epoch: int
    ) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current epoch's validation loss
            model: Model to checkpoint
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self._save_checkpoint(model, epoch, val_loss)
            return False
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: {self.counter}/{self.patience} "
                f"(best loss: {self.best_loss:.4f} at epoch {self.best_epoch})"
            )
            return self.counter >= self.patience

    def _save_checkpoint(
        self, model: torch.nn.Module, epoch: int, val_loss: float
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(
            f"Checkpoint saved: epoch {epoch}, val_loss: {val_loss:.4f}"
        )
