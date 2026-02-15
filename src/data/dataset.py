"""
EuroSAT Dataset Module
======================
Handles dataset downloading, stratified splitting, class distribution
analysis, and DataLoader creation with optional weighted sampling.
"""

import logging
import os
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
from sklearn.model_selection import train_test_split

from src.data.preprocessing import get_preprocessing_transforms
from src.data.augmentation import get_augmentation_transforms

logger = logging.getLogger(__name__)

# EuroSAT class names (alphabetical order as in torchvision)
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


class EuroSATDataModule:
    """
    End-to-end data module for EuroSAT satellite image classification.

    Handles:
        - Dataset download and caching
        - Stratified train/val/test splitting
        - Class distribution analysis and logging
        - Weighted sampling for class imbalance
        - Appropriate transforms per split
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.train_cfg = config["training"]

        self.data_dir = self.data_cfg["data_dir"]
        self.seed = config["project"]["seed"]
        self.batch_size = self.train_cfg["batch_size"]
        self.num_workers = self.data_cfg.get("num_workers", 4)
        self.pin_memory = self.data_cfg.get("pin_memory", True)

        # Determine image size based on model type
        model_type = config["model"]["type"]
        if model_type == "resnet":
            self.image_size = self.data_cfg["image_size_resnet"]
        else:
            self.image_size = self.data_cfg["image_size"]

        self.use_class_weights = self.train_cfg.get("use_class_weights", True)

        # Will be populated by setup()
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.class_distribution: Optional[Dict[str, int]] = None
        self._train_labels: Optional[list] = None  # cached for sampler

    def setup(self) -> None:
        """Download dataset and create stratified splits."""
        logger.info("=" * 60)
        logger.info("DATASET SETUP: EuroSAT")
        logger.info("=" * 60)

        # Download full dataset (no transform needed for label extraction)
        full_dataset = datasets.EuroSAT(
            root=self.data_dir,
            download=True,
            transform=None
        )

        # Extract labels efficiently — avoid loading images
        # Try using internal attributes first (much faster)
        if hasattr(full_dataset, 'targets'):
            all_labels = list(full_dataset.targets)
        elif hasattr(full_dataset, '_labels'):
            all_labels = list(full_dataset._labels)
        else:
            # Fallback: extract from file paths (EuroSAT folder structure)
            all_labels = self._extract_labels_from_paths(full_dataset)

        all_indices = list(range(len(full_dataset)))

        # Log dataset metadata
        self._log_dataset_info(full_dataset, all_labels)

        # Stratified train/val/test split
        train_ratio = self.data_cfg["split"]["train"]
        val_ratio = self.data_cfg["split"]["val"]
        test_ratio = self.data_cfg["split"]["test"]

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

        # First split: train vs (val + test)
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            all_indices, all_labels,
            test_size=(val_ratio + test_ratio),
            stratify=all_labels,
            random_state=self.seed
        )

        # Second split: val vs test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_labels,
            test_size=relative_test_ratio,
            stratify=temp_labels,
            random_state=self.seed
        )

        logger.info(f"Split sizes — Train: {len(train_idx)} | "
                     f"Val: {len(val_idx)} | Test: {len(test_idx)}")

        # Build transforms
        train_transform = get_augmentation_transforms(self.data_cfg, self.image_size)
        eval_transform = get_preprocessing_transforms(self.data_cfg, self.image_size)

        # Create transformed datasets
        train_full = datasets.EuroSAT(
            root=self.data_dir, download=False, transform=train_transform
        )
        eval_full = datasets.EuroSAT(
            root=self.data_dir, download=False, transform=eval_transform
        )

        self.train_dataset = Subset(train_full, train_idx)
        self.val_dataset = Subset(eval_full, val_idx)
        self.test_dataset = Subset(eval_full, test_idx)

        # Cache train labels for fast sampler creation
        self._train_labels = train_labels

        # Compute class weights for imbalance handling
        self._compute_class_weights(train_labels)

        logger.info("Dataset setup complete.\n")

    def _extract_labels_from_paths(self, dataset) -> list:
        """
        Extract labels from dataset without loading images.
        Uses the file path structure: .../ClassName/image.jpg
        """
        logger.info("Extracting labels from file paths (no image loading)...")
        labels = []
        # Build class-to-index mapping
        if hasattr(dataset, 'class_to_idx'):
            class_to_idx = dataset.class_to_idx
        else:
            class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        if hasattr(dataset, 'samples'):
            for path, label in dataset.samples:
                labels.append(label)
        elif hasattr(dataset, 'imgs'):
            for path, label in dataset.imgs:
                labels.append(label)
        else:
            # Last resort: load images
            logger.warning("No efficient label extraction available, loading images...")
            for i in range(len(dataset)):
                _, label = dataset[i]
                labels.append(label)

        return labels

    def _log_dataset_info(self, dataset, labels: list) -> None:
        """Log comprehensive dataset statistics."""
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Number of classes: {len(CLASS_NAMES)}")
        logger.info(f"Image format: RGB, native resolution 64×64")
        logger.info(f"Label format: Integer index (0-{len(CLASS_NAMES) - 1})")
        logger.info(f"Target image size: {self.image_size}×{self.image_size}")

        # Class distribution
        counter = Counter(labels)
        self.class_distribution = {}
        logger.info("\nClass Distribution:")
        logger.info("-" * 45)
        for idx in sorted(counter.keys()):
            name = CLASS_NAMES[idx]
            count = counter[idx]
            pct = 100.0 * count / len(labels)
            self.class_distribution[name] = count
            bar = "█" * int(pct / 2)
            logger.info(f"  {name:>25s}: {count:5d} ({pct:5.1f}%) {bar}")
        logger.info("-" * 45)

        # Imbalance ratio
        max_count = max(counter.values())
        min_count = min(counter.values())
        imbalance_ratio = max_count / min_count
        logger.info(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

        if imbalance_ratio > 2.0:
            logger.warning("Significant class imbalance detected — "
                           "weighted sampling will be applied.")
        else:
            logger.info("Class distribution is relatively balanced.")

    def _compute_class_weights(self, labels: list) -> None:
        """Compute inverse-frequency class weights for balanced training."""
        counter = Counter(labels)
        total = len(labels)
        num_classes = len(CLASS_NAMES)

        weights = torch.zeros(num_classes)
        for cls_idx in range(num_classes):
            count = counter.get(cls_idx, 1)
            weights[cls_idx] = total / (num_classes * count)

        self.class_weights = weights
        logger.info(f"Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")

    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader with optional weighted sampling."""
        if self.use_class_weights and self.class_weights is not None:
            # Use cached labels — no image loading needed
            sample_weights = [self.class_weights[label] for label in self._train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )

    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader (no augmentation, no shuffling)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader (no augmentation, no shuffling)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def get_class_names(self) -> list:
        """Return ordered class name list."""
        return CLASS_NAMES.copy()

    def get_sample_images(self, n: int = 5) -> Tuple[list, list]:
        """Return n sample images and labels from training set for visualization."""
        images, labels = [], []
        for i in range(min(n, len(self.train_dataset))):
            img, lbl = self.train_dataset[i]
            images.append(img)
            labels.append(lbl)
        return images, labels
