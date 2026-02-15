"""
Evaluation Metrics
==================
Comprehensive evaluation: accuracy, per-class precision/recall/F1,
confusion matrix visualization, and classification report.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: str = "./outputs",
    split_name: str = "test",
) -> Dict:
    """
    Compute full evaluation metrics on a dataset split.

    Args:
        model: Trained model
        loader: DataLoader for the evaluation split
        device: Compute device
        class_names: List of class names
        output_dir: Directory to save evaluation artifacts
        split_name: Name of the split (for logging/saving)

    Returns:
        Dictionary of evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Core metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision_macro = precision_score(all_labels, all_preds, average="macro") * 100
    recall_macro = recall_score(all_labels, all_preds, average="macro") * 100
    f1_macro = f1_score(all_labels, all_preds, average="macro") * 100
    f1_weighted = f1_score(all_labels, all_preds, average="weighted") * 100

    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULTS ({split_name.upper()} SET)")
    logger.info("=" * 60)
    logger.info(f"  Accuracy:         {accuracy:.2f}%")
    logger.info(f"  Precision (macro): {precision_macro:.2f}%")
    logger.info(f"  Recall (macro):    {recall_macro:.2f}%")
    logger.info(f"  F1 Score (macro):  {f1_macro:.2f}%")
    logger.info(f"  F1 Score (weighted): {f1_weighted:.2f}%")

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=3,
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Save report to file
    report_path = os.path.join(output_dir, f"classification_report_{split_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Classification Report — {split_name.upper()} Set\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Accuracy:           {accuracy:.2f}%\n")
        f.write(f"Precision (macro):  {precision_macro:.2f}%\n")
        f.write(f"Recall (macro):     {recall_macro:.2f}%\n")
        f.write(f"F1 Score (macro):   {f1_macro:.2f}%\n")
        f.write(f"F1 Score (weighted):{f1_weighted:.2f}%\n\n")
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names, output_dir, split_name
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
    }


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    output_dir: str,
    split_name: str,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title(f"Confusion Matrix — {split_name} (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title(f"Confusion Matrix — {split_name} (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")
