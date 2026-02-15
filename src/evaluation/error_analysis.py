"""
Error Analysis
==============
Qualitative and quantitative analysis of model failures:
  - Visualize most-confident misclassifications
  - Identify common confusion pairs
  - Generate improvement hypotheses
"""

import logging
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def analyze_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: str = "./outputs",
    num_samples: int = 25,
    mean: List[float] = None,
    std: List[float] = None,
) -> Dict:
    """
    Perform comprehensive error analysis.

    Args:
        model: Trained model
        loader: Test DataLoader
        device: Compute device
        class_names: List of class names
        output_dir: Directory to save analysis artifacts
        num_samples: Number of misclassified samples to visualize
        mean: Normalization mean (for un-normalizing images)
        std: Normalization std (for un-normalizing images)

    Returns:
        Dictionary with error analysis results
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Collect all predictions with confidence
    misclassified = []
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images_device = images.to(device, non_blocking=True)
            outputs = model(images_device)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)

            for i in range(len(labels)):
                total_count += 1
                if predicted[i].item() != labels[i].item():
                    misclassified.append({
                        "image": images[i].cpu(),
                        "true_label": labels[i].item(),
                        "pred_label": predicted[i].item(),
                        "confidence": confidences[i].item(),
                        "all_probs": probs[i].cpu().numpy(),
                    })
                else:
                    correct_count += 1

    logger.info(f"\nERROR ANALYSIS")
    logger.info(f"Total samples: {total_count}")
    logger.info(f"Correct: {correct_count} ({100*correct_count/total_count:.1f}%)")
    logger.info(f"Misclassified: {len(misclassified)} ({100*len(misclassified)/total_count:.1f}%)")

    # Sort by confidence (highest confidence errors are most interesting)
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)

    # Analyze confusion pairs
    confusion_pairs = _analyze_confusion_pairs(misclassified, class_names)

    # Visualize top misclassified samples
    if len(misclassified) > 0:
        _visualize_misclassified(
            misclassified[:num_samples], class_names, mean, std, output_dir
        )

    # Generate improvement hypotheses
    hypotheses = _generate_hypotheses(confusion_pairs, class_names, misclassified)

    # Save analysis report
    _save_analysis_report(
        output_dir, misclassified, confusion_pairs, hypotheses,
        class_names, total_count
    )

    return {
        "total_errors": len(misclassified),
        "error_rate": len(misclassified) / total_count,
        "confusion_pairs": confusion_pairs,
        "hypotheses": hypotheses,
    }


def _analyze_confusion_pairs(
    misclassified: List[Dict], class_names: List[str]
) -> List[Tuple[str, str, int]]:
    """Find the most common confusion pairs."""
    from collections import Counter

    pair_counts = Counter()
    for item in misclassified:
        true_name = class_names[item["true_label"]]
        pred_name = class_names[item["pred_label"]]
        pair_counts[(true_name, pred_name)] += 1

    # Sort by frequency
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    logger.info("\nTop Confusion Pairs:")
    logger.info("-" * 50)
    for (true_cls, pred_cls), count in sorted_pairs[:10]:
        logger.info(f"  {true_cls:>25s} → {pred_cls:<25s}: {count}")

    return [(t, p, c) for (t, p), c in sorted_pairs]


def _visualize_misclassified(
    samples: List[Dict],
    class_names: List[str],
    mean: List[float],
    std: List[float],
    output_dir: str,
) -> None:
    """Visualize the most-confident misclassified samples."""
    n = len(samples)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, item in enumerate(samples):
        ax = axes[i]
        img = _unnormalize(item["image"], mean, std)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        true_name = class_names[item["true_label"]]
        pred_name = class_names[item["pred_label"]]
        conf = item["confidence"] * 100

        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)",
            fontsize=9,
            color="red",
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Most Confident Misclassifications",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "misclassified_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Misclassified samples visualization saved to {save_path}")


def _unnormalize(
    img: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    img = img.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img


def _generate_hypotheses(
    confusion_pairs: List[Tuple[str, str, int]],
    class_names: List[str],
    misclassified: List[Dict],
) -> List[str]:
    """Generate data-driven improvement hypotheses based on error patterns."""
    hypotheses = []

    if not confusion_pairs:
        return ["Model performance is strong — no dominant failure modes detected."]

    # Hypothesis 1: Top confusion pair
    top_true, top_pred, top_count = confusion_pairs[0]
    hypotheses.append(
        f"Top confusion: '{top_true}' misclassified as '{top_pred}' "
        f"({top_count} times). These classes likely share visual features. "
        f"Consider: class-specific augmentation, higher-resolution inputs, "
        f"or attention mechanisms to focus on discriminative regions."
    )

    # Hypothesis 2: High-confidence errors
    high_conf_errors = [m for m in misclassified if m["confidence"] > 0.9]
    if high_conf_errors:
        hypotheses.append(
            f"{len(high_conf_errors)} errors with >90% confidence suggest "
            f"potential label noise or ambiguous samples in the dataset. "
            f"Manual review of these samples is recommended."
        )

    # Hypothesis 3: Agricultural class confusion
    agri_classes = {"AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"}
    agri_confusions = [
        (t, p, c) for t, p, c in confusion_pairs
        if t in agri_classes and p in agri_classes
    ]
    if agri_confusions:
        total_agri = sum(c for _, _, c in agri_confusions)
        hypotheses.append(
            f"Agricultural classes account for {total_agri} errors. "
            f"These classes have similar spectral signatures. Consider: "
            f"using multi-spectral bands (EuroSAT has 13 bands available), "
            f"temporal features, or ensemble methods."
        )

    # Hypothesis 4: Resolution
    hypotheses.append(
        "Fine-grained features may require higher resolution. "
        "Increasing input size from 64→128 or using super-resolution "
        "preprocessing could improve discrimination between similar classes."
    )

    return hypotheses


def _save_analysis_report(
    output_dir: str,
    misclassified: List[Dict],
    confusion_pairs: List[Tuple[str, str, int]],
    hypotheses: List[str],
    class_names: List[str],
    total_count: int,
) -> None:
    """Save a text-based error analysis report."""
    report_path = os.path.join(output_dir, "error_analysis_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total test samples: {total_count}\n")
        f.write(f"Total errors: {len(misclassified)}\n")
        f.write(f"Error rate: {100*len(misclassified)/total_count:.2f}%\n\n")

        f.write("TOP CONFUSION PAIRS\n")
        f.write("-" * 50 + "\n")
        for true_cls, pred_cls, count in confusion_pairs[:15]:
            f.write(f"  {true_cls:>25s} → {pred_cls:<25s}: {count}\n")

        f.write(f"\nIMPROVEMENT HYPOTHESES\n")
        f.write("-" * 50 + "\n")
        for i, hyp in enumerate(hypotheses, 1):
            f.write(f"\n{i}. {hyp}\n")

    logger.info(f"Error analysis report saved to {report_path}")
