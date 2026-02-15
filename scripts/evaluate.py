#!/usr/bin/env python3
"""
Evaluation Script
=================
CLI entry point for model evaluation and error analysis.

Usage:
    python scripts/evaluate.py --config configs/config.yaml
    python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_resnet.pth
"""

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import EuroSATDataModule, CLASS_NAMES
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet_finetune import ResNetFineTune
from src.evaluation.metrics import compute_metrics
from src.evaluation.error_analysis import analyze_errors
from src.training.utils import set_seed, get_device

import torch


def setup_logging(config: dict) -> None:
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, "evaluation.log"), mode="a"
            ),
        ],
    )


def load_model(config: dict, checkpoint_path: str, device: torch.device):
    """Load model architecture and checkpoint weights."""
    model_type = config["model"]["type"]
    num_classes = config["model"]["num_classes"]

    if model_type == "baseline":
        model_cfg = config["model"]["baseline"]
        model = BaselineCNN(
            num_classes=num_classes,
            channels=model_cfg.get("channels", [32, 64, 128]),
            dropout=model_cfg.get("dropout", 0.3),
        )
    elif model_type == "resnet":
        model = ResNetFineTune(
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    logger = logging.getLogger(__name__)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EuroSAT classification model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--model", type=str, choices=["baseline", "resnet"],
        help="Override model type from config"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.model:
        config["model"]["type"] = args.model

    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("EuroSAT LAND-USE CLASSIFICATION — EVALUATION")
    logger.info("=" * 60)

    set_seed(config["project"]["seed"])
    device = get_device(config["project"].get("device", "auto"))

    # Data setup
    data_module = EuroSATDataModule(config)
    data_module.setup()
    test_loader = data_module.get_test_loader()

    # Load model
    checkpoint_path = args.checkpoint or config["inference"]["model_path"]
    model = load_model(config, checkpoint_path, device)

    output_dir = config["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Compute metrics
    logger.info("\n--- Computing Evaluation Metrics ---")
    metrics = compute_metrics(
        model=model,
        loader=test_loader,
        device=device,
        class_names=CLASS_NAMES,
        output_dir=output_dir,
        split_name="test",
    )

    # Error analysis
    logger.info("\n--- Running Error Analysis ---")
    error_results = analyze_errors(
        model=model,
        loader=test_loader,
        device=device,
        class_names=CLASS_NAMES,
        output_dir=output_dir,
        num_samples=config["evaluation"].get("num_misclassified_samples", 25),
    )

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Test Accuracy:    {metrics['accuracy']:.2f}%")
    logger.info(f"  F1 (macro):       {metrics['f1_macro']:.2f}%")
    logger.info(f"  F1 (weighted):    {metrics['f1_weighted']:.2f}%")
    logger.info(f"  Total errors:     {error_results['total_errors']}")
    logger.info(f"  Error rate:       {error_results['error_rate']*100:.2f}%")
    logger.info(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
