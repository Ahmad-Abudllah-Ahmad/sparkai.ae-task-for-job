#!/usr/bin/env python3
"""
Training Script
===============
CLI entry point for model training.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --model baseline
    python scripts/train.py --config configs/config.yaml --model resnet --epochs 30
"""

import argparse
import logging
import os
import sys

import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import EuroSATDataModule
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet_finetune import ResNetFineTune
from src.training.trainer import Trainer
from src.training.utils import set_seed, get_device


def setup_logging(config: dict) -> None:
    """Configure logging with file and console handlers."""
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    log_level = config.get("logging", {}).get("level", "INFO")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, "training.log"), mode="a"
            ),
        ],
    )


def build_model(config: dict) -> object:
    """Build model based on configuration."""
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
        model_cfg = config["model"]["resnet"]
        model = ResNetFineTune(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            unfreeze_layers=model_cfg.get("unfreeze_layers", 2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger = logging.getLogger(__name__)
    logger.info(f"Model: {model_type} | Trainable params: {model.count_parameters():,}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train EuroSAT classification model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model", type=str, choices=["baseline", "resnet"],
        help="Override model type from config"
    )
    parser.add_argument(
        "--epochs", type=int, help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, help="Override learning rate"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only setup data and model, don't train"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model:
        config["model"]["type"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["optimizer"]["learning_rate"] = args.lr

    # Setup
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("EuroSAT LAND-USE CLASSIFICATION — TRAINING")
    logger.info("=" * 60)

    set_seed(config["project"]["seed"])
    device = get_device(config["project"].get("device", "auto"))

    # Data setup
    data_module = EuroSATDataModule(config)
    data_module.setup()

    if args.dry_run:
        logger.info("Dry run complete — data pipeline verified.")
        return

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    # Model
    model = build_model(config)

    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        class_weights=data_module.class_weights if config["training"].get("use_class_weights") else None,
    )

    # Train
    history = trainer.train(train_loader, val_loader)

    logger.info("\nTraining complete!")
    logger.info(f"Best val accuracy: {max(history['val_acc']):.2f}%")
    logger.info(f"Checkpoint saved to: {config['training']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
