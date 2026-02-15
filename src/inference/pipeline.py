"""
Inference Pipeline
==================
Clean, production-ready inference: load model → preprocess → predict.
Completely decoupled from training code.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from src.models.baseline_cnn import BaselineCNN
from src.models.resnet_finetune import ResNetFineTune

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


class InferencePipeline:
    """
    Production inference pipeline for EuroSAT classification.

    Usage:
        pipeline = InferencePipeline.from_config(config)
        result = pipeline.predict("path/to/image.jpg")
        # {'class': 'Forest', 'confidence': 0.97, 'probabilities': {...}}
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transform: transforms.Compose,
        class_names: List[str],
        confidence_threshold: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.transform = transform
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        self.model.eval()
        logger.info(
            f"InferencePipeline ready | Device: {device} | "
            f"Classes: {len(class_names)} | "
            f"Confidence threshold: {confidence_threshold}"
        )

    @classmethod
    def from_config(cls, config: dict) -> "InferencePipeline":
        """
        Factory method: build pipeline from config dictionary.

        Args:
            config: Full configuration dictionary

        Returns:
            Ready-to-use InferencePipeline instance
        """
        from src.training.utils import get_device

        device = get_device(config["project"].get("device", "auto"))
        model_type = config["model"]["type"]
        num_classes = config["model"]["num_classes"]

        # Build model architecture
        if model_type == "baseline":
            model_cfg = config["model"]["baseline"]
            model = BaselineCNN(
                num_classes=num_classes,
                channels=model_cfg.get("channels", [32, 64, 128]),
                dropout=model_cfg.get("dropout", 0.3),
            )
            image_size = config["data"]["image_size"]
        elif model_type == "resnet":
            model_cfg = config["model"]["resnet"]
            model = ResNetFineTune(
                num_classes=num_classes,
                pretrained=False,  # Don't download weights for inference
                freeze_backbone=False,
            )
            image_size = config["data"]["image_size_resnet"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint
        checkpoint_path = config["inference"]["model_path"]
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        logger.info(f"Model loaded from {checkpoint_path}")
        if "epoch" in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if "val_loss" in checkpoint:
            logger.info(f"  Checkpoint val_loss: {checkpoint['val_loss']:.4f}")

        # Build transform
        norm_cfg = config["data"]["normalization"]
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_cfg["imagenet_mean"],
                std=norm_cfg["imagenet_std"],
            ),
        ])

        confidence_threshold = config["inference"].get("confidence_threshold", 0.5)

        return cls(
            model=model,
            device=device,
            transform=transform,
            class_names=CLASS_NAMES,
            confidence_threshold=confidence_threshold,
        )

    def predict(self, image_input) -> Dict:
        """
        Run inference on a single image.

        Args:
            image_input: File path (str), PIL Image, or torch.Tensor

        Returns:
            {
                'class': str,
                'class_index': int,
                'confidence': float,
                'above_threshold': bool,
                'probabilities': {class_name: probability, ...}
            }
        """
        # Load and preprocess
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, torch.Tensor):
            # Already a tensor, skip transform
            tensor = image_input.unsqueeze(0) if image_input.dim() == 3 else image_input
            return self._predict_tensor(tensor)
        else:
            raise ValueError(
                f"Unsupported input type: {type(image_input)}. "
                f"Expected str, PIL.Image, or torch.Tensor."
            )

        tensor = self.transform(image).unsqueeze(0)
        return self._predict_tensor(tensor)

    def _predict_tensor(self, tensor: torch.Tensor) -> Dict:
        """Run inference on a preprocessed tensor."""
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)

        pred_idx = predicted.item()
        pred_confidence = confidence.item()
        pred_class = self.class_names[pred_idx]

        # Build probability map
        prob_dict = {
            name: round(probs[0][i].item(), 4)
            for i, name in enumerate(self.class_names)
        }

        return {
            "class": pred_class,
            "class_index": pred_idx,
            "confidence": round(pred_confidence, 4),
            "above_threshold": pred_confidence >= self.confidence_threshold,
            "probabilities": prob_dict,
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Run inference on multiple images."""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result["image_path"] = path
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({"image_path": path, "error": str(e)})
        return results
