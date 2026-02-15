"""
Tests for Inference Pipeline
=============================
Unit tests verifying model loading, preprocessing, and prediction format.
"""

import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_cnn import BaselineCNN
from src.models.resnet_finetune import ResNetFineTune
from src.inference.pipeline import InferencePipeline, CLASS_NAMES
from src.data.preprocessing import get_preprocessing_transforms
from src.training.utils import set_seed


class TestBaselineCNN:
    """Test baseline model architecture."""

    def test_forward_pass(self):
        model = BaselineCNN(num_classes=10)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    def test_parameter_count(self):
        model = BaselineCNN(num_classes=10)
        params = model.count_parameters()
        assert params > 0, "Model should have trainable parameters"
        assert params < 1_000_000, f"Baseline should be lightweight, got {params:,}"

    def test_different_input_sizes(self):
        """Model should handle different input sizes via adaptive pooling."""
        model = BaselineCNN(num_classes=10)
        for size in [32, 64, 96, 128]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 10), f"Failed for input size {size}"


class TestResNetFineTune:
    """Test ResNet fine-tuning model."""

    def test_forward_pass(self):
        model = ResNetFineTune(num_classes=10, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    def test_freeze_unfreeze(self):
        model = ResNetFineTune(
            num_classes=10, pretrained=False,
            freeze_backbone=True, unfreeze_layers=2
        )
        frozen_params = model.count_parameters()

        model.unfreeze_last_layers()
        unfrozen_params = model.count_parameters()

        assert unfrozen_params > frozen_params, \
            "Unfreezing should increase trainable params"


class TestInferencePipeline:
    """Test inference pipeline logic."""

    def setup_method(self):
        """Create a simple pipeline with untrained model for testing."""
        set_seed(42)
        model = BaselineCNN(num_classes=10)
        device = torch.device("cpu")
        data_config = {
            "normalization": {
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
            }
        }
        transform = get_preprocessing_transforms(data_config, image_size=64)

        self.pipeline = InferencePipeline(
            model=model,
            device=device,
            transform=transform,
            class_names=CLASS_NAMES,
            confidence_threshold=0.5,
        )

    def test_predict_pil_image(self):
        """Test prediction with PIL Image input."""
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = self.pipeline.predict(img)

        assert "class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "class_index" in result
        assert "above_threshold" in result

        assert result["class"] in CLASS_NAMES
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 10

    def test_predict_tensor(self):
        """Test prediction with tensor input."""
        tensor = torch.randn(3, 64, 64)
        result = self.pipeline.predict(tensor)

        assert result["class"] in CLASS_NAMES
        assert 0 <= result["confidence"] <= 1

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to approximately 1."""
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = self.pipeline.predict(img)

        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}"

    def test_batch_predict(self):
        """Test batch prediction handles errors gracefully."""
        results = self.pipeline.predict_batch(["nonexistent.jpg"])
        assert len(results) == 1
        assert "error" in results[0]


class TestPreprocessing:
    """Test data preprocessing transforms."""

    def test_transform_output_shape(self):
        data_config = {
            "normalization": {
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
            }
        }
        transform = get_preprocessing_transforms(data_config, image_size=64)

        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        tensor = transform(img)

        assert tensor.shape == (3, 64, 64), f"Expected (3, 64, 64), got {tensor.shape}"
        assert tensor.dtype == torch.float32


class TestSeedReproducibility:
    """Test that seeding produces reproducible results."""

    def test_reproducible_model_output(self):
        set_seed(42)
        model1 = BaselineCNN(num_classes=10)
        x = torch.randn(1, 3, 64, 64)
        out1 = model1(x)

        set_seed(42)
        model2 = BaselineCNN(num_classes=10)
        x = torch.randn(1, 3, 64, 64)
        out2 = model2(x)

        assert torch.allclose(out1, out2), "Same seed should produce same output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
