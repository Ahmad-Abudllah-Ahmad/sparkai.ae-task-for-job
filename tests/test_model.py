import torch
import pytest
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet_finetune import ResNetFineTune

def test_baseline_cnn_instantiation():
    """Test if BaselineCNN initializes correctly."""
    model = BaselineCNN(num_classes=10)
    assert isinstance(model, torch.nn.Module)

def test_baseline_cnn_forward():
    """Test forward pass of BaselineCNN."""
    model = BaselineCNN(num_classes=10)
    # EuroSAT images are 64x64
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    assert output.shape == (1, 10)

def test_resnet_instantiation():
    """Test if ResNetFineTune initializes correctly."""
    # Mocking weights to avoid downloading during tests if possible, 
    # but for simplicity we'll just let it load or use pretrained=False for speed
    model = ResNetFineTune(num_classes=10, pretrained=False)
    assert isinstance(model, torch.nn.Module)

def test_resnet_forward():
    """Test forward pass of ResNetFineTune."""
    model = ResNetFineTune(num_classes=10, pretrained=False)
    # ResNet needs 224x224 usually, but can adapt. 
    # Our data pipeline resizes to 224 for ResNet.
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 10)
