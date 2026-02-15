"""
Evaluation Script
=================
Run full evaluation on a trained model checkpoint.
"""

import argparse
import logging
import torch
import yaml
from pathlib import Path

from src.data.dataset import EuroSATDataModule
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet_finetune import ResNetFineTune
from src.evaluation.metrics import compute_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model_path: str, model_type: str = "baseline", config_path: str = "configs/config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args
    config["model"]["type"] = model_type
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data
    data_module = EuroSATDataModule(config)
    data_module.setup()
    test_loader = data_module.get_test_loader()
    class_names = data_module.get_class_names()
    
    # Model initialization
    if model_type == "resnet":
        model = ResNetFineTune(num_classes=10)
    else:
        model = BaselineCNN(num_classes=10)
        
    logger.info(f"Loading checkpoint from {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle state dict loading
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Evaluate
    logger.info(f"Starting evaluation for {model_type} model...")
    metrics = compute_metrics(
        model=model,
        loader=test_loader,
        device=device,
        class_names=class_names,
        output_dir="./outputs",
        split_name=f"test_{model_type}"
    )
    
    print("\n" + "="*40)
    print(f"{model_type.upper()} FINAL METRICS")
    print("="*40)
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1 Macro: {metrics['f1_macro']:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EuroSAT Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "resnet"], help="Model type")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    evaluate(args.model_path, args.model_type, args.config)
