#!/usr/bin/env python3
"""
Inference Script
================
CLI entry point for single-image inference.

Usage:
    python scripts/infer.py --image path/to/image.jpg
    python scripts/infer.py --image path/to/image.jpg --config configs/config.yaml
"""

import argparse
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import InferencePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a satellite image"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output result as JSON"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate input
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        config["inference"]["model_path"] = args.checkpoint

    # Build pipeline
    pipeline = InferencePipeline.from_config(config)

    # Run inference
    result = pipeline.predict(args.image)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 40)
        print("PREDICTION RESULT")
        print("=" * 40)
        print(f"  Image:      {args.image}")
        print(f"  Class:      {result['class']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Threshold:  {'PASS' if result['above_threshold'] else 'BELOW'}")
        print("\n  All probabilities:")
        for cls, prob in sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar = "█" * int(prob * 40)
            print(f"    {cls:>25s}: {prob*100:5.1f}% {bar}")
        print("=" * 40)


if __name__ == "__main__":
    main()
