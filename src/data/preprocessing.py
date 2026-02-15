"""
Preprocessing Transforms
========================
Validation/test-time transforms: resize + normalize only.
No data augmentation applied — ensures consistent evaluation.
"""

from torchvision import transforms


def get_preprocessing_transforms(data_config: dict, image_size: int) -> transforms.Compose:
    """
    Build evaluation/inference transforms.

    Args:
        data_config: Data section of config.yaml
        image_size: Target image size (64 for baseline, 224 for ResNet)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    norm_cfg = data_config["normalization"]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_cfg["imagenet_mean"],
            std=norm_cfg["imagenet_std"]
        ),
    ])
