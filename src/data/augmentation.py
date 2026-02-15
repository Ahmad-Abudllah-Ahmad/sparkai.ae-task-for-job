"""
Data Augmentation Transforms
=============================
Training-time augmentation pipeline to improve generalization.
Augmentations are chosen to be physically meaningful for satellite imagery:
  - Flips: satellite images have no canonical orientation
  - Rotation: land features are rotation-invariant
  - Color jitter: accounts for seasonal / atmospheric variation
"""

from torchvision import transforms


def get_augmentation_transforms(data_config: dict, image_size: int) -> transforms.Compose:
    """
    Build training augmentation pipeline.

    Args:
        data_config: Data section of config.yaml
        image_size: Target image size

    Returns:
        torchvision.transforms.Compose with augmentations + normalization
    """
    aug_cfg = data_config["augmentation"]
    norm_cfg = data_config["normalization"]

    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]

    if aug_cfg.get("enabled", True):
        if aug_cfg.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if aug_cfg.get("vertical_flip", True):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))

        rotation = aug_cfg.get("random_rotation", 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(degrees=rotation))

        cj = aug_cfg.get("color_jitter", {})
        if cj:
            transform_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            ))

    # Always apply ToTensor + Normalize after augmentations
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_cfg["imagenet_mean"],
            std=norm_cfg["imagenet_std"]
        ),
    ])

    return transforms.Compose(transform_list)
