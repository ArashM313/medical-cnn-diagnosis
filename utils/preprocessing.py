"""
Shared preprocessing transforms for all three modalities (X-ray, CT, MRI).

Medical grayscale-ish images (X-ray, CT, MRI) are handled uniformly here:
we always force 3 channels (RGB) even if the source image is single-channel,
because our CNN's first conv layer expects 3 input channels. This also lets
us reuse ImageNet-style pretrained backbones later if desired, without any
architecture change.
"""

import torch
from torchvision import transforms

# ImageNet mean/std work fine even for medical grayscale-as-RGB images because
# we replicate the single channel across R/G/B -- using these standard values
# keeps compatibility if we later swap in a pretrained backbone.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, train: bool, strong: bool = False) -> transforms.Compose:
    """
    Build the transform pipeline.

    train=True  -> includes light augmentation (flip, rotation, jitter) since
                   medical scans are usually roughly upright but patient
                   positioning/framing varies a bit.
    train=False -> deterministic resize + normalize only, used for val/test
                   and for single-image inference in the dashboard.
    strong=True -> adds RandomResizedCrop on top of the standard augmentation
                   (only meaningful when train=True). Used for modalities
                   with very little training data (CT: ~613 images), where
                   squeezing more effective variety out of each image matters
                   more than it does for X-ray/MRI's larger datasets.
    """
    if train:
        aug = [
            transforms.Grayscale(num_output_channels=3),  # force 3ch regardless of source
        ]
        if strong:
            # Crop a random 80-100% region before resizing -- effectively
            # trains on many slightly different framings of the same scan.
            aug.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
        else:
            aug.append(transforms.Resize((image_size, image_size)))

        aug += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),         # small: don't distort anatomy unrealistically
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
        return transforms.Compose(aug)
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])


def unnormalize(tensor):
    """
    Inverse of Normalize(), for turning a model-input tensor back into a
    displayable [0,1] image (used by Grad-CAM overlay and the dashboard).
    """
    mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(3, 1, 1)
    return tensor.cpu() * std + mean