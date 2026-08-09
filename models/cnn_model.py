"""
A single CNN architecture reused for all three modalities (X-ray, CT, MRI).
Only the final classification layer's output size changes (num_classes),
driven by config.MODALITIES[modality]['classes'].

Design notes:
  - 4 conv blocks (32 -> 64 -> 128 -> 256 channels), each: Conv -> BatchNorm ->
    ReLU -> MaxPool. BatchNorm stabilizes training given our datasets are
    modest in size (CT has only ~613 training images).
  - Dropout before the final FC layer combats overfitting on the smaller
    datasets (CT especially).
  - AdaptiveAvgPool2d before flattening: makes the network robust to input
    size changes (not strictly needed since we fix IMAGE_SIZE=128, but it's
    good practice and means Grad-CAM's spatial map has a clean, known shape).
  - The last conv block is exposed as `self.features_final` (a named layer,
    not buried in nn.Sequential) specifically so Grad-CAM (Step 5) can hook
    into it directly by name to get gradients/activations for the heatmap.
"""

import torch
import torch.nn as nn
from torchvision import models as tv_models


class MedicalCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Block 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
        )

        # Block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        # Block 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        # Block 4 (final conv block): 128 -> 256 channels.
        # Named explicitly (not nested in a bigger nn.Sequential with the
        # other blocks) so Grad-CAM can register hooks on it by name:
        # model.features_final
        self.features_final = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # NOTE: no MaxPool here -- Grad-CAM wants the last conv feature
            # map at reasonable spatial resolution (16x16) for a meaningful
            # heatmap; pooling further would make it too coarse (e.g. 8x8).
        )

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # 16x16 -> 4x4, fixed size regardless of input
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.features_final(x)      # <- Grad-CAM hooks read activations/gradients here
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int, device, pretrained: bool = False) -> nn.Module:
    """
    Convenience constructor used by both training.py and the dashboard.

    pretrained=False (default) -> MedicalCNN, trained from scratch. Used for
        X-ray and MRI, which have enough training data (4000+ images) that a
        from-scratch CNN learns well without needing pretrained features.

    pretrained=True -> TransferResNetCNN, a ResNet18 pretrained on ImageNet
        with early layers frozen. Used for CT, which has only ~613 training
        images -- far too few to learn good low-level features (edges,
        textures) from scratch. The pretrained backbone already knows these
        generic features; we only fine-tune the last block + classifier.
    """
    if pretrained:
        model = TransferResNetCNN(num_classes=num_classes)
    else:
        model = MedicalCNN(num_classes=num_classes)
    return model.to(device)


class TransferResNetCNN(nn.Module):
    """
    ResNet18 pretrained on ImageNet, adapted for our small CT dataset.

    Early layers (conv1, bn1, layer1, layer2, layer3) are FROZEN -- they
    already encode generic low-level features (edges, textures, shapes)
    learned from 1.2M ImageNet images, which transfer well to CT scans even
    though CT looks nothing like natural photos; low-level filters like edge
    detectors are useful regardless of image domain.

    Only layer4 (the last residual block) and the final classifier are
    trainable -- this is what actually specializes the model to distinguish
    adenocarcinoma / large.cell.carcinoma / normal / squamous.cell.carcinoma,
    using far fewer images than training all layers from scratch would need.

    `self.features_final` points at `layer4` (same name/purpose as in
    MedicalCNN) so the Grad-CAM code in Step 5 can hook it identically
    regardless of which architecture a given modality uses.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze everything first...
        for param in backbone.parameters():
            param.requires_grad = False

        # ...then unfreeze layer2/layer3/layer4 (not just layer4). CT scans
        # have a much bigger visual domain gap from ImageNet photos than a
        # typical fine-tuning target -- only the earliest generic edge
        # detectors (conv1, layer1) are kept frozen; everything from layer2
        # onward is retrained to adapt to CT-specific textures/patterns.
        for layer in [backbone.layer2, backbone.layer3, backbone.layer4]:
            for param in layer.parameters():
                param.requires_grad = True

        # Replace the ImageNet 1000-class head with ours; new layers are
        # trainable by default (requires_grad=True on creation).
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        self.backbone = backbone
        self.features_final = backbone.layer4  # named reference for Grad-CAM hooks

    def forward(self, x):
        # Manually replay ResNet18's forward pass up through avgpool + fc.
        # (Not using self.backbone(x) directly because we want the exact
        # same structural pattern as MedicalCNN.forward for clarity.)
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)

        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)  # == self.features_final(previous_x); same tensor path

        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        x = b.fc(x)
        return x


if __name__ == "__main__":
    # Sanity check: run this file directly in PyCharm to confirm both
    # architectures build and produce the expected output shape.
    import config

    for modality, cfg in config.MODALITIES.items():
        use_pretrained = bool(cfg.get("use_pretrained", False))
        model = build_model(num_classes=len(cfg["classes"]), device=config.DEVICE, pretrained=use_pretrained)
        dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
        output = model(dummy_input)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        arch = "TransferResNetCNN" if use_pretrained else "MedicalCNN"
        print(f"[{modality}] arch={arch}  output shape={tuple(output.shape)}  "
              f"expected=(2, {len(cfg['classes'])})  "
              f"params={n_params:,} (trainable={n_trainable:,})")