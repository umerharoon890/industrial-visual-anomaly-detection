"""Frozen ResNet-18 feature extractor used by the anomaly detectors."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18PatchExtractor(nn.Module):
    """Return spatial layer2 features from ImageNet-pretrained ResNet-18."""

    def __init__(self) -> None:
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.imagenet_mean) / self.imagenet_std
        return self.feature_extractor(images)
