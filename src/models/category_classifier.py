"""Prototype-based category recognition for MVTec AD images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from src.models.resnet_feature_extractor import ResNet18PatchExtractor


class CategoryPrototypeClassifier(nn.Module):
    """Assign an image to the closest saved category prototype.

    The returned match score is useful for ranking categories, but it should not
    be treated as a calibrated real-world probability.
    """

    def __init__(
        self,
        category_names: list[str],
        prototypes: torch.Tensor,
        image_size: int = 256,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()

        if prototypes.ndim != 2:
            raise ValueError("prototypes must have shape [num_categories, feature_dim]")
        if len(category_names) != prototypes.shape[0]:
            raise ValueError("category_names and prototypes do not match")

        self.category_names = list(category_names)
        self.image_size = int(image_size)
        self.temperature = float(temperature)
        self.feature_extractor = ResNet18PatchExtractor()

        self.register_buffer("prototypes", F.normalize(prototypes.float(), p=2, dim=1))
        self.transform = transforms.Compose(
            [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()]
        )

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.feature_extractor(images)
        embeddings = feature_maps.mean(dim=(2, 3))
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.extract_embeddings(images)
        return embeddings @ self.prototypes.T

    def predict(self, images: torch.Tensor) -> dict[str, Any]:
        self.eval()

        with torch.inference_mode():
            similarities = self(images)
            match_scores = torch.softmax(similarities / self.temperature, dim=1)
            top_scores, top_indices = match_scores.max(dim=1)

        predicted_categories = [self.category_names[int(index)] for index in top_indices.cpu()]
        top_scores = top_scores.cpu()

        return {
            "similarities": similarities.cpu(),
            "match_scores": match_scores.cpu(),
            "probabilities": match_scores.cpu(),  # kept for older notebook/app cells
            "predicted_indices": top_indices.cpu(),
            "predicted_categories": predicted_categories,
            "category_match_scores": top_scores,
            "confidence_scores": top_scores,  # backward-compatible alias
        }

    def load_image(self, image_path: str | Path) -> torch.Tensor:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def predict_image(self, image_path: str | Path, device: str | torch.device | None = None) -> dict[str, Any]:
        device = self.prototypes.device if device is None else torch.device(device)
        image_tensor = self.load_image(image_path).to(device)
        return self.predict(image_tensor)

    @classmethod
    def from_saved_file(
        cls,
        classifier_path: str | Path,
        device: str | torch.device,
    ) -> "CategoryPrototypeClassifier":
        classifier_path = Path(classifier_path)
        if not classifier_path.exists():
            raise FileNotFoundError(f"Missing category prototype file: {classifier_path}")

        checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
        classifier = cls(
            category_names=list(checkpoint["category_names"]),
            prototypes=checkpoint["prototypes"],
            image_size=int(checkpoint.get("image_size", 256)),
            temperature=float(checkpoint.get("temperature", 0.05)),
        )
        return classifier.to(device).eval()
