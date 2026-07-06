"""Feature-memory anomaly detector used for the final model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.resnet_feature_extractor import ResNet18PatchExtractor


class FeatureAnomalyDetector(nn.Module):
    """Compare ResNet patch features against a category-specific normal memory bank."""

    def __init__(
        self,
        memory_bank: torch.Tensor,
        image_size: int,
        image_score_threshold: float,
        pixel_threshold: float,
        minimum_component_area: int = 64,
        opening_kernel_size: int = 3,
        closing_kernel_size: int = 5,
        top_fraction: float = 0.01,
        query_chunk_size: int = 1024,
        memory_chunk_size: int = 2000,
    ) -> None:
        super().__init__()

        if memory_bank.ndim != 2:
            raise ValueError("memory_bank must have shape [num_patches, feature_dim]")
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in the range (0, 1]")

        self.feature_extractor = ResNet18PatchExtractor()
        self.register_buffer("memory_bank", F.normalize(memory_bank.float(), p=2, dim=1))

        self.image_size = int(image_size)
        self.image_score_threshold = float(image_score_threshold)
        self.pixel_threshold = float(pixel_threshold)
        self.minimum_component_area = int(minimum_component_area)
        self.opening_kernel_size = int(opening_kernel_size)
        self.closing_kernel_size = int(closing_kernel_size)
        self.top_fraction = float(top_fraction)
        self.query_chunk_size = int(query_chunk_size)
        self.memory_chunk_size = int(memory_chunk_size)

    def _nearest_normal_patch_distances(self, query_patches: torch.Tensor) -> torch.Tensor:
        """Return one distance per query patch without building a huge similarity matrix."""
        distances = []

        for query_start in range(0, len(query_patches), self.query_chunk_size):
            query_chunk = query_patches[query_start : query_start + self.query_chunk_size]
            best_similarity = torch.full((len(query_chunk),), -1.0, device=query_chunk.device)

            for memory_start in range(0, len(self.memory_bank), self.memory_chunk_size):
                memory_chunk = self.memory_bank[memory_start : memory_start + self.memory_chunk_size]
                chunk_similarity = query_chunk @ memory_chunk.T
                best_similarity = torch.maximum(best_similarity, chunk_similarity.max(dim=1).values)

            distances.append(torch.sqrt(torch.clamp(2.0 - 2.0 * best_similarity, min=0.0)))

        return torch.cat(distances, dim=0)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.feature_extractor(images)
        batch_size, channels, height, width = feature_maps.shape

        patches = feature_maps.permute(0, 2, 3, 1).reshape(-1, channels)
        patches = F.normalize(patches, p=2, dim=1)

        patch_distances = self._nearest_normal_patch_distances(patches)
        low_res_maps = patch_distances.reshape(batch_size, 1, height, width)
        full_res_maps = F.interpolate(
            low_res_maps,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        flat_maps = low_res_maps.flatten(start_dim=1)
        top_k = max(1, int(flat_maps.shape[1] * self.top_fraction))
        image_scores = torch.topk(flat_maps, k=top_k, dim=1).values.mean(dim=1)

        return image_scores, full_res_maps

    def _postprocess_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        mask = binary_mask.astype(np.uint8) * 255
        opening_kernel = np.ones((self.opening_kernel_size, self.opening_kernel_size), dtype=np.uint8)
        closing_kernel = np.ones((self.closing_kernel_size, self.closing_kernel_size), dtype=np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask, dtype=np.uint8)

        for component_index in range(1, num_components):
            area = stats[component_index, cv2.CC_STAT_AREA]
            if area >= self.minimum_component_area:
                cleaned[labels == component_index] = 1

        return cleaned

    def predict(self, images: torch.Tensor) -> dict[str, Any]:
        self.eval()

        with torch.inference_mode():
            image_scores, anomaly_maps = self(images)

        predicted_labels = (image_scores > self.image_score_threshold).long()
        raw_masks = (anomaly_maps.squeeze(1) > self.pixel_threshold).to(torch.uint8)
        processed_masks = np.stack([self._postprocess_mask(mask) for mask in raw_masks.cpu().numpy()])

        return {
            "image_scores": image_scores.cpu(),
            "predicted_labels": predicted_labels.cpu(),
            "anomaly_maps": anomaly_maps.squeeze(1).cpu(),
            "raw_masks": raw_masks.cpu(),
            "processed_masks": torch.from_numpy(processed_masks),
        }

    @classmethod
    def from_saved_files(
        cls,
        memory_bank_path: str | Path,
        configuration_path: str | Path,
        device: str | torch.device,
    ) -> "FeatureAnomalyDetector":
        memory_bank_path = Path(memory_bank_path)
        configuration_path = Path(configuration_path)

        if not memory_bank_path.exists():
            raise FileNotFoundError(f"Missing memory bank: {memory_bank_path}")
        if not configuration_path.exists():
            raise FileNotFoundError(f"Missing detector config: {configuration_path}")

        checkpoint = torch.load(memory_bank_path, map_location="cpu", weights_only=True)
        with configuration_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        post = config.get("postprocessing", {})
        detector = cls(
            memory_bank=checkpoint["memory_bank"],
            image_size=config["image_size"],
            image_score_threshold=config["image_score_threshold"],
            pixel_threshold=config["pixel_threshold"],
            minimum_component_area=post.get("minimum_component_area", 64),
            opening_kernel_size=post.get("opening_kernel_size", 3),
            closing_kernel_size=post.get("closing_kernel_size", 5),
        )

        return detector.to(device).eval()
