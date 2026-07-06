"""Single-image inference helpers for the Streamlit app and CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.feature_anomaly_detector import FeatureAnomalyDetector


class IndustrialAnomalyInference:
    """Load a saved detector and run it on individual image files."""

    def __init__(
        self,
        memory_bank_path: str | Path,
        configuration_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.detector = FeatureAnomalyDetector.from_saved_files(
            memory_bank_path=memory_bank_path,
            configuration_path=configuration_path,
            device=self.device,
        )
        self.image_size = self.detector.image_size
        self.transform = transforms.Compose(
            [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()]
        )

    def load_image(self, image_path: str | Path) -> tuple[Image.Image, torch.Tensor]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, tensor

    @staticmethod
    def tensor_to_image_array(image_tensor: torch.Tensor) -> np.ndarray:
        return image_tensor.detach().cpu().permute(1, 2, 0).numpy()

    @staticmethod
    def normalize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
        lower = np.percentile(anomaly_map, 1)
        upper = np.percentile(anomaly_map, 99)
        normalized = (anomaly_map - lower) / (upper - lower + 1e-8)
        return np.clip(normalized, 0, 1)

    def create_heatmap_overlay(
        self,
        image_array: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: float = 0.40,
    ) -> np.ndarray:
        heatmap = plt.get_cmap("inferno")(self.normalize_anomaly_map(anomaly_map))[..., :3]
        return np.clip((1 - alpha) * image_array + alpha * heatmap, 0, 1)

    @staticmethod
    def create_mask_overlay(
        image_array: np.ndarray,
        predicted_mask: np.ndarray,
        alpha: float = 0.60,
    ) -> np.ndarray:
        mask = predicted_mask.astype(bool)
        overlay = image_array.copy()
        red = np.zeros_like(image_array)
        red[..., 0] = 1.0
        overlay[mask] = (1 - alpha) * image_array[mask] + alpha * red[mask]
        return np.clip(overlay, 0, 1)

    def predict_image(self, image_path: str | Path) -> dict[str, Any]:
        original_image, image_tensor = self.load_image(image_path)
        result = self.detector.predict(image_tensor)

        resized_image = self.tensor_to_image_array(image_tensor.squeeze(0))
        image_score = float(result["image_scores"][0].item())
        predicted_label = int(result["predicted_labels"][0].item())
        anomaly_map = result["anomaly_maps"][0].detach().cpu().numpy()
        raw_mask = result["raw_masks"][0].detach().cpu().numpy()
        processed_mask = result["processed_masks"][0].detach().cpu().numpy()

        return {
            "image_path": str(image_path),
            "original_image": original_image,
            "resized_image": resized_image,
            "image_score": image_score,
            "image_threshold": self.detector.image_score_threshold,
            "predicted_label": predicted_label,
            "prediction_text": "defective" if predicted_label else "normal",
            "anomaly_map": anomaly_map,
            "raw_mask": raw_mask,
            "processed_mask": processed_mask,
            "heatmap_overlay": self.create_heatmap_overlay(resized_image, anomaly_map),
            "mask_overlay": self.create_mask_overlay(resized_image, processed_mask),
        }

    def save_prediction_outputs(
        self,
        prediction: dict[str, Any],
        output_dir: str | Path,
        prefix: str = "prediction",
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "resized_image": prediction["resized_image"],
            "heatmap_overlay": prediction["heatmap_overlay"],
            "mask_overlay": prediction["mask_overlay"],
            "processed_mask": prediction["processed_mask"],
        }
        saved_paths: dict[str, Path] = {}

        for name, array in outputs.items():
            output_path = output_dir / f"{prefix}_{name}.png"
            if name == "processed_mask":
                image = Image.fromarray((array * 255).astype(np.uint8))
            else:
                image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
            image.save(output_path)
            saved_paths[name] = output_path

        return saved_paths
