"""Small PyTorch dataset wrapper for MVTec AD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MVTecDataset(Dataset):
    """Load images, labels, and masks from one MVTec AD category."""

    def __init__(
        self,
        dataset_root: str | Path,
        category: str,
        split: str,
        image_size: int = 256,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.category = category
        self.split = split.lower()
        self.image_size = int(image_size)

        if self.split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.category_path = self.dataset_root / category
        self.split_path = self.category_path / self.split
        self.ground_truth_path = self.category_path / "ground_truth"

        if not self.category_path.exists():
            raise FileNotFoundError(f"Missing category folder: {self.category_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"Missing split folder: {self.split_path}")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No images found in {self.split_path}")

    def _collect_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        defect_dirs = sorted(path for path in self.split_path.iterdir() if path.is_dir())

        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            is_anomalous = defect_type != "good"

            image_paths = sorted(
                path
                for path in defect_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                mask_path: Path | None = None
                if self.split == "test" and is_anomalous:
                    mask_path = self.ground_truth_path / defect_type / f"{image_path.stem}_mask.png"
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

                samples.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "defect_type": defect_type,
                        "label": int(is_anomalous),
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path: Path = sample["image_path"]
        mask_path: Path | None = sample["mask_path"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        if mask_path is None:
            mask_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = (self.mask_transform(mask) > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "defect_type": sample["defect_type"],
            "image_path": str(image_path),
        }
