"""Build the prototype file used for automatic MVTec category detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mvtec_dataset import MVTecDataset
from src.models.resnet_feature_extractor import ResNet18PatchExtractor
from src.project_config import DATASET_ROOT, MVTEC_CATEGORIES, RESULTS_DIR, category_classifier_path

IMAGE_SIZE = 256
BATCH_SIZE = 8
RANDOM_SEED = 42
TEMPERATURE = 0.05
SUMMARY_PATH = RESULTS_DIR / "metrics" / "category_prototype_summary.csv"


def extract_category_embeddings(
    feature_extractor: torch.nn.Module,
    category: str,
    device: torch.device,
) -> torch.Tensor:
    dataset = MVTecDataset(
        dataset_root=DATASET_ROOT,
        category=category,
        split="train",
        image_size=IMAGE_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings = []
    feature_extractor.eval()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Building prototype: {category}"):
            images = batch["image"].to(device, non_blocking=True)
            feature_maps = feature_extractor(images)
            image_embeddings = F.normalize(feature_maps.mean(dim=(2, 3)), p=2, dim=1)
            embeddings.append(image_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = category_classifier_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Output file  : {output_path}")
    print(f"Device       : {device}")

    feature_extractor = ResNet18PatchExtractor().to(device).eval()
    prototypes = []
    summary_rows = []

    for category in MVTEC_CATEGORIES:
        embeddings = extract_category_embeddings(feature_extractor, category, device)
        prototype = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
        prototypes.append(prototype)

        summary_rows.append(
            {
                "category": category,
                "training_images": len(embeddings),
                "embedding_dimension": int(embeddings.shape[1]),
                "prototype_norm": float(prototype.norm(dim=1).item()),
            }
        )

    prototype_tensor = torch.cat(prototypes, dim=0).contiguous()
    torch.save(
        {
            "category_names": MVTEC_CATEGORIES,
            "prototypes": prototype_tensor,
            "image_size": IMAGE_SIZE,
            "feature_dimension": int(prototype_tensor.shape[1]),
            "temperature": TEMPERATURE,
            "random_seed": RANDOM_SEED,
        },
        output_path,
    )

    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    print(f"Saved prototype classifier: {output_path}")
    print(f"Saved summary CSV        : {SUMMARY_PATH}")
    print(f"Prototype tensor shape   : {tuple(prototype_tensor.shape)}")


if __name__ == "__main__":
    main()
