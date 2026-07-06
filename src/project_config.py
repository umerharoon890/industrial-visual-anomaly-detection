"""Shared paths and category names used across the project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def detector_paths(category: str) -> tuple[Path, Path]:
    """Return the memory-bank and JSON config paths for one category."""
    memory_bank = MODELS_DIR / f"resnet18_memory_bank_{category}.pt"
    config = MODELS_DIR / f"feature_detector_{category}_config.json"
    return memory_bank, config


def category_classifier_path() -> Path:
    """Path used by the prototype-based category recognizer."""
    return MODELS_DIR / "category_prototypes_resnet18.pt"
