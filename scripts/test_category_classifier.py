"""Quick smoke test for automatic category detection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.category_classifier import CategoryPrototypeClassifier
from src.project_config import DATASET_ROOT, category_classifier_path

TEST_IMAGES = {
    "bottle": DATASET_ROOT / "bottle" / "test" / "good" / "000.png",
    "cable": DATASET_ROOT / "cable" / "test" / "good" / "000.png",
    "screw": DATASET_ROOT / "screw" / "test" / "good" / "000.png",
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = CategoryPrototypeClassifier.from_saved_file(category_classifier_path(), device=device)

    print("Loaded category classifier")
    print("Device:", device)
    print("Categories:", ", ".join(classifier.category_names))
    print("Prototype shape:", tuple(classifier.prototypes.shape))
    print()

    for expected_category, image_path in TEST_IMAGES.items():
        result = classifier.predict_image(image_path)
        predicted_category = result["predicted_categories"][0]
        match_score = float(result["category_match_scores"][0].item())

        print("=" * 60)
        print(f"Image     : {image_path}")
        print(f"Expected  : {expected_category}")
        print(f"Predicted : {predicted_category}")
        print(f"Match score: {match_score:.4f}")


if __name__ == "__main__":
    main()
