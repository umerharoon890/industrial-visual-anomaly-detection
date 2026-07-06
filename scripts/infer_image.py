"""Run anomaly detection on one image from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.single_image_inference import IndustrialAnomalyInference
from src.models.category_classifier import CategoryPrototypeClassifier
from src.project_config import RESULTS_DIR, category_classifier_path, detector_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run industrial anomaly detection on one image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--category",
        default="bottle",
        help="MVTec category to use, or 'auto' to detect it first. Default: bottle.",
    )
    parser.add_argument("--memory-bank", default=None, help="Optional explicit memory-bank path.")
    parser.add_argument("--config", default=None, help="Optional explicit detector config path.")
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "cli_outputs"),
        help="Where result images and JSON should be saved.",
    )
    parser.add_argument("--prefix", default="sample", help="Prefix for saved output files.")
    parser.add_argument("--device", default=None, help="Use 'cuda' or 'cpu'. Default chooses automatically.")
    return parser.parse_args()


def resolve_detector_files(args: argparse.Namespace, device: torch.device) -> tuple[str, Path, Path, float | None]:
    if args.memory_bank and args.config:
        return "custom", Path(args.memory_bank), Path(args.config), None

    if args.category.lower() == "auto":
        classifier = CategoryPrototypeClassifier.from_saved_file(category_classifier_path(), device=device)
        category_result = classifier.predict_image(args.image)
        category = category_result["predicted_categories"][0]
        match_score = float(category_result["category_match_scores"][0].item())
    else:
        category = args.category
        match_score = None

    memory_bank_path, config_path = detector_paths(category)
    return category, memory_bank_path, config_path, match_score


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    category, memory_bank_path, config_path, match_score = resolve_detector_files(args, device)

    inference = IndustrialAnomalyInference(
        memory_bank_path=memory_bank_path,
        configuration_path=config_path,
        device=device,
    )
    prediction = inference.predict_image(args.image)
    saved_paths = inference.save_prediction_outputs(prediction, args.output_dir, args.prefix)

    result = {
        "image_path": str(args.image),
        "category": category,
        "category_match_score": match_score,
        "prediction": prediction["prediction_text"],
        "predicted_label": prediction["predicted_label"],
        "image_score": prediction["image_score"],
        "image_threshold": prediction["image_threshold"],
        "device": str(device),
        "saved_outputs": {name: str(path) for name, path in saved_paths.items()},
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json_path = output_dir / f"{args.prefix}_result.json"
    result_json_path.write_text(json.dumps(result, indent=4), encoding="utf-8")

    print("=" * 60)
    print("ANOMALY DETECTION RESULT")
    print("=" * 60)
    print(f"Image      : {args.image}")
    print(f"Category   : {category}")
    if match_score is not None:
        print(f"Match score: {match_score:.4f}")
    print(f"Prediction : {prediction['prediction_text']}")
    print(f"Score      : {prediction['image_score']:.6f}")
    print(f"Threshold  : {prediction['image_threshold']:.6f}")
    print(f"Device     : {device}")
    print("\nSaved outputs:")
    for name, path in saved_paths.items():
        print(f"{name:<18}: {path}")
    print(f"result_json       : {result_json_path}")


if __name__ == "__main__":
    main()
