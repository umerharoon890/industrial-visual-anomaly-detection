"""Streamlit demo for automatic industrial anomaly detection."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.single_image_inference import IndustrialAnomalyInference
from src.models.category_classifier import CategoryPrototypeClassifier
from src.project_config import DATASET_ROOT, MVTEC_CATEGORIES, category_classifier_path, detector_paths

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@st.cache_resource
def load_category_classifier() -> CategoryPrototypeClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CategoryPrototypeClassifier.from_saved_file(category_classifier_path(), device=device)


@st.cache_resource
def load_anomaly_detector(category: str) -> tuple[IndustrialAnomalyInference, dict]:
    memory_bank_path, config_path = detector_paths(category)
    inference = IndustrialAnomalyInference(memory_bank_path, config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return inference, config


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def dataset_samples() -> dict[str, Path]:
    samples: dict[str, Path] = {}

    for category in MVTEC_CATEGORIES:
        test_root = DATASET_ROOT / category / "test"
        if not test_root.exists():
            continue

        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            images = sorted(
                path for path in defect_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if images:
                samples[f"{category} / {defect_dir.name} / {images[0].name}"] = images[0]

    return samples


def colored_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    low, high = np.percentile(anomaly_map, [1, 99])
    normalized = np.clip((anomaly_map - low) / (high - low + 1e-8), 0, 1)
    return plt.get_cmap("inferno")(normalized)[..., :3]


def to_png_bytes(image_array: np.ndarray) -> bytes:
    image = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def top_category_rows(category_result: dict, top_k: int = 5) -> list[dict]:
    scores = category_result["match_scores"][0]
    similarities = category_result["similarities"][0]
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

    rows = []
    for rank, (score, index) in enumerate(zip(top_scores, top_indices), start=1):
        index = int(index.item())
        rows.append(
            {
                "rank": rank,
                "category": MVTEC_CATEGORIES[index],
                "match_score": round(float(score.item()), 4),
                "cosine_similarity": round(float(similarities[index].item()), 4),
            }
        )
    return rows


st.set_page_config(
    page_title="Industrial Anomaly Detector",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #172033 0%, #111827 55%, #0b1020 100%);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; font-size: 2.5rem; margin-bottom: 0.3rem; }
    .hero p { color: #cbd5e1; font-size: 1.05rem; max-width: 950px; }
    .badge {
        padding: 0.9rem;
        border-radius: 14px;
        font-weight: 800;
        text-align: center;
        font-size: 1.2rem;
    }
    .badge-category { background: rgba(59,130,246,0.16); color: #93c5fd; border: 1px solid rgba(147,197,253,0.28); }
    .badge-normal { background: rgba(34,197,94,0.16); color: #4ade80; border: 1px solid rgba(74,222,128,0.28); }
    .badge-defect { background: rgba(239,68,68,0.16); color: #f87171; border: 1px solid rgba(248,113,113,0.28); }
    .note { color: #94a3b8; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Industrial Visual Anomaly Detection</h1>
        <p>
            Upload an inspection image. The app first finds the closest MVTec category,
            then loads that category's anomaly detector and produces the score, heatmap,
            and predicted defect mask.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model status")
    try:
        category_classifier = load_category_classifier()
        st.success("Category recognizer loaded")
        st.write(f"Device: `{category_classifier.prototypes.device}`")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.caption("Supported categories")
    st.write(", ".join(MVTEC_CATEGORIES))
    st.caption("The category match score ranks known categories; it is not a calibrated probability.")

st.info(
    "This app assumes the input image belongs to one of the 15 MVTec AD categories. "
    "Other objects will still be assigned to the closest known category."
)

input_mode = st.radio("Input source", ["Upload image", "Use dataset sample"], horizontal=True)
image_path: Path | None = None
input_caption = ""

if input_mode == "Upload image":
    uploaded_file = st.file_uploader("Upload an inspection image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        image_path = save_upload(uploaded_file)
        input_caption = uploaded_file.name
else:
    samples = dataset_samples()
    if not samples:
        st.warning("No local MVTec samples found. Use image upload instead.")
    else:
        selected_sample = st.selectbox("Sample image", list(samples.keys()))
        image_path = samples[selected_sample]
        input_caption = selected_sample

if image_path is None:
    st.warning("Upload or select an image to run inference.")
    st.stop()

input_image = Image.open(image_path).convert("RGB")

with st.spinner("Detecting category..."):
    category_result = category_classifier.predict_image(image_path)

predicted_category = category_result["predicted_categories"][0]
category_match_score = float(category_result["category_match_scores"][0].item())
category_rows = top_category_rows(category_result)

memory_bank_path, config_path = detector_paths(predicted_category)
if not memory_bank_path.exists() or not config_path.exists():
    st.error(
        f"Predicted category `{predicted_category}`, but its detector files are missing. "
        "Run Notebook 6 to build all memory banks and configs."
    )
    st.stop()

with st.spinner(f"Running {predicted_category} anomaly detector..."):
    inference, config = load_anomaly_detector(predicted_category)
    prediction = inference.predict_image(image_path)

prediction_text = prediction["prediction_text"]
image_score = prediction["image_score"]
threshold = prediction["image_threshold"]
score_margin = image_score - threshold
heatmap = colored_heatmap(prediction["anomaly_map"])

st.subheader("Result")
category_col, result_col, score_col, threshold_col, margin_col = st.columns(5)

with category_col:
    st.markdown(f'<div class="badge badge-category">{predicted_category}</div>', unsafe_allow_html=True)
    st.caption(f"Match score: {category_match_score:.4f}")

with result_col:
    badge_class = "badge-defect" if prediction_text == "defective" else "badge-normal"
    st.markdown(f'<div class="badge {badge_class}">{prediction_text.upper()}</div>', unsafe_allow_html=True)

with score_col:
    st.metric("Anomaly score", f"{image_score:.6f}")
with threshold_col:
    st.metric("Threshold", f"{threshold:.6f}")
with margin_col:
    st.metric("Margin", f"{score_margin:+.6f}")

st.markdown(
    f'<div class="note">Input: <b>{input_caption}</b> | Auto category: <b>{predicted_category}</b></div>',
    unsafe_allow_html=True,
)
st.divider()

overview_tab, visual_tab, category_tab, download_tab, detail_tab = st.tabs(
    ["Overview", "Visuals", "Category detection", "Downloads", "Details"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.image(input_image, caption="Original image", use_container_width=True)
    with right:
        st.image(prediction["mask_overlay"], caption="Predicted mask overlay", use_container_width=True)

    if prediction_text == "defective":
        st.error("The anomaly score is above the learned category threshold.")
    else:
        st.success("The anomaly score is below the learned category threshold.")

with visual_tab:
    cols = st.columns(4)
    with cols[0]:
        st.image(prediction["resized_image"], caption="Resized input", use_container_width=True)
    with cols[1]:
        st.image(heatmap, caption="Anomaly heatmap", use_container_width=True)
    with cols[2]:
        st.image(prediction["heatmap_overlay"], caption="Heatmap overlay", use_container_width=True)
    with cols[3]:
        st.image(prediction["mask_overlay"], caption="Mask overlay", use_container_width=True)

with category_tab:
    st.subheader("Top category matches")
    st.dataframe(category_rows, use_container_width=True)
    st.write(
        "The top category decides which anomaly detector is used. The score ranks the known "
        "MVTec categories, but should not be interpreted as a real-world confidence value."
    )

with download_tab:
    cols = st.columns(4)
    with cols[0]:
        st.download_button(
            "Resized input",
            data=to_png_bytes(prediction["resized_image"]),
            file_name=f"{predicted_category}_input.png",
            mime="image/png",
        )
    with cols[1]:
        st.download_button(
            "Heatmap",
            data=to_png_bytes(heatmap),
            file_name=f"{predicted_category}_heatmap.png",
            mime="image/png",
        )
    with cols[2]:
        st.download_button(
            "Heatmap overlay",
            data=to_png_bytes(prediction["heatmap_overlay"]),
            file_name=f"{predicted_category}_heatmap_overlay.png",
            mime="image/png",
        )
    with cols[3]:
        st.download_button(
            "Mask overlay",
            data=to_png_bytes(prediction["mask_overlay"]),
            file_name=f"{predicted_category}_mask_overlay.png",
            mime="image/png",
        )

    result_json = {
        "auto_detected_category": predicted_category,
        "category_match_score": category_match_score,
        "top_category_matches": category_rows,
        "prediction": prediction_text,
        "image_score": image_score,
        "image_threshold": threshold,
        "score_margin": score_margin,
        "device": str(inference.device),
    }
    st.download_button(
        "JSON result",
        data=json.dumps(result_json, indent=4),
        file_name=f"{predicted_category}_result.json",
        mime="application/json",
    )

with detail_tab:
    st.subheader("Detector configuration")
    st.json(
        {
            "category": predicted_category,
            "backbone": config.get("backbone"),
            "feature_layer": config.get("feature_layer"),
            "image_size": config.get("image_size"),
            "memory_bank_size": config.get("memory_bank_size"),
            "image_score_threshold": config.get("image_score_threshold"),
            "pixel_threshold": config.get("pixel_threshold"),
            "postprocessing": config.get("postprocessing"),
        }
    )
    st.subheader("Saved metrics")
    st.json(
        {
            "image_metrics": config.get("final_image_metrics", {}),
            "pixel_metrics": config.get("final_pixel_metrics", {}),
        }
    )
