# Industrial Visual Anomaly Detection

This is an end-to-end industrial anomaly detection project using the MVTec AD dataset.

The project started with a simple autoencoder baseline and then moved to a stronger feature-memory approach using a frozen ResNet-18 backbone. The final workflow supports all 15 MVTec AD categories, generates heatmaps/masks, includes few-shot defect-type experiments, and has both CLI and Streamlit inference.

## What it can do

Given an inspection image, the system can:

- estimate the closest MVTec product category automatically,
- classify the image as normal or defective,
- generate an anomaly heatmap,
- create a predicted binary defect mask,
- show visual overlays for interpretation,
- run from notebooks, the command line, or a Streamlit app.

## Why this approach

In real inspection tasks, normal samples are usually easier to collect than defective samples. Because of that, the main detector is built around normal-reference learning instead of standard supervised defect classification.

For each product category, the system stores patch-level ResNet features from normal training images. During inference, image patches are compared against this normal memory bank. Patches that are far from the learned normal patterns become the anomaly heatmap.

This makes the method useful when there are few or no labelled defect examples.

## Dataset

The project uses the MVTec Anomaly Detection dataset with 15 categories:

```text
bottle, cable, capsule, carpet, grid, hazelnut, leather,
metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
```

The dataset is not included in this repository. Download it separately and place it here:

```text
data/raw/mvtec_ad/
```

Expected structure:

```text
data/raw/mvtec_ad/
  bottle/
    train/
    test/
    ground_truth/
  cable/
    train/
    test/
    ground_truth/
  ...
```

## Method summary

### 1. Autoencoder baseline

A convolutional autoencoder was trained on normal images. It detects defects using reconstruction error. This was useful as a baseline, but it struggled with subtle defects.

### 2. ResNet patch-feature detector

The final detector uses a frozen pretrained ResNet-18 model:

1. Extract spatial features from normal training images.
2. Build a compact normal memory bank for each category.
3. Compare each test patch against the memory bank.
4. Use the top 1% most anomalous patches for image-level scoring.
5. Resize the patch-distance map to create a full-resolution anomaly heatmap.

### 3. Pixel-level localization

Category-specific thresholds are calculated from normal validation images. The continuous heatmap is thresholded and post-processed to produce a predicted defect mask.

### 4. Few-shot defect-type classification

After anomaly detection, a small prototype-based classifier is used to test defect-type classification using 1-shot, 5-shot, and 10-shot support examples.

### 5. Robustness tests

The detector was also tested under brightness changes, contrast changes, Gaussian noise, Gaussian blur, and small rotations.

## Selected results

### Image-level detection

| Category | ROC-AUC | Average Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| bottle | 0.9992 | 0.9998 | 1.0000 | 0.9844 |
| cable | 0.9314 | 0.9626 | 0.7717 | 0.8554 |
| carpet | 1.0000 | 1.0000 | 1.0000 | 0.9622 |
| leather | 1.0000 | 1.0000 | 1.0000 | 0.9787 |
| screw | 0.5802 | 0.7983 | 0.1008 | 0.1805 |

### Pixel-level localization

| Category | Pixel ROC-AUC | Pixel Average Precision |
|---|---:|---:|
| bottle | 0.9797 | 0.7506 |
| cable | 0.8850 | 0.3994 |
| carpet | 0.9795 | 0.6218 |
| leather | 0.9958 | 0.5418 |
| screw | 0.9727 | 0.0996 |
| transistor | 0.7367 | 0.3139 |

The detector performs strongly on several categories such as bottle, carpet, leather, wood, grid, and hazelnut. Screw, transistor, capsule, and cable are more difficult. I kept these weaker results because they show the actual limitations of the approach.

## Project structure

```text
app/
  streamlit_app.py

notebooks/
  01_environment_setup.ipynb
  02_dataset_exploration.ipynb
  03_autoencoder_baseline.ipynb
  04_feature_based_anomaly_detection.ipynb
  05_few_shot_anomaly_detection.ipynb
  06_multicategory_and_robustness.ipynb
  07_final_demo_and_inference.ipynb

scripts/
  build_category_prototypes.py
  infer_image.py
  test_category_classifier.py

src/
  data/
    mvtec_dataset.py
  inference/
    single_image_inference.py
  models/
    autoencoder.py
    category_classifier.py
    feature_anomaly_detector.py
    resnet_feature_extractor.py
  project_config.py
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

For CUDA support, install the PyTorch build that matches your GPU/driver from the official PyTorch installation page.

## Build the category prototype classifier

After the detector memory banks and JSON configs have been created, build the auto-category prototype file:

```powershell
python scripts\build_category_prototypes.py
```

Quick test:

```powershell
python scripts\test_category_classifier.py
```

## CLI inference

Use a known category:

```powershell
python scripts\infer_image.py `
  --image "data\raw\mvtec_ad\bottle\test\broken_large\000.png" `
  --category bottle `
  --prefix bottle_test
```

Or let the CLI detect the category first:

```powershell
python scripts\infer_image.py `
  --image "data\raw\mvtec_ad\bottle\test\broken_large\000.png" `
  --category auto `
  --prefix auto_test
```

Outputs are saved under:

```text
results/cli_outputs/
```

## Streamlit app

```powershell
streamlit run app\streamlit_app.py
```

The app supports automatic category detection, anomaly scoring, normal/defective prediction, heatmaps, mask overlays, and result downloads.

## Important limitation

This is not a universal anomaly detector for every object. It compares an image against normal examples from known categories.

For a new object type, the correct workflow is:

1. collect normal images of that object,
2. build a normal memory bank,
3. set thresholds using normal validation images,
4. run inference on new inspection images.

Defective examples are useful for evaluation, but they are not required for the main anomaly detector.

## Future work

- Add stronger augmentation for robustness.
- Try coreset memory-bank selection instead of random sampling.
- Compare different ResNet layers and backbones.
- Add an interface for creating custom categories from normal reference images.
- Deploy the Streamlit app online.
