# Portfolio Write-Up Outline

## 1. Problem Statement
Industrial quality inspection requires identifying defective products from visual images. Since defective examples are rare, the system should learn from mostly normal data.

## 2. Dataset
MVTec AD with all 15 categories. Each category contains normal training images and test images containing normal and defective samples.

## 3. Methods
- Autoencoder baseline
- ResNet-18 patch-feature detector
- Normal memory bank
- Top-1% patch anomaly scoring
- Pixel-level heatmap and binary mask generation
- Few-shot defect-type classification
- Automatic category detection using category prototypes

## 4. Key Results
Summarize image-level ROC-AUC, F1, recall, pixel ROC-AUC, and pixel AP across categories.

## 5. Robustness
Explain performance under brightness, contrast, noise, blur, and rotation.

## 6. Deployment
Discuss CLI inference and Streamlit web application.

## 7. Limitations
Discuss category-specific nature, false positives under distribution shift, and difficult categories.

## 8. Future Work
Augmentations, coreset memory banks, online deployment, and custom category creation.
