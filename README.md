# Medical Image Disease Classifier (React + Vite)

A modern, responsive React web application rewritten from the original Dash/PyTorch pipeline for medical image disease classification and Grad-CAM visual explanations across X-Ray, CT, and MRI scans.

## Features

- **Multi-Modality Support:**
  - **Chest X-Ray:** Binary classification between `NORMAL` and `PNEUMONIA` using custom 4-block ConvNet (MedicalCNN) with calibrated decision threshold ($0.53$).
  - **Chest CT Scan:** 4-class pulmonary differential (`adenocarcinoma`, `large.cell.carcinoma`, `normal`, `squamous.cell.carcinoma`) based on ResNet18 transfer learning.
  - **Brain MRI:** 4-class cranial neoplasm differential (`glioma`, `meningioma`, `notumor`, `pituitary`) using deep MedicalCNN.
- **Grad-CAM Visual Explanations:**
  - Generates class activation maps directly from the final convolutional layer (`model.features_final`).
  - Standard JET colormap visualization with dynamic alpha opacity blending slider ($0.10$ to $0.90$, default $0.45$).
- **Model Performance Dashboard:**
  - Test set accuracy, macro F1, and multi-class ROC-AUC (OVR).
  - Interactive Confusion Matrix heatmaps with sample distributions and percentage breakdowns.
  - Per-class precision, recall, F1-score, and support metrics.
  - Architecture and training hyperparameters specifications.
- **Diagnostic Upload & Clinical Presets:**
  - Drag-and-drop or file upload for arbitrary PNG, JPG, and DICOM export scans.
  - 1-click verified clinical reference cases for instant evaluation.

## Tech Stack

- **Framework:** React 19 + TypeScript + Vite
- **Styling:** Tailwind CSS
- **Icons:** Lucide React
- **Interpretability:** Client-side Grad-CAM feature attribution with JET colormap synthesis

## Development

```bash
# Install dependencies
npm install

# Start development server on port 3000
npm run dev

# Production build
npm run build
```
