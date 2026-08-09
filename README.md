# Medical Image Disease Classifier

A CNN-based diagnostic tool that classifies diseases from X-ray, CT, and MRI
scans, with Grad-CAM visual explanations and an interactive Dash dashboard.

## Features
- Upload and preprocess medical images
- CNN-based disease classification (from-scratch CNN for X-ray/MRI, ResNet18
  transfer learning for CT)
- Support for X-ray, MRI, and CT scans, each with its own trained model
- Accuracy, precision, recall, F1, confusion matrix, and ROC-AUC per modality
- Grad-CAM heatmaps showing which regions drove each prediction
- Interactive Dash web dashboard for uploading scans and viewing results

## Project structure
```
config.py              Central configuration (paths, hyperparameters, per-modality settings)
utils/                  Shared preprocessing and inference helpers
models/                 CNN architectures (MedicalCNN, TransferResNetCNN)
training/               Dataset loading, training, evaluation, threshold calibration
interpretability/       Grad-CAM implementation
dashboard/              Dash web app (layout, callbacks, entry point)
checkpoints/            Trained model weights and evaluation metrics (weights gitignored)
```

## Setup
1. Clone the repo and open it in PyCharm.
2. Mark the project root as **Sources Root** (right-click → Mark Directory as → Sources Root).
3. Install dependencies: `pip install -r requirements.txt`
4. Download the datasets (see Datasets section below) and place them under `data_sets/`
   matching the paths in `config.py`.

## Datasets
This project uses three public Kaggle datasets (not included in this repo due to size):
- Chest X-Ray (Pneumonia): [Kaggle link]
- Chest CT-Scan images: [Kaggle link]
- Brain Tumor MRI Dataset: [Kaggle link]

## Running the pipeline
1. `training/dataset.py` — sanity-check that dataset paths resolve correctly
2. `models/cnn_model.py` — sanity-check both architectures build correctly
3. `training/train.py` — train the CNN(s); edit `MODALITIES_TO_TRAIN` to control which
4. `training/evaluate.py` — evaluate on the held-out test set, saving metrics JSON
5. `dashboard/app.py` — launch the interactive dashboard at `http://127.0.0.1:8050`

## Notes
- Trained model weights are excluded from version control (see `.gitignore`) since
  they're large and fully reproducible by running `training/train.py`.
- X-ray uses a calibrated decision threshold (see `training/recalibrate.py`) rather
  than a naive 0.5 cutoff, tuned on the validation set for balanced accuracy.
