"""
Evaluates a trained modality's checkpoint on its held-out TEST set (never
seen during training or validation-based checkpoint selection) and saves a
metrics report as JSON. The dashboard (Step 6) reads these JSON files
directly rather than recomputing evaluation on every page load, which would
be slow and pointless since the model doesn't change between dashboard runs.

Run this after training/train.py has produced a checkpoint for a modality.
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)

import config
from models.cnn_model import build_model
from training.dataset import build_dataloaders
from utils.inference import predict_class_index


def evaluate_modality(modality: str) -> dict:
    cfg = config.MODALITIES[modality]

    if not os.path.exists(cfg["checkpoint"]):
        raise FileNotFoundError(
            f"No checkpoint for '{modality}' at {cfg['checkpoint']}. "
            f"Run training/train.py for this modality first."
        )

    checkpoint = torch.load(cfg["checkpoint"], map_location=config.DEVICE)
    classes = checkpoint["classes"]  # trust the checkpoint's own class list, not config
    num_classes = len(classes)
    decision_threshold = checkpoint.get("decision_threshold", 0.5)

    use_pretrained = bool(cfg.get("use_pretrained", False))
    model = build_model(num_classes=num_classes, device=config.DEVICE, pretrained=use_pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = build_dataloaders(modality)

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Apply the calibrated decision threshold for binary modalities, instead
    # of the naive argmax used for the running preds above. For multiclass,
    # argmax (already computed) is correct and unchanged.
    if num_classes == 2:
        all_preds = np.array([predict_class_index(p, decision_threshold) for p in all_probs])

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # ROC-AUC: binary case needs the positive class's probability column;
    # multiclass uses one-vs-rest averaging. Wrapped in try/except because
    # it can fail if the test set happens to be missing a class entirely.
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        roc_auc = None

    per_class = {
        classes[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }

    report = {
        "modality": modality,
        "classes": classes,
        "test_accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1)),
        "roc_auc_ovr": float(roc_auc) if roc_auc is not None else None,
        "decision_threshold": float(decision_threshold),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "checkpoint_val_acc": float(checkpoint.get("val_acc", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_set_size": int(len(all_labels)),
    }

    out_path = os.path.join(config.CHECKPOINT_DIR, f"{modality}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{modality}] test_accuracy={accuracy:.4f}  macro_f1={report['macro_f1']:.4f}  "
          f"roc_auc={report['roc_auc_ovr']}  -> saved {out_path}")

    return report


if __name__ == "__main__":
    for modality_name, cfg in config.MODALITIES.items():
        if os.path.exists(cfg["checkpoint"]):
            evaluate_modality(modality_name)
        else:
            print(f"[{modality_name}] No checkpoint yet -- skipping evaluation. "
                  f"Run training/train.py first.")