"""
Trains one CNN per modality ("xray", "ct", "mri") using the shared
MedicalCNN architecture. Run this file directly in PyCharm; it trains all
three modalities in sequence and saves the best checkpoint (by validation
accuracy) for each to config.MODALITIES[modality]['checkpoint'].

Handles class imbalance (X-ray has ~3x more PNEUMONIA than NORMAL) via
class-weighted CrossEntropyLoss, computed automatically from the training
set's actual class counts -- no hardcoded weights, so this also works
correctly for CT/MRI even though they're roughly balanced.
"""

import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from models.cnn_model import build_model
from training.dataset import build_dataloaders, build_datasets
from utils.inference import find_best_threshold


def compute_class_weights(train_ds, num_classes) -> torch.Tensor:
    """
    Inverse-frequency class weights: rarer classes get a higher weight so
    the loss doesn't let the model just predict the majority class.
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = Counter(label for _, label in train_ds.samples)
    total = len(train_ds.samples)
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Shared loop body for both training and validation/test passes."""
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)  # restore default
    return total_loss / total, correct / total


def train_modality(modality: str, epochs: int = None):
    print(f"\n{'='*60}\nTraining modality: {modality}\n{'='*60}")

    cfg = config.MODALITIES[modality]
    num_classes = len(cfg["classes"])
    epochs = epochs or cfg.get("epochs") or config.EPOCHS

    train_ds, val_ds, test_ds = build_datasets(modality)
    train_loader, val_loader, test_loader = build_dataloaders(modality)

    print(f"Classes: {cfg['classes']}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    class_weights = compute_class_weights(train_ds, num_classes).to(config.DEVICE)
    print(f"Class weights (inverse frequency): {class_weights.tolist()}")

    use_pretrained = bool(cfg.get("use_pretrained", False))
    model = build_model(num_classes=num_classes, device=config.DEVICE, pretrained=use_pretrained)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if use_pretrained:
        # Differential learning rates: the newly-unfrozen backbone layers
        # (layer2/3/4, carrying pretrained ImageNet weights) get a much
        # lower LR so fine-tuning nudges them gently toward CT-specific
        # features rather than overwriting the useful pretrained weights.
        # The freshly-initialized classifier head has no pretrained
        # knowledge to protect, so it trains at the normal (higher) LR.
        backbone_params = [p for name, p in model.named_parameters()
                            if p.requires_grad and "backbone.fc" not in name]
        head_params = [p for name, p in model.named_parameters()
                        if p.requires_grad and "backbone.fc" in name]
        optimizer = Adam([
            {"params": backbone_params, "lr": config.TRANSFER_LEARNING_LR},
            {"params": head_params, "lr": config.LEARNING_RATE},
        ])
        print(f"Architecture: TransferResNetCNN (ResNet18, fine-tuning layer2-4), "
              f"backbone_lr={config.TRANSFER_LEARNING_LR}, head_lr={config.LEARNING_RATE}")
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(trainable_params, lr=config.LEARNING_RATE)
        print(f"Architecture: MedicalCNN (from scratch), lr={config.LEARNING_RATE}")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = config.TRANSFER_EARLY_STOP_PATIENCE if use_pretrained else config.EARLY_STOP_PATIENCE
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, config.DEVICE, train=False)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        # Save checkpoint only when validation accuracy improves -- this is
        # what "best model" means here, not just the final epoch's weights.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": cfg["classes"],
                "val_acc": val_acc,
                "epoch": epoch,
            }, cfg["checkpoint"])
            print(f"  -> New best val_acc={val_acc:.4f}, checkpoint saved to {cfg['checkpoint']}")

        # Early stopping: track val_loss (not val_acc) since loss is a
        # smoother signal -- accuracy can plateau/tie for several epochs
        # while the model is still genuinely improving or degrading underneath.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"  -> Early stopping: val_loss hasn't improved in {early_stop_patience} epochs.")
                break

    print(f"\nFinished training '{modality}'. Best val_acc={best_val_acc:.4f}")

    # --- Decision threshold calibration (binary modalities only) ---
    # Reload the BEST checkpoint (not necessarily the in-memory model's
    # current weights, since the last epoch trained isn't always the best
    # one) and, for binary modalities, find the optimal classification
    # threshold on the validation set. This is what fixes the "high ROC-AUC
    # but mediocre accuracy" symptom: the model's probabilities are fine,
    # the default 0.5 cutoff just wasn't the best place to draw the line.
    decision_threshold = 0.5
    if num_classes == 2:
        print("Calibrating decision threshold on validation set...")
        best_checkpoint = torch.load(cfg["checkpoint"], map_location=config.DEVICE)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.eval()

        val_probs, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                probs = torch.softmax(model(images), dim=1)
                val_probs.extend(probs[:, 1].cpu().numpy().tolist())
                val_labels.extend(labels.numpy().tolist())

        decision_threshold = find_best_threshold(np.array(val_labels), np.array(val_probs))
        print(f"Best decision threshold: {decision_threshold:.2f} (default would be 0.50)")

        # Re-save the checkpoint with the calibrated threshold included.
        best_checkpoint["decision_threshold"] = decision_threshold
        torch.save(best_checkpoint, cfg["checkpoint"])
    else:
        # Multiclass: no single threshold applies; store 0.5 for a
        # consistent checkpoint schema across all modalities (unused downstream).
        checkpoint = torch.load(cfg["checkpoint"], map_location=config.DEVICE)
        checkpoint["decision_threshold"] = decision_threshold
        torch.save(checkpoint, cfg["checkpoint"])

    return history


if __name__ == "__main__":
    # Trains all three modalities one after another. On CPU this can take a
    # while -- feel free to comment out modalities you don't want to
    # (re)train right now, e.g. run just ["xray"] first to sanity check.
    for modality_name in config.MODALITIES:
        train_modality(modality_name)