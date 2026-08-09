"""
Builds train/val/test DataLoaders for a given modality ("xray", "ct", "mri"),
handling the folder-structure quirks of each dataset:

  - CT: train-split class folders have staging text appended to the class
    name (e.g. "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib") -> matched by
    prefix against the canonical class list.
  - X-ray: shipped "val" folder is only 16 images -> merged into the train
    pool, then a fresh validation split is carved out.
  - MRI: no "val" folder at all -> a validation split is carved out of train.

Design choice: rather than writing three separate dataset-loading functions
(one per modality), we implement ONE generic scanner driven entirely by
config.MODALITIES, so adding a 4th modality later requires zero new code here.
"""

import os
import random
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

import config
from utils.preprocessing import get_transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _resolve_class_folder(folder_name: str, classes: List[str], use_prefix: bool) -> str | None:
    """Map a folder name to a canonical class name, or None if it doesn't match any class."""
    if use_prefix:
        for cls in classes:
            if folder_name.startswith(cls):
                return cls
        return None
    else:
        return folder_name if folder_name in classes else None


def _scan_folder(root: str, classes: List[str], use_prefix: bool) -> List[Tuple[str, int]]:
    """
    Walk one split directory (e.g. .../train) and return a list of
    (filepath, label_index) pairs for every image found under a recognized
    class subfolder.
    """
    samples = []
    if root is None or not os.path.isdir(root):
        return samples

    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue

        matched_class = _resolve_class_folder(entry, classes, use_prefix)
        if matched_class is None:
            continue  # unrecognized folder -> skip rather than crash

        label = classes.index(matched_class)
        for fname in os.listdir(entry_path):
            if fname.lower().endswith(IMG_EXTENSIONS):
                samples.append((os.path.join(entry_path, fname), label))

    return samples


class MedicalImageDataset(Dataset):
    """Generic dataset over a list of (filepath, label) pairs."""

    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_datasets(modality: str, seed: int = None):
    """
    Returns (train_ds, val_ds, test_ds) for the given modality key
    ("xray" | "ct" | "mri"), applying the merge/carve logic described above.
    """
    seed = seed or config.SEED
    cfg = config.MODALITIES[modality]
    classes = cfg["classes"]
    use_prefix = bool(cfg.get("class_folder_prefixes"))

    train_samples = _scan_folder(cfg["split_dirs"]["train"], classes, use_prefix)
    val_samples = _scan_folder(cfg["split_dirs"].get("val"), classes, use_prefix)
    test_samples = _scan_folder(cfg["split_dirs"]["test"], classes, use_prefix)

    if not train_samples:
        raise RuntimeError(
            f"No training images found for modality '{modality}'. "
            f"Check config.MODALITIES['{modality}']['split_dirs']['train']."
        )

    # Merge an unreliable/too-small shipped val set into the train pool (X-ray case)
    if cfg.get("merge_val_into_train") and val_samples:
        train_samples = train_samples + val_samples
        val_samples = []

    # If there's still no val set (X-ray after merge, or MRI which never had one),
    # carve one out of the train pool deterministically.
    if not val_samples:
        rng = random.Random(seed)
        shuffled = train_samples[:]
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * config.VAL_SPLIT_FROM_TRAIN)
        val_samples = shuffled[:n_val]
        train_samples = shuffled[n_val:]

    train_tf = get_transforms(config.IMAGE_SIZE, train=True, strong=bool(cfg.get("strong_augmentation", False)))
    eval_tf = get_transforms(config.IMAGE_SIZE, train=False)

    train_ds = MedicalImageDataset(train_samples, train_tf)
    val_ds = MedicalImageDataset(val_samples, eval_tf)
    test_ds = MedicalImageDataset(test_samples, eval_tf)

    return train_ds, val_ds, test_ds


def build_dataloaders(modality: str):
    """Convenience wrapper: dataset objects -> ready-to-use DataLoaders."""
    train_ds, val_ds, test_ds = build_datasets(modality)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check you can run directly in PyCharm (right-click -> Run)
    # to confirm every modality's folders resolve correctly before training.
    for mod in config.MODALITIES:
        train_ds, val_ds, test_ds = build_datasets(mod)
        print(f"[{mod}] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
              f"classes={config.MODALITIES[mod]['classes']}")