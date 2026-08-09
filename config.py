import os
import torch

# --- Base paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Root of your local dataset collection ---
# Change this ONE line if you ever move the data_sets folder.
DATA_ROOT = r"C:\Users\asus\OneDrive\Documents\Python AI-ML\medical_image_classifier\data_sets"

# --- Per-modality configuration ---
# Each modality defines:
#   - split_dirs: explicit train/val/test folder paths (these datasets already
#     come pre-split, so we do NOT do our own random_split like the old notebook did)
#   - classes: fixed class order -> stable label indices across train/eval/dashboard
#   - class_folder_prefixes: for datasets where folder names differ slightly
#     across splits (e.g. ct_scan01/train has staging info appended), we match
#     by PREFIX instead of exact name.
#   - checkpoint: where the trained weights for this modality are saved
MODALITIES = {
    "xray": {
        "split_dirs": {
            # NOTE: the shipped "val" folder only has 16 images total -> too small
            # to trust for model selection. We fold it into "train" and instead
            # carve a proper validation split out of the combined train pool
            # (see training/dataset.py -> build_datasets). Listed here anyway so
            # the loader knows to merge it in.
            "train": os.path.join(DATA_ROOT, "chest_xray", "chest_xray", "train"),
            "val":   os.path.join(DATA_ROOT, "chest_xray", "chest_xray", "val"),
            "test":  os.path.join(DATA_ROOT, "chest_xray", "chest_xray", "test"),
        },
        "merge_val_into_train": True,   # val folder too small to use standalone
        "classes": ["NORMAL", "PNEUMONIA"],
        "class_folder_prefixes": None,  # exact folder names match class names
        "checkpoint": os.path.join(CHECKPOINT_DIR, "xray_cnn.pth"),
    },
    "ct": {
        "split_dirs": {
            "train": os.path.join(DATA_ROOT, "ct_scan01", "train"),
            "val":   os.path.join(DATA_ROOT, "ct_scan01", "valid"),
            "test":  os.path.join(DATA_ROOT, "ct_scan01", "test"),
        },
        "classes": ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"],
        "class_folder_prefixes": True,  # train folders have extra staging text appended
        "merge_val_into_train": False,  # ct_scan01's "valid" folder is a normal, adequately-sized split
        "use_pretrained": True,  # small dataset (613 train images) -> use ResNet18 transfer learning
        "strong_augmentation": True,  # extra RandomResizedCrop -- squeeze more variety from few images
        "epochs": 40,  # slower effective LR on backbone -> allow more epochs than the other modalities
        "checkpoint": os.path.join(CHECKPOINT_DIR, "ct_cnn.pth"),
    },
    "mri": {
        "split_dirs": {
            "train": os.path.join(DATA_ROOT, "mri", "Training"),
            "val":   None,  # no separate val folder shipped -> carved out of train below
            "test":  os.path.join(DATA_ROOT, "mri", "Testing"),
        },
        "merge_val_into_train": False,  # nothing to merge; val is just None
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "class_folder_prefixes": None,
        "checkpoint": os.path.join(CHECKPOINT_DIR, "mri_cnn.pth"),
    },
}

# --- Shared training hyperparameters ---
IMAGE_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRANSFER_LEARNING_LR = 1e-4  # lower LR for fine-tuning pretrained backbones (used when use_pretrained=True)
EPOCHS = 30
EARLY_STOP_PATIENCE = 8            # default patience (val_loss plateau) for from-scratch models
TRANSFER_EARLY_STOP_PATIENCE = 14  # more patience for fine-tuned models -- the low backbone LR
                                    # means genuine improvement can be slower/noisier epoch-to-epoch
VAL_SPLIT_FROM_TRAIN = 0.15  # used only when a modality has no dedicated val folder (MRI)
SEED = 42