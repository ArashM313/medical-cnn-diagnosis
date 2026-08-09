"""
Recalibrates the decision threshold for an already-trained binary modality
WITHOUT retraining -- just reloads the existing checkpoint, runs it on the
validation set, and finds/saves the best threshold. Fast (a few val-set
forward passes), unlike train.py which retrains from scratch.

Use this when a modality's architecture/data hasn't changed but you want to
(re)tune its threshold -- e.g. X-ray right now, after we added threshold
calibration to the pipeline but without needing to redo the full ~1 hour
training run that already produced a good model.
"""

import config
from training.train import calibrate_threshold_for_modality

if __name__ == "__main__":
    # Edit this list if you want to recalibrate other binary modalities later.
    MODALITIES_TO_RECALIBRATE = ["xray"]

    for modality_name in MODALITIES_TO_RECALIBRATE:
        cfg = config.MODALITIES[modality_name]
        if len(cfg["classes"]) != 2:
            print(f"[{modality_name}] Skipping -- threshold calibration only applies to binary modalities.")
            continue
        calibrate_threshold_for_modality(modality_name)