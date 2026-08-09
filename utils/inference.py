"""
Small shared helpers for turning model probabilities into a final predicted
class, and for finding the best decision threshold for binary modalities.

Why this exists: a model can have excellent ROC-AUC (meaning it ranks the
two classes well) while still having mediocre accuracy at the default 0.5
threshold, if that threshold isn't actually where the two classes separate
best for this particular model/dataset. This is common when training uses
class-weighted loss (as we do, for X-ray's imbalance) -- the weighting
shifts the model's natural probability outputs, so 0.5 is no longer the
optimal cutoff. Tuning the threshold on VALIDATION data (never test data,
to avoid leaking test information into the decision rule) fixes this
without changing the model at all.

Only meaningful for binary (2-class) modalities. For 3+ classes, standard
argmax is used -- a single scalar threshold doesn't generalize cleanly to
multiclass decision rules.
"""

import numpy as np


def predict_class_index(probabilities: np.ndarray, threshold: float = 0.5) -> int:
    """
    probabilities: 1D array of per-class softmax probabilities for ONE sample.
    threshold: only used when len(probabilities) == 2. Compares the
        probability of class index 1 (by our convention, the "positive"/
        disease class, e.g. PNEUMONIA) against this threshold.
    """
    if len(probabilities) == 2:
        return 1 if probabilities[1] >= threshold else 0
    return int(np.argmax(probabilities))


def find_best_threshold(y_true: np.ndarray, y_prob_positive: np.ndarray) -> float:
    """
    Sweeps thresholds from 0.05 to 0.95 and returns the one that maximizes
    BALANCED accuracy: the average of sensitivity (recall on class 1) and
    specificity (recall on class 0).

    Balanced accuracy (rather than plain accuracy) is the right metric to
    optimize here because plain accuracy can be maximized by a threshold
    that just favors the majority class -- balanced accuracy forces the
    threshold to treat both classes' error rates as equally important,
    which is the appropriate framing for a diagnostic tool where missing
    a disease case (false negative) and a false alarm (false positive)
    are both costly, just in different ways.
    """
    best_threshold, best_score = 0.5, -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (y_prob_positive >= threshold).astype(int)

        # Sensitivity: of actual positives (y_true==1), fraction predicted positive
        positive_mask = (y_true == 1)
        sensitivity = preds[positive_mask].mean() if positive_mask.any() else 0.0

        # Specificity: of actual negatives (y_true==0), fraction predicted negative
        negative_mask = (y_true == 0)
        specificity = (1 - preds[negative_mask]).mean() if negative_mask.any() else 0.0

        balanced_acc = (sensitivity + specificity) / 2

        if balanced_acc > best_score:
            best_score = balanced_acc
            best_threshold = float(threshold)

    return best_threshold