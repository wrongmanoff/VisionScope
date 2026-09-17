"""Reusable, non-fabricating evaluation metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 for binary labels."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def segmentation_metrics(mask_true: np.ndarray, mask_pred: np.ndarray) -> dict[str, float]:
    """Return IoU and Dice for binary segmentation masks."""
    a = np.asarray(mask_true).astype(bool)
    b = np.asarray(mask_pred).astype(bool)
    if a.shape != b.shape:
        raise ValueError("Masks must have identical shapes.")
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    dice_den = a.sum() + b.sum()
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / dice_den) if dice_den else 1.0,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE and RMSE for continuous predictions."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Arrays must have identical shapes.")
    err = a - b
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }
