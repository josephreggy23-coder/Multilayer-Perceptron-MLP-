"""Transparent multiclass evaluation statistics implemented with NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    """Count actual-class rows against predicted-class columns."""
    _validate_class_indices(labels, classes, "labels")
    _validate_class_indices(predictions, classes, "predictions")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have identical shapes")
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def classification_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Return accuracy, per-class scores, confusion matrix, and calibration data."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    classes = probabilities.shape[1]
    _validate_class_indices(labels, classes, "labels")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("labels and probabilities must contain the same number of rows")

    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(labels, predictions, classes)
    supports = matrix.sum(axis=1)
    true_positives = np.diag(matrix)
    predicted_totals = matrix.sum(axis=0)
    precision = _safe_divide(true_positives, predicted_totals)
    recall = _safe_divide(true_positives, supports)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = float(np.mean(predictions == labels))
    confidence = np.max(probabilities, axis=1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "mean_confidence": float(np.mean(confidence)),
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {
                "class": int(class_index),
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(supports[class_index]),
            }
            for class_index in range(classes)
        ],
    }


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Estimate confidence calibration error using equally spaced bins."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correctness = predictions == labels
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        in_bin = (confidences >= lower) & ((confidences < upper) if upper < 1.0 else (confidences <= upper))
        if np.any(in_bin):
            error += float(np.mean(in_bin) * abs(np.mean(correctness[in_bin]) - np.mean(confidences[in_bin])))
    return error


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def _validate_class_indices(values: np.ndarray, classes: int, name: str) -> None:
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= classes):
        raise ValueError(f"{name} must contain values from zero through {classes - 1}")
