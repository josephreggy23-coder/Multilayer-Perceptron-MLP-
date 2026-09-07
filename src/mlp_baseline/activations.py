"""Numerically stable activation functions and their derivatives."""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Apply the rectified linear unit elementwise."""
    return np.maximum(x, 0.0)


def relu_gradient(x: np.ndarray) -> np.ndarray:
    """Return d ReLU(x) / dx elementwise."""
    return (x > 0.0).astype(float)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert class logits to probabilities without numerical overflow."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean multiclass cross-entropy for integer class labels."""
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(labels.size), labels])))

