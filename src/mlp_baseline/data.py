"""Synthetic datasets and preprocessing utilities for MLP experiments."""

from __future__ import annotations

import numpy as np


def make_spiral(samples_per_class: int = 150, classes: int = 3, noise: float = 0.20, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create a 2D, non-linearly separable spiral classification dataset."""
    if samples_per_class <= 0 or classes < 2 or noise < 0:
        raise ValueError("samples_per_class > 0, classes >= 2, and noise >= 0 are required")

    generator = np.random.default_rng(seed)
    total = samples_per_class * classes
    features = np.empty((total, 2), dtype=float)
    labels = np.empty(total, dtype=int)
    for class_index in range(classes):
        rows = slice(class_index * samples_per_class, (class_index + 1) * samples_per_class)
        radius = np.linspace(0.0, 1.0, samples_per_class)
        angle = np.linspace(class_index * 4.0, (class_index + 1) * 4.0, samples_per_class)
        angle += generator.normal(0.0, noise, samples_per_class)
        features[rows] = np.column_stack((radius * np.sin(angle), radius * np.cos(angle)))
        labels[rows] = class_index
    return features, labels


def train_test_split(features: np.ndarray, labels: np.ndarray, test_fraction: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return shuffled train/test partitions, preserving matching row order."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have equal numbers of rows")
    indices = np.random.default_rng(seed).permutation(features.shape[0])
    boundary = int(features.shape[0] * (1.0 - test_fraction))
    train_indices, test_indices = indices[:boundary], indices[boundary:]
    return features[train_indices], features[test_indices], labels[train_indices], labels[test_indices]


def standardize(train_features: np.ndarray, *other_features: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardize arrays using only training-set statistics."""
    mean = np.mean(train_features, axis=0, keepdims=True)
    scale = np.std(train_features, axis=0, keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    return tuple((array - mean) / scale for array in (train_features, *other_features))
