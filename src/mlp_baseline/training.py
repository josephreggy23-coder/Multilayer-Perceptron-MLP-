"""Deterministic mini-batch training for the educational MLP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import cross_entropy
from .model import MLP


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.08
    seed: int = 42


def train(model: MLP, features: np.ndarray, labels: np.ndarray, config: TrainingConfig) -> list[dict[str, float]]:
    """Train a model and return per-epoch loss and accuracy records."""
    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    generator = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        indices = generator.permutation(features.shape[0])
        for start in range(0, features.shape[0], config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            model.train_batch(features[batch_indices], labels[batch_indices], config.learning_rate)
        probabilities = model.predict_proba(features)
        history.append(
            {
                "epoch": float(epoch),
                "loss": cross_entropy(probabilities, labels),
                "accuracy": model.score(features, labels),
            }
        )
    return history
