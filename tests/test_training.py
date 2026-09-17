"""Tests for the mini-batch training loop.

Verifies the training module's convergence properties, epoch bookkeeping,
deterministic shuffling, and parameter validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlp_baseline.data import make_spiral
from mlp_baseline.model import MLP
from mlp_baseline.training import TrainingConfig, train


# ---- helpers ---------------------------------------------------------------

def _small_problem(seed: int = 42):
    """Return standardized spiral data and a fresh MLP."""
    features, labels = make_spiral(samples_per_class=30, classes=3, seed=seed)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    features = (features - mean) / std
    model = MLP((2, 16, 3), seed=seed)
    return model, features, labels


# ---- convergence -----------------------------------------------------------

class TestConvergence:
    def test_loss_decreases_over_training(self) -> None:
        """Training loss should decrease monotonically on a learnable problem."""
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=80, batch_size=16, learning_rate=0.1)
        history = train(model, features, labels, config)

        early_loss = np.mean([h["loss"] for h in history[:10]])
        late_loss = np.mean([h["loss"] for h in history[-10:]])
        assert late_loss < early_loss * 0.6

    def test_accuracy_improves_over_training(self) -> None:
        """Accuracy should increase to a reasonable level."""
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=150, batch_size=16, learning_rate=0.1)
        history = train(model, features, labels, config)

        assert history[-1]["accuracy"] > 0.8
        assert history[-1]["accuracy"] > history[0]["accuracy"]

    def test_longer_training_reaches_lower_loss(self) -> None:
        """More epochs should yield lower or equal loss."""
        model_short, features, labels = _small_problem()
        model_long = MLP((2, 16, 3), seed=42)

        config_short = TrainingConfig(epochs=30, batch_size=16, learning_rate=0.1)
        config_long = TrainingConfig(epochs=150, batch_size=16, learning_rate=0.1)

        history_short = train(model_short, features, labels, config_short)
        history_long = train(model_long, features, labels, config_long)

        assert history_long[-1]["loss"] <= history_short[-1]["loss"]


# ---- history structure and epoch bookkeeping --------------------------------

class TestHistory:
    def test_history_length_matches_epoch_count(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=25, batch_size=32)
        history = train(model, features, labels, config)
        assert len(history) == 25

    def test_epoch_numbers_are_sequential(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=10, batch_size=32)
        history = train(model, features, labels, config)
        epochs = [h["epoch"] for h in history]
        assert epochs == [float(e) for e in range(1, 11)]

    def test_history_contains_required_keys(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=3, batch_size=32)
        history = train(model, features, labels, config)
        for record in history:
            assert "epoch" in record
            assert "loss" in record
            assert "accuracy" in record

    def test_loss_is_finite_and_positive(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=20, batch_size=32)
        history = train(model, features, labels, config)
        for record in history:
            assert np.isfinite(record["loss"])
            assert record["loss"] > 0

    def test_accuracy_is_in_zero_one_range(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=20, batch_size=32)
        history = train(model, features, labels, config)
        for record in history:
            assert 0.0 <= record["accuracy"] <= 1.0


# ---- determinism and reproducibility ----------------------------------------

class TestDeterminism:
    def test_same_seed_same_history(self) -> None:
        """Identical seeds should produce identical training trajectories."""
        model_a = MLP((2, 16, 3), seed=42)
        model_b = MLP((2, 16, 3), seed=42)
        features, labels = make_spiral(samples_per_class=30, seed=42)

        config = TrainingConfig(epochs=20, batch_size=16, seed=99)
        history_a = train(model_a, features, labels, config)
        history_b = train(model_b, features, labels, config)

        for a, b in zip(history_a, history_b):
            assert a["loss"] == b["loss"]
            assert a["accuracy"] == b["accuracy"]

    def test_different_seeds_different_history(self) -> None:
        """Different shuffle seeds should produce different trajectories."""
        model_a = MLP((2, 16, 3), seed=42)
        model_b = MLP((2, 16, 3), seed=42)
        features, labels = make_spiral(samples_per_class=30, seed=42)

        history_a = train(model_a, features, labels, TrainingConfig(epochs=10, seed=1))
        history_b = train(model_b, features, labels, TrainingConfig(epochs=10, seed=2))

        # Losses should differ by at least a small amount due to shuffle order
        losses_a = [h["loss"] for h in history_a]
        losses_b = [h["loss"] for h in history_b]
        assert losses_a != losses_b


# ---- batch size effects -----------------------------------------------------

class TestBatchSize:
    def test_full_batch_uses_all_samples(self) -> None:
        """Setting batch_size >= n_samples should use all data per epoch."""
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=20, batch_size=features.shape[0])
        history = train(model, features, labels, config)
        # Should still converge, just with fewer updates per epoch
        assert history[-1]["loss"] < history[0]["loss"]

    def test_batch_size_one_still_converges(self) -> None:
        """Online SGD (batch_size=1) should eventually converge."""
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=100, batch_size=1, learning_rate=0.01)
        history = train(model, features, labels, config)
        assert history[-1]["loss"] < history[0]["loss"] * 0.5


# ---- input validation -------------------------------------------------------

class TestValidation:
    def test_zero_epochs_raises(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(epochs=0)
        with pytest.raises(ValueError, match="positive"):
            train(model, features, labels, config)

    def test_negative_learning_rate_raises(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(learning_rate=-0.01)
        with pytest.raises(ValueError, match="positive"):
            train(model, features, labels, config)

    def test_zero_batch_size_raises(self) -> None:
        model, features, labels = _small_problem()
        config = TrainingConfig(batch_size=0)
        with pytest.raises(ValueError, match="positive"):
            train(model, features, labels, config)
