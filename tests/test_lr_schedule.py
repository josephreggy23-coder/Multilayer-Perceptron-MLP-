"""Tests for learning rate schedule implementations.

Covers step decay, exponential decay, and cosine annealing with and
without warm restarts, verifying the closed-form expressions from the
module docstring.
"""

import numpy as np
import pytest

from mlp_baseline.lr_schedule import (
    CosineAnnealing,
    ExponentialDecay,
    StepDecay,
)


# ------------------------------------------------------------------
# StepDecay
# ------------------------------------------------------------------
class TestStepDecay:
    def test_initial_rate_at_epoch_zero(self) -> None:
        schedule = StepDecay(0.1, step_size=10, gamma=0.5)
        assert schedule(0) == pytest.approx(0.1)

    def test_rate_drops_at_step_boundary(self) -> None:
        schedule = StepDecay(0.1, step_size=10, gamma=0.5)
        assert schedule(9) == pytest.approx(0.1)
        assert schedule(10) == pytest.approx(0.05)

    def test_rate_drops_multiple_times(self) -> None:
        schedule = StepDecay(0.1, step_size=5, gamma=0.5)
        assert schedule(0) == pytest.approx(0.1)
        assert schedule(5) == pytest.approx(0.05)
        assert schedule(10) == pytest.approx(0.025)
        assert schedule(15) == pytest.approx(0.0125)

    def test_gamma_one_means_constant(self) -> None:
        schedule = StepDecay(0.1, step_size=10, gamma=1.0)
        for epoch in [0, 10, 50, 100]:
            assert schedule(epoch) == pytest.approx(0.1)

    def test_monotone_non_increasing(self) -> None:
        schedule = StepDecay(0.1, step_size=7, gamma=0.3)
        rates = [schedule(e) for e in range(100)]
        assert all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))

    def test_always_positive(self) -> None:
        schedule = StepDecay(0.1, step_size=5, gamma=0.1)
        for epoch in range(200):
            assert schedule(epoch) > 0.0

    def test_invalid_step_size_raises(self) -> None:
        with pytest.raises(ValueError, match="step_size"):
            StepDecay(0.1, step_size=0, gamma=0.5)

    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            StepDecay(0.1, step_size=10, gamma=0.0)
        with pytest.raises(ValueError, match="gamma"):
            StepDecay(0.1, step_size=10, gamma=1.5)

    def test_invalid_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_lr"):
            StepDecay(-0.1, step_size=10, gamma=0.5)


# ------------------------------------------------------------------
# ExponentialDecay
# ------------------------------------------------------------------
class TestExponentialDecay:
    def test_initial_rate_at_epoch_zero(self) -> None:
        schedule = ExponentialDecay(0.1, gamma=0.95)
        assert schedule(0) == pytest.approx(0.1)

    def test_known_value(self) -> None:
        """lr(t) = lr_0 * gamma^t, so lr(10) = 0.1 * 0.95^10."""
        schedule = ExponentialDecay(0.1, gamma=0.95)
        assert schedule(10) == pytest.approx(0.1 * 0.95**10, rel=1e-10)

    def test_strictly_decreasing(self) -> None:
        schedule = ExponentialDecay(0.1, gamma=0.9)
        rates = [schedule(e) for e in range(100)]
        assert all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))

    def test_gamma_one_means_constant(self) -> None:
        schedule = ExponentialDecay(0.1, gamma=1.0)
        for epoch in [0, 50, 100]:
            assert schedule(epoch) == pytest.approx(0.1)

    def test_always_positive(self) -> None:
        schedule = ExponentialDecay(0.1, gamma=0.5)
        for epoch in range(500):
            assert schedule(epoch) > 0.0

    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            ExponentialDecay(0.1, gamma=0.0)


# ------------------------------------------------------------------
# CosineAnnealing
# ------------------------------------------------------------------
class TestCosineAnnealing:
    def test_starts_at_initial_lr(self) -> None:
        schedule = CosineAnnealing(0.1, total_epochs=100)
        assert schedule(0) == pytest.approx(0.1)

    def test_approaches_lr_min_at_end(self) -> None:
        """At the last epoch of a cycle, the rate should be near lr_min."""
        schedule = CosineAnnealing(0.1, total_epochs=100, lr_min=0.0)
        # Epoch 99 is the last epoch within the single cycle.
        assert schedule(99) < 0.001

    def test_midpoint_is_average(self) -> None:
        """At the midpoint of a cosine half-wave, lr = (lr_0 + lr_min) / 2."""
        schedule = CosineAnnealing(0.1, total_epochs=100, lr_min=0.01)
        expected = (0.1 + 0.01) / 2.0
        assert schedule(50) == pytest.approx(expected, rel=1e-6)

    def test_custom_lr_min(self) -> None:
        schedule = CosineAnnealing(0.1, total_epochs=100, lr_min=0.01)
        # Should never go below lr_min
        for epoch in range(101):
            assert schedule(epoch) >= 0.01 - 1e-12

    def test_warm_restarts_repeat_cycle(self) -> None:
        """With one restart, the schedule should repeat with period T/2."""
        schedule = CosineAnnealing(0.1, total_epochs=100, lr_min=0.0, n_restarts=1)
        # Cycle length = 100 / 2 = 50
        # Epoch 0 and epoch 50 should both be at initial_lr
        assert schedule(0) == pytest.approx(0.1)
        assert schedule(50) == pytest.approx(0.1, abs=1e-10)
        # Epoch 25 and epoch 75 should both be at the midpoint
        assert schedule(25) == pytest.approx(schedule(75), rel=1e-6)

    def test_warm_restarts_peak_at_cycle_start(self) -> None:
        schedule = CosineAnnealing(0.1, total_epochs=90, lr_min=0.0, n_restarts=2)
        # Three cycles of length 30 each
        for start in [0, 30, 60]:
            assert schedule(start) == pytest.approx(0.1, abs=1e-10)

    def test_monotone_within_single_cycle(self) -> None:
        """Within a single cycle (epochs 0..T-1), the rate is non-increasing."""
        schedule = CosineAnnealing(0.1, total_epochs=100, lr_min=0.0, n_restarts=0)
        rates = [schedule(e) for e in range(100)]
        assert all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))

    def test_always_in_range(self) -> None:
        schedule = CosineAnnealing(0.1, total_epochs=100, lr_min=0.01, n_restarts=3)
        for epoch in range(200):
            lr = schedule(epoch)
            assert 0.01 - 1e-12 <= lr <= 0.1 + 1e-12

    def test_invalid_total_epochs_raises(self) -> None:
        with pytest.raises(ValueError, match="total_epochs"):
            CosineAnnealing(0.1, total_epochs=0)

    def test_invalid_lr_min_raises(self) -> None:
        with pytest.raises(ValueError, match="lr_min"):
            CosineAnnealing(0.1, total_epochs=100, lr_min=-0.01)
        with pytest.raises(ValueError, match="lr_min"):
            CosineAnnealing(0.1, total_epochs=100, lr_min=0.1)

    def test_invalid_n_restarts_raises(self) -> None:
        with pytest.raises(ValueError, match="n_restarts"):
            CosineAnnealing(0.1, total_epochs=100, n_restarts=-1)
