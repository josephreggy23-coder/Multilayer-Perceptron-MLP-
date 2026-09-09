"""Tests for the regularization module.

Covers L2 penalty computation, weight decay gradient correctness, and
inverted dropout behavior at training and inference time.
"""

import numpy as np
import pytest

from mlp_baseline.regularization import (
    DropoutMask,
    apply_weight_decay,
    l2_gradient,
    l2_penalty,
)


# ---------------------------------------------------------------
# L2 penalty
# ---------------------------------------------------------------
class TestL2Penalty:
    def test_zero_lambda_returns_zero(self) -> None:
        weights = [np.ones((3, 4)), np.ones((4, 2))]
        assert l2_penalty(weights, lam=0.0) == 0.0

    def test_known_value(self) -> None:
        # Single 2x2 identity matrix: sum of squares = 2, penalty = lam/2 * 2
        weights = [np.eye(2)]
        assert np.isclose(l2_penalty(weights, lam=1.0), 1.0)

    def test_multiple_layers(self) -> None:
        w1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # sum sq = 30
        w2 = np.array([[1.0], [1.0]])               # sum sq = 2
        expected = 0.5 * 0.01 * (30.0 + 2.0)
        assert np.isclose(l2_penalty([w1, w2], lam=0.01), expected)

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            l2_penalty([np.ones((2, 2))], lam=-0.1)


# ---------------------------------------------------------------
# L2 gradient
# ---------------------------------------------------------------
class TestL2Gradient:
    def test_gradient_is_lambda_times_weight(self) -> None:
        w = np.array([[1.0, -2.0], [3.0, 0.5]])
        grad = l2_gradient(w, lam=0.1)
        np.testing.assert_allclose(grad, 0.1 * w)

    def test_zero_lambda_gives_zero_gradient(self) -> None:
        w = np.ones((3, 3))
        np.testing.assert_array_equal(l2_gradient(w, lam=0.0), np.zeros((3, 3)))


# ---------------------------------------------------------------
# Weight decay application
# ---------------------------------------------------------------
class TestApplyWeightDecay:
    def test_gradients_are_augmented_in_place(self) -> None:
        weights = [np.array([[2.0, 1.0]])]
        grads = [np.array([[0.5, -0.3]])]
        result = apply_weight_decay(weights, grads, lam=0.1)

        expected = np.array([[0.5 + 0.1 * 2.0, -0.3 + 0.1 * 1.0]])
        np.testing.assert_allclose(result[0], expected)

    def test_zero_lambda_leaves_gradients_unchanged(self) -> None:
        weights = [np.ones((2, 3))]
        grads = [np.full((2, 3), 0.5)]
        original = grads[0].copy()
        apply_weight_decay(weights, grads, lam=0.0)
        np.testing.assert_array_equal(grads[0], original)


# ---------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------
class TestDropoutMask:
    def test_inference_returns_activations_unchanged(self) -> None:
        mask = DropoutMask(drop_prob=0.5, seed=0)
        a = np.ones((4, 8))
        result = mask.apply(a, training=False)
        np.testing.assert_array_equal(result, a)

    def test_training_zeros_some_activations(self) -> None:
        mask = DropoutMask(drop_prob=0.5, seed=0)
        a = np.ones((100, 50))
        result = mask.apply(a, training=True)

        # Some values should be zero (dropped), others scaled up.
        assert np.any(result == 0.0)
        assert np.any(result > 1.0)  # scaled by 1/(1-p) = 2

    def test_inverted_scaling_preserves_expected_value(self) -> None:
        """Over many trials, the mean output should approximate the input."""
        mask = DropoutMask(drop_prob=0.3, seed=42)
        a = np.full((1000, 100), 5.0)
        total = np.zeros_like(a)
        n_trials = 500
        for _ in range(n_trials):
            total += mask.apply(a, training=True)
        # Averaging across both trials and all elements gives a tight estimate.
        grand_mean = total.mean() / n_trials
        np.testing.assert_allclose(grand_mean, 5.0, atol=0.05)

    def test_zero_drop_prob_returns_activations_unchanged(self) -> None:
        mask = DropoutMask(drop_prob=0.0, seed=0)
        a = np.ones((4, 8))
        result = mask.apply(a, training=True)
        np.testing.assert_array_equal(result, a)

    def test_backward_applies_same_mask(self) -> None:
        mask = DropoutMask(drop_prob=0.5, seed=0)
        a = np.ones((4, 8))
        dropped = mask.apply(a, training=True)

        grad = np.ones_like(a)
        back = mask.backward(grad)

        # Where activation was dropped, gradient should also be zero.
        zero_positions = dropped == 0.0
        assert np.all(back[zero_positions] == 0.0)

    def test_backward_without_training_is_identity(self) -> None:
        mask = DropoutMask(drop_prob=0.5, seed=0)
        a = np.ones((4, 8))
        mask.apply(a, training=False)

        grad = np.ones_like(a) * 3.0
        back = mask.backward(grad)
        np.testing.assert_array_equal(back, grad)

    def test_invalid_drop_prob_raises(self) -> None:
        with pytest.raises(ValueError, match="drop_prob"):
            DropoutMask(drop_prob=1.0)
        with pytest.raises(ValueError, match="drop_prob"):
            DropoutMask(drop_prob=-0.1)
