"""Tests for numerical gradient verification.

These tests confirm that the backpropagation implementation in ``MLP`` computes
exact gradients by comparing them against centered finite differences. A passing
gradient check is the single strongest evidence that backprop is implemented
correctly; a failing one pinpoints the layer and parameter type where the
analytic gradient diverges from the true derivative.
"""

import numpy as np
import pytest

from mlp_baseline import MLP
from mlp_baseline.gradient_check import GradientCheckResult, check_gradients


def test_two_layer_gradients_match_finite_differences() -> None:
    """Simplest case: one hidden layer, three output classes."""
    model = MLP((2, 4, 3), seed=0)
    features = np.array([[0.5, -0.3], [1.0, 0.8], [-0.2, 0.6]])
    labels = np.array([0, 2, 1])

    results = check_gradients(model, features, labels)

    assert len(results) == 4  # 2 layers x (weights + biases)
    for result in results:
        assert result.passed, (
            f"{result.param_name}: max relative error {result.max_relative_error:.2e} "
            f"exceeds tolerance"
        )
        assert result.max_relative_error < 1e-5


def test_three_layer_gradients_match_finite_differences() -> None:
    """Deeper network: two hidden layers to test gradient flow through ReLU."""
    model = MLP((3, 8, 6, 4), seed=1)
    features = np.array([
        [0.1, -0.5, 0.9],
        [-0.4, 0.7, -0.2],
        [0.8, 0.3, 0.6],
        [-0.1, -0.8, 0.4],
    ])
    labels = np.array([1, 3, 0, 2])

    results = check_gradients(model, features, labels)

    assert len(results) == 6  # 3 layers x 2
    for result in results:
        assert result.passed, f"{result.param_name} failed gradient check"


def test_gradient_check_detects_corrupted_gradient() -> None:
    """When we deliberately perturb the analytic gradient, the check should fail."""
    model = MLP((2, 4, 3), seed=2)
    features = np.array([[0.5, -0.3], [1.0, 0.8]])
    labels = np.array([0, 2])

    # Corrupt the first weight matrix gradient by overriding loss_and_gradients
    original_method = model.loss_and_gradients

    def corrupted_gradients(f, l):
        loss, wg, bg = original_method(f, l)
        wg[0] = wg[0] + 0.5  # add a large constant error
        return loss, wg, bg

    model.loss_and_gradients = corrupted_gradients

    results = check_gradients(model, features, labels)

    # The first weight gradient should fail
    assert not results[0].passed
    assert results[0].max_relative_error > 0.1


def test_single_sample_gradient() -> None:
    """Gradient check on a single training example (batch size = 1)."""
    model = MLP((4, 5, 2), seed=3)
    features = np.array([[0.1, 0.2, 0.3, 0.4]])
    labels = np.array([1])

    results = check_gradients(model, features, labels)

    for result in results:
        assert result.passed, f"{result.param_name} failed on single sample"


def test_result_fields_populated() -> None:
    """Check that the result dataclass contains meaningful values."""
    model = MLP((2, 3, 2), seed=4)
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])

    results = check_gradients(model, features, labels)

    for result in results:
        assert isinstance(result, GradientCheckResult)
        assert result.mean_relative_error <= result.max_relative_error
        assert result.mean_relative_error >= 0.0
        assert "weights" in result.param_name or "biases" in result.param_name


def test_rejects_invalid_eps() -> None:
    model = MLP((2, 3, 2), seed=5)
    features = np.array([[1.0, 0.0]])
    labels = np.array([0])

    with pytest.raises(ValueError, match="eps must be positive"):
        check_gradients(model, features, labels, eps=0.0)

    with pytest.raises(ValueError, match="eps must be positive"):
        check_gradients(model, features, labels, eps=-1e-5)


def test_rejects_invalid_tolerance() -> None:
    model = MLP((2, 3, 2), seed=6)
    features = np.array([[1.0, 0.0]])
    labels = np.array([0])

    with pytest.raises(ValueError, match="tolerance must be positive"):
        check_gradients(model, features, labels, tolerance=0.0)
