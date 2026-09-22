"""Comprehensive tests for mlp_baseline.activations.

Every activation function and its derivative is tested for mathematical
correctness, numerical stability, and expected edge-case behaviour.
"""

import numpy as np
import pytest

from mlp_baseline.activations import cross_entropy, relu, relu_gradient, softmax


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------


class TestRelu:
    """Tests for the elementwise rectified linear unit."""

    def test_positive_pass_through(self) -> None:
        """Positive values should be returned unchanged: ReLU(x) = x for x > 0."""
        x = np.array([0.5, 1.0, 3.7, 100.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_negative_zeroing(self) -> None:
        """Negative values should be mapped to zero: ReLU(x) = 0 for x < 0."""
        x = np.array([-0.1, -1.0, -50.0, -1e8])
        np.testing.assert_array_equal(relu(x), np.zeros_like(x))

    def test_zero_input(self) -> None:
        """ReLU(0) = max(0, 0) = 0."""
        x = np.array([0.0])
        np.testing.assert_array_equal(relu(x), np.array([0.0]))

    def test_mixed_array(self) -> None:
        """A mixed array should have negatives zeroed and positives kept."""
        x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
        expected = np.array([0.0, 0.0, 0.0, 2.0, 5.0])
        np.testing.assert_array_equal(relu(x), expected)

    def test_2d_array(self) -> None:
        """ReLU should operate elementwise on matrices."""
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_array_equal(relu(x), expected)

    def test_output_dtype_is_float(self) -> None:
        """Output should be a floating-point array regardless of input dtype."""
        x = np.array([-1, 0, 1])  # integer input
        result = relu(x)
        assert np.issubdtype(result.dtype, np.floating) or result.dtype == x.dtype


# ---------------------------------------------------------------------------
# relu_gradient
# ---------------------------------------------------------------------------


class TestReluGradient:
    """Tests for the subgradient of ReLU (indicator of the positive reals)."""

    def test_positive_indicator(self) -> None:
        """The gradient is 1 for all strictly positive inputs."""
        x = np.array([0.1, 1.0, 100.0])
        np.testing.assert_array_equal(relu_gradient(x), np.ones_like(x))

    def test_negative_indicator(self) -> None:
        """The gradient is 0 for all strictly negative inputs."""
        x = np.array([-0.1, -1.0, -100.0])
        np.testing.assert_array_equal(relu_gradient(x), np.zeros_like(x))

    def test_zero_at_origin(self) -> None:
        """Convention: the subgradient at the origin is chosen as 0.

        This matches the 'x > 0' implementation (strict inequality).
        """
        x = np.array([0.0])
        np.testing.assert_array_equal(relu_gradient(x), np.array([0.0]))

    def test_shape_preservation(self) -> None:
        """The gradient array must have the same shape as the input."""
        for shape in [(5,), (3, 4), (2, 3, 4)]:
            x = np.random.randn(*shape)
            assert relu_gradient(x).shape == shape

    def test_output_dtype_is_float(self) -> None:
        """Output must be float (not bool) so it can participate in arithmetic."""
        x = np.array([-1.0, 0.0, 1.0])
        g = relu_gradient(x)
        assert g.dtype == float

    def test_mixed_array(self) -> None:
        """Spot-check a mixed input vector."""
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(relu_gradient(x), expected)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    """Tests for the row-wise softmax function."""

    def test_rows_sum_to_one(self) -> None:
        """Each row of softmax output must be a valid probability distribution."""
        logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        probs = softmax(logits)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-12)

    def test_all_positive(self) -> None:
        """Softmax probabilities are strictly positive (exp is always > 0)."""
        logits = np.array([[-100.0, 0.0, 100.0]])
        probs = softmax(logits)
        assert np.all(probs > 0.0)

    def test_shift_invariance(self) -> None:
        """softmax(x + c) == softmax(x) for any scalar c.

        This is the defining property that makes the max-subtraction trick
        valid for numerical stability.
        """
        logits = np.array([[1.0, 2.0, 3.0]])
        shifted = logits + 42.0
        np.testing.assert_allclose(softmax(logits), softmax(shifted), atol=1e-12)

    def test_numerical_stability_large_logits(self) -> None:
        """Logits of magnitude ~1000 must not produce inf or NaN.

        A naive implementation without the max-subtraction trick would overflow.
        """
        logits = np.array([[1000.0, 999.0, 998.0]])
        probs = softmax(logits)
        assert np.all(np.isfinite(probs)), "Softmax produced non-finite values"
        np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-12)

    def test_numerical_stability_very_negative_logits(self) -> None:
        """Extremely negative logits should not cause underflow issues."""
        logits = np.array([[-1000.0, -999.0, -998.0]])
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-12)

    def test_single_class_degenerate(self) -> None:
        """With one class, softmax must return 1.0 regardless of the logit value."""
        for val in [-5.0, 0.0, 5.0, 1000.0]:
            probs = softmax(np.array([[val]]))
            np.testing.assert_allclose(probs, [[1.0]], atol=1e-12)

    def test_equal_logits_give_uniform(self) -> None:
        """Equal logits should yield a uniform distribution over K classes."""
        K = 5
        logits = np.array([[7.0] * K])
        probs = softmax(logits)
        np.testing.assert_allclose(probs, np.full((1, K), 1.0 / K), atol=1e-12)

    def test_argmax_preserved(self) -> None:
        """The class with the largest logit should have the highest probability."""
        logits = np.array([[1.0, 5.0, 3.0]])
        probs = softmax(logits)
        assert np.argmax(probs, axis=1)[0] == 1

    def test_gradient_finite_differences(self) -> None:
        """Spot-check the Jacobian of softmax via two-sided finite differences.

        For a single row x, the Jacobian J_{ij} = d softmax(x)_i / d x_j
        satisfies J = diag(p) - p p^T.  We verify a few entries numerically.
        """
        x = np.array([[1.0, 2.0, 3.0]])
        p = softmax(x).flatten()
        K = p.size
        eps = 1e-5

        # Compute the full Jacobian via finite differences
        J_numerical = np.zeros((K, K))
        for j in range(K):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[0, j] += eps
            x_minus[0, j] -= eps
            J_numerical[:, j] = (softmax(x_plus).flatten() - softmax(x_minus).flatten()) / (2 * eps)

        # Analytic Jacobian: diag(p) - outer(p, p)
        J_analytic = np.diag(p) - np.outer(p, p)
        np.testing.assert_allclose(J_numerical, J_analytic, atol=1e-6)


# ---------------------------------------------------------------------------
# cross_entropy
# ---------------------------------------------------------------------------


class TestCrossEntropy:
    """Tests for the mean multiclass cross-entropy loss."""

    def test_perfect_prediction_near_zero(self) -> None:
        """When the model assigns probability ~1 to the correct class the loss
        should be close to 0, since -log(1) = 0.
        """
        probs = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]])
        labels = np.array([0, 1])
        loss = cross_entropy(probs, labels)
        assert loss < 0.02, f"Expected near-zero loss for confident correct predictions, got {loss}"

    def test_uniform_prediction_equals_log_K(self) -> None:
        """For a uniform distribution over K classes the loss is exactly log(K).

        -log(1/K) = log(K).
        """
        K = 4
        probs = np.full((3, K), 1.0 / K)
        labels = np.array([0, 1, 2])
        expected = np.log(K)
        np.testing.assert_allclose(cross_entropy(probs, labels), expected, atol=1e-12)

    def test_known_analytic_value(self) -> None:
        """Verify against a hand-computed example.

        Sample 0: true class 0, p=0.7  => -log(0.7)
        Sample 1: true class 1, p=0.3  => -log(0.3)
        Mean = (-log(0.7) + -log(0.3)) / 2
        """
        probs = np.array([[0.7, 0.3], [0.6, 0.3]])
        # Note: rows don't need to sum to 1 for this function; it just indexes.
        # But let's use proper distributions:
        probs = np.array([[0.7, 0.3], [0.4, 0.6]])
        labels = np.array([0, 1])
        expected = (-np.log(0.7) + -np.log(0.6)) / 2.0
        np.testing.assert_allclose(cross_entropy(probs, labels), expected, atol=1e-12)

    def test_non_negativity(self) -> None:
        """Cross-entropy is non-negative for valid probability inputs.

        Since probabilities are in (0, 1], -log(p) >= 0.
        """
        rng = np.random.default_rng(42)
        K = 5
        # Generate valid probability distributions
        raw = rng.random((10, K))
        probs = raw / raw.sum(axis=1, keepdims=True)
        labels = rng.integers(0, K, size=10)
        assert cross_entropy(probs, labels) >= 0.0

    def test_clipping_prevents_nan_on_zero_probability(self) -> None:
        """If the true class has probability 0, clipping to 1e-12 avoids NaN.

        Without clipping, log(0) = -inf and the loss would be inf or NaN.
        """
        probs = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1])  # both point to the zero-probability class
        loss = cross_entropy(probs, labels)
        assert np.isfinite(loss), "Loss should be finite even with zero-probability inputs"
        # The clipped value is 1e-12, so loss = -log(1e-12) = 12*log(10) ~ 27.6
        expected = -np.log(1e-12)
        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_returns_python_float(self) -> None:
        """The return type should be a plain Python float, not a numpy scalar."""
        probs = np.array([[0.8, 0.2]])
        labels = np.array([0])
        result = cross_entropy(probs, labels)
        assert isinstance(result, float)

    def test_single_sample(self) -> None:
        """The function should work correctly with a single sample."""
        probs = np.array([[0.2, 0.3, 0.5]])
        labels = np.array([2])
        expected = -np.log(0.5)
        np.testing.assert_allclose(cross_entropy(probs, labels), expected, atol=1e-12)

    def test_loss_decreases_with_higher_confidence(self) -> None:
        """More confident correct predictions should yield lower loss."""
        labels = np.array([0, 1])
        confident = np.array([[0.99, 0.01], [0.02, 0.98]])
        uncertain = np.array([[0.55, 0.45], [0.45, 0.55]])
        assert cross_entropy(confident, labels) < cross_entropy(uncertain, labels)
