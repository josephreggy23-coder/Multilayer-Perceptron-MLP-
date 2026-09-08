"""Numerical gradient verification for backpropagation correctness.

Finite-difference gradient checking compares the analytically computed
gradients from ``MLP.loss_and_gradients`` against a two-sided numerical
approximation:

    dL/dw_ij ~ (L(w_ij + eps) - L(w_ij - eps)) / (2 * eps)

The relative error between the two should be on the order of eps (machine
precision of float64 ~ 1e-16, so eps ~ 1e-5 gives relative error ~ 1e-6
when the implementation is correct).

Why two-sided?  A one-sided difference has O(eps) truncation error, while
the centered (two-sided) difference cancels the linear term and achieves
O(eps^2). For eps = 1e-5, that is the difference between 1e-5 and 1e-10
truncation error, which makes the check far more sensitive to real bugs
in the gradient computation.

Reference:
    Bengio, Y. (2012). Practical recommendations for gradient-based
    training of deep architectures, Sec. 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import MLP


@dataclass(frozen=True)
class GradientCheckResult:
    """Outcome of a numerical gradient check on one parameter array.

    Attributes
    ----------
    max_relative_error : float
        Largest relative error across all entries of the parameter matrix.
    mean_relative_error : float
        Average relative error, useful for diagnosing diffuse imprecision.
    param_name : str
        Human-readable label for the parameter being checked.
    passed : bool
        Whether max_relative_error fell below the chosen tolerance.
    """

    max_relative_error: float
    mean_relative_error: float
    param_name: str
    passed: bool


def _relative_error(analytic: float, numerical: float) -> float:
    """Relative error with a denominator guard against division by zero.

    Uses ``max(|a|, |n|, 1e-8)`` in the denominator so that a pair of
    near-zero gradients does not inflate the relative error.
    """
    denom = max(abs(analytic), abs(numerical), 1e-8)
    return abs(analytic - numerical) / denom


def check_gradients(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-5,
    tolerance: float = 1e-5,
) -> list[GradientCheckResult]:
    """Check all weight and bias gradients in *model* by finite differences.

    Parameters
    ----------
    model
        An ``MLP`` instance whose ``loss_and_gradients`` will be verified.
    features
        Input batch, shape ``(n_samples, n_features)``.
    labels
        Integer class labels, shape ``(n_samples,)``.
    eps
        Perturbation magnitude for the centered difference. Values near
        1e-5 balance truncation error (O(eps^2)) against round-off error
        (O(u/eps), where u ~ 1e-16 for float64).
    tolerance
        Maximum acceptable relative error per parameter entry. The
        default of 1e-5 is conservative; correct implementations
        typically achieve < 1e-7.

    Returns
    -------
    list[GradientCheckResult]
        One result per parameter array (weights and biases for each layer).
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    _, weight_grads, bias_grads = model.loss_and_gradients(features, labels)
    results: list[GradientCheckResult] = []

    for layer_index in range(len(model.weights)):
        # --- check weights ---
        param = model.weights[layer_index]
        analytic_grad = weight_grads[layer_index]
        errors: list[float] = []

        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                original = param[i, j]

                param[i, j] = original + eps
                loss_plus, _, _ = model.loss_and_gradients(features, labels)

                param[i, j] = original - eps
                loss_minus, _, _ = model.loss_and_gradients(features, labels)

                param[i, j] = original

                numerical = (loss_plus - loss_minus) / (2.0 * eps)
                errors.append(_relative_error(analytic_grad[i, j], numerical))

        results.append(GradientCheckResult(
            max_relative_error=float(max(errors)),
            mean_relative_error=float(np.mean(errors)),
            param_name=f"weights[{layer_index}] ({param.shape[0]}x{param.shape[1]})",
            passed=max(errors) < tolerance,
        ))

        # --- check biases ---
        bias = model.biases[layer_index]
        analytic_bias_grad = bias_grads[layer_index]
        errors = []

        for j in range(bias.shape[1]):
            original = bias[0, j]

            bias[0, j] = original + eps
            loss_plus, _, _ = model.loss_and_gradients(features, labels)

            bias[0, j] = original - eps
            loss_minus, _, _ = model.loss_and_gradients(features, labels)

            bias[0, j] = original

            numerical = (loss_plus - loss_minus) / (2.0 * eps)
            errors.append(_relative_error(analytic_bias_grad[0, j], numerical))

        results.append(GradientCheckResult(
            max_relative_error=float(max(errors)),
            mean_relative_error=float(np.mean(errors)),
            param_name=f"biases[{layer_index}] (1x{bias.shape[1]})",
            passed=max(errors) < tolerance,
        ))

    return results
