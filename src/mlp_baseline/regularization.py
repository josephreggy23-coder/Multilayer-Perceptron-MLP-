"""Regularization utilities for the educational MLP.

Overfitting is the central failure mode of flexible classifiers.
A model that memorizes the training set has zero training loss but poor
generalization; regularization penalizes complexity so the optimizer
prefers simpler parameter configurations that transfer to unseen data.

This module provides two orthogonal regularization mechanisms:

L2 weight decay (Tikhonov regularization)
-----------------------------------------
Adds a penalty proportional to the squared Frobenius norm of the weight
matrices to the loss:

    L_reg = L_data + (lambda / 2) * sum_l ||W_l||_F^2

The gradient of the penalty with respect to each weight matrix is simply:

    dL_reg / dW_l = dL_data / dW_l + lambda * W_l

so the update rule becomes:

    W_l <- W_l - lr * (dL/dW_l + lambda * W_l)
         = (1 - lr * lambda) * W_l - lr * dL/dW_l

The factor ``(1 - lr * lambda)`` shrinks every weight toward zero at
each step, hence the name "weight decay."  Note that biases are not
penalized: they control only the intercept, not the curvature, so
penalizing them adds bias without reducing variance.

Inverted dropout (Srivastava et al. 2014)
------------------------------------------
At training time, each hidden activation is independently set to zero
with probability ``p`` and the surviving activations are scaled by
``1 / (1 - p)`` so the expected activation magnitude is unchanged.
At inference time, all activations are used without modification.

The inverted scaling is the key implementation detail: it means the
forward pass at test time is identical to the non-dropout forward pass,
with no extra scaling step.  Mathematically, the mask ``m ~ Bernoulli(1-p)``
and scaling factor ``1/(1-p)`` produce:

    a_dropped = (a * m) / (1 - p)

    E[a_dropped] = a * E[m] / (1-p) = a * (1-p) / (1-p) = a

so the expected output of each layer is the same during training and
inference.
"""

from __future__ import annotations

import numpy as np


def l2_penalty(weights: list[np.ndarray], lam: float) -> float:
    """Compute the L2 regularization penalty.

    Parameters
    ----------
    weights
        List of weight matrices (one per layer).
    lam
        Regularization strength (lambda >= 0).

    Returns
    -------
    Penalty value: ``(lam / 2) * sum_l ||W_l||_F^2``.
    """
    if lam < 0:
        raise ValueError("Regularization strength lambda must be non-negative")
    if lam == 0.0:
        return 0.0
    return 0.5 * lam * sum(float(np.sum(w ** 2)) for w in weights)


def l2_gradient(weight: np.ndarray, lam: float) -> np.ndarray:
    """Gradient of the L2 penalty for a single weight matrix.

    Parameters
    ----------
    weight
        Weight matrix W_l.
    lam
        Regularization strength.

    Returns
    -------
    Gradient: ``lam * W_l``.
    """
    if lam < 0:
        raise ValueError("Regularization strength lambda must be non-negative")
    return lam * weight


def apply_weight_decay(
    weights: list[np.ndarray],
    weight_gradients: list[np.ndarray],
    lam: float,
) -> list[np.ndarray]:
    """Add the L2 penalty gradient to each weight gradient in place.

    This modifies the gradient list so the caller can proceed with the
    standard update rule ``W -= lr * grad`` and get the weight-decay
    effect automatically.

    Parameters
    ----------
    weights
        Current weight matrices.
    weight_gradients
        Gradients of the data loss with respect to each weight matrix.
    lam
        Regularization strength.

    Returns
    -------
    The same gradient list, with each entry incremented by ``lam * W_l``.
    """
    if lam < 0:
        raise ValueError("Regularization strength lambda must be non-negative")
    for i in range(len(weights)):
        weight_gradients[i] = weight_gradients[i] + l2_gradient(weights[i], lam)
    return weight_gradients


class DropoutMask:
    """Generate and apply inverted dropout masks for hidden layers.

    Parameters
    ----------
    drop_prob
        Probability of zeroing each activation (0 = no dropout, 1 = drop all).
    seed
        Random seed for reproducibility.

    Examples
    --------
    >>> mask = DropoutMask(drop_prob=0.5, seed=42)
    >>> activations = np.ones((4, 8))
    >>> dropped = mask.apply(activations, training=True)
    >>> dropped.shape
    (4, 8)
    >>> np.allclose(mask.apply(activations, training=False), activations)
    True
    """

    def __init__(self, drop_prob: float = 0.5, seed: int = 42) -> None:
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob
        self._rng = np.random.default_rng(seed)
        self._last_mask: np.ndarray | None = None

    def apply(self, activations: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply inverted dropout to a hidden-layer activation matrix.

        Parameters
        ----------
        activations
            Hidden layer activations, shape (batch_size, hidden_dim).
        training
            If True, apply dropout and cache the mask for the backward pass.
            If False, return activations unchanged (test-time behavior).

        Returns
        -------
        Masked (and scaled) activations during training, or unmodified
        activations during inference.
        """
        if not training or self.drop_prob == 0.0:
            self._last_mask = None
            return activations

        keep_prob = 1.0 - self.drop_prob
        mask = (self._rng.random(activations.shape) < keep_prob).astype(float)
        mask /= keep_prob  # inverted scaling
        self._last_mask = mask
        return activations * mask

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backpropagate through the dropout mask.

        The gradient of the dropout operation is simply the mask itself
        (with the inverted scaling), since:

            d(a * m / (1-p)) / da = m / (1-p)

        which is exactly what ``self._last_mask`` stores.

        Parameters
        ----------
        gradient
            Upstream gradient, same shape as the activations.

        Returns
        -------
        Gradient with dropout mask applied.
        """
        if self._last_mask is None:
            return gradient
        return gradient * self._last_mask

    @property
    def last_mask(self) -> np.ndarray | None:
        """The most recently generated mask, for inspection or testing."""
        return self._last_mask
