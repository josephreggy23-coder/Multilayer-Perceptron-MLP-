"""Weight initialization strategies for multilayer perceptrons.

This module provides three well-known initialization strategies that each
preserve the variance of activations (and, in the Xavier case, gradients)
as signals propagate through a deep network.  Without careful
initialization the activations either explode (growing exponentially with
depth) or vanish (shrinking to zero), making training slow or impossible.

The core idea behind all three strategies is identical:

    Choose Var(W) so that Var(output) = Var(input) for every layer.

Starting from the linear pre-activation  z = W @ x + b  (biases are zero
at init) and assuming the weights and inputs are independent with zero
mean, the variance of one output neuron is

    Var(z_j) = fan_in * Var(W) * E[x_i^2].

The three strategies differ only in how they estimate E[x_i^2] given the
non-linearity that precedes the current layer.

References
----------
He, K. et al.  "Delving Deep into Rectifiers", ICCV 2015.
Glorot, X. & Bengio, Y.  "Understanding the difficulty of training deep
    feedforward neural networks", AISTATS 2010.
LeCun, Y. et al.  "Efficient BackProp", 1998.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def he_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""He (Kaiming) initialization -- designed for ReLU networks.

    Mathematical derivation
    -----------------------
    ReLU zeroes out roughly half of its inputs, so

        E[x_i^2] = 0.5 * Var(input_i)        (for zero-mean pre-ReLU input).

    Substituting into the variance equation and requiring
    Var(z_j) = Var(input_j):

        fan_in * Var(W) * 0.5 * Var(input) = Var(input)
        Var(W) = 2 / fan_in.

    Weights are therefore drawn from

        W ~ N(0,  sqrt(2 / fan_in)).

    When to use
    -----------
    Pair with ReLU (or Leaky ReLU with a small slope) hidden layers.
    It is the default in most modern deep-learning frameworks for
    convolutional and fully connected layers that use ReLU.

    Parameters
    ----------
    fan_in : int
        Number of input units to the layer.
    fan_out : int
        Number of output units (used only for shape; does not affect the
        variance formula).
    rng : numpy.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight matrix of shape ``(fan_in, fan_out)``.
    """
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, (fan_in, fan_out))


def xavier_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Xavier / Glorot initialization -- designed for sigmoid and tanh.

    Mathematical derivation
    -----------------------
    For activation functions that are approximately linear around zero
    (sigmoid and tanh near their origin), the derivative is close to 1
    and E[x_i^2] ~ Var(input_i).  To preserve variance in *both* the
    forward and backward passes simultaneously, Glorot & Bengio average
    the two constraints:

        Forward:   Var(W) = 1 / fan_in
        Backward:  Var(W) = 1 / fan_out

    Taking the harmonic-style compromise:

        Var(W) = 2 / (fan_in + fan_out).

    Weights are therefore drawn from

        W ~ N(0,  sqrt(2 / (fan_in + fan_out))).

    When to use
    -----------
    Pair with sigmoid or tanh hidden layers, or any activation whose
    derivative near zero is approximately 1.  Not ideal for ReLU because
    it ignores the factor-of-two correction that ReLU needs.

    Parameters
    ----------
    fan_in : int
        Number of input units to the layer.
    fan_out : int
        Number of output units to the layer.
    rng : numpy.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight matrix of shape ``(fan_in, fan_out)``.
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, (fan_in, fan_out))


def lecun_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""LeCun initialization -- designed for SELU (and classic sigmoid).

    Mathematical derivation
    -----------------------
    Under the assumption that the activation function preserves variance
    exactly (which SELU is specifically engineered to do), the full
    input variance passes through:

        E[x_i^2] = Var(input_i).

    Substituting into the variance equation:

        fan_in * Var(W) * Var(input) = Var(input)
        Var(W) = 1 / fan_in.

    Weights are therefore drawn from

        W ~ N(0,  sqrt(1 / fan_in)).

    This is also the initialization originally recommended by LeCun for
    networks using the standard sigmoid, predating both He and Xavier.

    When to use
    -----------
    Pair with SELU activations (for self-normalizing networks) or
    with classic sigmoid when a simpler formula than Xavier is
    acceptable.

    Parameters
    ----------
    fan_in : int
        Number of input units to the layer.
    fan_out : int
        Number of output units (used only for shape; does not affect the
        variance formula).
    rng : numpy.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight matrix of shape ``(fan_in, fan_out)``.
    """
    std = np.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, (fan_in, fan_out))


_STRATEGIES = {
    "he": he_init,
    "xavier": xavier_init,
    "glorot": xavier_init,
    "lecun": lecun_init,
}


def initialize_weights(
    layer_sizes: Sequence[int],
    strategy: str = "he",
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create weight matrices and bias vectors for every layer transition.

    This is the main entry point for initializing a full network.
    It iterates over consecutive ``(fan_in, fan_out)`` pairs in
    *layer_sizes*, applies the requested initialization strategy to
    produce the weight matrices, and returns zero-initialized biases.

    Parameters
    ----------
    layer_sizes : sequence of int
        Sizes of each layer from input to output.  Must contain at least
        two positive integers, e.g. ``(784, 128, 10)``.
    strategy : str, optional
        Name of the initialization strategy.  One of ``"he"`` (default),
        ``"xavier"`` (alias ``"glorot"``), or ``"lecun"``.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    weights : list of numpy.ndarray
        One ``(fan_in, fan_out)`` matrix per layer transition.
    biases : list of numpy.ndarray
        One ``(1, fan_out)`` zero vector per layer transition.

    Raises
    ------
    ValueError
        If *layer_sizes* has fewer than two elements, any element is
        non-positive, or *strategy* is not recognized.
    """
    if len(layer_sizes) < 2:
        raise ValueError(
            "layer_sizes must contain at least two elements, "
            f"got {len(layer_sizes)}"
        )
    if any(s <= 0 for s in layer_sizes):
        raise ValueError("all layer sizes must be positive integers")

    key = strategy.lower()
    if key not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; "
            f"choose from {sorted(set(_STRATEGIES.keys()) - {'glorot'})}"
        )

    init_fn = _STRATEGIES[key]
    rng = np.random.default_rng(seed)

    weights = [
        init_fn(fan_in, fan_out, rng)
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    biases = [np.zeros((1, fan_out)) for fan_out in layer_sizes[1:]]

    return weights, biases
