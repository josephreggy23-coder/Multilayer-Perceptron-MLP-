"""Tests for mlp_baseline.weight_init."""

import numpy as np
import pytest

from mlp_baseline.weight_init import (
    he_init,
    initialize_weights,
    lecun_init,
    xavier_init,
)


# ---- shape tests -----------------------------------------------------------

def test_he_init_shape() -> None:
    rng = np.random.default_rng(0)
    w = he_init(128, 64, rng)
    assert w.shape == (128, 64)


def test_xavier_init_shape() -> None:
    rng = np.random.default_rng(0)
    w = xavier_init(256, 128, rng)
    assert w.shape == (256, 128)


def test_lecun_init_shape() -> None:
    rng = np.random.default_rng(0)
    w = lecun_init(512, 10, rng)
    assert w.shape == (512, 10)


# ---- empirical variance tests ----------------------------------------------
# With large matrices the sample variance should tightly match the
# theoretical value.  We use a relative tolerance of 5 %.

LARGE_FAN_IN = 4096
LARGE_FAN_OUT = 2048
RTOL = 0.05


def test_he_variance() -> None:
    rng = np.random.default_rng(1)
    w = he_init(LARGE_FAN_IN, LARGE_FAN_OUT, rng)
    expected_var = 2.0 / LARGE_FAN_IN
    np.testing.assert_allclose(w.var(), expected_var, rtol=RTOL)


def test_xavier_variance() -> None:
    rng = np.random.default_rng(2)
    w = xavier_init(LARGE_FAN_IN, LARGE_FAN_OUT, rng)
    expected_var = 2.0 / (LARGE_FAN_IN + LARGE_FAN_OUT)
    np.testing.assert_allclose(w.var(), expected_var, rtol=RTOL)


def test_lecun_variance() -> None:
    rng = np.random.default_rng(3)
    w = lecun_init(LARGE_FAN_IN, LARGE_FAN_OUT, rng)
    expected_var = 1.0 / LARGE_FAN_IN
    np.testing.assert_allclose(w.var(), expected_var, rtol=RTOL)


# ---- factory function tests ------------------------------------------------

def test_initialize_weights_shapes() -> None:
    sizes = (784, 128, 64, 10)
    weights, biases = initialize_weights(sizes, strategy="he", seed=0)

    assert len(weights) == 3
    assert len(biases) == 3

    assert weights[0].shape == (784, 128)
    assert weights[1].shape == (128, 64)
    assert weights[2].shape == (64, 10)

    assert biases[0].shape == (1, 128)
    assert biases[1].shape == (1, 64)
    assert biases[2].shape == (1, 10)


def test_biases_are_zeros() -> None:
    _, biases = initialize_weights((32, 16, 8), strategy="xavier")
    for b in biases:
        np.testing.assert_array_equal(b, 0.0)


def test_glorot_alias_matches_xavier() -> None:
    w_xavier, b_xavier = initialize_weights((64, 32), strategy="xavier", seed=99)
    w_glorot, b_glorot = initialize_weights((64, 32), strategy="glorot", seed=99)

    np.testing.assert_array_equal(w_xavier[0], w_glorot[0])
    np.testing.assert_array_equal(b_xavier[0], b_glorot[0])


def test_seed_reproducibility() -> None:
    w1, _ = initialize_weights((100, 50, 10), strategy="lecun", seed=7)
    w2, _ = initialize_weights((100, 50, 10), strategy="lecun", seed=7)
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_weights() -> None:
    w1, _ = initialize_weights((100, 50, 10), strategy="he", seed=0)
    w2, _ = initialize_weights((100, 50, 10), strategy="he", seed=1)
    assert not np.array_equal(w1[0], w2[0])


# ---- edge cases -------------------------------------------------------------

def test_two_layer_network() -> None:
    """Minimal network: just one weight matrix."""
    weights, biases = initialize_weights((4, 2), strategy="he")
    assert len(weights) == 1
    assert weights[0].shape == (4, 2)
    assert biases[0].shape == (1, 2)


def test_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        initialize_weights((10, 5), strategy="nonexistent")


def test_too_few_layers_raises() -> None:
    with pytest.raises(ValueError, match="at least two elements"):
        initialize_weights((10,))


def test_non_positive_layer_size_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        initialize_weights((10, 0, 5))


def test_strategy_is_case_insensitive() -> None:
    weights, _ = initialize_weights((8, 4), strategy="He", seed=0)
    assert weights[0].shape == (8, 4)
