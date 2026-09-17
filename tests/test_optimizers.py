"""Tests for SGD with momentum, Nesterov momentum, and Adam.

Each optimizer is tested against a small MLP on synthetic data:
  - convergence (loss decreases)
  - correctness of momentum accumulation
  - Adam's bias correction behaves as expected
  - parameter validation
"""

from __future__ import annotations

import numpy as np
import pytest

from mlp_baseline.model import MLP
from mlp_baseline.optimizers import Adam, SGDMomentum


# ---- helpers ---------------------------------------------------------------

def _easy_problem(seed: int = 7):
    """Return a tiny MLP and linearly separable data."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(40, 4))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    model = MLP((4, 8, 2), seed=seed)
    return model, features, labels


def _run_training(model, features, labels, optimizer, steps=200):
    """Run the training loop and return the loss history."""
    losses = []
    for _ in range(steps):
        loss, wg, bg = model.loss_and_gradients(features, labels)
        losses.append(loss)
        optimizer.step(model.weights, model.biases, wg, bg)
    return losses


# ---- SGD with momentum ----------------------------------------------------

class TestSGDMomentum:
    def test_zero_momentum_is_vanilla_sgd(self) -> None:
        """With mu=0 the update should equal plain gradient descent."""
        model_a, features, labels = _easy_problem()
        model_b = MLP((4, 8, 2), seed=7)

        sgd = SGDMomentum(lr=0.05, momentum=0.0)

        for _ in range(10):
            loss_a, wg_a, bg_a = model_a.loss_and_gradients(features, labels)
            sgd.step(model_a.weights, model_a.biases, wg_a, bg_a)

            loss_b, wg_b, bg_b = model_b.loss_and_gradients(features, labels)
            for i in range(len(model_b.weights)):
                model_b.weights[i] -= 0.05 * wg_b[i]
                model_b.biases[i] -= 0.05 * bg_b[i]

        for wa, wb in zip(model_a.weights, model_b.weights):
            np.testing.assert_allclose(wa, wb, atol=1e-12)

    def test_momentum_converges_faster_than_vanilla(self) -> None:
        """Momentum should reach a lower loss in the same number of steps."""
        model_vanilla, features, labels = _easy_problem()
        model_momentum = MLP((4, 8, 2), seed=7)

        vanilla = SGDMomentum(lr=0.05, momentum=0.0)
        momentum = SGDMomentum(lr=0.05, momentum=0.9)

        losses_v = _run_training(model_vanilla, features, labels, vanilla, 100)
        losses_m = _run_training(model_momentum, features, labels, momentum, 100)

        assert losses_m[-1] < losses_v[-1]

    def test_nesterov_converges(self) -> None:
        """Nesterov variant should also converge to low loss."""
        model, features, labels = _easy_problem()
        opt = SGDMomentum(lr=0.05, momentum=0.9, nesterov=True)
        losses = _run_training(model, features, labels, opt, 200)
        assert losses[-1] < losses[0] * 0.1

    def test_momentum_velocity_accumulates(self) -> None:
        """After several steps, velocity should be nonzero."""
        model, features, labels = _easy_problem()
        opt = SGDMomentum(lr=0.01, momentum=0.9)

        for _ in range(5):
            _, wg, bg = model.loss_and_gradients(features, labels)
            opt.step(model.weights, model.biases, wg, bg)

        assert all(np.any(v != 0) for v in opt._v_w)
        assert all(np.any(v != 0) for v in opt._v_b)

    def test_invalid_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr must be positive"):
            SGDMomentum(lr=-0.01)

    def test_invalid_momentum_raises(self) -> None:
        with pytest.raises(ValueError, match="momentum"):
            SGDMomentum(lr=0.01, momentum=1.0)


# ---- Adam ------------------------------------------------------------------

class TestAdam:
    def test_converges_on_easy_problem(self) -> None:
        model, features, labels = _easy_problem()
        opt = Adam(lr=0.01)
        losses = _run_training(model, features, labels, opt, 200)
        assert losses[-1] < losses[0] * 0.05

    def test_bias_correction_is_significant_early(self) -> None:
        """At t=1, the bias correction factor for beta1=0.9 should be ~10x."""
        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        model, features, labels = _easy_problem()
        _, wg, bg = model.loss_and_gradients(features, labels)
        opt.step(model.weights, model.biases, wg, bg)

        # After one step, bc1 = 1 - 0.9 = 0.1
        # so m_hat = m / 0.1 = 10 * m
        assert opt._t == 1
        bc1 = 1 - 0.9 ** 1
        assert abs(bc1 - 0.1) < 1e-12

    def test_step_counter_increments(self) -> None:
        model, features, labels = _easy_problem()
        opt = Adam(lr=0.001)

        for expected_t in range(1, 6):
            _, wg, bg = model.loss_and_gradients(features, labels)
            opt.step(model.weights, model.biases, wg, bg)
            assert opt._t == expected_t

    def test_moment_estimates_are_nonzero_after_update(self) -> None:
        model, features, labels = _easy_problem()
        opt = Adam(lr=0.001)
        _, wg, bg = model.loss_and_gradients(features, labels)
        opt.step(model.weights, model.biases, wg, bg)

        assert all(np.any(m != 0) for m in opt._m_w)
        assert all(np.any(v != 0) for v in opt._v_w)

    def test_different_from_sgd(self) -> None:
        """Adam and SGD should not produce identical parameter trajectories."""
        model_adam, features, labels = _easy_problem()
        model_sgd = MLP((4, 8, 2), seed=7)

        adam = Adam(lr=0.01)
        sgd = SGDMomentum(lr=0.01, momentum=0.0)

        _run_training(model_adam, features, labels, adam, 50)
        _run_training(model_sgd, features, labels, sgd, 50)

        # At least one weight matrix should differ significantly
        diffs = [np.max(np.abs(wa - ws))
                 for wa, ws in zip(model_adam.weights, model_sgd.weights)]
        assert max(diffs) > 0.01

    def test_invalid_beta1_raises(self) -> None:
        with pytest.raises(ValueError, match="beta1"):
            Adam(beta1=1.0)

    def test_invalid_beta2_raises(self) -> None:
        with pytest.raises(ValueError, match="beta2"):
            Adam(beta2=-0.1)

    def test_invalid_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps must be positive"):
            Adam(eps=0.0)
