"""First-order optimizers with momentum and adaptive learning rates.

Gradient descent with a fixed learning rate is a reasonable starting
point, but two well-understood modifications make it far more practical
on real loss surfaces.

SGD with Momentum (Polyak 1964)
-------------------------------
Vanilla SGD updates parameters as

    theta <- theta - lr * g

where g = dL/d(theta).  The trajectory oscillates across narrow valleys
because each gradient points toward the nearest wall, not along the
valley floor.

Momentum accumulates a running average of past gradients:

    v <- mu * v + g
    theta <- theta - lr * v

where mu in [0, 1) is the momentum coefficient and v is the velocity.
The velocity integrates the consistent component of the gradient while
cancelling oscillations, much like a heavy ball rolling downhill.  In
the quadratic bowl model, momentum reduces the number of iterations
needed to cross a valley of condition number kappa from O(kappa) to
O(sqrt(kappa)).

The Nesterov variant (Nesterov 1983) evaluates the gradient at the
"lookahead" position theta - lr * mu * v rather than at theta.  This
gives the correction a chance to brake before overshooting:

    v <- mu * v + grad(theta - lr * mu * v)
    theta <- theta - lr * v

In practice, Nesterov momentum is implemented by applying the momentum
step first, computing the gradient at the displaced position, and then
applying the gradient step.  This module uses the standard formulation,
which is algebraically equivalent and avoids the lookahead evaluation.

Adam (Kingma & Ba 2015)
-----------------------
Adam maintains per-parameter running estimates of the first moment (mean)
and the second raw moment (uncentered variance) of the gradient:

    m <- beta1 * m + (1 - beta1) * g          (first moment estimate)
    v <- beta2 * v + (1 - beta2) * g^2        (second moment estimate)

Because m and v are initialized at zero, they are biased toward zero
during the early iterations.  Bias-corrected estimates are:

    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)

The parameter update divides the bias-corrected first moment by the
square root of the bias-corrected second moment:

    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

This achieves two things simultaneously:
  1. The ratio m_hat / sqrt(v_hat) normalizes the step so that
     parameters with large gradients take smaller steps and vice versa.
  2. The first-moment tracking provides momentum-like behavior.

The default hyperparameters (beta1=0.9, beta2=0.999, eps=1e-8) work
well across a wide range of problems.

References
----------
Polyak, B. T. (1964).  Some methods of speeding up the convergence of
    iteration methods.  USSR Computational Mathematics and Mathematical
    Physics.
Nesterov, Y. (1983).  A method for unconstrained convex minimization
    problem with the rate of convergence O(1/k^2).  Doklady AN USSR.
Kingma, D. P. & Ba, J. (2015).  Adam: A Method for Stochastic
    Optimization.  ICLR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class Optimizer(ABC):
    """Base class for parameter optimizers.

    Subclasses implement ``step``, which updates weights and biases given
    their respective gradients.
    """

    @abstractmethod
    def step(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        weight_grads: list[np.ndarray],
        bias_grads: list[np.ndarray],
    ) -> None:
        """Apply one parameter update in place."""


class SGDMomentum(Optimizer):
    """Stochastic gradient descent with (optional Nesterov) momentum.

    Parameters
    ----------
    lr : float
        Learning rate (step size).
    momentum : float
        Momentum coefficient mu in [0, 1). Zero recovers vanilla SGD.
    nesterov : bool
        If True, use the Nesterov accelerated gradient variant.

    Mathematical note
    -----------------
    With nesterov=False the update is:

        v_w[l] = mu * v_w[l] + dL/dW[l]
        W[l]  -= lr * v_w[l]

    With nesterov=True, the effective update (Sutskever et al. 2013
    reformulation that avoids the lookahead evaluation) is:

        v_w[l] = mu * v_w[l] + dL/dW[l]
        W[l]  -= lr * (mu * v_w[l] + dL/dW[l])

    The Nesterov form applies the momentum term twice in one step,
    effectively anticipating where the next gradient will be.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self._v_w: list[np.ndarray] | None = None
        self._v_b: list[np.ndarray] | None = None

    def _init_state(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
    ) -> None:
        self._v_w = [np.zeros_like(w) for w in weights]
        self._v_b = [np.zeros_like(b) for b in biases]

    def step(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        weight_grads: list[np.ndarray],
        bias_grads: list[np.ndarray],
    ) -> None:
        if self._v_w is None:
            self._init_state(weights, biases)

        for i in range(len(weights)):
            self._v_w[i] = self.momentum * self._v_w[i] + weight_grads[i]
            self._v_b[i] = self.momentum * self._v_b[i] + bias_grads[i]

            if self.nesterov:
                weights[i] -= self.lr * (
                    self.momentum * self._v_w[i] + weight_grads[i]
                )
                biases[i] -= self.lr * (
                    self.momentum * self._v_b[i] + bias_grads[i]
                )
            else:
                weights[i] -= self.lr * self._v_w[i]
                biases[i] -= self.lr * self._v_b[i]


class Adam(Optimizer):
    r"""Adam optimizer with bias-corrected moment estimates.

    Parameters
    ----------
    lr : float
        Step size alpha.  Unlike SGD, the effective step is normalized by
        the second moment, so values around 1e-3 are typical.
    beta1 : float
        Exponential decay rate for the first moment (gradient mean).
    beta2 : float
        Exponential decay rate for the second moment (gradient variance).
    eps : float
        Small constant for numerical stability in the denominator.

    Mathematical note
    -----------------
    Per-parameter update at step t (for weight matrix W[l]):

        m[l] = beta1 * m[l] + (1 - beta1) * dL/dW[l]
        v[l] = beta2 * v[l] + (1 - beta2) * (dL/dW[l])^2

        m_hat[l] = m[l] / (1 - beta1^t)
        v_hat[l] = v[l] / (1 - beta2^t)

        W[l] -= lr * m_hat[l] / (sqrt(v_hat[l]) + eps)

    The bias correction compensates for the zero initialization of
    m and v.  Without it, the early estimates are systematically too
    small: at t=1, m = (1-beta1)*g, which for beta1=0.9 is only 10%
    of the true gradient mean.  Dividing by (1-beta1^t) = 0.1 recovers
    the full scale.  As t grows, beta1^t vanishes and the correction
    becomes negligible.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t: int = 0
        self._m_w: list[np.ndarray] | None = None
        self._m_b: list[np.ndarray] | None = None
        self._v_w: list[np.ndarray] | None = None
        self._v_b: list[np.ndarray] | None = None

    def _init_state(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
    ) -> None:
        self._m_w = [np.zeros_like(w) for w in weights]
        self._m_b = [np.zeros_like(b) for b in biases]
        self._v_w = [np.zeros_like(w) for w in weights]
        self._v_b = [np.zeros_like(b) for b in biases]

    def step(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        weight_grads: list[np.ndarray],
        bias_grads: list[np.ndarray],
    ) -> None:
        if self._m_w is None:
            self._init_state(weights, biases)

        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        for i in range(len(weights)):
            # --- weights ---
            self._m_w[i] = self.beta1 * self._m_w[i] + (1 - self.beta1) * weight_grads[i]
            self._v_w[i] = self.beta2 * self._v_w[i] + (1 - self.beta2) * weight_grads[i] ** 2

            m_hat = self._m_w[i] / bc1
            v_hat = self._v_w[i] / bc2
            weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # --- biases ---
            self._m_b[i] = self.beta1 * self._m_b[i] + (1 - self.beta1) * bias_grads[i]
            self._v_b[i] = self.beta2 * self._v_b[i] + (1 - self.beta2) * bias_grads[i] ** 2

            m_hat_b = self._m_b[i] / bc1
            v_hat_b = self._v_b[i] / bc2
            biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
