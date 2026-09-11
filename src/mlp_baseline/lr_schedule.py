"""Learning rate schedules for gradient-descent training.

A fixed learning rate is a strong baseline for small problems, but most
real training benefits from reducing the step size over time.  A large
initial rate explores aggressively; a shrinking rate lets the optimizer
settle into a narrow basin.

This module provides three standard schedules, each parameterized so
that the user chooses an initial rate and a decay shape:

Step decay
----------
The rate drops by a constant factor every ``step_size`` epochs:

    lr(t) = lr_0 * gamma^(floor(t / step_size))

This is piecewise-constant: the rate is held for ``step_size`` epochs,
then multiplied by ``gamma``.  It is the simplest schedule and the
easiest to tune: the user chooses when to drop and by how much.

Exponential decay
-----------------
The rate decays smoothly by a fixed fraction each epoch:

    lr(t) = lr_0 * gamma^t

This is the continuous analogue of step decay.  It produces a geometric
sequence of rates, so the log-rate decreases linearly with epoch.

Cosine annealing (Loshchilov & Hutter, 2017)
---------------------------------------------
The rate follows a half-cosine from ``lr_0`` down to ``lr_min``:

    lr(t) = lr_min + (lr_0 - lr_min) * (1 + cos(pi * t / T_max)) / 2

where ``T_max`` is the total number of epochs.  Unlike step and
exponential decay, cosine annealing is non-monotone at the boundary of
a restart cycle.  With ``n_restarts > 0``, the schedule repeats the
cosine wave multiple times (warm restarts), which has been shown to
help escape sharp minima early in training.

Each restart divides the total budget ``T_max`` into ``n_restarts + 1``
equal sub-cycles of length ``T_cycle = T_max / (n_restarts + 1)``, and
the cosine runs from ``lr_0`` to ``lr_min`` within each sub-cycle:

    lr(t) = lr_min + (lr_0 - lr_min) * (1 + cos(pi * t_local / T_cycle)) / 2

where ``t_local = t mod T_cycle``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LRSchedule(ABC):
    """Base class for learning rate schedules."""

    def __init__(self, initial_lr: float) -> None:
        if initial_lr <= 0.0:
            raise ValueError("initial_lr must be positive")
        self.initial_lr = initial_lr

    @abstractmethod
    def __call__(self, epoch: int) -> float:
        """Return the learning rate for the given epoch (0-indexed)."""


class StepDecay(LRSchedule):
    """Drop the learning rate by ``gamma`` every ``step_size`` epochs.

    Parameters
    ----------
    initial_lr
        Starting learning rate.
    step_size
        Number of epochs between rate reductions.
    gamma
        Multiplicative decay factor (0 < gamma <= 1).

    Examples
    --------
    >>> schedule = StepDecay(0.1, step_size=30, gamma=0.1)
    >>> schedule(0)
    0.1
    >>> schedule(30)
    0.010000000000000002
    >>> schedule(60)
    0.0010000000000000002
    """

    def __init__(self, initial_lr: float, step_size: int = 30, gamma: float = 0.1) -> None:
        super().__init__(initial_lr)
        if step_size <= 0:
            raise ValueError("step_size must be a positive integer")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, epoch: int) -> float:
        return self.initial_lr * self.gamma ** (epoch // self.step_size)


class ExponentialDecay(LRSchedule):
    """Multiply the learning rate by ``gamma`` each epoch.

    Parameters
    ----------
    initial_lr
        Starting learning rate.
    gamma
        Per-epoch multiplicative factor (0 < gamma <= 1).

    Examples
    --------
    >>> schedule = ExponentialDecay(0.1, gamma=0.95)
    >>> f"{schedule(0):.4f}"
    '0.1000'
    >>> f"{schedule(10):.4f}"
    '0.0599'
    """

    def __init__(self, initial_lr: float, gamma: float = 0.95) -> None:
        super().__init__(initial_lr)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

    def __call__(self, epoch: int) -> float:
        return self.initial_lr * self.gamma ** epoch


class CosineAnnealing(LRSchedule):
    """Cosine annealing with optional warm restarts.

    Parameters
    ----------
    initial_lr
        Peak learning rate at the start of each cycle.
    total_epochs
        Total training budget in epochs.
    lr_min
        Minimum learning rate at the bottom of each cosine trough.
    n_restarts
        Number of warm restarts. 0 means a single cosine decay over the
        full budget; ``k`` means ``k + 1`` equal cosine sub-cycles.

    Examples
    --------
    >>> schedule = CosineAnnealing(0.1, total_epochs=100, lr_min=0.0)
    >>> f"{schedule(0):.4f}"
    '0.1000'
    >>> f"{schedule(50):.4f}"
    '0.0500'
    >>> f"{schedule(100):.4f}"
    '0.0000'
    """

    def __init__(
        self,
        initial_lr: float,
        total_epochs: int,
        lr_min: float = 0.0,
        n_restarts: int = 0,
    ) -> None:
        super().__init__(initial_lr)
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if lr_min < 0.0:
            raise ValueError("lr_min must be non-negative")
        if lr_min >= initial_lr:
            raise ValueError("lr_min must be less than initial_lr")
        if n_restarts < 0:
            raise ValueError("n_restarts must be non-negative")
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.n_restarts = n_restarts
        self.cycle_length = total_epochs / (n_restarts + 1)

    def __call__(self, epoch: int) -> float:
        t_local = epoch % self.cycle_length
        cosine = 0.5 * (1.0 + np.cos(np.pi * t_local / self.cycle_length))
        return self.lr_min + (self.initial_lr - self.lr_min) * cosine
