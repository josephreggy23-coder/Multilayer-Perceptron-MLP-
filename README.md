# Multilayer Perceptron (MLP), from scratch

[![Tests](https://github.com/josephreggy23-coder/Multilayer-Perceptron-MLP-/actions/workflows/tests.yml/badge.svg)](https://github.com/josephreggy23-coder/Multilayer-Perceptron-MLP-/actions/workflows/tests.yml)

An educational multiclass classifier implemented in **NumPy only**. It trains a
fully connected neural network on a non-linearly separable spiral dataset,
without relying on PyTorch, TensorFlow, or scikit-learn.

The project is deliberately compact enough to audit. It exposes the mathematics
behind a standard MLP: affine transformations, ReLU activations, softmax,
cross-entropy, and reverse-mode differentiation (backpropagation).

## Model

For hidden layer \(l\), the MLP applies

\[
z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}, \qquad
a^{(l)} = \max(0, z^{(l)}).
\]

The final layer produces a categorical probability distribution:

\[
p_k = \frac{\exp(z_k - \max_j z_j)}{\sum_j \exp(z_j - \max_j z_j)}.
\]

It minimizes mean cross-entropy,

\[
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i},
\]

using mini-batch gradient descent. The output-layer derivative simplifies to
\(\partial \mathcal{L}/\partial z = (p - y)/N\); all earlier derivatives are
calculated by the chain rule.

## Features

- Vectorized forward pass and exact backpropagation.
- He parameter initialization for ReLU layers.
- Numerically stable softmax and clipped cross-entropy.
- Deterministic synthetic spiral data, train/test split, and training loop.
- Input validation and automated unit tests.

## Quick start

Use Python 3.10+ and install the package in editable mode:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest -q
```

Train the default architecture, `2 → 32 → 32 → 3`:

```bash
python -m mlp_baseline.train --epochs 300
```

The runner prints a JSON experiment summary and writes it to
`artifacts/metrics.json`. With the default deterministic seed, a 100-epoch
smoke run reaches approximately **97–99% test accuracy** on the held-out spiral
data. Exact numbers can vary across NumPy versions.

## Project layout

```text
src/mlp_baseline/
  activations.py  # ReLU, stable softmax, cross-entropy
  model.py        # MLP parameters, forward pass, backpropagation
  data.py         # spiral generator, split, standardization
  training.py     # mini-batch optimization loop
  train.py        # command-line experiment
tests/            # math, model, and data tests
```

## Limits and next steps

This is a transparent baseline, not a production deep-learning framework. Good
extensions include momentum/Adam, validation-based early stopping, decision
boundary plots, and comparisons with logistic regression.
