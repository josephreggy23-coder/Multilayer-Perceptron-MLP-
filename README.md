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
- Full evaluation: confusion matrix, per-class precision/recall/F1, mean
  confidence, and expected calibration error (ECE).
- Input validation and automated unit tests.

## How it works

The experiment follows this reproducible pipeline:

1. **Generate data.** Three intertwined spirals create 450 two-dimensional
   observations. This is intentionally non-linear, so a linear classifier is
   not enough.
2. **Hold out data.** A seeded shuffle reserves 20% (90 samples) for testing.
   Standardization is fit on training data only, preventing test-set leakage.
3. **Initialize the network.** The default `2 → 32 → 32 → 3` architecture has
   two ReLU hidden layers. Weights use He initialization, which keeps the scale
   of signals stable through ReLU layers.
4. **Learn by backpropagation.** Each epoch shuffles the training set, takes
   mini-batches of 32, calculates the loss gradient, and moves every parameter
   in the direction that lowers the loss.
5. **Evaluate honestly.** The held-out samples are never used to update the
   weights. The project reports class-by-class performance and whether the
   model's confidence matches its observed accuracy.

## Baseline results

The following is a reproducible 300-epoch run with the repository's default
seed and configuration. The test set contains 90 held-out samples.

| Statistic | Result | Meaning |
| --- | ---: | --- |
| Final training loss | 0.0230 | Cross-entropy after optimization; lower is better. |
| Training accuracy | 99.44% | Correct classifications on the 360 training samples. |
| Test accuracy | 98.89% | Correct classifications on unseen data. |
| Macro precision | 98.67% | Across classes, predicted labels were usually correct. |
| Macro recall | 99.17% | Across classes, true examples were found. |
| Macro F1 | 98.90% | Balanced precision/recall summary. |
| Mean confidence | 97.96% | Average probability assigned to the predicted class. |
| ECE (10 bins) | 2.75% | Average confidence/accuracy mismatch; lower is better. |

The resulting confusion matrix uses **rows = actual class** and **columns =
predicted class**:

```text
          predicted
actual      0   1   2
      0    39   0   1
      1     0  26   0
      2     0   0  24
```

Per-class results are `F1 = 98.73%` (class 0), `100.00%` (class 1), and
`97.96%` (class 2). These are benchmark results for synthetic data, not a
claim about performance on a real-world dataset.

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
`artifacts/metrics.json`. It includes accuracy, loss, macro precision/recall/
F1, calibration error, a confusion matrix, and per-class scores. Exact numbers
can vary slightly across NumPy versions.

## Project layout

```text
src/mlp_baseline/
  activations.py  # ReLU, stable softmax, cross-entropy
  model.py        # MLP parameters, forward pass, backpropagation
  data.py         # spiral generator, split, standardization
  metrics.py      # confusion matrix, F1, calibration statistics
  training.py     # mini-batch optimization loop
  train.py        # command-line experiment
tests/            # math, model, and data tests
```

## Limits and next steps

This is a transparent baseline, not a production deep-learning framework. Good
extensions include momentum/Adam, validation-based early stopping, decision
boundary plots, and comparisons with logistic regression.
