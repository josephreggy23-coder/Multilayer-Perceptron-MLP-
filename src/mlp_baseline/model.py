"""A fully connected classifier trained with vectorized backpropagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import cross_entropy, relu, relu_gradient, softmax


@dataclass(frozen=True)
class BatchMetrics:
    """Metrics reported after one parameter update."""

    loss: float
    accuracy: float


class MLP:
    """Multiclass MLP with ReLU hidden layers and a softmax output layer.

    Parameters are initialized with He initialization.  Given a batch X, a
    layer computes ``X @ W + b``; hidden layers apply ReLU and the output
    probabilities are the softmax of the final logits.
    """

    def __init__(self, layer_sizes: tuple[int, ...], seed: int = 42) -> None:
        if len(layer_sizes) < 2 or any(size <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive sizes")

        self.layer_sizes = layer_sizes
        generator = np.random.default_rng(seed)
        self.weights = [
            generator.normal(0.0, np.sqrt(2.0 / input_size), (input_size, output_size))
            for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros((1, output_size)) for output_size in layer_sizes[1:]]

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Return probabilities plus the cached activations and pre-activations."""
        if features.ndim != 2 or features.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"features must have shape (n, {self.layer_sizes[0]})")

        activations = [features]
        pre_activations: list[np.ndarray] = []
        current = features
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            linear = current @ weight + bias
            pre_activations.append(linear)
            current = softmax(linear) if index == len(self.weights) - 1 else relu(linear)
            activations.append(current)
        return current, activations, pre_activations

    def loss_and_gradients(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """Compute cross-entropy and exact vectorized gradients for one batch."""
        labels = self._validate_labels(labels, features.shape[0])
        probabilities, activations, pre_activations = self.forward(features)
        loss = cross_entropy(probabilities, labels)
        samples = features.shape[0]

        weight_gradients = [np.empty_like(weight) for weight in self.weights]
        bias_gradients = [np.empty_like(bias) for bias in self.biases]
        gradient = probabilities.copy()
        gradient[np.arange(samples), labels] -= 1.0
        gradient /= samples

        for index in range(len(self.weights) - 1, -1, -1):
            weight_gradients[index] = activations[index].T @ gradient
            bias_gradients[index] = np.sum(gradient, axis=0, keepdims=True)
            if index > 0:
                gradient = (gradient @ self.weights[index].T) * relu_gradient(pre_activations[index - 1])

        return loss, weight_gradients, bias_gradients

    def train_batch(self, features: np.ndarray, labels: np.ndarray, learning_rate: float) -> BatchMetrics:
        """Perform one gradient-descent update and return metrics before it."""
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        loss, weight_gradients, bias_gradients = self.loss_and_gradients(features, labels)
        for index in range(len(self.weights)):
            self.weights[index] -= learning_rate * weight_gradients[index]
            self.biases[index] -= learning_rate * bias_gradients[index]
        return BatchMetrics(loss=loss, accuracy=self.score(features, labels))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return a probability vector for every feature row."""
        probabilities, _, _ = self.forward(features)
        return probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the most likely class index for every feature row."""
        return np.argmax(self.predict_proba(features), axis=1)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy."""
        labels = self._validate_labels(labels, features.shape[0])
        return float(np.mean(self.predict(features) == labels))

    def _validate_labels(self, labels: np.ndarray, samples: int) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.shape[0] != samples:
            raise ValueError("labels must be a one-dimensional array matching features")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must contain integer class indices")
        if np.any(labels < 0) or np.any(labels >= self.layer_sizes[-1]):
            raise ValueError("labels must be within the output class range")
        return labels
