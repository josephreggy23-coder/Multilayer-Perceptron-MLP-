"""A compact, educational multilayer perceptron implementation."""

from .model import MLP
from .weight_init import initialize_weights

__all__ = ["MLP", "initialize_weights"]
