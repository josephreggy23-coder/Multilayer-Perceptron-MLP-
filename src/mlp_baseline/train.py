"""Run a complete, reproducible MLP classification experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import make_spiral, standardize, train_test_split
from .model import MLP
from .training import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on spiral data.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics.json"))
    arguments = parser.parse_args()

    features, labels = make_spiral()
    x_train, x_test, y_train, y_test = train_test_split(features, labels)
    x_train, x_test = standardize(x_train, x_test)
    model = MLP((2, 32, 32, 3))
    history = train(model, x_train, y_train, TrainingConfig(epochs=arguments.epochs))
    summary = {
        "architecture": list(model.layer_sizes),
        "epochs": arguments.epochs,
        "final_train_loss": history[-1]["loss"],
        "final_train_accuracy": history[-1]["accuracy"],
        "test_accuracy": model.score(x_test, y_test),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
