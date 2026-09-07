import numpy as np

from mlp_baseline.activations import cross_entropy, softmax


def test_softmax_rows_are_probabilities() -> None:
    probabilities = softmax(np.array([[1000.0, 999.0], [-2.0, 1.0]]))

    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    assert np.all(probabilities > 0.0)


def test_cross_entropy_rewards_correct_predictions() -> None:
    labels = np.array([0, 1])
    confident = np.array([[0.99, 0.01], [0.02, 0.98]])
    uncertain = np.array([[0.55, 0.45], [0.45, 0.55]])

    assert cross_entropy(confident, labels) < cross_entropy(uncertain, labels)
