import numpy as np

from mlp_baseline import MLP


def test_predictions_have_the_expected_shape() -> None:
    model = MLP((2, 6, 3), seed=1)
    probabilities = model.predict_proba(np.array([[0.2, -0.5], [1.0, 0.0]]))

    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))


def test_one_update_reduces_loss_on_an_easy_batch() -> None:
    features = np.array([[-1.0, -1.0], [-0.8, -1.2], [1.0, 1.0], [1.2, 0.9]])
    labels = np.array([0, 0, 1, 1])
    model = MLP((2, 8, 2), seed=7)

    initial_loss, _, _ = model.loss_and_gradients(features, labels)
    for _ in range(100):
        model.train_batch(features, labels, learning_rate=0.1)
    final_loss, _, _ = model.loss_and_gradients(features, labels)

    assert final_loss < initial_loss
    assert model.score(features, labels) == 1.0
