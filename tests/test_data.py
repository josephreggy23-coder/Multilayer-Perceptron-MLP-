import numpy as np

from mlp_baseline.data import make_spiral, standardize, train_test_split


def test_spiral_data_has_all_requested_classes() -> None:
    features, labels = make_spiral(samples_per_class=10, classes=3, seed=1)

    assert features.shape == (30, 2)
    np.testing.assert_array_equal(np.bincount(labels), np.array([10, 10, 10]))


def test_standardization_uses_training_statistics() -> None:
    train = np.array([[1.0, 3.0], [3.0, 7.0]])
    test = np.array([[5.0, 11.0]])
    standardized_train, standardized_test = standardize(train, test)

    np.testing.assert_allclose(standardized_train.mean(axis=0), np.zeros(2))
    np.testing.assert_allclose(standardized_test, np.array([[3.0, 3.0]]))


def test_split_covers_every_input_row() -> None:
    features = np.arange(20, dtype=float).reshape(10, 2)
    labels = np.arange(10)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_fraction=0.3, seed=2)

    assert x_train.shape[0] + x_test.shape[0] == 10
    np.testing.assert_array_equal(np.sort(np.concatenate((y_train, y_test))), labels)
