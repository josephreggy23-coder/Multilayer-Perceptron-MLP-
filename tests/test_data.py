"""Tests for mlp_baseline.data: make_spiral, train_test_split, standardize."""

import numpy as np
import pytest

from mlp_baseline.data import make_spiral, standardize, train_test_split


# ---------------------------------------------------------------------------
# make_spiral
# ---------------------------------------------------------------------------

class TestMakeSpiral:
    def test_output_shape(self) -> None:
        features, labels = make_spiral(samples_per_class=50, classes=3)
        assert features.shape == (150, 2)
        assert labels.shape == (150,)

    def test_label_counts(self) -> None:
        samples, classes = 40, 4
        _, labels = make_spiral(samples_per_class=samples, classes=classes)
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, np.full(classes, samples))

    def test_deterministic_with_same_seed(self) -> None:
        f1, l1 = make_spiral(seed=99)
        f2, l2 = make_spiral(seed=99)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds_differ(self) -> None:
        f1, _ = make_spiral(seed=0)
        f2, _ = make_spiral(seed=1)
        assert not np.array_equal(f1, f2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"samples_per_class": 0},
            {"samples_per_class": -1},
            {"classes": 1},
            {"noise": -0.1},
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            make_spiral(**kwargs)


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_total_count_preserved(self) -> None:
        n = 100
        X = np.random.default_rng(0).standard_normal((n, 3))
        y = np.arange(n)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_fraction=0.25)
        assert X_tr.shape[0] + X_te.shape[0] == n
        assert y_tr.shape[0] + y_te.shape[0] == n

    def test_split_fraction(self) -> None:
        n = 200
        X = np.ones((n, 2))
        y = np.zeros(n)
        _, X_te, _, _ = train_test_split(X, y, test_fraction=0.3)
        expected_test = n - int(n * 0.7)
        assert X_te.shape[0] == expected_test

    def test_row_correspondence_preserved(self) -> None:
        X = np.arange(20).reshape(10, 2).astype(float)
        y = np.arange(10)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_fraction=0.4, seed=7)
        # Each label k should map to row [2k, 2k+1] in the original features.
        for features, labels in [(X_tr, y_tr), (X_te, y_te)]:
            for feat_row, label in zip(features, labels):
                np.testing.assert_array_equal(feat_row, [label * 2, label * 2 + 1])

    @pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_test_fraction_raises(self, bad_frac: float) -> None:
        X = np.ones((10, 2))
        y = np.ones(10)
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(X, y, test_fraction=bad_frac)

    def test_mismatched_shapes_raise(self) -> None:
        X = np.ones((10, 2))
        y = np.ones(8)
        with pytest.raises(ValueError, match="equal"):
            train_test_split(X, y)


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------

class TestStandardize:
    def test_training_set_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(42)
        train = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
        (result,) = standardize(train)
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-12)

    def test_test_set_uses_training_stats(self) -> None:
        train = np.array([[2.0, 4.0], [4.0, 8.0]])
        test = np.array([[3.0, 6.0]])
        _, test_std = standardize(train, test)
        # train mean = [3, 6], train std = [1, 2]
        # test standardized = (3-3)/1, (6-6)/2 = [0, 0]
        np.testing.assert_allclose(test_std, [[0.0, 0.0]], atol=1e-12)

    def test_constant_feature_not_nan(self) -> None:
        train = np.array([[1.0, 5.0], [1.0, 3.0]])
        (result,) = standardize(train)
        assert not np.any(np.isnan(result))
        # Constant column (std=0) should keep original offset: (val - mean) / 1
        np.testing.assert_allclose(result[:, 0], [0.0, 0.0])
