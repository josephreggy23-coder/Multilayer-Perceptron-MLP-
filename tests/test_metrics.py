import numpy as np

from mlp_baseline.metrics import classification_summary, confusion_matrix, expected_calibration_error


def test_confusion_matrix_counts_actual_rows_and_prediction_columns() -> None:
    labels = np.array([0, 0, 1, 1, 2])
    predictions = np.array([0, 1, 1, 1, 0])

    np.testing.assert_array_equal(
        confusion_matrix(labels, predictions, classes=3),
        np.array([[1, 1, 0], [0, 2, 0], [1, 0, 0]]),
    )


def test_classification_summary_reports_expected_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.4, 0.6]])
    summary = classification_summary(labels, probabilities)

    assert summary["accuracy"] == 0.75
    assert np.isclose(summary["macro_precision"], 5.0 / 6.0)
    assert summary["macro_recall"] == 0.75
    assert summary["per_class"][0]["support"] == 2


def test_perfectly_calibrated_predictions_have_zero_ece() -> None:
    labels = np.array([0, 1])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert expected_calibration_error(labels, probabilities) == 0.0
