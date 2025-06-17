import numpy as np
import pytest

import neural_cryptanalyst as nc


def test_align_traces_empty():
    """align_traces should raise IndexError for empty input."""
    empty = np.empty((0, 10))
    with pytest.raises(IndexError):
        nc.align_traces(empty)


def test_preprocess_constant_traces():
    """preprocess_traces should handle constant traces without NaN."""
    traces = np.ones((5, 20)) * 7
    processed = nc.preprocess_traces(traces)
    assert np.all(np.isfinite(processed))


def test_guessing_entropy_invalid_shape():
    """calculate_guessing_entropy returns default value for wrong shape."""
    preds = np.random.rand(10, 10)
    result = nc.calculate_guessing_entropy(preds, 0, [1, 5])
    assert np.all(result == 128)


def test_success_rate_zero_preds():
    """calculate_success_rate should handle zero predictions gracefully."""
    preds = np.zeros((1, 5, 256))
    ge = nc.calculate_success_rate(preds, 0, [1, 3])
    assert np.all(ge == 0)


def test_select_poi_no_traces():
    """select_points_of_interest returns fallback indices for empty input."""
    traces = np.empty((0, 5))
    labels = np.array([], dtype=int)
    poi_idx, poi_traces = nc.select_points_of_interest(traces, labels)
    assert poi_idx.size == 5
    assert poi_traces.shape == traces.shape
