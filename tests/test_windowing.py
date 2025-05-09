import numpy as np
import pytest

from ml_manufacturing.windowing import effective_window, make_windows, stack_windows


def _toy(n=6, f=2):
    features = np.arange(n * f, dtype=float).reshape(n, f)
    labels = np.arange(n) * 10
    return features, labels


def test_window_layout_and_label_of_newest_sample():
    features, labels = _toy()
    x, y = make_windows(features, labels, 3)
    assert x.shape == (4, 6)
    np.testing.assert_array_equal(x[0], np.concatenate(features[0:3]))  # [x(t), x(t+1), x(t+2)]
    np.testing.assert_array_equal(x[-1], np.concatenate(features[3:6]))
    np.testing.assert_array_equal(y, labels[2:])


def test_window_zero_means_single_samples():
    features, labels = _toy()
    for size in (0, 1):
        x, y = make_windows(features, labels, size)
        np.testing.assert_array_equal(x, features)
        np.testing.assert_array_equal(y, labels)
    assert effective_window(0) == effective_window(1) == 1
    with pytest.raises(ValueError):
        effective_window(-1)


def test_short_recordings_yield_no_windows():
    features, labels = _toy(n=2)
    x, y = make_windows(features, labels, 5)
    assert x.shape == (0, 10) and y.shape == (0,)


def test_windows_never_cross_recordings():
    a, b = _toy(n=4), _toy(n=4)
    b = (b[0] + 100, b[1] + 1)
    x, _ = stack_windows([a, b], 3)
    assert len(x) == 2 + 2  # (4 - 3 + 1) windows per recording
    np.testing.assert_array_equal(x[1], np.concatenate(a[0][1:4]))
    np.testing.assert_array_equal(x[2], np.concatenate(b[0][0:3]))


def test_label_never_leaks_into_features(rp_train):
    """Each window holds exactly window x 51 signal values - the label column is never an input."""
    for size in (0, 3, 5):
        x, y = stack_windows(rp_train, size)
        assert x.shape[1] == max(1, size) * 51
        assert len(x) == len(y)


def test_mismatched_lengths_are_rejected():
    features, labels = _toy()
    with pytest.raises(ValueError):
        make_windows(features, labels[:-1], 3)
    with pytest.raises(ValueError):
        stack_windows([(features[:1], labels[:1])], 3)
