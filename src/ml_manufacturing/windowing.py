"""Sliding windows over event-driven recordings.

A window of size ``w`` turns ``w`` consecutive samples into one feature vector
``[x(t), x(t+1), ..., x(t+w-1)]`` and takes the label of the newest sample. Windows never
cross recording boundaries. Window size 0 means "no windowing" (one sample per vector),
matching the naming used in the project experiments.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def effective_window(window: int) -> int:
    """Number of samples per feature vector (window 0 and 1 both mean a single sample)."""
    if window < 0:
        raise ValueError("window size must be >= 0")
    return max(1, int(window))


def make_windows(features: np.ndarray, labels: np.ndarray | None, window: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Window one recording. Returns ``(X, y)`` with ``X.shape == (n - w + 1, w * n_features)``."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2-D array (samples x signals)")
    size = effective_window(window)
    n_samples, n_features = features.shape
    if n_samples < size:
        empty_y = None if labels is None else np.empty(0, dtype=int)
        return np.empty((0, size * n_features)), empty_y

    windows = sliding_window_view(features, (size, n_features))[:, 0]
    x = windows.reshape(n_samples - size + 1, size * n_features)
    if labels is None:
        return x, None
    labels = np.asarray(labels)
    if len(labels) != n_samples:
        raise ValueError("labels and features must have the same length")
    return x, labels[size - 1 :]


def stack_windows(recordings: list[tuple[np.ndarray, np.ndarray]], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Window every recording separately and stack the results."""
    parts = [make_windows(features, labels, window) for features, labels in recordings]
    parts = [(x, y) for x, y in parts if len(x)]
    if not parts:
        raise ValueError("no recording is long enough for the requested window size")
    return np.vstack([x for x, _ in parts]), np.concatenate([y for _, y in parts])
