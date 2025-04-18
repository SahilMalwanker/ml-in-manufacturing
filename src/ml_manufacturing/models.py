"""Classifier catalogue with the hyperparameters used in the study."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# class_weight="balanced" counteracts the dominance of normal-behaviour samples.
_FACTORIES: dict[str, tuple[str, Callable[[int], ClassifierMixin]]] = {
    "svc": (
        "Support vector classifier (RBF kernel)",
        lambda seed: SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
    ),
    "dt": (
        "Decision tree",
        lambda seed: DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed),
    ),
    "rf": (
        "Random forest",
        lambda seed: RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
    ),
    # same 128-64 dense layout as the project's Keras network, without the TensorFlow dependency
    "mlp": (
        "Neural network (dense 128-64)",
        lambda seed: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=seed),
    ),
}

MODEL_NAMES: tuple[str, ...] = tuple(_FACTORIES)


def describe(name: str) -> str:
    return _FACTORIES[name][0]


def build_model(name: str, random_state: int = 42) -> ClassifierMixin:
    """A fresh, unfitted classifier."""
    if name not in _FACTORIES:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
    return _FACTORIES[name][1](random_state)
