"""The fault detector: window size + MinMax scaler + classifier, trained and deployed as one unit."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler

from . import __version__
from .models import build_model
from .schema import FEATURE_COLUMNS, N_FEATURES, fault_name
from .windowing import effective_window, stack_windows

BUNDLE_FORMAT = "ml-manufacturing-detector/1"


@dataclass
class Evaluation:
    """Test-set metrics of a detector (precision, recall and F1 are macro-averaged)."""

    y_true: np.ndarray
    y_pred: np.ndarray
    predict_seconds: float

    @property
    def labels(self) -> list[int]:
        return sorted(set(self.y_true.tolist()) | set(self.y_pred.tolist()))

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_pred))

    @property
    def precision(self) -> float:
        return float(precision_score(self.y_true, self.y_pred, average="macro", zero_division=0))

    @property
    def recall(self) -> float:
        return float(recall_score(self.y_true, self.y_pred, average="macro", zero_division=0))

    @property
    def f1(self) -> float:
        return float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))

    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred, labels=self.labels)

    def report(self) -> str:
        names = [f"{code:>2} {fault_name(code)}" for code in self.labels]
        return classification_report(self.y_true, self.y_pred, labels=self.labels, target_names=names, zero_division=0)


class FaultDetector:
    """Classifies windows of press-station samples into fault codes.

    The scaler is fitted on training data only and stored with the model, so offline
    evaluation and online inference apply exactly the same transformation. With ``clip``
    (default) inputs outside the training range are clipped to it: a signal that never varied
    in the training data (e.g. the analog valve pressures, constant in the digital twin) then
    cannot push a real-plant window far outside the region the classifier has seen.
    """

    def __init__(self, model: str = "svc", window: int = 3, random_state: int = 42, clip: bool = True):
        self.model_name = model
        self.window = int(window)
        self.random_state = random_state
        self.clip = clip
        self.scaler: MinMaxScaler | None = None
        self.model = None
        self.fit_seconds: float | None = None

    @property
    def samples_per_window(self) -> int:
        return effective_window(self.window)

    @property
    def n_inputs(self) -> int:
        return self.samples_per_window * N_FEATURES

    @property
    def classes(self) -> list[int]:
        self._check_fitted()
        return [int(c) for c in self.model.classes_]

    def fit(self, recordings: list[tuple[np.ndarray, np.ndarray]]) -> FaultDetector:
        """Train on ``(signals, labels)`` pairs, one pair per recording."""
        x, y = stack_windows(recordings, self.window)
        started = time.perf_counter()
        self.scaler = MinMaxScaler(clip=self.clip).fit(x)
        self.model = build_model(self.model_name, self.random_state).fit(self.scaler.transform(x), y)
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Fault codes for already windowed inputs of shape ``(n, samples_per_window * 51)``."""
        self._check_fitted()
        windows = np.atleast_2d(np.asarray(windows, dtype=float))
        if windows.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs per window, got {windows.shape[1]}")
        return self.model.predict(self.scaler.transform(windows)).astype(int)

    def confidence(self, windows: np.ndarray) -> np.ndarray | None:
        """Probability of the predicted class, if the classifier provides probabilities."""
        self._check_fitted()
        if not hasattr(self.model, "predict_proba"):
            return None
        windows = np.atleast_2d(np.asarray(windows, dtype=float))
        return self.model.predict_proba(self.scaler.transform(windows)).max(axis=1)

    def evaluate(self, recordings: list[tuple[np.ndarray, np.ndarray]]) -> Evaluation:
        x, y = stack_windows(recordings, self.window)
        started = time.perf_counter()
        y_pred = self.predict(x)
        return Evaluation(y_true=y.astype(int), y_pred=y_pred, predict_seconds=time.perf_counter() - started)

    def save(self, path: str | Path) -> Path:
        """Persist the detector (joblib/pickle format: only load bundles from trusted sources)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format": BUNDLE_FORMAT,
            "model_name": self.model_name,
            "window": self.window,
            "random_state": self.random_state,
            "features": list(FEATURE_COLUMNS),
            "scaler": self.scaler,
            "model": self.model,
            "versions": {"ml-in-manufacturing": __version__, "scikit-learn": sklearn.__version__},
        }
        joblib.dump(bundle, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> FaultDetector:
        """Load a trusted detector bundle, including bundles saved before the project rename."""
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, dict) or bundle.get("format") not in {BUNDLE_FORMAT, "twinsight-detector/1"}:
            raise ValueError(f"{path} is not a manufacturing ML detector bundle")
        if bundle["features"] != list(FEATURE_COLUMNS):
            raise ValueError(f"{path} was trained on a different signal schema")
        detector = cls(bundle["model_name"], bundle["window"], bundle["random_state"], bundle["scaler"].clip)
        detector.scaler, detector.model = bundle["scaler"], bundle["model"]
        return detector

    def _check_fitted(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("detector is not trained yet")
