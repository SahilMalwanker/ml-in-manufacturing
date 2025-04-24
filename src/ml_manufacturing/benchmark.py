"""Train/test every scenario x model x window combination of the study on a manifest."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO

import pandas as pd

from .datasets import SCENARIOS, Recording, scenario_data
from .detector import FaultDetector
from .models import MODEL_NAMES
from .windowing import stack_windows

COLUMNS = (
    "scenario",
    "model",
    "window",
    "train_windows",
    "test_windows",
    "test_classes",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "train_s",
    "predict_us_per_window",
)


def run_benchmark(
    recordings: list[Recording],
    scenarios: Iterable[str] = tuple(SCENARIOS),
    models: Iterable[str] = MODEL_NAMES,
    windows: Iterable[int] = (0, 3, 5, 10),
    random_state: int = 42,
    clip: bool = True,
    out: TextIO | None = sys.stdout,
) -> pd.DataFrame:
    """Return one row of test-set metrics per combination (precision/recall/F1 macro-averaged).

    Recordings shorter than the window yield no windows, so ``test_classes`` reports how many
    fault classes are actually present in the test windows of each row.
    """
    rows = []
    for scenario in scenarios:
        train, test = scenario_data(recordings, scenario)
        for window in windows:
            n_train = len(stack_windows(train, window)[1])
            for model in models:
                detector = FaultDetector(model, window, random_state, clip).fit(train)
                result = detector.evaluate(test)
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "window": window,
                        "train_windows": n_train,
                        "test_windows": len(result.y_true),
                        "test_classes": len(set(result.y_true.tolist())),
                        "accuracy": round(result.accuracy, 4),
                        "precision": round(result.precision, 4),
                        "recall": round(result.recall, 4),
                        "f1": round(result.f1, 4),
                        "train_s": round(detector.fit_seconds, 4),
                        "predict_us_per_window": round(1e6 * result.predict_seconds / len(result.y_true), 2),
                    }
                )
                if out is not None:
                    print(
                        f"{scenario}  {model:>3}  w={window:<2d}  acc={result.accuracy:.3f}  "
                        f"f1={result.f1:.3f}  ({len(result.y_true)} test windows)",
                        file=out,
                        flush=True,
                    )
    return pd.DataFrame(rows, columns=list(COLUMNS))
