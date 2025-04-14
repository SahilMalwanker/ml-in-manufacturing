"""Reading DAQ recordings, parsing MQTT payloads and labelling samples.

Two on-disk formats are supported:

* **raw DAQ recordings** (``.txt``): exactly what the data acquisition client writes,
  one MQTT payload per line, 7 timestamp fields + 51 signals, comma separated, no header;
* **labelled CSV files**: a header with the signal names (timestamps optional) and an
  optional ``label`` column, comma or semicolon separated.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FAULTS, FEATURE_COLUMNS, LABEL_COLUMN, PAYLOAD_FIELDS, TIMESTAMP_COLUMNS

LABEL_METHODS = ("auto", "rule", "delay", "whole")


def parse_payload(payload: str | bytes) -> np.ndarray:
    """Convert one MQTT payload into the 51 signal values (timestamp fields are dropped)."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    fields = [field.strip() for field in text.strip().split(",")]
    if len(fields) != PAYLOAD_FIELDS:
        raise ValueError(f"expected {PAYLOAD_FIELDS} comma-separated fields, got {len(fields)}")
    return np.asarray(fields[len(TIMESTAMP_COLUMNS) :], dtype=float)


def _read_labelled_csv(path: Path, first_line: str) -> pd.DataFrame:
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle, delimiter=sep)
        header = next(rows)
        first_row = next(rows, header)
    # pandas would silently use the first column as index if rows are wider than the header
    if len(first_row) != len(header):
        raise ValueError(
            f"{path}: rows have {len(first_row)} fields but the header names {len(header)}; "
            "refusing to guess the column alignment"
        )
    frame = pd.read_csv(path, sep=sep, skipinitialspace=True)
    # original exports quote tag names like Stationsdaten."Station sicher" and call the label "Label"
    frame.columns = [str(col).replace('"', "").strip() for col in frame.columns]
    frame = frame.rename(columns={col: LABEL_COLUMN for col in frame.columns if col.lower() == LABEL_COLUMN})
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} signal columns, e.g. {missing[:3]}")
    return frame


def read_recording(path: str | Path) -> pd.DataFrame:
    """Load a raw DAQ recording or a labelled CSV file into a DataFrame."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    if not first_line.strip():
        raise ValueError(f"{path} is empty")

    if first_line.lstrip()[0].isalpha():
        frame = _read_labelled_csv(path, first_line)
    else:
        names = [*TIMESTAMP_COLUMNS, *FEATURE_COLUMNS]
        frame = pd.read_csv(path, header=None, skipinitialspace=True)
        if frame.shape[1] != PAYLOAD_FIELDS:
            raise ValueError(f"{path}: expected {PAYLOAD_FIELDS} fields per line, got {frame.shape[1]}")
        frame.columns = names

    frame[list(FEATURE_COLUMNS)] = frame[list(FEATURE_COLUMNS)].astype(float)
    return frame


def has_timestamps(frame: pd.DataFrame) -> bool:
    return all(col in frame.columns for col in TIMESTAMP_COLUMNS)


def elapsed_ms(frame: pd.DataFrame) -> np.ndarray:
    """Milliseconds since the first sample, computed from the PLC timestamp fields."""
    if not has_timestamps(frame):
        raise ValueError("recording has no timestamp columns")
    stamps = pd.to_datetime(
        {
            "year": frame["Year"],
            "month": frame["Month"],
            "day": frame["Day"],
            "hour": frame["Hour"],
            "minute": frame["Minute"],
            "second": frame["Second"],
        }
    ) + pd.to_timedelta(frame["Nanosecond"], unit="ns")
    seconds = (stamps - stamps.iloc[0]).dt.total_seconds()
    return seconds.to_numpy() * 1000.0


def rule_mask(frame: pd.DataFrame, fault: int) -> np.ndarray:
    """Rows in which the indicator signals of ``fault`` are in their fault state."""
    rule = FAULTS[fault].rule
    if not rule:
        raise ValueError(f"fault {fault} ({FAULTS[fault].name}) has no signal rule")
    mask = np.ones(len(frame), dtype=bool)
    for signal, value in rule:
        mask &= frame[signal].to_numpy() == value
    return mask


def label_recording(
    frame: pd.DataFrame,
    fault: int,
    fault_delay_ms: float | None = None,
    method: str = "auto",
) -> np.ndarray:
    """Per-sample labels for a recording in which ``fault`` was injected.

    A fault is held until the acquisition ends, so every sample from its onset onwards
    carries the fault code and every sample before it is normal (code 0).

    method
        ``rule``  -- onset = first sample in which the fault's indicator signals match;
        ``delay`` -- onset = first sample at least ``fault_delay_ms`` after the first one;
        ``whole`` -- the complete recording carries the fault code;
        ``auto``  -- ``rule`` if the fault has one, else ``delay`` if given, else ``whole``.
    """
    if method not in LABEL_METHODS:
        raise ValueError(f"method must be one of {LABEL_METHODS}")
    if fault not in FAULTS:
        raise ValueError(f"unknown fault code {fault}")
    labels = np.zeros(len(frame), dtype=int)
    if fault == 0 or len(frame) == 0:
        return labels

    if method == "auto":
        if FAULTS[fault].rule:
            method = "rule"
        elif fault_delay_ms is not None:
            method = "delay"
        else:
            method = "whole"

    if method == "whole":
        onset_hits = np.ones(len(frame), dtype=bool)
    elif method == "rule":
        onset_hits = rule_mask(frame, fault)
    else:
        if fault_delay_ms is None:
            raise ValueError("method 'delay' needs fault_delay_ms")
        onset_hits = elapsed_ms(frame) >= fault_delay_ms

    if onset_hits.any():
        labels[int(np.argmax(onset_hits)) :] = fault
    return labels


def to_labelled_csv(frame: pd.DataFrame, labels: np.ndarray, path: str | Path) -> Path:
    """Write signals (+ timestamps when available) and labels as a comma-separated CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*(TIMESTAMP_COLUMNS if has_timestamps(frame) else ()), *FEATURE_COLUMNS]
    out = frame[columns].copy()
    out[LABEL_COLUMN] = np.asarray(labels, dtype=int)
    out.to_csv(path, index=False, float_format="%.6g")
    return path


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The 51 signals of a recording as a float matrix (timestamps and labels excluded)."""
    return frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
