"""Recording manifests and the train/test scenarios of the real-plant vs. digital-twin study."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .daq import feature_matrix, label_recording, read_recording
from .schema import FAULTS, LABEL_COLUMN

DOMAINS = ("real_plant", "digital_twin")
SPLITS = ("train", "test")

# scenario id -> (training domain, test domain); "combined" = both domains
SCENARIOS: dict[str, tuple[str, str]] = {
    "R-R": ("real_plant", "real_plant"),
    "D-D": ("digital_twin", "digital_twin"),
    "R-D": ("real_plant", "digital_twin"),
    "D-R": ("digital_twin", "real_plant"),
    "C-C": ("combined", "combined"),
}


@dataclass(frozen=True)
class Recording:
    """One DAQ recording listed in a manifest."""

    path: Path
    domain: str
    split: str
    fault: int
    fault_delay_ms: float | None = None

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Signals (n x 51) and per-sample labels of the recording."""
        frame = read_recording(self.path)
        if LABEL_COLUMN in frame.columns:
            labels = frame[LABEL_COLUMN].to_numpy(dtype=int)
        else:
            labels = label_recording(frame, self.fault, self.fault_delay_ms)
        return feature_matrix(frame), labels


def load_manifest(path: str | Path) -> list[Recording]:
    """Read a manifest CSV (``file,domain,split,fault`` + optional ``fault_delay_ms``); paths are relative to it.

    Further columns (run number, sample counts, notes) are informational and ignored here.
    """
    path = Path(path)
    recordings = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            domain, split, fault = row["domain"], row["split"], int(row["fault"])
            if domain not in DOMAINS or split not in SPLITS or fault not in FAULTS:
                raise ValueError(f"invalid manifest row: {row}")
            delay = row.get("fault_delay_ms", "").strip()
            recordings.append(
                Recording(
                    path=(path.parent / row["file"]).resolve(),
                    domain=domain,
                    split=split,
                    fault=fault,
                    fault_delay_ms=float(delay) if delay else None,
                )
            )
    return recordings


def select(
    recordings: list[Recording],
    domain: str,
    split: str,
    faults: set[int] | None = None,
) -> list[Recording]:
    """Recordings of one domain (or ``combined``) and split, optionally restricted to fault codes."""
    if domain not in (*DOMAINS, "combined"):
        raise ValueError(f"unknown domain {domain!r}")
    return [
        rec
        for rec in recordings
        if rec.split == split
        and (domain == "combined" or rec.domain == domain)
        and (faults is None or rec.fault in faults)
    ]


def scenario_data(
    recordings: list[Recording], scenario: str
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Loaded train and test recordings of a scenario.

    The test side is restricted to faults seen in training, so a cross-domain model is only
    asked to recognise classes it was trained on.
    """
    train_domain, test_domain = SCENARIOS[scenario]
    train = select(recordings, train_domain, "train")
    seen = {rec.fault for rec in train}
    test = select(recordings, test_domain, "test", faults=seen)
    if not train or not test:
        raise ValueError(f"scenario {scenario}: no train or test recordings in the manifest")
    return [rec.load() for rec in train], [rec.load() for rec in test]
