"""Provenance: how data/ was built from the project's working files.

The original labelled recordings (``Data/RealPlant/RealPlant_{Train,Test}Data`` and
``Data/DigTwin/DigitalTwin_{Train,Test}Data``, one semicolon-separated CSV per DAQ run) are
not part of this repository. This script documents every transformation applied to them:

1. Header names are replaced by the canonical PLC tag names of ``ml_manufacturing.schema``
   (the originals carry nested CSV quoting such as ``Stationsdaten.""Station sicher""``).
2. Digital-twin rows contain one more field than their header: a constant ``0`` between the
   last signal and the label. It is verified to be zero and dropped. Reading those files with
   a header-based CSV reader silently shifts every twin column by one position.
3. Digital-twin emergency-stop runs were labelled as faulty from their first row, although the
   stop is only injected later. They are relabelled with the signal rule used for every other
   safety fault (``KeinNotHaltExtern von St10 == 0``), which reproduces the real-plant labels.
4. Files are renamed ``<rp|dt>_c<fault>_run<n>.csv`` (runs numbered by acquisition time) and
   written comma-separated with a ``label`` column; ``data/manifest.csv`` lists them.

Usage: python scripts/build_dataset.py --source <working dir>/Data --out data
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ml_manufacturing.schema import FAULTS, FEATURE_COLUMNS, LABEL_COLUMN, TIMESTAMP_COLUMNS

SOURCES = {
    ("real_plant", "train"): "RealPlant/RealPlant_TrainData",
    ("real_plant", "test"): "RealPlant/RealPlant_TestData",
    ("digital_twin", "train"): "DigTwin/DigitalTwin_TrainData",
    ("digital_twin", "test"): "DigTwin/DigitalTwin_TestData",
}
PREFIX = {"real_plant": "rp", "digital_twin": "dt"}
N_FIELDS = len(TIMESTAMP_COLUMNS) + len(FEATURE_COLUMNS) + 1
TWIN_EXTRA_FIELD = len(TIMESTAMP_COLUMNS) + len(FEATURE_COLUMNS)
NAME = re.compile(
    r"^(?P<code>\d+)(?:_\d+)?_edSeqDataStation60_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>[\d-]+)(?P<note>.*)$"
)
DELAY = re.compile(r"^_?(?P<whole>\d+)(?:p(?P<frac>\d))?\s*_?(?:sec(?:ond)?s?)?$", re.IGNORECASE)
ESTOP_SIGNAL = "KeinNotHaltExtern von St10"


@dataclass
class Run:
    source: Path
    domain: str
    split: str
    fault: int
    recorded_at: datetime
    fault_delay_ms: int | None
    note: str
    rows: list[list[str]]
    labels: np.ndarray
    relabelled: int = 0
    run: int = 0

    @property
    def file(self) -> str:
        return f"{self.domain}/{self.split}/{PREFIX[self.domain]}_c{self.fault:02d}_run{self.run}.csv"


def _number(token: str) -> str:
    value = float(token)
    return str(int(value)) if value.is_integer() else repr(value)


def read_source(path: Path, domain: str) -> tuple[list[list[str]], np.ndarray]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0].split(";")
    if len(header) != N_FIELDS or header[-1].strip() != "Label":
        raise ValueError(f"{path.name}: unexpected header with {len(header)} fields")
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        fields = [field.strip() for field in line.split(";")]
        if domain == "digital_twin" and len(fields) == N_FIELDS + 1:
            if fields[TWIN_EXTRA_FIELD] != "0":
                raise ValueError(f"{path.name}:{number}: extra twin field is {fields[TWIN_EXTRA_FIELD]!r}, not 0")
            del fields[TWIN_EXTRA_FIELD]
        if len(fields) != N_FIELDS:
            raise ValueError(f"{path.name}:{number}: {len(fields)} fields, expected {N_FIELDS}")
        rows.append([_number(field) for field in fields])
    labels = np.array([int(row[-1]) for row in rows])
    return [row[:-1] for row in rows], labels


def describe(path: Path, domain: str, split: str) -> Run:
    match = NAME.match(path.stem)
    if not match:
        raise ValueError(f"unexpected file name {path.name}")
    rows, labels = read_source(path, domain)
    faults = set(labels.tolist()) - {0}
    if len(faults) > 1:
        raise ValueError(f"{path.name}: several fault codes {sorted(faults)}")
    fault = faults.pop() if faults else 0
    expected = 0 if int(match["code"]) in (0, 1) else int(match["code"])  # real-plant normal runs are numbered 1
    if fault != expected or fault not in FAULTS:
        raise ValueError(f"{path.name}: labels say fault {fault}, file name says {match['code']}")

    hh, mm, ss = (int(part) for part in match["time"].split("-"))
    recorded_at = datetime.fromisoformat(match["date"]).replace(hour=hh, minute=mm, second=ss)
    suffix = match["note"].removesuffix("_modified_modified_modified")
    delay, note = None, ""
    if timing := DELAY.match(suffix):
        delay = int(timing["whole"]) * 1000 + int(timing["frac"] or 0) * 100
    elif "after_it_enters" in suffix:
        note = "RFID presence forced off after the shuttle entered"
    elif "before_it_enters" in suffix:
        note = "RFID presence forced off before the shuttle entered"
    elif suffix:
        raise ValueError(f"{path.name}: unknown file-name suffix {suffix!r}")
    run = Run(path, domain, split, fault, recorded_at, delay, note, rows, labels)

    if domain == "digital_twin" and fault == 6:
        column = len(TIMESTAMP_COLUMNS) + FEATURE_COLUMNS.index(ESTOP_SIGNAL)
        rule = np.array([float(row[column]) == 0 for row in rows])
        if not rule.any():
            raise ValueError(f"{path.name}: emergency stop never visible in {ESTOP_SIGNAL}")
        corrected = np.where(rule, 6, 0)
        run.relabelled = int((corrected != labels).sum())
        run.labels = corrected
        run.note = "labels before the emergency stop corrected"
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the project's original Data folder")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    runs = [
        describe(path, domain, split)
        for (domain, split), folder in SOURCES.items()
        for path in sorted((args.source / folder).iterdir())
        if path.suffix.lower() == ".csv"
    ]
    for domain in PREFIX:
        for fault in sorted({r.fault for r in runs}):
            same = sorted((r for r in runs if r.domain == domain and r.fault == fault), key=lambda r: r.recorded_at)
            for number, run in enumerate(same, start=1):
                run.run = number

    header = [*TIMESTAMP_COLUMNS, *FEATURE_COLUMNS, LABEL_COLUMN]
    for run in runs:
        target = args.out / run.file
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows([*row, int(label)] for row, label in zip(run.rows, run.labels, strict=True))

    runs.sort(key=lambda r: (r.domain != "real_plant", r.split != "train", r.fault, r.run))
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "file",
                "domain",
                "split",
                "fault",
                "run",
                "samples",
                "fault_samples",
                "fault_delay_ms",
                "recorded_at",
                "note",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.file,
                    run.domain,
                    run.split,
                    run.fault,
                    run.run,
                    len(run.rows),
                    int((run.labels != 0).sum()),
                    "" if run.fault_delay_ms is None else run.fault_delay_ms,
                    run.recorded_at.isoformat(timespec="seconds"),
                    run.note,
                ]
            )
    relabelled = sum(run.relabelled for run in runs)
    print(f"{len(runs)} recordings -> {args.out}; {relabelled} twin emergency-stop labels corrected")


if __name__ == "__main__":
    main()
