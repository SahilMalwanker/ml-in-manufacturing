from __future__ import annotations

from pathlib import Path

import pytest

from ml_manufacturing.datasets import Recording, load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
RAW_EXAMPLES = ROOT / "data" / "raw_examples"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def manifest_path() -> Path:
    return MANIFEST


@pytest.fixture(scope="session")
def manifest() -> list[Recording]:
    return load_manifest(MANIFEST)


@pytest.fixture(scope="session")
def dt_normal_path() -> Path:
    """Unprocessed DAQ output of one complete normal press cycle in the digital twin."""
    return RAW_EXAMPLES / "dt_normal_cycle.txt"


@pytest.fixture(scope="session")
def dt_stuck_sensor_path() -> Path:
    """Unprocessed DAQ output of a twin run whose press-arm position sensor freezes after 2000 ms."""
    return RAW_EXAMPLES / "dt_press_arm_sensor_2000ms.txt"


@pytest.fixture(scope="session")
def rp_train(manifest):
    return [rec.load() for rec in manifest if rec.domain == "real_plant" and rec.split == "train"]


@pytest.fixture(scope="session")
def rp_test(manifest):
    return [rec.load() for rec in manifest if rec.domain == "real_plant" and rec.split == "test"]
