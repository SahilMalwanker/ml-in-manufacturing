import csv

import numpy as np
import pytest

from ml_manufacturing.datasets import DOMAINS, SCENARIOS, load_manifest, scenario_data, select

SIMULATED_FAULTS = {0, 2, 3, 4, 5, 6, 8, 9, 10}


def test_manifest_lists_existing_recordings(manifest):
    assert len(manifest) == 97
    assert all(rec.path.is_file() for rec in manifest)
    assert {rec.domain for rec in manifest} == set(DOMAINS)


def test_manifest_counts_match_the_files(manifest_path):
    """Sample counts in the manifest agree with the files, and each run holds one fault only."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    for rec in load_manifest(manifest_path):
        row = rows[rec.path.relative_to(manifest_path.parent).as_posix()]
        _, labels = rec.load()
        assert len(labels) == int(row["samples"])
        assert int(np.count_nonzero(labels)) == int(row["fault_samples"])
        assert set(labels.tolist()) <= {0, rec.fault}


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("split", ["train", "test"])
def test_every_split_covers_all_faults(manifest, domain, split):
    assert {rec.fault for rec in select(manifest, domain, split)} == SIMULATED_FAULTS


def test_select_by_domain_and_split(manifest):
    assert len(select(manifest, "real_plant", "train")) == 21
    assert len(select(manifest, "digital_twin", "train")) == 29
    assert len(select(manifest, "combined", "train")) == 50
    assert len(select(manifest, "combined", "test")) == 47
    assert {rec.fault for rec in select(manifest, "real_plant", "test", faults={8})} == {8}
    with pytest.raises(ValueError):
        select(manifest, "plant", "train")


def test_cross_domain_tests_only_cover_trained_faults(manifest):
    train, test = scenario_data(manifest, "D-R")
    trained = {int(code) for _, labels in train for code in set(labels)}
    tested = {int(code) for _, labels in test for code in set(labels)}
    assert trained == SIMULATED_FAULTS
    assert tested <= trained
    assert set(SCENARIOS) == {"R-R", "D-D", "R-D", "D-R", "C-C"}


def test_invalid_manifest_rows_are_rejected(tmp_path):
    bad = tmp_path / "manifest.csv"
    bad.write_text("file,domain,split,fault,fault_delay_ms\nx.txt,real_plant,validation,0,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(bad)
