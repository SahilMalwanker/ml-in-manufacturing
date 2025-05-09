import numpy as np
import pytest

from ml_manufacturing.daq import (
    elapsed_ms,
    feature_matrix,
    has_timestamps,
    label_recording,
    parse_payload,
    read_recording,
    to_labelled_csv,
)
from ml_manufacturing.schema import FEATURE_COLUMNS, LABEL_COLUMN, TIMESTAMP_COLUMNS


def test_read_raw_recording(dt_normal_path):
    frame = read_recording(dt_normal_path)
    assert list(frame.columns) == [*TIMESTAMP_COLUMNS, *FEATURE_COLUMNS]
    assert len(frame) == 82
    assert has_timestamps(frame)
    t = elapsed_ms(frame)
    assert t[0] == 0 and np.all(np.diff(t) >= 0)
    assert 7000 < t[-1] < 8000  # a complete normal press cycle takes about 8 s


def test_parse_payload_matches_file_reader(dt_normal_path):
    frame = read_recording(dt_normal_path)
    first_line = dt_normal_path.read_text(encoding="ascii").splitlines()[0]
    values = parse_payload(first_line)
    assert values.shape == (51,)
    np.testing.assert_array_equal(values, feature_matrix(frame)[0])
    np.testing.assert_array_equal(parse_payload(first_line.encode()), values)


@pytest.mark.parametrize("payload", ["1,2,3", "", "2025,1,1,0,0,0,0," + ",".join(["0"] * 52)])
def test_parse_payload_rejects_wrong_length(payload):
    with pytest.raises(ValueError):
        parse_payload(payload)


def test_read_labelled_csv(manifest):
    rec = next(r for r in manifest if r.domain == "real_plant")
    frame = read_recording(rec.path)
    assert list(frame.columns) == [*TIMESTAMP_COLUMNS, *FEATURE_COLUMNS, LABEL_COLUMN]
    assert has_timestamps(frame)
    assert feature_matrix(frame).shape == (len(frame), 51)


def test_original_export_format_is_understood(tmp_path):
    """Semicolons, nested quotes in tag names and a capitalised 'Label' column, as in the project exports."""
    names = [
        f'"Stationsdaten.""{n.split(".", 1)[1]}"""' if n.startswith("Stationsdaten.") else n for n in FEATURE_COLUMNS
    ]
    row = ";".join(["1"] * len(FEATURE_COLUMNS))
    path = tmp_path / "export.CSV"
    path.write_text(";".join([*names, "Label"]) + f"\n{row};3\n{row};3\n", encoding="utf-8")
    frame = read_recording(path)
    assert list(frame.columns) == [*FEATURE_COLUMNS, LABEL_COLUMN]
    assert frame[LABEL_COLUMN].tolist() == [3, 3]


def test_rows_wider_than_the_header_are_rejected(tmp_path):
    """A header-based reader would otherwise shift every column by one position."""
    path = tmp_path / "shifted.csv"
    row = ",".join(["0"] * (len(FEATURE_COLUMNS) + 2))
    path.write_text(",".join([*FEATURE_COLUMNS, LABEL_COLUMN]) + f"\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column alignment"):
        read_recording(path)


def test_rule_labels_are_latched(dt_normal_path):
    frame = read_recording(dt_normal_path)
    frame.loc[10:19, "LichtvorhangFrei"] = 0  # light curtain interrupted for ten samples
    labels = label_recording(frame, fault=3, method="rule")
    assert labels[:10].tolist() == [0] * 10
    assert set(labels[10:].tolist()) == {3}  # held until the end of the acquisition


def test_delay_labels_follow_the_injection_time(dt_stuck_sensor_path):
    frame = read_recording(dt_stuck_sensor_path)
    labels = label_recording(frame, 8, 2000, method="delay")
    t = elapsed_ms(frame)
    np.testing.assert_array_equal(labels, np.where(t >= 2000, 8, 0))
    assert 0 < labels.sum() < 8 * len(labels)


def test_auto_method_selection(dt_normal_path):
    frame = read_recording(dt_normal_path)
    assert set(label_recording(frame, 8).tolist()) == {8}  # no rule, no delay -> whole recording
    assert set(label_recording(frame, 0).tolist()) == {0}
    assert set(label_recording(frame, 3).tolist()) == {0}  # light curtain never interrupted
    with pytest.raises(ValueError):
        label_recording(frame, 8, method="rule")
    with pytest.raises(ValueError):
        label_recording(frame, 8, method="delay")
    with pytest.raises(ValueError):
        label_recording(frame, 3, method="nope")


@pytest.mark.parametrize("fault", [2, 3, 4, 5, 6, 9, 10])
def test_rules_reproduce_real_plant_labels(manifest, fault):
    """The signal rules recover the per-sample labels of every real-plant run exactly."""
    recordings = [r for r in manifest if r.domain == "real_plant" and r.fault == fault]
    assert recordings
    for rec in recordings:
        frame = read_recording(rec.path)
        expected = frame[LABEL_COLUMN].to_numpy()
        np.testing.assert_array_equal(label_recording(frame, fault, method="rule"), expected)


def test_labelled_csv_round_trip(dt_normal_path, tmp_path):
    frame = read_recording(dt_normal_path)
    labels = np.zeros(len(frame), dtype=int)
    labels[40:] = 8
    path = to_labelled_csv(frame, labels, tmp_path / "out.csv")
    back = read_recording(path)
    assert has_timestamps(back)
    np.testing.assert_allclose(feature_matrix(back), feature_matrix(frame))
    np.testing.assert_array_equal(back[LABEL_COLUMN].to_numpy(), labels)
