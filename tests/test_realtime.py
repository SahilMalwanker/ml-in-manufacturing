import numpy as np
import pytest

from ml_manufacturing.daq import elapsed_ms, feature_matrix, parse_payload, read_recording
from ml_manufacturing.detector import FaultDetector
from ml_manufacturing.realtime import OnlinePredictor, Prediction, format_prediction
from ml_manufacturing.replay import iter_payloads
from ml_manufacturing.schema import FEATURE_COLUMNS, LABEL_COLUMN
from ml_manufacturing.windowing import make_windows


@pytest.fixture(scope="module")
def tree(rp_train):
    return FaultDetector("dt", window=3).fit(rp_train)


def test_online_predictions_match_offline_evaluation(tree, manifest):
    """Streaming payloads one by one gives exactly the offline windowed predictions."""
    rec = next(r for r in manifest if r.path.name == "rp_c09_run3.csv")
    features, _ = rec.load()
    offline = tree.predict(make_windows(features, None, 3)[0])

    predictor = OnlinePredictor(tree, dedupe=False)
    online = [p.code for _, payload in iter_payloads(rec.path) if (p := predictor.update(payload))]
    np.testing.assert_array_equal(online, offline)


def test_unchanged_payloads_are_ignored(tree, dt_normal_path):
    lines = dt_normal_path.read_text(encoding="ascii").splitlines()
    predictor = OnlinePredictor(tree, dedupe=True)
    assert predictor.update(lines[0]) is None
    assert predictor.update(lines[0]) is None  # same signals again -> no new sample
    assert predictor.samples == 1
    results = [predictor.update(line) for line in lines[1:3]]
    assert results[0] is None and isinstance(results[1], Prediction)
    assert results[1].sample == 3


def test_replay_payloads_of_raw_recordings(dt_normal_path):
    frame = read_recording(dt_normal_path)
    items = list(iter_payloads(dt_normal_path))
    assert len(items) == len(frame)
    np.testing.assert_allclose([offset for offset, _ in items], elapsed_ms(frame) / 1000.0)
    parsed = np.vstack([parse_payload(payload) for _, payload in items])
    np.testing.assert_array_equal(parsed, feature_matrix(frame))


def test_replay_payloads_of_csv_recordings(manifest):
    """Labelled CSV files keep the PLC timestamps, so the replay reproduces the original timing."""
    rec = next(r for r in manifest if r.domain == "real_plant")
    features, _ = rec.load()
    items = list(iter_payloads(rec.path))
    np.testing.assert_allclose([offset for offset, _ in items], elapsed_ms(read_recording(rec.path)) / 1000.0)
    parsed = np.vstack([parse_payload(payload) for _, payload in items])
    np.testing.assert_allclose(parsed, features)


def test_replay_paces_files_without_timestamps(manifest, tmp_path):
    rec = next(r for r in manifest if r.domain == "digital_twin")
    path = tmp_path / "no_timestamps.csv"
    read_recording(rec.path)[[*FEATURE_COLUMNS, LABEL_COLUMN]].to_csv(path, index=False)
    items = list(iter_payloads(path, interval_ms=50))
    np.testing.assert_allclose([offset for offset, _ in items], np.arange(len(items)) * 0.05)
    parsed = np.vstack([parse_payload(payload) for _, payload in items])
    np.testing.assert_allclose(parsed, rec.load()[0])


def test_prediction_formatting():
    line = format_prediction(
        Prediction(sample=7, code=8, name="Press arm position sensor failure", confidence=0.97, latency_ms=0.4)
    )
    assert "FAULT" in line and "[ 8]" in line and "p=0.97" in line
    assert "NORMAL" in format_prediction(Prediction(1, 0, "Normal behaviour", None, 0.1))
