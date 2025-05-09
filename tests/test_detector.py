import joblib
import numpy as np
import pytest

from ml_manufacturing.datasets import scenario_data
from ml_manufacturing.detector import FaultDetector
from ml_manufacturing.windowing import stack_windows


@pytest.fixture(scope="module")
def svc(rp_train):
    return FaultDetector("svc", window=3).fit(rp_train)


def test_real_plant_detector_generalises_to_unseen_runs(svc, rp_test):
    result = svc.evaluate(rp_test)
    assert svc.classes == [0, 2, 3, 4, 5, 6, 8, 9, 10]
    assert result.accuracy > 0.95
    assert result.f1 > 0.95
    assert result.confusion().sum() == len(result.y_true)
    assert "Shuttle RFID presence detection failure" in result.report()


def test_clipping_keeps_a_twin_trained_model_usable_on_the_plant(manifest):
    """Analog signals that are constant in the twin would push plant windows far out of range."""
    train, test = scenario_data(manifest, "D-R")
    clipped = FaultDetector("svc", window=3).fit(train).evaluate(test)
    unclipped = FaultDetector("svc", window=3, clip=False).fit(train).evaluate(test)
    assert len(set(unclipped.y_pred.tolist())) == 1  # every plant window ends up in one class
    assert clipped.f1 > unclipped.f1 + 0.5


def test_save_load_round_trip(svc, rp_test, tmp_path):
    path = svc.save(tmp_path / "detector.joblib")
    assert joblib.load(path)["format"] == "ml-manufacturing-detector/1"
    restored = FaultDetector.load(path)
    x, _ = stack_windows(rp_test, 3)
    np.testing.assert_array_equal(restored.predict(x), svc.predict(x))
    assert (restored.model_name, restored.window, restored.clip) == ("svc", 3, True)


def test_load_bundle_saved_before_project_rename(svc, rp_test, tmp_path):
    path = svc.save(tmp_path / "previous_name.joblib")
    bundle = joblib.load(path)
    bundle["format"] = "twinsight-detector/1"
    joblib.dump(bundle, path)
    restored = FaultDetector.load(path)
    windows, _ = stack_windows(rp_test, 3)
    np.testing.assert_array_equal(restored.predict(windows), svc.predict(windows))


def test_load_rejects_foreign_files(tmp_path):
    path = tmp_path / "other.joblib"
    joblib.dump({"model": None}, path)
    with pytest.raises(ValueError):
        FaultDetector.load(path)


def test_input_width_is_checked(svc):
    with pytest.raises(ValueError):
        svc.predict(np.zeros((1, 51)))


def test_untrained_detector_refuses_to_predict():
    with pytest.raises(RuntimeError):
        FaultDetector().predict(np.zeros((1, 153)))


def test_confidence_only_for_probabilistic_models(svc, rp_train):
    x, _ = stack_windows(rp_train[:1], 3)
    assert svc.confidence(x) is None
    tree = FaultDetector("dt", window=3).fit(rp_train)
    confidence = tree.confidence(x)
    assert confidence.shape == (len(x),) and np.all((confidence > 0) & (confidence <= 1))


def test_unknown_model_name():
    with pytest.raises(ValueError):
        FaultDetector("xgb").fit([(np.zeros((5, 51)), np.zeros(5, dtype=int))])
