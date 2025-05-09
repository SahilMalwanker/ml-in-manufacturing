import pandas as pd

from ml_manufacturing.cli import main
from ml_manufacturing.daq import read_recording
from ml_manufacturing.schema import LABEL_COLUMN


def test_convert_labels_raw_recordings(dt_stuck_sensor_path, tmp_path):
    args = ["convert", str(dt_stuck_sensor_path), "--fault", "8", "--delay-ms", "2000", "--out-dir", str(tmp_path)]
    assert main(args) == 0
    frame = read_recording(tmp_path / f"{dt_stuck_sensor_path.stem}.csv")
    assert set(frame[LABEL_COLUMN]) == {0, 8}


def test_train_and_evaluate(manifest_path, tmp_path):
    model = tmp_path / "detector.joblib"
    figure = tmp_path / "confusion.png"
    common = ["--manifest", str(manifest_path), "--domain", "real_plant"]
    assert main(["train", *common, "--model", "dt", "--window", "5", "--out", str(model)]) == 0
    assert main(["evaluate", "--model", str(model), *common, "--confusion-matrix", str(figure)]) == 0
    assert model.stat().st_size > 0 and figure.stat().st_size > 0


def test_benchmark_writes_a_table(manifest_path, tmp_path):
    out = tmp_path / "bench.csv"
    args = ["benchmark", "--manifest", str(manifest_path), "--scenarios", "R-R", "--models", "dt"]
    assert main([*args, "--windows", "0", "3", "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert list(table["window"]) == [0, 3]
    assert table["accuracy"].between(0, 1).all()
    assert (table["test_classes"] == 9).all()
