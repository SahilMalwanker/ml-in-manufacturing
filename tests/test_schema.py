import re
from pathlib import Path

from ml_manufacturing.schema import (
    FAULTS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    N_FEATURES,
    PAYLOAD_FIELDS,
    SIGNALS,
    TIMESTAMP_COLUMNS,
    fault_name,
)


def test_payload_layout():
    assert len(TIMESTAMP_COLUMNS) == 7
    assert N_FEATURES == 51
    assert PAYLOAD_FIELDS == 58
    assert len(set(FEATURE_COLUMNS)) == N_FEATURES
    assert LABEL_COLUMN not in FEATURE_COLUMNS


def test_signal_metadata_is_complete():
    for signal in SIGNALS:
        assert signal.dtype in {"Bool", "Byte", "Int", "Word", "Real"}
        assert signal.group and signal.description


def test_fault_rules_reference_known_signals():
    for fault in FAULTS.values():
        for signal, _value in fault.rule:
            assert signal in FEATURE_COLUMNS, (fault.code, signal)


def test_fault_catalogue():
    assert sorted(FAULTS) == list(range(18))
    simulated = {code for code, fault in FAULTS.items() if fault.simulated}
    assert simulated == {0, 2, 3, 4, 5, 6, 8, 9, 10}
    assert fault_name(3) == "Light curtain interrupted"
    assert fault_name(99) == "Unknown fault (99)"


def test_plc_reference_matches_signal_order():
    reference = Path(__file__).resolve().parents[1] / "plc/MQTT_PublishMsgBuild.scl"
    arguments = re.findall(r"(?:bool|byte|int|real|word)Val := (.*?), str :=", reference.read_text())
    assert [argument.replace('"', "") for argument in arguments] == list(FEATURE_COLUMNS)
