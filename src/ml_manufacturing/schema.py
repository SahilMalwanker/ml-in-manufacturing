"""Signal schema of the ST60 MQTT payload and the fault catalogue of the press station.

The press-station PLC publishes one comma-separated message per scan via the function
``MQTT_PublishMsgBuild`` (FC9): seven timestamp fields followed by 51 process signals.
Column names are the original PLC tag names so that data, PLC project and
documentation can be cross-referenced one-to-one.
"""

from __future__ import annotations

from dataclasses import dataclass

TIMESTAMP_COLUMNS: tuple[str, ...] = (
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Second",
    "Nanosecond",
)
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class Signal:
    """One process signal of the MQTT payload."""

    name: str
    group: str
    dtype: str
    address: str
    description: str


SIGNALS: tuple[Signal, ...] = (
    # Shuttle hand-over with station 10
    Signal("Shuttle_in_Station_von_St10", "shuttle", "Bool", "%I101.1", "Shuttle present in the station (from St10)"),
    Signal("LockShuttle_an_St10", "shuttle", "Bool", "%Q101.4", "Lock shuttle (command to St10)"),
    Signal("StartShuttle_an_St10", "shuttle", "Bool", "%Q101.5", "Release shuttle (command to St10)"),
    Signal("ShuttleInitFlagsByteVonSt10", "shuttle", "Byte", "%IB104", "Shuttle initialisation flags (from St10)"),
    # Station state
    Signal("Stationsdaten.Betriebszustand", "station", "Int", "%DB1.DBW0", "State of the station state machine"),
    Signal("Stationsdaten.Station sicher", "station", "Bool", "%DB1.DBX14.0", "Station safe"),
    Signal("Stationsdaten.StationIsWorking", "station", "Bool", "%DB1.DBX130.1", "Station working (cycle active)"),
    Signal("Stationsdaten.Station in Grundstellung", "station", "Bool", "%DB1.DBX14.1", "Station in home position"),
    Signal(
        "Stationsdaten.StartWorkVonAblaufketteShuttle",
        "station",
        "Bool",
        "%DB1.DBX130.0",
        "Start work (hand-over from the shuttle sequence)",
    ),
    Signal("Stationsdaten.Auftragssimulation aktiv", "station", "Bool", "%DB1.DBX15.2", "Order simulation active"),
    Signal("S13 Hand (links)", "station", "Bool", "%I0.3", "Mode selector S13 in manual (left)"),
    Signal("S13 Auto (rechts)", "station", "Bool", "%I0.4", "Mode selector S13 in automatic (right)"),
    Signal("St10_in_AUTO", "station", "Bool", "%I101.2", "Station 10 in automatic mode"),
    # Safety signals
    Signal("LichtvorhangFrei", "safety", "Bool", "%I1000.3", "Light curtain clear (1 = not interrupted)"),
    Signal("Bereichshalt Nicht", "safety", "Bool", "%I1000.0", "Area stop not pressed (1 = OK)"),
    Signal("SchalterTuerZu", "safety", "Bool", "%I1000.2", "Door safety switch closed (1 = OK)"),
    Signal("SchaltmatteGedrueckt", "safety", "Bool", "%I1006.0", "Safety mat pressed (1 = person detected)"),
    Signal(
        "VentilRueckfuehrungKeineDruckl",
        "safety",
        "Bool",
        "%I1000.1",
        "Valve feedback: circuit behind the safety valve vented",
    ),
    Signal("VentilAnsteuerung", "safety", "Bool", "%Q1012.0", "Safety valve on: circuit behind the valve pressurised"),
    Signal("KeinNotHaltExtern von St10", "safety", "Bool", "%I200.0", "No external emergency stop from St10 (1 = OK)"),
    Signal("SchluesselFreigabe", "safety", "Bool", "%I1000.5", "Key switch released"),
    Signal("F01000_4/8F-DIDC24V.QBAD", "safety", "Bool", "-", "F-DI module 1 outputs substitute values"),
    Signal("F01006_4/8F-DIDC24V_2.QBAD", "safety", "Bool", "-", "F-DI module 2 outputs substitute values"),
    Signal("F01012_4F-DODC24V/2A.QBAD", "safety", "Bool", "-", "F-DO module outputs substitute values"),
    Signal(
        "Stationsdaten.Safety Status Bits.Safety Ok - Aktoren freigegeben",
        "safety",
        "Bool",
        "%DB1.DBX2.0",
        "Safety OK, actuators enabled",
    ),
    Signal(
        "Stationsdaten.Safety Status Bits.ET200S - Baugruppe passiviert",
        "safety",
        "Bool",
        "%DB1.DBX2.4",
        "ET200S safety module passivated",
    ),
    # Press arm
    Signal("B11 Pressenarm untere Endlage", "press", "Bool", "%I2.0", "Press arm at lower end position"),
    Signal("B12 Pressenarm obere Endlage", "press", "Bool", "%I2.1", "Press arm at upper end position (rest)"),
    Signal("Stationsdaten.Anfahrt_Oben", "press", "Bool", "%DB1.DBX78.4", "Move press arm to top"),
    Signal("Stationsdaten.Anfahrt_50Proz", "press", "Bool", "%DB1.DBX78.5", "Move press arm to 50 %"),
    Signal("Stationsdaten.Anfahrt_Unten", "press", "Bool", "%DB1.DBX78.6", "Move press arm to bottom"),
    Signal("Stationsdaten.Sollpos_percent", "press", "Real", "%DB1.DBD44", "Target position of the press arm [%]"),
    Signal(
        "Stationsdaten.ActPosPercCtrlOutp",
        "press",
        "Real",
        "%DB1.DBD74",
        "Actual press-arm position seen by the position controller [%]",
    ),
    # Proportional pressure valves (raw analog values, S7 nominal range 0..27648)
    Signal("M1 Ventil Ist-Druck", "valves", "Word", "%IW4", "Valve M1 actual pressure (raw analog)"),
    Signal("M1 Ventil Soll-Druck", "valves", "Word", "%QW4", "Valve M1 target pressure (raw analog)"),
    Signal("M2 Ventil Ist-Druck", "valves", "Word", "%IW6", "Valve M2 actual pressure (raw analog)"),
    Signal("M2 Ventil Soll-Druck", "valves", "Word", "%QW6", "Valve M2 target pressure (raw analog)"),
    # RFID reader (RF180C)
    Signal("RFIDInterface.RFID_busyRead", "rfid", "Bool", "-", "RFID read in progress"),
    Signal("RFIDInterface.RFID_errorRead", "rfid", "Bool", "-", "RFID read error"),
    Signal("RFIDInterface.RFID_presenceRead", "rfid", "Bool", "-", "RFID tag present (read channel)"),
    Signal("RFIDInterface.RFID_busyWrite", "rfid", "Bool", "-", "RFID write in progress"),
    Signal("RFIDInterface.RFID_errorWrite", "rfid", "Bool", "-", "RFID write error"),
    Signal("RFIDInterface.RFID_presenceWrite", "rfid", "Bool", "-", "RFID tag present (write channel)"),
    Signal("RFIDInterface.RFID_busyReset", "rfid", "Bool", "-", "RFID reset in progress"),
    Signal("RFIDInterface.RFID_doneReset", "rfid", "Bool", "-", "RFID reset done"),
    Signal("RFIDInterface.RFID_errorReset", "rfid", "Bool", "-", "RFID reset error"),
    Signal("RFIDInterface.RFID_resetOk", "rfid", "Bool", "-", "RFID reset OK"),
    # Field-device diagnostics
    Signal("Field_Device_Diagnostic_DB.Main_PLC_60", "diagnostics", "Bool", "-", "Main PLC of St60 reachable"),
    Signal("Field_Device_Diagnostic_DB.ET200s_St_60", "diagnostics", "Bool", "-", "ET200S of St60 reachable"),
    Signal("Field_Device_Diagnostic_DB.State_PROFI_60", "diagnostics", "Bool", "-", "PROFINET state of St60"),
    Signal("Field_Device_Diagnostic_DB.HMI_60", "diagnostics", "Bool", "-", "HMI of St60 reachable"),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(s.name for s in SIGNALS)
N_FEATURES = len(FEATURE_COLUMNS)
PAYLOAD_FIELDS = len(TIMESTAMP_COLUMNS) + N_FEATURES


@dataclass(frozen=True)
class Fault:
    """One entry of the press-station fault catalogue.

    ``rule`` lists ``(signal, value)`` pairs that all hold while the fault is present.
    Faults without a rule (e.g. a stuck position sensor) are labelled from their
    injection time instead.
    """

    code: int
    name: str
    group: str
    simulated: bool
    rule: tuple[tuple[str, float], ...] = ()


FAULTS: dict[int, Fault] = {
    f.code: f
    for f in (
        Fault(0, "Normal behaviour", "normal", True),
        Fault(1, "Station not safe", "safety", False),
        Fault(2, "Area stop", "safety", True, (("Bereichshalt Nicht", 0),)),
        Fault(3, "Light curtain interrupted", "safety", True, (("LichtvorhangFrei", 0),)),
        Fault(4, "Safety mat pressed", "safety", True, (("SchaltmatteGedrueckt", 1),)),
        Fault(5, "Door safety switch opened", "safety", True, (("SchalterTuerZu", 0),)),
        Fault(6, "Emergency stop", "safety", True, (("KeinNotHaltExtern von St10", 0),)),
        Fault(7, "Safety components passivated", "safety", False),
        Fault(8, "Press arm position sensor failure", "press arm", True),
        Fault(
            9,
            "Shuttle RFID presence detection failure",
            "shuttle",
            True,
            (("Shuttle_in_Station_von_St10", 1), ("RFIDInterface.RFID_presenceRead", 0)),
        ),
        Fault(
            10,
            "Shuttle-in-station sensor failure",
            "shuttle",
            True,
            (("Shuttle_in_Station_von_St10", 0), ("RFIDInterface.RFID_presenceRead", 1)),
        ),
        Fault(11, "No connection to ET200S", "communication", False),
        Fault(12, "No connection to the control station", "communication", False),
        Fault(13, "No connection to PROFINET", "communication", False),
        Fault(14, "No connection to the RFID reader", "communication", False),
        Fault(15, "Station not started in automatic mode", "operation", False),
        Fault(16, "Press not in starting position", "operation", False),
        Fault(17, "Software error during main process execution", "operation", False),
    )
}


def fault_name(code: int) -> str:
    """Human-readable name of a fault code (unknown codes are reported as such)."""
    fault = FAULTS.get(int(code))
    return fault.name if fault else f"Unknown fault ({code})"
