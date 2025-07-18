# SIMIT Scenario References

[System connections](../docs/architecture.md) | [Data and labels](../data/README.md)

These text files transcribe selected scenario scripts from the project's SIMIT screenshots. They preserve the virtual commissioning workflow, but are **not a self-contained simulation project**. They have not been compiled or executed against SIMIT as part of this repository's Python checks.

**Simulation only.** Use an isolated PLCSIM Advanced instance with no physical outputs connected. These files are not instructions for overriding safety signals on a real machine. Do not import or execute them on a live plant.

## Included Scenarios

| Case | Script | Purpose |
|---|---|---|
| 0 | [test_normal.txt](scripts/test_normal.txt) | Initialize and acquire a normal cycle |
| 3 | [test_light_curtain.txt](scripts/test_light_curtain.txt) | Change the simulated light-curtain indication |
| 4 | [test_safety_mat.txt](scripts/test_safety_mat.txt) | Change the simulated safety-mat indication |
| 6 | [test_emergency_stop.txt](scripts/test_emergency_stop.txt) | Change the virtual external-stop indication |
| 8 | [test_press_arm_sensor.txt](scripts/test_press_arm_sensor.txt) | Capture position and forward a frozen value |
| 9 | [test_shuttle_rfid_error.txt](scripts/test_shuttle_rfid_error.txt) | Suppress simulated RFID presence |
| 10 | [test_shuttle_in_station.txt](scripts/test_shuttle_in_station.txt) | Alter simulated shuttle presence |

Area-stop and door-switch recordings exist in the dataset, but complete script bodies were not available in the reviewed screenshots. No substitute scripts have been invented for them. Timings shown in these examples are not a complete enumeration of all recorded injection delays; consult the data manifest for known per-run delays.

## Dependencies

The original setup requires licensed SIMIT, NX MCD, PLCSIM Advanced, the TIA Portal project, matching tag/coupling definitions, and a DAQ handshake. The scripts reference helpers that are **not included**: `STOP_DA`, `START_DA`, `OPENFILE`, `WAIT_END_OF_DA`, `InitNXSimulation`, `SwitchMode_Abort`, and `SwitchMode_Auto`. Their names alone are not enough to reconstruct their implementation.

Shared-memory paths and PLC coupling identifiers must match the original engineering configuration. The press-arm experiment uses the forwarding chart's selector and `SIMIT_SHM_StuckPosition` / `SIMIT_SHM_ProvPosFault`, not an ML model.

## Experiment Lifecycle

```mermaid
flowchart TD
    Stop[Stop acquisition and reset simulation] --> Init[Initialize NX and simulated interfaces]
    Init --> Mode[Run abort and automatic-mode initialization]
    Mode --> Ready[Allow simulation to settle]
    Ready --> DAQ[Start acquisition handshake and open recording]
    DAQ --> Delay[Advance to the chosen injection time]
    Delay --> Inject[Apply simulated fault]
    Inject --> Record[Record until acquisition completes or times out]
    Record --> Finish[Stop acquisition]
    Finish --> Release[Release forced simulation values]
    Release --> Review[Check trace and annotate the recording]
```

This is a conceptual lifecycle; individual transcriptions differ in their final stop/wait statements. They are historical references, not a validated orchestration framework. A new experiment must verify cleanup and helper behavior in the simulator before collecting data.

The original acquisition configuration used shuttle-presence start/stop conditions and a maximum capture interval of approximately 30 seconds. A fixed scenario delay is not necessarily the same as elapsed time from the first saved event; that distinction matters when assigning targets from timings.