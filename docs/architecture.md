# System Architecture

[Overview](../README.md) | [MQTT operation](mqtt.md) | [Data](../data/README.md)

## Physical and Virtual Station

ST60 receives a shuttle carrying a relay/socket assembly, locks it, performs the pressing cycle, returns the press arm, and releases the shuttle. The physical PLC remains responsible for motion and safety. The ML detector only observes telemetry alongside that control system.

![Original NX MCD station model](images/press_station.png)

The cropped image above shows the project digital twin, not a photograph of the physical station.

```mermaid
flowchart TB
    Engineering[TIA Portal V17 engineering project]
    subgraph Real[Physical Station]
        RPLC[CPU 1516F-3 PN/DP]
        IO[Distributed I/O and field devices]
        Mechanics[Press arm / valves / shuttle / RFID]
        Safety[Safety switches and interlocks]
        RPLC <-->|Fieldbus I/O| IO
        IO <--> Mechanics
        Safety --> IO
    end
    subgraph Twin[Virtual Commissioning]
        VPLC[PLCSIM Advanced virtual PLC]
        Coupling[SIMIT PLC coupling]
        Signals[SIMIT forwarding and fault logic]
        Memory[Shared-memory signal mapping]
        MCD[NX MCD mechanical simulation]
        VPLC <--> Coupling
        Coupling <--> Signals
        Signals <--> Memory
        Memory <--> MCD
        Scripts[Simulation scenario scripts] --> Signals
    end
    Engineering -.-> RPLC
    Engineering -.-> VPLC
    RPLC --> Publisher[FC9 payload construction / MQTT client]
    VPLC --> Publisher
    Publisher --> Broker[Mosquitto broker]
    Broker --> Acquisition[DAQ recording client]
    Broker --> Inference[Python inference subscriber]
    Acquisition --> Files[Event-driven recordings]
    Files --> Offline[Offline training and evaluation]
    Offline --> Bundle[Model + scaler + window + schema]
    Bundle --> Inference
    Inference --> Display[Console / optional prediction topic]
```

The shared publisher box represents equivalent telemetry logic in the two PLCs, not a third PLC. Run one selected source on the topic; do not interleave two machines into one inference buffer. Dotted arrows indicate engineering configuration, not telemetry. External engineering tools, the original DAQ executable, device configuration, and licensed project containers are not included.

## Signal Exchange

![Original SIMIT forwarding chart](images/simit_forwarding.png)

The forwarding chart passes lower/upper end-position signals, press-position feedback, and shuttle presence from MCD toward the virtual PLC. In the opposite direction, shuttle release and the simulated position setpoint reach MCD. For the stuck-sensor experiment, a SIMIT selector substitutes a captured position value for live position feedback.

```mermaid
flowchart LR
    Position[Live MCD position] --> Selector{SIMIT fault selector}
    Captured[Captured stuck position] --> Selector
    Enable[Simulation fault enable] -.-> Selector
    Selector --> Feedback[Position feedback to virtual PLC]
    Feedback --> Control[PLC sequence and position control]
    Control --> Setpoint[Position setpoint to MCD]
    Setpoint --> Motion[MCD mechanical response]
    Motion --> Position
```

The substituted value changes what the controller observes; it does not make every mechanical or pneumatic fault realistic. In the supplied recordings, actual valve-pressure feedback is not reproduced by the twin.

## Signal Contract

The project's later FC9 export uses **seven timestamp fields plus 51 signals**. The earlier background report uses a different 56-feature schema and must not be used to decode these recordings.

| Feature positions, 1-based | Group | Count |
|---|---|---:|
| 1-4 | Shuttle hand-over | 4 |
| 5-13 | Station state and operating mode | 9 |
| 14-26 | Safety status and feedback | 13 |
| 27-33 | Press end positions and position control | 7 |
| 34-37 | M1/M2 target and actual valve pressure | 4 |
| 38-47 | RFID read/write/reset | 10 |
| 48-51 | Field-device diagnostics | 4 |

Canonical names, addresses, types and English descriptions live in [schema.py](../src/ml_manufacturing/schema.py). The [PLC reference](../plc/README.md) includes the payload-building SCL body. The last four signals follow **executable append order**, including main PLC before ET200S; the historical FC9 feature-number comments reverse those two numbers.

Raw timestamps have a nanosecond fractional field. Wire messages are numeric, comma-separated, and contain no label. The Python parser requires the exact field count, removes the timestamp prefix and converts the 51 signals to numeric input. A same-width but reordered PLC payload cannot be detected automatically: verify tag order before connecting a new publisher.

## Training Flow

```mermaid
flowchart TD
    Manifest[Read recording manifest] --> Split[Select source training runs and destination test runs]
    Split --> Parse[Read recordings and select canonical signal columns]
    Parse --> Window[Build windows separately inside each recording]
    Window --> Features[Flatten w x 51 signal values; target is last-row label]
    Features --> Fit[Fit MinMax scaler on training windows only]
    Fit --> Train[Fit selected classifier on scaled training windows]
    Train --> Save[Save trusted detector bundle]
    Train --> Test[Transform and predict held-out test windows]
    Test --> Metrics[Accuracy / macro precision / recall / F1]
    Metrics --> Evidence[Confusion matrices and per-fault transfer analysis]
```

At window 3, each input is `[sample t-2: 51 signals, sample t-1: 51 signals, sample t: 51 signals]`, giving 153 features. Neither the target nor absolute time is included. The model classifies the current state; it does not forecast a future event.

## Streaming Flow

```mermaid
flowchart TD
    Message[MQTT payload received] --> Parse[Parse 58 numeric fields]
    Parse --> Valid{Valid payload and usable values?}
    Valid -->|No| Warn[Log malformed payload warning]
    Warn --> Message
    Valid -->|Yes| Changed{Changed process signals or dedupe disabled?}
    Changed -->|No| Message
    Changed -->|Yes| Buffer[Append signals to bounded window]
    Buffer --> Full{Window full?}
    Full -->|No| Message
    Full -->|Yes| Transform[Flatten and apply saved scaler]
    Transform --> Predict[Predict class and optional score]
    Predict --> Console[Print result, optionally only class changes]
    Predict --> JSON[Optionally publish prediction JSON]
    Console --> Message
```

The parser detects wrong widths and nonnumeric signal values; classifier/scaler validation handles invalid numeric input. There is no physical plausibility validator or automatic sensor-schema discovery. The live buffer spans the continuous stream, unlike offline windows split by recording. Restart the subscriber between independently replayed experiments when exact offline parity is needed. A connection break or a long silence does not currently clear that buffer automatically.

![Offline replay through the streaming predictor](images/realtime_timeline.png)

This trace passes four held-out recordings sequentially through one local predictor with deduplication disabled and a continuous buffer. It demonstrates the inference path, not broker transport, live hardware validation, or end-to-end latency measurement.

## Code Ownership

| Module | Responsibility |
|---|---|
| `schema` | Signal order, metadata and case names |
| `daq` | File/wire parsing and optional raw-data labelling |
| `datasets` | Manifest paths and experiment selection |
| `windowing` | Per-recording windows and target alignment |
| `models`, `detector` | Classifier factories, fitted preprocessing, bundles and metrics |
| `benchmark`, `plotting` | Scenario comparisons and confusion-matrix rendering |
| `realtime`, `replay` | Stream buffering, MQTT subscriber and recording publisher |
| `cli` | User-facing commands |

## Safety Boundary

The inference path is advisory and has no actuator-write interface. Communication loss, a normal label, or a high probability score must never release a safety interlock. Any plant use needs an independently engineered watchdog, access control, payload validation, operational alarm policy, and fresh hardware validation. Fault-injection scripts belong only in an isolated simulation environment.