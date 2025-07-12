# PLC Telemetry Reference

[Architecture](../docs/architecture.md) | [Python signal schema](../src/ml_manufacturing/schema.py)

[MQTT_PublishMsgBuild.scl](MQTT_PublishMsgBuild.scl) is a compact transcription of the **body** of FC9 from the project's August 2025 TIA Portal export. It documents the numeric wire-message order used by the 51-signal dataset. It is not a standalone, importable PLC program or a complete MQTT client.

The original function returns `Void` and uses these temporary variables:

| Variable | Type |
|---|---|
| `tempString` | `WString[1024]` |
| `tempLength` | `Int` |
| `MsgCnt` | `Int` |

Dependencies not bundled here are the PLC tag table, `Stationsdaten`, `RFIDInterface`, field-device diagnostic data, the F-I/O symbols, `MQTT_Data`, and the helper functions `appendSysTime`, `appendBoolValue`, `appendByteValue`, `appendIntValue`, `appendRealvalue`, and `appendWordValue`. In the source project `MQTT_Data` is DB13, with a payload buffer and message length consumed by the separate MQTT publishing logic.

The function appends seven time fields, then 51 process values, then copies the assembled string into the message buffer. Its final helper uses an empty separator, so the message has **no trailing empty CSV field**. At 51 features, the real-valued position feedback is `Stationsdaten.ActPosPercCtrlOutp`; do not insert the older raw B10 field from the background report.

Executable append order is authoritative. The source export numbers the last main-PLC and ET200S entries inconsistently in comments; this transcription preserves the actual main-PLC-first order and omits the misleading comments.

The signal-order contract is checked against the Python schema. **TIA compilation, helper behavior, buffer bounds, PLC cycle timing, and hardware operation are not validated here.** The original engineering software is required for those checks. Do not deploy this reference to physical equipment without the facility's normal engineering and safety review.