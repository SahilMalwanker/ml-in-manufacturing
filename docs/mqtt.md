# MQTT Operation

[Overview](../README.md) | [Architecture](architecture.md)

## Local Demonstration

First train a bundle from the repository root:

```bash
ml-manufacturing train --domain combined --model svc --window 3 --out models/press.joblib
```

Install a broker separately, for example Eclipse Mosquitto. For an isolated demonstration, start its local-only default configuration with `mosquitto -p 1883`. Keep that broker terminal open. Broker software is not a Python package dependency and is not bundled.

Start the subscriber in a second terminal:

```bash
ml-manufacturing serve --model models/press.joblib --host localhost --changes-only --publish-topic ml-manufacturing/predictions
```

After the subscriber reports that it is connected, replay one held-out recording in a third terminal:

```bash
ml-manufacturing replay data/real_plant/test/rp_c08_run4.csv --host localhost
```

The default input topic is `Station60_Press_MQTT_Topic`; both processes accept `--topic` and `--port`. The prediction topic must be separate from the input topic. The tools use fixed client IDs, so run only one subscriber and one replayer per broker. Never run the replayer against a live plant's telemetry topic.

Use `--speed 2` for twice-speed playback or `--speed 0` to remove pacing. Timestamps, when present, determine inter-message delays. Only files without timestamps use `--interval-ms` (100 ms by default). The replay publisher omits labels and formats signal values like the PLC.

## Prediction Output

The console includes an accepted-sample counter, class code, class name, and local processing time. A nonzero class indicates one of the known fault labels, not a command to the station. `--changes-only` suppresses repeated console classes but does not suppress JSON predictions.

An illustrative JSON message is:

```json
{
  "sample": 12,
  "code": 8,
  "name": "Press arm position sensor failure",
  "confidence": null,
  "latency_ms": 0.8
}
```

This is an example, not a measured latency guarantee. The SVC does not enable probability estimation, so `confidence` is null. Other included classifiers expose a maximum class-probability score; it is not calibrated confidence or out-of-distribution detection. The counter counts accepted samples, excluding unchanged messages discarded by deduplication. The JSON has no source timestamp or machine ID.

## Event Handling

By default, unchanged signal arrays are skipped even if timestamps differ. The first prediction appears once the rolling buffer contains `w` accepted samples. `--no-dedupe` keeps every message and is useful when comparing a single recording's online output exactly with its offline windows.

The buffer is not reset between replay files or on broker reconnection. Restart the detector when switching experiments or sources. Malformed messages produce warnings; silence produces no new prediction. Retained or delayed messages are not treated as a special stale-data state. There is no heartbeat alarm or timestamp-order enforcement. All of these need additional engineering before unattended plant monitoring.

## Authentication and Network Scope

Use `--username` when the broker requires credentials. The corresponding password is read from the `ML_MANUFACTURING_MQTT_PASSWORD` environment variable, not a command-line password flag. Set it locally without placing a secret in code, a checked-in file, or shell history. This project does not load `.env` files automatically.

The current CLI supports plain MQTT/TCP, **not TLS**. Use it only on localhost or an approved isolated lab network. Do not expose an unauthenticated listener to the Internet. Broker authorization should permit the inference client to subscribe only to the telemetry topic and, if needed, publish only to its separate prediction topic. Authentication without TLS does not encrypt credentials.

Messages use QoS 0 with no durable processing guarantee. Reconnect attempts back off from 1 to 30 seconds. Use Ctrl+C to stop the subscriber. Broker delivery, buffering, log output, and acquisition delay are outside the reported local prediction timing.

## Model Trust

Detector bundles use joblib/pickle serialization. **Load only bundles you trained or otherwise fully trust:** deserialization can execute code before format checks run. Schema and format checks are compatibility checks, not a security sandbox. Retrain in the same dependency environment used for inference rather than transferring legacy model binaries across scikit-learn versions.

## Troubleshooting

| Symptom | Check |
|---|---|
| Cannot connect | Start the broker and verify host, port, local firewall and authentication |
| Connected, no predictions | Confirm topic/source, then wait for enough changed samples; repeated states are skipped |
| Wrong field-count warning | This project expects 58 wire fields; do not send labels or the older 56-feature schema |
| Classifications differ from offline output | Use one recording, restart the subscriber, use `--no-dedupe`, and verify model/schema |
| Unexpected transfer results | Review missing twin signals and clipping in the results guide |
| Disconnect loops with two clients | Fixed client IDs conflict; stop duplicate subscribers or replayers |

The automated tests validate parsing, replay formatting, buffer warm-up, deduplication, and offline/online prediction parity without requiring a broker. A broker-backed smoke test is a separate verification step; neither substitutes for the original PLC/SIMIT setup.