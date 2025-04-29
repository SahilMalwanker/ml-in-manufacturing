"""Real-time fault detection on the live MQTT stream of the press station.

The PLC (real or virtual) publishes one payload per scan. Like the data acquisition client,
the subscriber keeps only payloads whose signals changed (event-driven data), buffers the
last ``w`` samples and classifies the window as soon as the buffer is full. The model is
advisory: all safety functions stay in the PLC.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import TextIO

import numpy as np

from .daq import parse_payload
from .detector import FaultDetector
from .schema import fault_name

DEFAULT_TOPIC = "Station60_Press_MQTT_Topic"
PASSWORD_ENV = "ML_MANUFACTURING_MQTT_PASSWORD"


@dataclass(frozen=True)
class Prediction:
    sample: int
    code: int
    name: str
    confidence: float | None
    latency_ms: float


class OnlinePredictor:
    """Turns a stream of MQTT payloads into fault predictions."""

    def __init__(self, detector: FaultDetector, dedupe: bool = True):
        self.detector = detector
        self.dedupe = dedupe
        self._buffer: deque[np.ndarray] = deque(maxlen=detector.samples_per_window)
        self._last: np.ndarray | None = None
        self.samples = 0

    def update(self, payload: str | bytes) -> Prediction | None:
        """Feed one payload; returns a prediction once enough changed samples are buffered."""
        started = time.perf_counter()
        values = parse_payload(payload)
        if self.dedupe and self._last is not None and np.array_equal(values, self._last):
            return None
        self._last = values
        self._buffer.append(values)
        self.samples += 1
        if len(self._buffer) < self._buffer.maxlen:
            return None

        window = np.concatenate(self._buffer)[None, :]
        code = int(self.detector.predict(window)[0])
        confidence = self.detector.confidence(window)
        return Prediction(
            sample=self.samples,
            code=code,
            name=fault_name(code),
            confidence=None if confidence is None else float(confidence[0]),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def format_prediction(prediction: Prediction) -> str:
    status = "NORMAL" if prediction.code == 0 else "FAULT "
    confidence = "" if prediction.confidence is None else f"  p={prediction.confidence:.2f}"
    return (
        f"#{prediction.sample:<5d} {status} [{prediction.code:>2}] {prediction.name}"
        f"{confidence}  ({prediction.latency_ms:.1f} ms)"
    )


def serve(
    model_path: str,
    host: str = "localhost",
    port: int = 1883,
    topic: str = DEFAULT_TOPIC,
    username: str | None = None,
    publish_topic: str | None = None,
    dedupe: bool = True,
    changes_only: bool = False,
    out: TextIO = sys.stdout,
) -> None:
    """Subscribe to the press-station topic and print (and optionally publish) predictions."""
    import paho.mqtt.client as mqtt

    detector = FaultDetector.load(model_path)
    predictor = OnlinePredictor(detector, dedupe=dedupe)
    last_code: list[int | None] = [None]

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            print(f"[mqtt] connection refused: {reason_code}", file=out)
            return
        print(f"[mqtt] connected to {host}:{port}, subscribing to '{topic}'", file=out)
        client.subscribe(topic)

    def on_message(client, userdata, message):
        try:
            prediction = predictor.update(message.payload)
        except ValueError as exc:
            print(f"[warn] skipped malformed payload: {exc}", file=out)
            return
        if prediction is None:
            return
        if not changes_only or prediction.code != last_code[0]:
            print(format_prediction(prediction), file=out, flush=True)
        last_code[0] = prediction.code
        if publish_topic:
            client.publish(publish_topic, json.dumps(asdict(prediction)))

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        print(f"[mqtt] disconnected ({reason_code}), reconnecting ...", file=out)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ml-manufacturing-inference")
    if username:
        client.username_pw_set(username, os.environ.get(PASSWORD_ENV))
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    print(
        f"[ml-manufacturing] {detector.model_name.upper()} detector, window {detector.window}, "
        f"{len(detector.classes)} classes: {detector.classes}",
        file=out,
    )
    print(f"[mqtt] connecting to {host}:{port} ...", file=out)
    client.connect_async(host, port, keepalive=60)
    try:
        client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        print("[ml-manufacturing] stopped", file=out)
    finally:
        client.disconnect()
