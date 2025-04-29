"""Replay DAQ recordings over MQTT so the real-time detector can be demonstrated without a PLC."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

import numpy as np

from .daq import elapsed_ms, has_timestamps, read_recording
from .realtime import DEFAULT_TOPIC, PASSWORD_ENV
from .schema import FEATURE_COLUMNS, TIMESTAMP_COLUMNS


def _format_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def iter_payloads(path: str | Path, interval_ms: float = 100.0) -> Iterator[tuple[float, str]]:
    """Yield ``(seconds since first sample, payload)`` in the PLC's MQTT message format.

    Recordings with PLC timestamps keep their original timing; files without timestamps are
    paced at ``interval_ms`` and stamped with the current UTC time.
    """
    frame = read_recording(path)
    if has_timestamps(frame):
        offsets = elapsed_ms(frame) / 1000.0
        stamps = frame[list(TIMESTAMP_COLUMNS)].astype(int).to_numpy().tolist()
    else:
        offsets = np.arange(len(frame)) * interval_ms / 1000.0
        start = datetime.now(timezone.utc)
        moments = [start + timedelta(seconds=float(offset)) for offset in offsets]
        stamps = [[m.year, m.month, m.day, m.hour, m.minute, m.second, m.microsecond * 1000] for m in moments]
    values = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    for offset, stamp, row in zip(offsets, stamps, values, strict=True):
        fields = [f"{int(stamp[0]):04d}", *(f"{int(part):02d}" for part in stamp[1:6]), str(int(stamp[6]))]
        fields += [_format_value(value) for value in row]
        yield float(offset), ",".join(fields)


def replay(
    paths: list[str | Path],
    host: str = "localhost",
    port: int = 1883,
    topic: str = DEFAULT_TOPIC,
    speed: float = 1.0,
    interval_ms: float = 100.0,
    username: str | None = None,
    out: TextIO = sys.stdout,
) -> int:
    """Publish recordings to ``topic``; ``speed`` 2.0 plays twice as fast, 0 as fast as possible."""
    import paho.mqtt.client as mqtt

    if speed < 0:
        raise ValueError("speed must be >= 0")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ml-manufacturing-replay")
    if username:
        client.username_pw_set(username, os.environ.get(PASSWORD_ENV))
    client.connect(host, port, keepalive=60)
    client.loop_start()
    sent = 0
    try:
        for path in paths:
            payloads = list(iter_payloads(path, interval_ms))
            print(f"[replay] {Path(path).name}: {len(payloads)} payloads -> '{topic}'", file=out, flush=True)
            previous = 0.0
            for offset, payload in payloads:
                if speed > 0 and offset > previous:
                    time.sleep((offset - previous) / speed)
                previous = offset
                client.publish(topic, payload, qos=0).wait_for_publish(timeout=5)
                sent += 1
    finally:
        client.loop_stop()
        client.disconnect()
    print(f"[replay] done, {sent} payloads published", file=out)
    return sent
