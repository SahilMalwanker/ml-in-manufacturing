"""Animated SVG artwork for the README and docs.

pip install -e .
python scripts/make_readme_art.py

Writes docs/images/: banner.svg (header), pipeline.svg (data flow), results_bars.svg (transfer
results), live_terminal.svg (streaming console) and daq_handshake.svg (how a twin run is recorded).
The SVGs are self-contained (no scripts, web fonts or external links), so GitHub renders and
animates them inline. Numbers come from data/manifest.csv and results/benchmark.csv; the console
lines are produced by the package's own streaming code on a held-out recording, and the measured
per-message latencies therefore differ slightly between runs.
"""

from __future__ import annotations

import csv
import math
from html import escape
from pathlib import Path
from string import Template

from ml_manufacturing.datasets import Recording, load_manifest, scenario_data
from ml_manufacturing.detector import FaultDetector
from ml_manufacturing.realtime import DEFAULT_TOPIC, OnlinePredictor, format_prediction
from ml_manufacturing.replay import iter_payloads
from ml_manufacturing.schema import N_FEATURES

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"

PALETTE = {
    "FONT": "'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, 'Liberation Sans', sans-serif",
    "MONO": "'Cascadia Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    "TEAL": "#2DD4BF",
    "TEAL_LIGHT": "#5EEAD4",
    "ORANGE": "#FB923C",
    "ORANGE_LIGHT": "#FDBA74",
    "WHITE": "#F8FAFC",
    "TEXT": "#E2E8F0",
    "MUTED": "#94A3B8",
}
TEAL, TEAL_LIGHT = PALETTE["TEAL"], PALETTE["TEAL_LIGHT"]
ORANGE, ORANGE_LIGHT = PALETTE["ORANGE"], PALETTE["ORANGE_LIGHT"]
WHITE = PALETTE["WHITE"]

BASE_CSS = """
text { font-family: $FONT; }
.rise { animation: rise .9s cubic-bezier(.2, .7, .2, 1) both; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } }
.kicker { font-size: 14px; font-weight: 700; letter-spacing: 3px; fill: $TEAL_LIGHT; }
.legend { font-size: 13px; fill: $MUTED; }
"""
REDUCED_MOTION = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"

BANNER_CSS = """
.title { font-size: 56px; font-weight: 800; letter-spacing: -1px; }
.sub { font-size: 17px; fill: #B6C2CF; }
.chip { fill: #FFFFFF; fill-opacity: .05; stroke: $TEAL; stroke-opacity: .45; }
.chip-text { font-size: 15px; font-weight: 600; fill: $TEXT; }
.lane { font-size: 13px; font-weight: 700; letter-spacing: 2px; }
.lane-note { font-size: 12px; fill: #7F92A6; }
.frozen { font-size: 11px; font-weight: 700; letter-spacing: 1px; fill: $ORANGE_LIGHT; }
.trace { fill: none; stroke-width: 2.4; stroke-linejoin: round; stroke-linecap: round;
         animation: scroll 5s linear infinite; }
@keyframes scroll { to { transform: translateX(-240px); } }
.wire { fill: none; stroke: #CBD5E1; stroke-opacity: .25; stroke-width: 1.6; }
.core { fill: #0D1B26; stroke: $TEAL; stroke-width: 2.5; animation: ring 7s infinite; }
.core-text { font-size: 22px; font-weight: 800; letter-spacing: 1px; fill: $WHITE; }
.pulse { fill: none; stroke: $TEAL; stroke-width: 2; transform-box: fill-box; transform-origin: center;
         animation: pulse 2.4s ease-out infinite, ring 7s infinite; }
@keyframes pulse { from { opacity: .6; } to { transform: scale(1.9); opacity: 0; } }
@keyframes ring { 0%, 56% { stroke: $TEAL; } 60%, 96% { stroke: $ORANGE; } 100% { stroke: $TEAL; } }
.ok { animation: ok 7s infinite; }
.bad { opacity: 0; animation: bad 7s infinite; }
@keyframes ok { 0%, 56% { opacity: 1; } 60%, 96% { opacity: 0; } 100% { opacity: 1; } }
@keyframes bad { 0%, 56% { opacity: 0; } 60%, 96% { opacity: 1; } 100% { opacity: 0; } }
.state { font-size: 14px; font-weight: 800; letter-spacing: 2px; }
"""

PIPELINE_CSS = """
.card { fill: #FFFFFF; fill-opacity: .05; stroke: #FFFFFF; stroke-opacity: .1; }
.card-title { font-size: 17px; font-weight: 700; fill: $WHITE; }
.card-note { font-size: 13px; fill: $MUTED; }
.edge { fill: none; stroke: #CBD5E1; stroke-opacity: .35; stroke-width: 1.6; }
.bundle { fill: none; stroke: $TEAL_LIGHT; stroke-opacity: .7; stroke-width: 1.6; stroke-dasharray: 6 6;
          animation: dash 1.2s linear infinite; }
@keyframes dash { to { stroke-dashoffset: -12; } }
.blink { animation: blink 1.4s ease-in-out infinite; }
@keyframes blink { 50% { opacity: .2; } }
.note { font-size: 12.5px; font-style: italic; fill: $MUTED; }
"""

RESULTS_CSS = """
.caption { font-size: 15px; fill: #B6C2CF; }
.row { font-size: 16px; font-weight: 600; fill: $TEXT; }
.value { font-size: 17px; font-weight: 800; }
.track { fill: #FFFFFF; fill-opacity: .06; }
.grow { transform-box: fill-box; transform-origin: left center;
        animation: grow 1.4s cubic-bezier(.2, .8, .2, 1) both; }
@keyframes grow { from { transform: scaleX(0); } }
.sweep { animation: sweep 6s ease-in-out infinite both; }
@keyframes sweep { 45%, 100% { transform: translateX(1000px); } }
.grid { stroke: #FFFFFF; stroke-opacity: .1; stroke-dasharray: 3 5; }
.tick { font-size: 12px; fill: $MUTED; }
"""

TERMINAL_CSS = """
.window { fill: #0A0F1A; stroke: #FFFFFF; stroke-opacity: .1; }
.titlebar { fill: #FFFFFF; fill-opacity: .04; }
.chrome { font-size: 14px; fill: $MUTED; }
.pane { font-size: 12px; font-weight: 700; letter-spacing: 2px; fill: #64748B; }
.mono { font-family: $MONO; font-size: 16px; fill: #CBD5E1; }
.cmd { fill: $TEXT; }
.prompt { fill: $TEAL; font-weight: 700; }
.tag { fill: #7F92A6; }
.normal { fill: $TEAL_LIGHT; font-weight: 700; }
.fault { fill: $ORANGE_LIGHT; font-weight: 700; }
.dim { fill: #64748B; }
.caret { fill: $TEAL_LIGHT; animation: caret 1s steps(1) infinite; }
@keyframes caret { 50% { opacity: 0; } }
.status { font-size: 13px; font-weight: 800; letter-spacing: 1.5px; }
.wire-label { font-size: 12px; font-weight: 700; letter-spacing: 2px; fill: $MUTED; }
"""

HANDSHAKE_CSS = """
.caption { font-size: 16px; fill: #B6C2CF; }
.head { fill: #FFFFFF; fill-opacity: .05; stroke: #FFFFFF; stroke-opacity: .12; }
.head-title { font-size: 17px; font-weight: 700; fill: $WHITE; }
.head-note { font-size: 13.5px; fill: $MUTED; }
.life { stroke: #FFFFFF; stroke-opacity: .16; stroke-dasharray: 4 6; }
.msg { fill: none; stroke-width: 2; }
.label { font-size: 15px; fill: $TEXT; }
.byte { font-family: $MONO; font-size: 13px; font-weight: 700; }
.pill { fill: #0D1B26; stroke-width: 1.5; }
.pill-text { font-family: $MONO; font-size: 14px; font-weight: 700; fill: $WHITE; }
.note { font-size: 15px; fill: #B6C2CF; }
.band { fill: $TEAL; fill-opacity: .08; }
.foot { font-size: 13.5px; fill: #7F92A6; }
"""

CARD_WIDTH, CARD_HEIGHT = 200, 74
MONO_ADVANCE = 9.6  # px per character at 16 px; every terminal line is fitted to it with textLength
DEMO_RECORDING = "rp_c06_run3"  # held-out real-plant emergency-stop run for the console animation


def frame(width: int, height: int, label: str, css: str, body: str, defs: str = "") -> str:
    """Dark rounded card with a faint engineering grid, shared by all README artwork."""
    style = Template(BASE_CSS + css).substitute(PALETTE) + REDUCED_MOTION
    return f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>{style}</style>
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#0B1220"/><stop offset="1" stop-color="#0F2E35"/>
</linearGradient>
<linearGradient id="accent" x1="0" x2="1">
<stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{ORANGE}"/>
</linearGradient>
<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
<path d="M32 0H0V32" fill="none" stroke="#FFFFFF" stroke-opacity=".045"/>
</pattern>
<filter id="soft" x="-2" y="-2" width="5" height="5">
<feGaussianBlur stdDeviation="2.2" result="blur"/>
<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
<clipPath id="card"><rect width="{width}" height="{height}" rx="18"/></clipPath>
{defs}
</defs>
<g clip-path="url(#card)">
<rect width="{width}" height="{height}" fill="url(#bg)"/>
<rect width="{width}" height="{height}" fill="url(#grid)"/>
{body}
</g>
</svg>
"""


def packet(path: str, color: str, seconds: float, offset: float = 0.0) -> str:
    """Glowing dot travelling along ``path``; a negative begin staggers dots without a start-up gap."""
    begin = f"-{offset}s" if offset else "0s"
    return (
        f'<circle r="3.6" fill="{color}" filter="url(#soft)">'
        f'<animateMotion path="{path}" dur="{seconds}s" begin="{begin}" repeatCount="indefinite"/></circle>'
    )


def legend(items: list[tuple[str, str]], right: float, y: float) -> str:
    """Right-aligned row of coloured dots with labels (text widths are estimated)."""
    parts, x = [], right
    for label, color in reversed(items):
        x -= len(label) * 7.2
        parts.append(f'<text x="{x:g}" y="{y + 5:g}" class="legend">{label}</text>')
        parts.append(f'<circle cx="{x - 11:g}" cy="{y:g}" r="5" fill="{color}"/>')
        x -= 36
    return "".join(parts)


def wave(noisy: bool) -> str:
    """Two periods of a press cycle (rest, press, hold, return); seamless when shifted by one period."""
    if not noisy:
        return "M0 50 L34 50 L52 14 L150 14 L168 50 L274 50 L292 14 L390 14 L408 50 L480 50"
    points = []
    for x in range(0, 481, 3):
        phase = x % 240
        if 34 <= phase < 52:
            y = 50 - 36 * (phase - 34) / 18
        elif 52 <= phase < 150:
            y = 14 - 6 * max(0.0, 1 - (phase - 52) / 14)
        elif 150 <= phase < 168:
            y = 14 + 36 * (phase - 150) / 18
        else:
            y = 50 + 5 * max(0.0, 1 - (phase - 168) / 14) if phase >= 168 else 50.0
        y += 1.4 * math.sin(2 * math.pi * 19 * x / 240) + 0.8 * math.sin(2 * math.pi * 47 * x / 240)
        points.append(f"{x} {y:.1f}")
    return "M" + " L".join(points)


def lane(y: int, label: str, note: str, color: str, path: str, seconds: float, freezes: bool = False) -> str:
    """Scrolling signal trace; with ``freezes`` the reading locks at 63 % while the fault pill is shown."""
    trace = f'<path d="{path}" class="trace" stroke="{color}" style="animation-duration:{seconds}s"/>'
    border = ' style="animation: ring 7s infinite"' if freezes else ""
    if freezes:
        level = 50 - 0.63 * 36
        trace = (
            f'<g class="ok">{trace}</g><g class="bad">'
            f'<path d="M0 {level:.1f} H240" class="trace" stroke="{ORANGE}" style="animation:none"/>'
            f'<text x="230" y="56" text-anchor="end" class="frozen">READING FROZEN AT 63 %</text></g>'
        )
    return (
        f'<g transform="translate(700 {y})">'
        f'<text y="-12" class="lane" fill="{color}">{label}</text>'
        f'<text x="240" y="-12" text-anchor="end" class="lane-note">{note}</text>'
        f'<rect width="240" height="64" rx="10" fill="#FFFFFF" fill-opacity=".04" '
        f'stroke="{color}" stroke-opacity=".4"{border}/>'
        f'<g clip-path="url(#lane)">{trace}</g></g>'
    )


def status_pill(kind: str, text: str, color: str, light: str) -> str:
    return (
        f'<g class="{kind}"><rect x="940" y="258" width="200" height="34" rx="17" fill="{color}" '
        f'fill-opacity=".14" stroke="{color}"/><circle cx="964" cy="275" r="5" fill="{color}"/>'
        f'<text x="1050" y="280" text-anchor="middle" class="state" fill="{light}">{text}</text></g>'
    )


def banner(recordings: int, signals: int, classes: int, runs: int) -> str:
    chips, x = [], 56
    labels = (f"{recordings} recordings", f"{signals} PLC signals", f"{classes} classes", f"{runs} benchmarks")
    for i, text in enumerate(labels):
        width = round(len(text) * 9.4 + 30)
        chips.append(
            f'<g class="rise" style="animation-delay:{0.55 + 0.1 * i:.2f}s">'
            f'<rect x="{x}" y="284" width="{width}" height="34" rx="17" class="chip"/>'
            f'<text x="{x + width / 2:g}" y="306" text-anchor="middle" class="chip-text">{text}</text></g>'
        )
        x += width + 12
    real_wire, twin_wire, out_wire = (
        "M940 128 C975 128 972 176 1000 179",
        "M940 238 C975 238 972 190 1000 187",
        "M1040 223 L1040 258",
    )
    defs = f"""<clipPath id="lane"><rect x="1" y="1" width="238" height="62" rx="9"/></clipPath>
<radialGradient id="glow">
<stop offset="0" stop-color="{TEAL}" stop-opacity=".22"/><stop offset="1" stop-color="{TEAL}" stop-opacity="0"/>
</radialGradient>
<linearGradient id="shine" x1="0" x2="1">
<stop offset="0" stop-color="#FFFFFF" stop-opacity="0"/><stop offset=".42" stop-color="#FFFFFF" stop-opacity="0"/>
<stop offset=".5" stop-color="#FFFFFF" stop-opacity=".55"/><stop offset=".58" stop-color="#FFFFFF" stop-opacity="0"/>
<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>
<animateTransform attributeName="gradientTransform" type="translate" values="-1 0;-1 0;1 0;1 0"
 keyTimes="0;.18;.5;1" dur="7s" repeatCount="indefinite"/>
</linearGradient>"""
    body = f"""
<text x="56" y="84" class="kicker rise">HOCHSCHULE DARMSTADT · AUTFAB ST60 PRESS STATION</text>
<g class="rise" style="animation-delay:.12s">
<text x="53" y="150" class="title" fill="{WHITE}">Machine Learning</text>
</g>
<g class="rise" style="animation-delay:.24s">
<text x="53" y="212" class="title" fill="url(#accent)">in Manufacturing</text>
<text x="53" y="212" class="title" fill="url(#shine)">in Manufacturing</text>
</g>
<text x="57" y="252" class="sub rise"
 style="animation-delay:.38s">Real-plant vs. digital-twin training data for fault detection</text>
{"".join(chips)}
<circle cx="1040" cy="183" r="130" fill="url(#glow)"/>
{lane(96, "REAL PLANT", "ST60 · S7-1500", TEAL, wave(noisy=True), 5.5, freezes=True)}
{lane(206, "DIGITAL TWIN", "SIMIT · NX MCD", ORANGE, wave(noisy=False), 5.0)}
<path d="{real_wire}" class="wire"/>
<path d="{twin_wire}" class="wire"/>
<path d="{out_wire}" class="wire"/>
<circle cx="1040" cy="183" r="40" class="pulse"/>
<circle cx="1040" cy="183" r="40" class="core"/>
<text x="1040" y="191" text-anchor="middle" class="core-text">ML</text>
{packet(real_wire, TEAL_LIGHT, 1.8)}
{packet(twin_wire, ORANGE_LIGHT, 1.8, 0.9)}
{packet(out_wire, WHITE, 1.2, 0.3)}
{status_pill("ok", "NORMAL", TEAL, TEAL_LIGHT)}
{status_pill("bad", "FAULT DETECTED", ORANGE, ORANGE_LIGHT)}
"""
    label = "Machine Learning in Manufacturing: real-plant vs. digital-twin training data for fault detection"
    return frame(1200, 360, label, BANNER_CSS, body, defs)


def card(x: int, y: int, title: str, note: str, accent: str, delay: float) -> str:
    return (
        f'<g class="rise" style="animation-delay:{delay:.2f}s">'
        f'<rect x="{x}" y="{y}" width="{CARD_WIDTH}" height="{CARD_HEIGHT}" rx="14" class="card"/>'
        f'<rect x="{x}" y="{y + 14}" width="4" height="{CARD_HEIGHT - 28}" rx="2" fill="{accent}"/>'
        f'<text x="{x + 20}" y="{y + 32}" class="card-title">{title}</text>'
        f'<text x="{x + 20}" y="{y + 54}" class="card-note">{note}</text></g>'
    )


def link(start: tuple[float, float], end: tuple[float, float]) -> str:
    """Horizontal S-curve between two card edges (a straight line when both are level)."""
    (x1, y1), (x2, y2) = start, end
    middle = (x1 + x2) / 2
    return f"M{x1:g} {y1:g} C{middle:g} {y1:g} {middle:g} {y2:g} {x2:g} {y2:g}"


def pipeline(recordings: int) -> str:
    shared = "url(#accentV)"
    boxes = {
        "plant": (24, 96, "Real plant", "ST60 · S7-1500 PLC", TEAL),
        "twin": (24, 236, "Digital twin", "NX MCD · SIMIT · PLCSIM", ORANGE),
        "telemetry": (260, 166, "PLC telemetry", f"{N_FEATURES} signals per message", shared),
        "broker": (496, 166, "MQTT broker", "Mosquitto · port 1883", shared),
        "recordings": (732, 96, "Recordings", f"{recordings} event-driven runs", shared),
        "live": (732, 236, "Live detector", "3-message window", shared),
        "training": (968, 96, "Training", "scaler + classifier", shared),
        "prediction": (968, 236, "Prediction", "console · JSON topic", shared),
    }

    def out(name: str) -> tuple[float, float]:
        x, y = boxes[name][:2]
        return x + CARD_WIDTH, y + CARD_HEIGHT / 2

    def into(name: str) -> tuple[float, float]:
        x, y = boxes[name][:2]
        return x, y + CARD_HEIGHT / 2

    flows = [
        ("plant", "telemetry", [(TEAL_LIGHT, 0.0)]),
        ("twin", "telemetry", [(ORANGE_LIGHT, 0.8)]),
        ("telemetry", "broker", [(WHITE, 0.3)]),
        ("broker", "recordings", [(WHITE, 0.6)]),
        ("broker", "live", [(WHITE, 1.1)]),
        ("recordings", "training", [(TEAL_LIGHT, 0.4)]),
        ("live", "prediction", [(TEAL_LIGHT, 0.2), (ORANGE_LIGHT, 1.0)]),
    ]
    edges, packets = [], []
    for source, target, dots in flows:
        path = link(out(source), into(target))
        edges.append(f'<path d="{path}" class="edge" marker-end="url(#arrow)"/>')
        packets.extend(packet(path, color, 1.6, offset) for color, offset in dots)

    train_x, train_y = boxes["training"][:2]
    live_x, live_y = boxes["live"][:2]
    start_x, start_y = train_x + 42, train_y + CARD_HEIGHT
    end_x = live_x + CARD_WIDTH - 62
    bundle = f"M{start_x} {start_y} C{start_x} {start_y + 36} {end_x} {live_y - 36} {end_x} {live_y}"
    cards = [
        card(x, y, title, note, accent, 0.05 + 0.08 * i) for i, (x, y, title, note, accent) in enumerate(boxes.values())
    ]
    defs = f"""<linearGradient id="accentV" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{ORANGE}"/>
</linearGradient>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0 1 L8 5 L0 9 z" fill="#CBD5E1" fill-opacity=".6"/>
</marker>
<marker id="arrowTeal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0 1 L8 5 L0 9 z" fill="{TEAL_LIGHT}"/>
</marker>"""
    items = [("real plant", TEAL), ("digital twin", ORANGE), ("shared pipeline", "url(#accent)")]
    body = f"""
<text x="24" y="52" class="kicker rise">HOW IT FITS TOGETHER</text>
{legend(items, 1176, 47)}
{"".join(edges)}
<path d="{bundle}" class="bundle" marker-end="url(#arrowTeal)"/>
<text x="{start_x + 12}" y="{(start_y + live_y) / 2 + 8:g}" class="note">detector bundle</text>
{"".join(cards)}
{"".join(packets)}
<circle cx="{live_x + CARD_WIDTH - 18}" cy="{live_y + 16}" r="4.5" fill="{TEAL}" class="blink"/>
"""
    label = (
        "Data flow: the real plant and the digital twin publish PLC telemetry over MQTT for training and live detection"
    )
    return frame(1200, 340, label, PIPELINE_CSS, body, defs)


def results_chart(accuracy: dict[str, float]) -> str:
    rows = (
        ("Real → Real", "R-R", "same"),
        ("Twin → Twin", "D-D", "same"),
        ("Real → Twin", "R-D", "cross"),
        ("Twin → Real", "D-R", "cross"),
        ("Combined → Combined", "C-C", "both"),
    )
    fills = {"same": "url(#same)", "cross": "url(#cross)", "both": "url(#accent)"}
    colors = {"same": TEAL_LIGHT, "cross": ORANGE_LIGHT, "both": WHITE}
    x0, span = 280, 780
    clips, bars = [], []
    for i, (name, scenario, group) in enumerate(rows):
        y = 108 + 46 * i
        width = accuracy[scenario] * span
        clips.append(
            f'<clipPath id="bar{i}"><rect x="{x0}" y="{y}" width="{width:.1f}" height="24" rx="12"/></clipPath>'
        )
        bars.append(f"""
<text x="{x0 - 18}" y="{y + 18}" text-anchor="end" class="row">{name}</text>
<rect x="{x0}" y="{y}" width="{span}" height="24" rx="12" class="track"/>
<rect x="{x0}" y="{y}" width="{width:.1f}" height="24" rx="12" fill="{fills[group]}" class="grow"
 style="animation-delay:{0.15 + 0.12 * i:.2f}s"/>
<g clip-path="url(#bar{i})"><rect x="{x0 - 140}" y="{y}" width="120" height="24" fill="url(#barShine)"
 class="sweep" style="animation-delay:{1.8 + 0.15 * i:.2f}s"/></g>
<text x="{x0 + span + 18}" y="{y + 18}" class="value rise" fill="{colors[group]}"
 style="animation-delay:{0.9 + 0.12 * i:.2f}s">{accuracy[scenario]:.1%}</text>""")
    defs = f"""<linearGradient id="same" x1="0" x2="1">
<stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{TEAL_LIGHT}"/>
</linearGradient>
<linearGradient id="cross" x1="0" x2="1">
<stop offset="0" stop-color="{ORANGE}"/><stop offset="1" stop-color="{ORANGE_LIGHT}"/>
</linearGradient>
<linearGradient id="barShine" x1="0" x2="1">
<stop offset="0" stop-color="#FFFFFF" stop-opacity="0"/><stop offset=".5" stop-color="#FFFFFF" stop-opacity=".45"/>
<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>
</linearGradient>
{"".join(clips)}"""
    items = [("same domain", TEAL), ("cross-domain", ORANGE), ("both domains", "url(#accent)")]
    middle, end = x0 + span / 2, x0 + span
    body = f"""
<text x="40" y="50" class="kicker rise">DOES TWIN DATA TRANSFER TO THE REAL PLANT?</text>
<text x="40" y="76" class="caption rise"
 style="animation-delay:.1s">SVC accuracy on held-out recordings · 3-message window</text>
{legend(items, 1160, 45)}
<path d="M{middle:g} 96 V324 M{end} 96 V324" class="grid"/>
{"".join(bars)}
<text x="{x0}" y="346" text-anchor="middle" class="tick">0%</text>
<text x="{middle:g}" y="346" text-anchor="middle" class="tick">50%</text>
<text x="{end}" y="346" text-anchor="middle" class="tick">100%</text>
"""
    return frame(
        1200, 366, "SVC test accuracy for each combination of training and test domain", RESULTS_CSS, body, defs
    )


def gate(start: float, end: float, period: float) -> str:
    """SMIL opacity switch: visible from ``start`` to ``end`` seconds of a looping ``period``."""
    if end >= period:
        values, times = "0;1", f"0;{start / period:.4f}"
    else:
        values, times = "0;1;0", f"0;{start / period:.4f};{end / period:.4f}"
    return (
        f'<animate attributeName="opacity" values="{values}" keyTimes="{times}" calcMode="discrete" '
        f'dur="{period}s" repeatCount="indefinite"/>'
    )


def mono_line(x: float, y: float, segments: list[tuple[str, str]]) -> str:
    """One terminal line from ``(css class, text)`` segments, fitted to the fixed character advance."""
    text = "".join(part for _, part in segments)
    spans = "".join(
        f'<tspan class="{cls}">{escape(part, quote=False)}</tspan>' if cls else escape(part, quote=False)
        for cls, part in segments
    )
    width = len(text) * MONO_ADVANCE
    return (
        f'<text x="{x}" y="{y}" class="mono" xml:space="preserve" textLength="{width:.1f}" '
        f'lengthAdjust="spacing">{spans}</text>'
    )


def typed_command(x: float, y: float, command: str, start: float, rate: float, period: float, key: str, focus: float):
    """``$ command`` typed character by character; returns (clip definition, body, time typing ends).

    The caret is shown from ``focus`` (when the pane becomes active) until the command is entered.
    """
    steps = [start + i / rate for i in range(1, len(command) + 1)]
    end = steps[-1]
    prompt = 2 * MONO_ADVANCE
    widths = [prompt + i * MONO_ADVANCE for i in range(1, len(command) + 1)]
    times = ";".join(f"{t / period:.4f}" for t in [0.0, *steps, period - 0.4])
    clip_values = ";".join(f"{w:.1f}" for w in [prompt, *widths, prompt])
    caret_values = ";".join(f"{x + w:.1f}" for w in [prompt, *widths, prompt])
    full = prompt + len(command) * MONO_ADVANCE + 4
    clip = (
        f'<clipPath id="{key}"><rect x="{x - 2}" y="{y - 18}" width="{full:.1f}" height="25">'
        f'<animate attributeName="width" values="{clip_values}" keyTimes="{times}" calcMode="discrete" '
        f'dur="{period}s" repeatCount="indefinite"/></rect></clipPath>'
    )
    body = (
        f'<g clip-path="url(#{key})">{mono_line(x, y, [("prompt", "$ "), ("cmd", command)])}</g>'
        f'<g opacity="0">{gate(focus, end + 0.3, period)}<rect x="{x + full - 4:.1f}" y="{y - 15}" width="9" '
        f'height="19" class="caret"><animate attributeName="x" values="{caret_values}" keyTimes="{times}" '
        f'calcMode="discrete" dur="{period}s" repeatCount="indefinite"/></rect></g>'
    )
    return clip, body, end


def prediction_segments(line: str) -> list[tuple[str, str]]:
    """Colour a ``format_prediction`` line: sample counter, status, class and latency."""
    head, rest = line.split("[", 1)
    sample, status = head[: head.index(" ")], head[head.index(" ") :]
    detail, latency = rest.rsplit("  (", 1)
    kind = "normal" if "NORMAL" in status else "fault"
    return [("dim", sample), (kind, status), (kind if kind == "fault" else "", "[" + detail), ("dim", "  (" + latency)]


def tagged_segments(line: str) -> list[tuple[str, str]]:
    tag, rest = line.split("]", 1)
    return [("tag", tag + "]"), ("", rest)]


def terminal_session(recordings: list[Recording], detector: FaultDetector) -> dict:
    """What ``serve --changes-only`` prints while ``replay`` streams one held-out recording."""
    rec = next(r for r in recordings if r.path.stem == DEMO_RECORDING)
    predictor = OnlinePredictor(detector)
    lines: list[tuple[float, int, str]] = []
    last, payloads, offset = None, 0, 0.0
    for offset, payload in iter_payloads(rec.path):
        payloads += 1
        prediction = predictor.update(payload)
        if prediction is None:
            continue
        if prediction.code != last:
            lines.append((offset, prediction.code, format_prediction(prediction)))
        last = prediction.code
    return {"recording": rec, "lines": lines, "payloads": payloads, "duration": offset}


def live_terminal(session: dict, detector: FaultDetector) -> str:
    rec, predictions = session["recording"], session["lines"]
    period, rate = 16.0, 48.0
    hide = period - 0.4
    x, left, right, top = 64, 40, 1160, 30
    serve_cmd = "ml-manufacturing serve --model models/press.joblib --host localhost --changes-only"
    replay_cmd = f"ml-manufacturing replay {rec.path.relative_to(ROOT).as_posix()} --host localhost"
    header = [
        f"[ml-manufacturing] {detector.model_name.upper()} detector, window {detector.window}, "
        f"{len(detector.classes)} classes: {detector.classes}",
        "[mqtt] connecting to localhost:1883 ...",
        f"[mqtt] connected to localhost:1883, subscribing to '{DEFAULT_TOPIC}'",
    ]
    step = 26
    y_serve = top + 66
    divider = y_serve + step * (4 + len(predictions)) + 14
    y_replay = divider + 36
    bottom = y_replay + 3 * step + 18
    height = bottom + 30

    clip_a, serve_body, serve_end = typed_command(x, y_serve, serve_cmd, 0.5, rate, period, "typeServe", 0.01)
    header_times = [serve_end + 0.35, serve_end + 0.6, serve_end + 1.0]
    replay_start = header_times[-1] + 0.6
    clip_b, replay_body, replay_end = typed_command(
        x, y_replay, replay_cmd, replay_start, rate, period, "typeReplay", serve_end + 0.3
    )
    stream_start = replay_end + 0.35
    stream_end = stream_start + session["duration"]

    parts = [serve_body, replay_body]
    for i, (line, t) in enumerate(zip(header, header_times, strict=True)):
        parts.append(
            f'<g opacity="0">{gate(t, hide, period)}{mono_line(x, y_serve + step * (i + 1), tagged_segments(line))}</g>'
        )
    states: list[tuple[float, str, str]] = [(0.0, "WAITING", "#64748B")]
    for i, (offset, code, line) in enumerate(predictions):
        t = stream_start + offset
        y = y_serve + step * (4 + i) + 6
        parts.append(f'<g opacity="0">{gate(t, hide, period)}{mono_line(x, y, prediction_segments(line))}</g>')
        states.append((t, "NORMAL" if code == 0 else f"FAULT {code}", TEAL if code == 0 else ORANGE))
    replay_lines = [
        (stream_start, f"[replay] {rec.path.name}: {session['payloads']} payloads -> '{DEFAULT_TOPIC}'"),
        (stream_end + 0.05, f"[replay] done, {session['payloads']} payloads published"),
    ]
    for i, (t, line) in enumerate(replay_lines):
        y = y_replay + step * (i + 1)
        parts.append(f'<g opacity="0">{gate(t, hide, period)}{mono_line(x, y, tagged_segments(line))}</g>')
    prompt_y = y_replay + step * 3
    parts.append(
        f'<g opacity="0">{gate(stream_end + 0.35, hide, period)}{mono_line(x, prompt_y, [("prompt", "$ ")])}'
        f'<rect x="{x + 2 * MONO_ADVANCE:.1f}" y="{prompt_y - 15}" width="9" height="19" class="caret"/></g>'
    )

    chips = []
    for i, (t, text, color) in enumerate(states):
        end = states[i + 1][0] if i + 1 < len(states) else hide
        chip_w = len(text) * 9.4 + 40
        cx = right - 20 - chip_w
        chips.append(
            f'<g opacity="0">{gate(max(t, 0.01), end, period)}'
            f'<rect x="{cx:.1f}" y="{top + 8}" width="{chip_w:.1f}" height="20" rx="10" fill="{color}" '
            f'fill-opacity=".16" stroke="{color}" stroke-opacity=".7"/>'
            f'<circle cx="{cx + 13:.1f}" cy="{top + 18}" r="4" fill="{color}"/>'
            f'<text x="{cx + 24:.1f}" y="{top + 22.5}" class="status" fill="{color}">{text}</text></g>'
        )
    wire_x = right - 48
    dots = "".join(
        f'<circle r="3.4" fill="{ORANGE_LIGHT}" filter="url(#soft)"><animateMotion path="M{wire_x} {bottom - 16} '
        f'V{y_serve + 30}" dur="1.1s" begin="-{0.37 * k:.2f}s" repeatCount="indefinite"/></circle>'
        for k in range(3)
    )
    body = f"""
<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" rx="12" class="window"/>
<path d="M{left} {top + 36} V{top + 12} Q{left} {top} {left + 12} {top} H{right - 12} Q{right} {top} {right} {top + 12}
 V{top + 36} Z" class="titlebar"/>
<circle cx="{left + 22}" cy="{top + 18}" r="6" fill="#FF5F57"/>
<circle cx="{left + 42}" cy="{top + 18}" r="6" fill="#FEBC2E"/>
<circle cx="{left + 62}" cy="{top + 18}" r="6" fill="#28C840"/>
<text x="{(left + right) / 2:g}" y="{top + 23}" text-anchor="middle"
 class="chrome">live fault detection · ml-in-manufacturing</text>
{"".join(chips)}
<text x="{wire_x - 22}" y="{y_serve - 4}" text-anchor="end" class="pane">DETECTOR</text>
<text x="{wire_x - 22}" y="{y_replay - 4}" text-anchor="end" class="pane">PLANT REPLAY</text>
<path d="M{left} {divider} H{right}" stroke="#FFFFFF" stroke-opacity=".1"/>
<path d="M{wire_x} {bottom - 16} V{y_serve + 30}" stroke="{ORANGE}" stroke-opacity=".25" stroke-dasharray="3 5"/>
<g opacity="0">{gate(stream_start, stream_end, period)}{dots}</g>
<rect x="{wire_x - 26}" y="{divider - 10}" width="52" height="20" rx="10" fill="#0A0F1A"
 stroke="{ORANGE}" stroke-opacity=".4"/>
<text x="{wire_x}" y="{divider + 4}" text-anchor="middle" class="wire-label">MQTT</text>
{"".join(parts)}
"""
    label = (
        f"Terminal session: the detector flags faults while a held-out plant recording ({rec.path.name}) is replayed"
    )
    return frame(1200, height, label, TERMINAL_CSS, body, clip_a + clip_b)


def handshake() -> str:
    """Sequence of one digital-twin recording, from the DAQ client configuration and handshake log."""
    period = 15.0
    hide = period - 0.5
    simit, shm, daq, mqtt = 180, 470, 760, 1040
    heads = (
        (simit, "SIMIT test script", "scenario sequence"),
        (shm, "Shared memory", "SIMITshm_DataAcquisition"),
        (daq, "DAQ client", "drives NX MCD · writes files"),
        (mqtt, "PLC telemetry", "MQTT · one payload per scan"),
    )
    rows = (
        ("call", "START_DA", ("Start NX", 10), ("NX started", 11), "starts the NX MCD simulation"),
        ("call", "OPENFILE", ("OpenFile", 30), ("File opened", 31), "opens the recording files"),
        ("trigger", "Shuttle_in_Station = 1", "start trigger: acquisition starts"),
        ("stream", "payload stream", "stores only changed payloads"),
        ("simit", "GO-FOR · FIX", "fault scenarios: after the scenario delay, force the fault signal"),
        ("trigger", "Shuttle_in_Station = 0 or 30 s", "stop trigger: acquisition stops"),
        ("closed", "WAIT_END_OF_DA", ("File closed", 41), "closes the files"),
        ("call", "STOP_DA", ("Stop NX", 20), ("NX stopped", 21), "stops the simulation"),
        ("simit", "UNFIX", "release the forced signals for the next scenario"),
    )
    row_height = {"call": 66, "closed": 50, "trigger": 46, "stream": 46, "simit": 46}
    centres, top = [], 166.0
    for row in rows:
        centres.append(top + row_height[row[0]] / 2)
        top += row_height[row[0]]
    starts = [0.5 + 1.3 * i for i in range(len(rows))]
    last_y = centres[-1]
    height = round(top + 50)

    def pill(y: float, text: str, color: str) -> tuple[str, float]:
        width = len(text) * 8.7 + 26
        return (
            f'<rect x="{simit - width / 2:.1f}" y="{y - 13}" width="{width:.1f}" height="26" rx="13" class="pill" '
            f'stroke="{color}"/><text x="{simit}" y="{y + 5}" text-anchor="middle" class="pill-text">{text}</text>'
        ), simit + width / 2

    def arrow(x1: float, x2: float, y: float, color: str, t: float, dashed: bool = False) -> str:
        """Line drawn from x1 to x2 starting at ``t``, with a glowing tip and an arrowhead on arrival."""
        tip = -1 if x2 < x1 else 1
        head = f"M{x2:g} {y:g} l{-9 * tip} -5 v10 z"
        times = f"0;{t / period:.4f};{(t + 0.4) / period:.4f};1"
        style = ' stroke-dasharray="5 5"' if dashed else ""
        line = (
            f'<path d="M{x1:g} {y:g} H{x2:g}" class="msg" stroke="{color}"{style}/>'
            if dashed
            else f'<path d="M{x1:g} {y:g} H{x2:g}" class="msg" stroke="{color}" pathLength="1" stroke-dasharray="1">'
            f'<animate attributeName="stroke-dashoffset" values="1;1;0;0" keyTimes="{times}" dur="{period}s" '
            f'repeatCount="indefinite"/></path>'
        )
        return (
            f"{line}"
            f'<path d="{head}" fill="{color}" opacity="0">{gate(t + 0.4, hide, period)}</path>'
            f'<circle r="4" fill="{color}" filter="url(#soft)" opacity="0">{gate(t, t + 0.4, period)}'
            f'<animateMotion path="M{x1:g} {y:g} H{x2:g}" keyPoints="0;0;1;1" keyTimes="{times}" calcMode="linear" '
            f'dur="{period}s" repeatCount="indefinite"/></circle>'
        )

    def byte(y: float, value: int, color: str) -> str:
        return (
            f'<rect x="{shm - 17}" y="{y - 10}" width="34" height="20" rx="6" fill="#0D1B26" stroke="{color}"/>'
            f'<text x="{shm}" y="{y + 4.5}" text-anchor="middle" class="byte" fill="{color}">{value}</text>'
        )

    groups = []
    stream_on = stream_off = 0.0
    for i, (row, t) in enumerate(zip(rows, starts, strict=True)):
        y = centres[i]
        kind, items = row[0], []
        if kind == "call":
            _, command, (ask, ask_byte), (ack, ack_byte), note = row
            shape, edge = pill(y, command, TEAL)
            items += [
                shape,
                arrow(edge + 6, daq - 6, y - 8, TEAL_LIGHT, t),
                f'<text x="{(edge + shm) / 2:g}" y="{y - 15}" text-anchor="middle" class="label">{ask}</text>',
                byte(y - 8, ask_byte, TEAL_LIGHT),
                arrow(daq - 6, edge + 6, y + 10, "#CBD5E1", t + 0.55),
                f'<text x="{(shm + daq) / 2:g}" y="{y + 30}" text-anchor="middle" class="label">{ack}</text>',
                byte(y + 10, ack_byte, "#CBD5E1"),
                f'<text x="{daq + 16}" y="{y + 6}" class="note">{note}</text>',
            ]
        elif kind == "closed":
            _, command, (ack, ack_byte), note = row
            shape, edge = pill(y, command, "#CBD5E1")
            items += [
                shape,
                arrow(daq - 6, edge + 6, y, "#CBD5E1", t),
                f'<text x="{(shm + daq) / 2:g}" y="{y - 10}" text-anchor="middle" class="label">{ack}</text>',
                byte(y, ack_byte, "#CBD5E1"),
                f'<text x="{daq + 16}" y="{y + 6}" class="note">{note}</text>',
            ]
        elif kind in ("trigger", "stream"):
            _, label, note = row
            items += [
                arrow(mqtt - 6, daq + 6, y, ORANGE, t, dashed=kind == "stream"),
                f'<text x="{(daq + mqtt) / 2:g}" y="{y - 10}" text-anchor="middle" class="byte" '
                f'fill="{ORANGE_LIGHT}">{label}</text>',
                f'<text x="{daq - 16}" y="{y + 6}" text-anchor="end" class="note">{note}</text>',
            ]
            if kind == "stream":
                stream_on = t
            elif stream_on:
                stream_off = t + 0.4
        else:
            _, command, note = row
            color = ORANGE if command.startswith("GO-FOR") else "#CBD5E1"
            shape, edge = pill(y, command, color)
            width = len(note) * 7.6 + 28
            items += [
                shape,
                f'<rect x="{edge + 12:.1f}" y="{y - 13}" width="{width:.1f}" height="26" rx="13" fill="#0D1B26" '
                f'stroke="{color}" stroke-opacity=".35"/>',
                f'<text x="{edge + 26:.1f}" y="{y + 5}" class="note">{note}</text>',
            ]
        half = row_height[kind] / 2 - 2
        band = (
            f'<rect x="24" y="{y - half:g}" width="1152" height="{2 * half:g}" rx="10" class="band" opacity="0">'
            f"{gate(t, t + 1.3, period)}</rect>"
        )
        groups.append(f'{band}<g opacity="0">{gate(t, hide, period)}{"".join(items)}</g>')

    stream_y = centres[[row[0] for row in rows].index("stream")]
    stream = "".join(
        f'<circle r="3.4" fill="{ORANGE_LIGHT}" filter="url(#soft)"><animateMotion path="M{mqtt - 6} {stream_y} '
        f'H{daq + 6}" dur="1s" begin="-{0.25 * k:.2f}s" repeatCount="indefinite"/></circle>'
        for k in range(4)
    )
    head_boxes = "".join(
        f'<rect x="{x - 122}" y="98" width="244" height="54" rx="12" class="head"/>'
        f'<text x="{x}" y="121" text-anchor="middle" class="head-title">{title}</text>'
        f'<text x="{x}" y="141" text-anchor="middle" class="head-note">{note}</text>'
        f'<path d="M{x} 152 V{last_y + 26}" class="life"/>'
        for x, title, note in heads
    )
    body = f"""
<text x="40" y="50" class="kicker rise">HOW A DIGITAL-TWIN RUN IS RECORDED</text>
<text x="40" y="76" class="caption rise" style="animation-delay:.1s">SIMIT and the DAQ client hand over control \
through shared-memory bytes; the shuttle signal in the PLC telemetry starts and stops each recording</text>
{head_boxes}
{"".join(groups)}
<g opacity="0">{gate(stream_on + 0.4, stream_off, period)}{stream}</g>
<text x="40" y="{height - 22}" class="foot">Byte offsets and triggers from the DAQ client configuration. \
Whether the fault step comes before or after the start trigger depends on the scenario delay.</text>
"""
    label = "Sequence diagram: SIMIT, shared memory, the DAQ client and the PLC telemetry during one twin recording"
    return frame(1200, height, label, HANDSHAKE_CSS, body)


def main() -> None:
    recordings = load_manifest(ROOT / "data" / "manifest.csv")
    with (ROOT / "results" / "benchmark.csv").open(newline="", encoding="utf-8") as handle:
        bench = list(csv.DictReader(handle))
    accuracy = {
        row["scenario"]: float(row["accuracy"]) for row in bench if row["model"] == "svc" and row["window"] == "3"
    }
    classes = len({rec.fault for rec in recordings})
    train, _ = scenario_data(recordings, "C-C")
    detector = FaultDetector("svc", 3).fit(train)
    session = terminal_session(recordings, detector)
    artwork = {
        "banner.svg": banner(len(recordings), N_FEATURES, classes, len(bench)),
        "pipeline.svg": pipeline(len(recordings)),
        "results_bars.svg": results_chart(accuracy),
        "live_terminal.svg": live_terminal(session, detector),
        "daq_handshake.svg": handshake(),
    }
    for name, svg in artwork.items():
        path = IMAGES / name
        path.write_text(svg, encoding="utf-8", newline="\n")
        print("wrote", path)
    for offset, _, line in session["lines"]:
        print(f"  {session['recording'].path.name} +{offset:.2f} s  {line}")


if __name__ == "__main__":
    main()
