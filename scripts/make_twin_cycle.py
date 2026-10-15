"""Animated GIF of the digital twin running one press cycle, next to the PLC telemetry of that run.

pip install -e . imageio-ffmpeg     (or have ffmpeg on PATH)
python scripts/make_twin_cycle.py --video "<NX MCD screen recording>.mp4"

The source is a 1920 x 1040 screen recording of the NX MCD digital twin made during the project;
it is not redistributed. Only the 3D viewport is used: window chrome, file names and tool panels
are cropped away, and a stray mouse pointer is replaced with the same pixels from a pointer-free
frame. The right-hand panel replays data/raw_examples/dt_normal_cycle.txt, the DAQ recording of
this very run: its first sample (shuttle-in-station start trigger) is 4.74 s into the screen
recording, and the press-arm motion in the video and the recorded position signal agree within
about 0.1 s.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import warnings
from functools import cache
from itertools import pairwise
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml_manufacturing.daq import elapsed_ms, read_recording

ROOT = Path(__file__).resolve().parents[1]
RECORDING = ROOT / "data" / "raw_examples" / "dt_normal_cycle.txt"
OUT = ROOT / "docs" / "images" / "twin_cycle.gif"

CROP = (330, 190, 1070, 830)  # x0, y0, x1, y1 of the 3D viewport area in the 1920 x 1040 recording
MOTION = (380, 215, 1010, 600)  # where the model moves; never touched by the pointer clean-up
POINTER_UNTIL = 5.0  # the mouse pointer rests in the viewport until here (s)
CLEAN_AT = 13.2  # pointer-free reference frame (s)
LOOP = (3.3, 13.3)  # the shuttle waits at the entry in both frames, so the clip loops seamlessly
DATA_START = 4.74  # recording time of the first DAQ sample (s)
FPS = 12.5
WIDTH, HEIGHT, S = 1000, 500, 2  # output size and supersampling factor for text and traces

VIDEO_BOX = (16, 20, 548, 480)
PANEL_X0, PANEL_X1 = 572, 984
TEAL, ORANGE, WHITE = (45, 212, 191), (251, 146, 60), (248, 250, 252)
TEXT, MUTED, DIM = (226, 232, 240), (148, 163, 184), (100, 116, 139)
DIGITAL = (
    ("Shuttle in station", "Shuttle_in_Station_von_St10", TEAL),
    ("Shuttle locked", "LockShuttle_an_St10", TEAL),
    ("Press down command", "Stationsdaten.Anfahrt_Unten", ORANGE),
    ("Press up command", "Stationsdaten.Anfahrt_Oben", ORANGE),
)
RELEASE = ("Release shuttle", "StartShuttle_an_St10", TEAL)
TARGET, ACTUAL = "Stationsdaten.Sollpos_percent", "Stationsdaten.ActPosPercCtrlOutp"


def ffmpeg_executable() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit("ffmpeg not found: put it on PATH or `pip install imageio-ffmpeg`") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(ffmpeg: str, video: Path, start: float, duration: float | None, fps: float | None) -> np.ndarray:
    """Cropped RGB frames of ``video`` from ``start`` (one frame when ``duration`` is None)."""
    x0, y0, x1, y1 = CROP
    filters = ([f"fps={fps}"] if fps else []) + [f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"]
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{start}", "-i", str(video)]
    command += ["-t", f"{duration}"] if duration else ["-frames:v", "1"]
    command += ["-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, y1 - y0, x1 - x0, 3)


def remove_pointer(frame: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Replace pixels that differ from the pointer-free frame outside the moving model."""
    changed = np.abs(frame.astype(np.int16) - clean.astype(np.int16)).max(axis=2) > 24
    x0, y0, x1, y1 = (v - o for v, o in zip(MOTION, (*CROP[:2], *CROP[:2]), strict=True))
    changed[y0:y1, x0:x1] = False
    grown = Image.fromarray(changed.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))
    mask = np.asarray(grown) > 0
    out = frame.copy()
    out[mask] = clean[mask]
    return out


@cache
def font(size: float, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    families = ["Cascadia Mono", "Consolas", "DejaVu Sans Mono"] if mono else ["Segoe UI", "DejaVu Sans"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        properties = font_manager.FontProperties(family=families, weight="bold" if bold else "normal")
        path = font_manager.findfont(properties, fallback_to_default=True)
    return ImageFont.truetype(path, round(size * S))


def background() -> Image.Image:
    """Dark gradient card with a faint grid, matching the SVG artwork."""
    w, h = WIDTH * S, HEIGHT * S
    ramp = (np.linspace(0, 1, w)[None, :] + np.linspace(0, 1, h)[:, None]) / 2
    start, end = np.array([11, 18, 32]), np.array([15, 46, 53])
    pixels = start + (end - start) * ramp[..., None]
    grid = np.zeros((h, w), bool)
    grid[:, :: 32 * S] = True
    grid[:: 32 * S, :] = True
    pixels[grid] = pixels[grid] * 0.955 + 255 * 0.045
    return Image.fromarray(pixels.astype(np.uint8)).convert("RGBA")


class Telemetry:
    """The DAQ recording of the filmed run, on the video's time axis."""

    def __init__(self) -> None:
        frame = read_recording(RECORDING)
        self.t = elapsed_ms(frame) / 1000.0 + DATA_START
        self.frame = frame
        self.end = float(self.t[-1])

        def first(mask: np.ndarray) -> float:
            return float(self.t[int(np.argmax(mask))])

        lock, down = self.signal(DIGITAL[1][1]), self.signal(DIGITAL[2][1])
        actual = self.signal(ACTUAL)
        self.stroke = float(actual.max())
        up = first(self.signal(DIGITAL[3][1]) == 1)
        self.phases = [
            (LOOP[0], "Shuttle arrives from station 10"),
            (DATA_START, "Shuttle in station: recording starts"),
            (first(lock == 1), "Shuttle locked"),
            (first(down == 1), "Press arm moves down"),
            (first(actual >= self.stroke - 0.5), f"Press arm holds at {self.stroke:.0f} % stroke"),
            (up, "Press arm returns"),
            (first((self.t > up) & (actual <= 0.5)), "Press arm back at rest"),
            (first(self.signal(RELEASE[1]) == 1), "Shuttle released"),
            (self.end, "Shuttle leaves: recording stops"),
        ]

    def signal(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy(dtype=float)

    def phase(self, now: float) -> str:
        return [text for start, text in self.phases if start <= now + 1e-6][-1]


def x_of(now: float) -> float:
    return PANEL_X0 + (now - LOOP[0]) / (LOOP[1] - LOOP[0]) * (PANEL_X1 - PANEL_X0)


def step_points(t: np.ndarray, v: np.ndarray, now: float, y_low: float, y_high: float, top: float) -> list:
    """Step-plot vertices up to ``now`` (values scaled from 0..top to y_low..y_high)."""
    points = []
    for i in range(len(t)):
        if t[i] > now:
            break
        y = y_low + (y_high - y_low) * v[i] / top
        if points:
            points.append((x_of(t[i]), points[-1][1]))
        points.append((x_of(t[i]), y))
    if points:
        points.append((x_of(min(now, t[-1])), points[-1][1]))
    return [(x * S, y * S) for x, y in points]


def static_layer() -> Image.Image:
    image = background()
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.text((PANEL_X0 * S, 24 * S), "PLC TELEMETRY OF THIS RUN", font=font(12, bold=True), fill=TEAL)
    draw.text((PANEL_X0 * S, 42 * S), "Normal press cycle in the digital twin", font=font(17, bold=True), fill=WHITE)
    draw.text((PANEL_X0 * S, 66 * S), "data/raw_examples/dt_normal_cycle.txt", font=font(10.5, mono=True), fill=MUTED)
    x0, y0, x1, y1 = VIDEO_BOX
    draw.rounded_rectangle([x0 * S - 2, y0 * S - 2, x1 * S + 1, y1 * S + 1], radius=14 * S, outline=(255, 255, 255, 40))
    legend_y = (lane_tops()[4] + 8) * S
    draw.line([(PANEL_X1 - 150) * S, legend_y, (PANEL_X1 - 132) * S, legend_y], fill=TEAL, width=2 * S)
    draw.text(((PANEL_X1 - 126) * S, legend_y - 8 * S), "actual", font=font(11), fill=MUTED)
    for k in range(3):
        start = (PANEL_X1 - 78 + 7 * k) * S
        draw.line([start, legend_y, start + 4 * S, legend_y], fill=TEXT, width=S)
    draw.text(((PANEL_X1 - 52) * S, legend_y - 8 * S), "target", font=font(11), fill=MUTED)
    image.alpha_composite(overlay)
    return image


def lane_tops() -> list[float]:
    return [92, 132, 172, 212, 252, 352]


def analog_box() -> tuple[float, float, float, float]:
    top = lane_tops()[4]
    return PANEL_X0, top + 18, PANEL_X1, top + 88


def render(index: int, now: float, video: np.ndarray, telemetry: Telemetry, base: Image.Image) -> Image.Image:
    image = base.copy()
    x0, y0, x1, y1 = VIDEO_BOX
    shot = Image.fromarray(video).resize(((x1 - x0) * S, (y1 - y0) * S), Image.LANCZOS)
    mask = Image.new("L", shot.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, shot.size[0] - 1, shot.size[1] - 1], radius=12 * S, fill=255)
    image.paste(shot, (x0 * S, y0 * S), mask)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    small = font(11, bold=True)
    tag = "NX MCD DIGITAL TWIN · REAL TIME"
    tag_width = small.getlength(tag) / S
    draw.rounded_rectangle(
        [(x0 + 12) * S, (y0 + 12) * S, (x0 + 50 + tag_width) * S, (y0 + 36) * S], radius=12 * S, fill=(11, 18, 32, 200)
    )
    draw.ellipse([(x0 + 22) * S, (y0 + 20) * S, (x0 + 30) * S, (y0 + 28) * S], fill=ORANGE)
    draw.text(((x0 + 38) * S, (y0 + 16) * S), tag, font=small, fill=WHITE)
    band = [x0 * S, (y1 - 46) * S, x1 * S, y1 * S]
    draw.rounded_rectangle(band, radius=12 * S, fill=(11, 18, 32, 190), corners=(False, False, True, True))
    recording = DATA_START <= now < telemetry.end
    draw.ellipse([(x0 + 16) * S, (y1 - 28) * S, (x0 + 26) * S, (y1 - 18) * S], fill=TEAL if recording else MUTED)
    draw.text(((x0 + 36) * S, (y1 - 35) * S), telemetry.phase(now), font=font(15, bold=True), fill=WHITE)

    # recording window and trigger markers
    lanes_top, lanes_bottom = lane_tops()[0] - 4, lane_tops()[-1] + 40
    if now >= DATA_START:
        right = x_of(min(now, telemetry.end))
        draw.rectangle([x_of(DATA_START) * S, lanes_top * S, right * S, lanes_bottom * S], fill=(45, 212, 191, 18))
        for edge in [DATA_START] + ([telemetry.end] if now >= telemetry.end else []):
            for y in range(int(lanes_top), int(lanes_bottom), 6):
                draw.line([x_of(edge) * S, y * S, x_of(edge) * S, (y + 3) * S], fill=(45, 212, 191, 150), width=S)
    pill_text, pill_color = (
        ("REC", ORANGE) if recording else (("SAVED", TEAL) if now >= telemetry.end else ("IDLE", DIM))
    )
    draw.rounded_rectangle(
        [(PANEL_X1 - 70) * S, 24 * S, PANEL_X1 * S, 44 * S], radius=10 * S, outline=pill_color, width=S
    )
    if not recording or index % 10 < 6:
        draw.ellipse([(PANEL_X1 - 60) * S, 30 * S, (PANEL_X1 - 52) * S, 38 * S], fill=pill_color)
    draw.text(((PANEL_X1 - 46) * S, 27 * S), pill_text, font=small, fill=pill_color)

    t = telemetry.t
    for top, (_, name, color) in zip(lane_tops()[:4] + lane_tops()[5:], [*DIGITAL, RELEASE], strict=True):
        low, high = top + 34, top + 20
        draw.line([PANEL_X0 * S, low * S, PANEL_X1 * S, low * S], fill=(255, 255, 255, 22), width=S)
        points = step_points(t, telemetry.signal(name), now, low, high, 1.0)
        if len(points) > 1:
            draw.line(points, fill=color, width=2 * S, joint="curve")
    ax0, ay0, ax1, ay1 = analog_box()
    for level in (0, 50, 100):
        y = ay1 - (ay1 - ay0) * level / 100
        draw.line([ax0 * S, y * S, ax1 * S, y * S], fill=(255, 255, 255, 18), width=S)
    target = step_points(t, telemetry.signal(TARGET), now, ay1, ay0, 100.0)
    for (xa, ya), (xb, yb) in pairwise(target):
        length = max(abs(xb - xa), abs(yb - ya))
        for k in range(0, int(length), 10 * S):
            f0, f1 = k / max(length, 1), min(1.0, (k + 5 * S) / max(length, 1))
            draw.line(
                [xa + (xb - xa) * f0, ya + (yb - ya) * f0, xa + (xb - xa) * f1, ya + (yb - ya) * f1], fill=TEXT, width=S
            )
    actual = [
        (x_of(ti) * S, (ay1 - (ay1 - ay0) * v / 100) * S)
        for ti, v in zip(t, telemetry.signal(ACTUAL), strict=True)
        if ti <= now
    ]
    if len(actual) > 1:
        draw.line(actual, fill=TEAL, width=2 * S, joint="curve")

    # one tick per recorded message: event-driven sampling is irregular in time
    axis = lane_tops()[-1] + 46
    draw.line([PANEL_X0 * S, axis * S, PANEL_X1 * S, axis * S], fill=(255, 255, 255, 40), width=S)
    shown = t[t <= now]
    for ti in shown:
        draw.line([x_of(ti) * S, (axis - 5) * S, x_of(ti) * S, axis * S], fill=(45, 212, 191, 200), width=S)
    count = f"{len(shown)} of {len(t)} messages recorded" if now >= DATA_START else "waiting for the start trigger"
    draw.text((PANEL_X0 * S, (axis + 8) * S), count, font=font(12), fill=TEXT)
    clock = f"t = {max(0.0, min(now, telemetry.end) - DATA_START):.2f} s"
    draw.text(((PANEL_X1 - 80) * S, (axis + 8) * S), clock, font=font(12, mono=True), fill=MUTED)
    draw.text((PANEL_X0 * S, (axis + 27) * S), "each tick is one stored message", font=font(11), fill=DIM)

    cursor = x_of(now)
    draw.line([cursor * S, (lanes_top - 2) * S, cursor * S, (axis + 2) * S], fill=(248, 250, 252, 150), width=S)
    label_font = font(12)
    for top, label in zip(
        lane_tops(), [lane[0] for lane in DIGITAL] + ["Press arm position [%]", RELEASE[0]], strict=True
    ):
        width = label_font.getlength(label) / S
        draw.rectangle(
            [PANEL_X0 * S, (top + 1) * S, (PANEL_X0 + width + 6) * S, (top + 17) * S], fill=(13, 30, 41, 235)
        )
        draw.text((PANEL_X0 * S, top * S), label, font=label_font, fill=MUTED)
    image.alpha_composite(overlay)
    return image.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True, help="NX MCD screen recording (1920 x 1040)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    ffmpeg = ffmpeg_executable()
    count = round((LOOP[1] - LOOP[0]) * FPS)
    frames = decode(ffmpeg, args.video, LOOP[0], LOOP[1] - LOOP[0], FPS)[:count]
    clean = decode(ffmpeg, args.video, CLEAN_AT, None, None)[0]
    telemetry = Telemetry()
    base = static_layer()
    with tempfile.TemporaryDirectory() as scratch:
        for index, video in enumerate(frames):
            now = LOOP[0] + index / FPS
            if now < POINTER_UNTIL:
                video = remove_pointer(video, clean)
            render(index, now, video, telemetry, base).save(Path(scratch) / f"{index:04d}.png")
        palette = (
            "split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        )
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{FPS}",
             "-i", str(Path(scratch) / "%04d.png"), "-vf", palette, "-loop", "0", str(args.out)],
            check=True,
        )  # fmt: skip
    print(f"{len(frames)} frames -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
