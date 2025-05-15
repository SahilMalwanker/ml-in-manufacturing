"""Regenerate the figures in docs/images and the derived tables in results/.

pip install -e .
ml-manufacturing benchmark --windows 0 3 5 10 20     # results/benchmark.csv (a few minutes)
python scripts/make_figures.py                # figures + per-fault recall and clipping tables
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_manufacturing.daq import elapsed_ms, read_recording
from ml_manufacturing.datasets import Recording, load_manifest, scenario_data, select
from ml_manufacturing.detector import FaultDetector
from ml_manufacturing.plotting import TEAL_CMAP, plot_confusion
from ml_manufacturing.realtime import OnlinePredictor
from ml_manufacturing.replay import iter_payloads
from ml_manufacturing.schema import FEATURE_COLUMNS, SIGNALS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMAGES = ROOT / "docs" / "images"
MANIFEST = ROOT / "data" / "manifest.csv"

PLANT, TWIN, INK, MUTED, GRID = "#0A8276", "#E0791F", "#1F2933", "#7B8794", "#D9E2EC"
CMAP = TEAL_CMAP
WINDOW = 3
CLASSES = (0, 2, 3, 4, 5, 6, 8, 9, 10)
SHORT = {
    0: "Normal",
    2: "Area stop",
    3: "Light curtain",
    4: "Safety mat",
    5: "Door switch",
    6: "Emergency stop",
    8: "Press-arm sensor",
    9: "RFID presence",
    10: "Shuttle sensor",
}
MODELS = {"svc": "SVC", "dt": "Decision tree", "rf": "Random forest", "mlp": "Neural net"}
MODEL_STYLE = {"svc": (PLANT, "o"), "dt": (TWIN, "s"), "rf": ("#3E4C59", "^"), "mlp": ("#B7950B", "D")}
SCENARIO_LABELS = {
    "R-R": "Real → Real",
    "D-D": "Twin → Twin",
    "R-D": "Real → Twin",
    "D-R": "Twin → Real",
    "C-C": "Combined → Combined",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": "#9AA5B1",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "figure.dpi": 100,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", IMAGES / name)


def _heatmap(ax, values: np.ndarray, rows: list[str], cols: list[str], fmt: str, vmin: float, vmax: float) -> None:
    masked = np.ma.masked_invalid(values)
    ax.imshow(masked, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_facecolor("#F0F2F5")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "n/a" if np.isnan(value) else format(value, fmt)
            shade = 0 if np.isnan(value) else (value - vmin) / (vmax - vmin)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if shade > 0.6 else INK,
                fontweight="bold" if shade > 0.9 else "normal",
            )
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def class_label(code: int) -> str:
    return f"{code:>2}  {SHORT[code]}"


def _title(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def _load(recordings: list[Recording], domain: str, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    return [rec.load() for rec in select(recordings, domain, split)]


def dataset_overview(recordings: list[Recording]) -> None:
    counts: dict[tuple[str, str], np.ndarray] = {}
    for rec in recordings:
        _, labels = rec.load()
        tally = counts.setdefault((rec.domain, rec.split), np.zeros(len(CLASSES), dtype=int))
        tally += [int(np.count_nonzero(labels == code)) for code in CLASSES]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    y = np.arange(len(CLASSES))
    panels = (("real_plant", PLANT, "Real plant"), ("digital_twin", TWIN, "Digital twin"))
    for ax, (domain, color, title) in zip(axes, panels, strict=True):
        train, test = counts[(domain, "train")], counts[(domain, "test")]
        top = max(train.max(), test.max())
        ax.barh(y - 0.2, train, 0.38, color=color, label="train runs")
        ax.barh(y + 0.2, test, 0.38, color=color, alpha=0.4, label="test runs")
        for yi, (a, b) in enumerate(zip(train, test, strict=True)):
            ax.text(a + 0.012 * top, yi - 0.2, f"{a:,}", va="center", fontsize=8, color=INK)
            ax.text(b + 0.012 * top, yi + 0.2, f"{b:,}", va="center", fontsize=8, color=INK)
        runs = sum(rec.domain == domain for rec in recordings)
        ax.set_title(f"{title}: {runs} recordings, {int(train.sum() + test.sum()):,} samples", loc="left")
        ax.set_xlim(0, top * 1.16)
        ax.set_xlabel("event-driven samples per class")
        ax.legend(frameon=False, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [class_label(code) for code in CLASSES])
    axes[0].invert_yaxis()
    fig.tight_layout()
    _save(fig, "dataset_overview.png")


def press_cycle() -> None:
    """One normal press cycle as logged from the real plant and from the digital twin."""
    runs = (
        ("Real plant", ROOT / "data/real_plant/train/rp_c00_run1.csv", PLANT),
        ("Digital twin", ROOT / "data/digital_twin/train/dt_c00_run1.csv", TWIN),
    )
    full_scale = 27648.0  # nominal range of S7 analog values
    fig, axes = plt.subplots(2, 2, figsize=(13, 5.6), sharex=True, sharey="col")
    for row, (title, path, color) in enumerate(runs):
        frame = read_recording(path)
        t = elapsed_ms(frame) / 1000.0
        t -= t[np.argmax(frame["Stationsdaten.StationIsWorking"].to_numpy() > 0)]
        position, valves = axes[row]
        target, actual = frame["Stationsdaten.Sollpos_percent"], frame["Stationsdaten.ActPosPercCtrlOutp"]
        position.step(t, target, where="post", color=MUTED, lw=1.4, ls="--", label="target position")
        position.step(t, actual, where="post", color=color, lw=2.2, label="actual position")
        position.plot(t, np.full(len(t), -14.0), "|", color=color, ms=8, alpha=0.8, label="logged sample")
        for valve, shade in (("M1", color), ("M2", INK)):
            valves.step(
                t, frame[f"{valve} Ventil Soll-Druck"] / full_scale * 100, where="post", color=shade, lw=1.2, ls="--"
            )
            valves.step(
                t,
                frame[f"{valve} Ventil Ist-Druck"] / full_scale * 100,
                where="post",
                color=shade,
                lw=2.2,
                label=f"{valve} actual",
            )
        valves.legend(loc="upper right", fontsize=8.5, framealpha=1.0)
        position.set_ylabel(f"{title}\nposition [%]", color=INK)
        valves.set_ylabel("pressure [% of range]")
        for ax in (position, valves):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
        position.set_ylim(-20, 108)
        position.text(
            0.99, 0.06, f"{len(t)} samples", transform=position.transAxes, ha="right", fontsize=8.5, color=color
        )
    axes[0, 0].set_title("Press arm", loc="left")
    axes[0, 1].set_title("Proportional valves M1 / M2 (dashed: target, solid: actual)", loc="left")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8.5, ncols=3)
    for ax in axes[1]:
        ax.set_xlabel("time since the press cycle started [s]")
    axes[0, 0].set_xlim(-1.0, 5.8)
    _title(fig, "Same PLC program, two worlds: a normal press cycle in the real plant and in the digital twin")
    fig.tight_layout()
    _save(fig, "press_cycle.png")


def results_overview(bench: pd.DataFrame) -> None:
    table = bench[bench.window == WINDOW]
    columns = [label.replace(" → ", "\n→ ") for label in SCENARIO_LABELS.values()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    for ax, metric, title, scale, fmt in (
        (axes[0], "accuracy", "Accuracy [%]", 100.0, ".1f"),
        (axes[1], "f1", "Macro F1 (every fault class weighs the same)", 1.0, ".2f"),
    ):
        grid = np.array(
            [
                [table[(table.model == model) & (table.scenario == code)][metric].iloc[0] for code in SCENARIO_LABELS]
                for model in MODELS
            ]
        )
        _heatmap(ax, grid * scale, list(MODELS.values()), columns, fmt, 0.0, scale)
        ax.set_title(title, loc="left")
    _title(fig, f"Test results on unseen recordings, window size {WINDOW} (training data → test data)")
    fig.tight_layout()
    _save(fig, "results_overview.png")


def reported_vs_reproduced(bench: pd.DataFrame) -> None:
    reported = pd.read_csv(RESULTS / "reported_benchmark.csv")
    domains = {"real_plant": "R", "digital_twin": "D", "combined": "C"}
    reported["scenario"] = reported.train_data.map(domains) + "-" + reported.test_data.map(domains)
    ours = bench[bench.window == WINDOW]
    x = np.arange(len(SCENARIO_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
    for ax, (name, model) in zip(axes, (("SVC", "svc"), ("DT", "dt"), ("RF", "rf")), strict=True):
        old = [
            reported[(reported.model == name) & (reported.scenario == s)].accuracy_pct.iloc[0] for s in SCENARIO_LABELS
        ]
        new = [100 * ours[(ours.model == model) & (ours.scenario == s)].accuracy.iloc[0] for s in SCENARIO_LABELS]
        ax.bar(x - 0.19, old, 0.36, color="#CBD2D9", label="project report")
        ax.bar(x + 0.19, new, 0.36, color=PLANT, label="reproduced here")
        for xi, (a, b) in enumerate(zip(old, new, strict=True)):
            ax.text(xi - 0.19, a + 1.5, f"{a:.0f}", ha="center", fontsize=8, color=MUTED)
            ax.text(xi + 0.19, b + 1.5, f"{b:.0f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, list(SCENARIO_LABELS), fontsize=9)
        ax.set_title(MODELS[model], loc="left")
        ax.set_ylim(0, 112)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test accuracy [%]")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    _title(fig, "Window size 3: accuracies from the project report vs. this re-implementation")
    fig.tight_layout()
    _save(fig, "reported_vs_reproduced.png")


def window_size(bench: pd.DataFrame) -> None:
    windows = sorted(bench.window.unique())
    fig, axes = plt.subplots(1, len(SCENARIO_LABELS), figsize=(15, 3.4), sharey=True)
    for ax, (code, label) in zip(axes, SCENARIO_LABELS.items(), strict=True):
        part = bench[bench.scenario == code]
        for model, name in MODELS.items():
            color, marker = MODEL_STYLE[model]
            rows = part[part.model == model].sort_values("window")
            complete = (rows.test_classes == len(CLASSES)).to_numpy()
            ax.plot(rows.window, rows.f1, color=color, lw=1.6, label=name)
            ax.scatter(rows.window[complete], rows.f1[complete], marker=marker, color=color, s=28, zorder=3)
            ax.scatter(rows.window[~complete], rows.f1[~complete], marker=marker, s=28, zorder=3, fc="white", ec=color)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xticks(windows)
        ax.set_xlabel("window size")
        ax.set_ylim(0, 1.04)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("macro F1")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.text(
        0.01,
        -0.04,
        "Window 0 = single samples. Hollow markers: the twin test runs of faults 8 and 9 have only 4 samples, "
        "so windows of 5 and more leave 7 of 9 classes to evaluate.",
        fontsize=8.5,
        color=MUTED,
    )
    _title(fig, "Effect of the window size on unseen test recordings")
    fig.tight_layout()
    _save(fig, "window_size.png")


def per_fault_recall(recordings: list[Recording]) -> pd.DataFrame:
    """Recall of every fault on the real-plant test runs for plant-, twin- and combined-trained models."""
    test = _load(recordings, "real_plant", "test")
    rows = []
    for train_domain in ("real_plant", "digital_twin", "combined"):
        train = _load(recordings, train_domain, "train")
        for model in MODELS:
            result = FaultDetector(model, WINDOW).fit(train).evaluate(test)
            for code in CLASSES:
                hits = result.y_pred[result.y_true == code] == code
                rows.append(
                    {
                        "train": train_domain,
                        "test": "real_plant",
                        "model": model,
                        "window": WINDOW,
                        "fault": code,
                        "support": len(hits),
                        "recall": round(float(hits.mean()), 4),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "per_fault_recall.csv", index=False)
    print("wrote", RESULTS / "per_fault_recall.csv")
    return table


def transfer_per_fault(table: pd.DataFrame) -> None:
    panels = (
        ("real_plant", "trained on the real plant"),
        ("digital_twin", "trained on the digital twin only"),
        ("combined", "trained on both domains"),
    )
    columns = [name.replace(" ", "\n") for name in MODELS.values()]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (domain, title) in zip(axes, panels, strict=True):
        part = table[table.train == domain]
        grid = np.array(
            [
                [part[(part.model == model) & (part.fault == code)].recall.iloc[0] for model in MODELS]
                for code in CLASSES
            ]
        )
        _heatmap(ax, grid * 100, [class_label(code) for code in CLASSES], columns, ".0f", 0, 100)
        ax.set_title(title, loc="left")
    _title(fig, f"Recall per fault on unseen real-plant runs [%], window size {WINDOW}")
    fig.tight_layout()
    _save(fig, "transfer_per_fault.png")


def confusion_figures(recordings: list[Recording]) -> None:
    for scenario, model, name, title in (
        ("C-C", "svc", "confusion_combined_svc.png", "SVC, trained and tested on both domains"),
        ("D-R", "dt", "confusion_twin_to_real_dt.png", "Decision tree, trained on the twin, tested on the plant"),
    ):
        train, test = scenario_data(recordings, scenario)
        result = FaultDetector(model, WINDOW).fit(train).evaluate(test)
        caption = f"{title}\naccuracy {result.accuracy:.1%}  |  macro F1 {result.f1:.2f}  |  window size {WINDOW}"
        print("wrote", plot_confusion(result, IMAGES / name, caption))


def clipping_ablation(recordings: list[Recording]) -> None:
    rows = []
    for scenario in ("D-R", "R-D"):
        train, test = scenario_data(recordings, scenario)
        for model in MODELS:
            for clip in (False, True):
                result = FaultDetector(model, WINDOW, clip=clip).fit(train).evaluate(test)
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "window": WINDOW,
                        "clip": clip,
                        "accuracy": round(result.accuracy, 4),
                        "f1": round(result.f1, 4),
                    }
                )
    pd.DataFrame(rows).to_csv(RESULTS / "clipping_ablation.csv", index=False)
    print("wrote", RESULTS / "clipping_ablation.csv")


GAP_NAMES = {
    "ShuttleInitFlagsByteVonSt10": "Shuttle init flags (byte)",
    "Stationsdaten.Station in Grundstellung": "Station in home position",
    "S13 Auto (rechts)": "Mode switch S13 in automatic",
    "VentilRueckfuehrungKeineDruckl": "Safety-valve feedback",
    "VentilAnsteuerung": "Safety-valve control",
    "SchluesselFreigabe": "Key switch released",
    "B11 Pressenarm untere Endlage": "B11 lower end switch",
    "B12 Pressenarm obere Endlage": "B12 upper end switch",
    "M1 Ventil Ist-Druck": "M1 valve actual pressure",
    "M2 Ventil Ist-Druck": "M2 valve actual pressure",
    "RFIDInterface.RFID_errorRead": "RFID read error",
    "RFIDInterface.RFID_busyReset": "RFID reset busy",
    "Field_Device_Diagnostic_DB.State_PROFI_60": "PROFINET state St60",
}


def domain_gap(recordings: list[Recording]) -> None:
    """Signals that behave differently during normal operation, and fault samples logged per run."""
    normal: dict[str, list[np.ndarray]] = {"real_plant": [], "digital_twin": []}
    for rec in recordings:
        features, labels = rec.load()
        normal[rec.domain].append(features[labels == 0])
    plant, twin = np.vstack(normal["real_plant"]), np.vstack(normal["digital_twin"])
    gap = pd.DataFrame(
        [
            {
                "signal": signal.name,
                "type": signal.dtype,
                "plant_distinct": len(np.unique(plant[:, i])),
                "twin_distinct": len(np.unique(twin[:, i])),
                "plant_min": plant[:, i].min(),
                "plant_max": plant[:, i].max(),
                "twin_min": twin[:, i].min(),
                "twin_max": twin[:, i].max(),
                "plant_mean": round(float(plant[:, i].mean()), 4),
                "twin_mean": round(float(twin[:, i].mean()), 4),
            }
            for i, signal in enumerate(SIGNALS)
        ]
    )
    gap.to_csv(RESULTS / "domain_gap_signals.csv", index=False)
    print("wrote", RESULTS / "domain_gap_signals.csv")
    plant_constant, twin_constant = gap.plant_distinct == 1, gap.twin_distinct == 1
    differs = gap[
        (plant_constant != twin_constant) | (plant_constant & twin_constant & (gap.plant_min != gap.twin_min))
    ]
    differs = differs.sort_values(["plant_distinct", "twin_distinct"])

    def value_range(low: float, high: float) -> str:
        return f"{low:g}" if low == high else f"{low:g}–{high:g}"

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(differs))
    left.barh(y - 0.2, differs.plant_distinct, 0.38, color=PLANT, label="real plant")
    left.barh(y + 0.2, differs.twin_distinct, 0.38, color=TWIN, label="digital twin")
    for yi, (_, row) in enumerate(differs.iterrows()):
        note = f"plant {value_range(row.plant_min, row.plant_max)}  ·  twin {value_range(row.twin_min, row.twin_max)}"
        left.text(max(row.plant_distinct, row.twin_distinct) * 1.3, yi, note, va="center", fontsize=8, color=INK)
    left.set_xscale("log")
    left.set_xlim(0.8, 30000)
    left.set_yticks(y, [GAP_NAMES.get(s, s) for s in differs.signal], fontsize=9)
    left.invert_yaxis()
    left.set_xlabel("distinct values in normal operation (log scale); text: value range")
    left.set_title("Signals that behave differently in the twin", loc="left")
    left.legend(frameon=False, loc="upper right")
    left.spines[["top", "right"]].set_visible(False)

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        runs = pd.DataFrame(csv.DictReader(handle)).astype({"fault": int, "fault_samples": int})
    faults = [code for code in CLASSES if code]
    x = np.arange(len(faults))
    for domain, color, shift, label in (
        ("real_plant", PLANT, -0.19, "real plant"),
        ("digital_twin", TWIN, 0.19, "twin"),
    ):
        stats = [runs[(runs.domain == domain) & (runs.fault == code)].fault_samples for code in faults]
        median = np.array([s.median() for s in stats])
        low, high = np.array([s.min() for s in stats]), np.array([s.max() for s in stats])
        right.bar(x + shift, median, 0.36, color=color, label=f"{label} (median)")
        right.errorbar(x + shift, median, yerr=[median - low, high - median], fmt="none", ecolor=INK, capsize=2, lw=0.8)
    right.set_yscale("log")
    right.set_xticks(x, [SHORT[code] for code in faults], rotation=35, ha="right", fontsize=9)
    right.set_ylabel("fault samples per run (min–max bars)")
    right.set_title("Event-driven samples logged while a fault is active", loc="left")
    right.legend(frameon=False, loc="upper left", fontsize=8.5)
    right.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.56,
        -0.035,
        "Few twin events remain after a fault stops signal changes.",
        fontsize=8,
        color=TWIN,
    )
    fig.tight_layout()
    _save(fig, "domain_gap.png")


TREE_QUESTIONS = {
    "SchaltmatteGedrueckt": "safety mat pressed",
    "Bereichshalt Nicht": "area stop not pressed",
    "SchalterTuerZu": "safety door closed",
    "LichtvorhangFrei": "light curtain clear",
    "KeinNotHaltExtern von St10": "no emergency stop from station 10",
    "St10_in_AUTO": "station 10 in automatic mode",
}


def decision_tree(recordings: list[Recording]) -> None:
    """The first splits of a decision tree trained on both domains (single samples)."""
    detector = FaultDetector("dt", window=0).fit(_load(recordings, "combined", "train"))
    result = detector.evaluate(_load(recordings, "combined", "test"))
    recall = {code: float(np.mean(result.y_pred[result.y_true == code] == code)) for code in CLASSES}
    model, tree = detector.model, detector.model.tree_
    low, high = detector.scaler.data_min_, detector.scaler.data_max_

    def is_leaf(node: int) -> bool:
        return tree.children_left[node] == -1

    def predicted(node: int) -> int:
        return int(model.classes_[np.argmax(tree.value[node][0])])

    # follow the cascade as long as every split peels off exactly one leaf
    steps, node = [], 0
    while not is_leaf(node):
        left, right = tree.children_left[node], tree.children_right[node]
        if is_leaf(left) == is_leaf(right):
            break
        feature = int(tree.feature[node])
        threshold = low[feature] + tree.threshold[node] * (high[feature] - low[feature])
        leaf_on_left = is_leaf(left)
        leaf = left if leaf_on_left else right
        steps.append(
            (FEATURE_COLUMNS[feature], threshold, leaf_on_left, predicted(leaf), int(tree.n_node_samples[leaf]))
        )
        node = right if leaf_on_left else left

    depth = np.zeros(tree.node_count, dtype=int)
    for parent in range(tree.node_count):  # children always have larger ids than their parent
        if not is_leaf(parent):
            depth[tree.children_left[parent]] = depth[tree.children_right[parent]] = depth[parent] + 1
    rest, stack = [], [node]
    while stack:
        current = stack.pop()
        rest.append(current)
        if not is_leaf(current):
            stack += [tree.children_left[current], tree.children_right[current]]
    rest_classes = sorted({predicted(n) for n in rest if is_leaf(n)})
    rest_signals = pd.Series([FEATURE_COLUMNS[int(tree.feature[n])] for n in rest if not is_leaf(n)]).value_counts()
    booleans = {signal.name for signal in SIGNALS if signal.dtype == "Bool"}

    def condition(signal: str, threshold: float, low_side: bool) -> str:
        if signal in booleans:
            return "= 0" if low_side else "= 1"
        return f"≤ {threshold:g}" if low_side else f"> {threshold:g}"

    n = len(steps)
    fig, ax = plt.subplots(figsize=(12, 1.3 * (n + 1) + 0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-n - 0.8, 0.7)
    ax.axis("off")
    node_box = {"boxstyle": "round,pad=0.6", "fc": "white", "ec": MUTED, "lw": 1.2}
    leaf_box = {"boxstyle": "round,pad=0.6", "fc": "#E6F4F1", "ec": PLANT, "lw": 1.2}
    rest_box = {"boxstyle": "round,pad=0.6", "fc": "#FDF0E6", "ec": TWIN, "lw": 1.2}
    arrow = {"arrowstyle": "-|>", "color": MUTED, "lw": 1.2}
    for i, (signal, threshold, leaf_on_left, code, samples) in enumerate(steps):
        y = -i
        question = TREE_QUESTIONS.get(signal, "process signal")
        ax.text(0.2, y, f"{signal}\n({question})", ha="center", va="center", fontsize=9.5, color=INK, bbox=node_box)
        ax.annotate("", xy=(0.56, y), xytext=(0.36, y), arrowprops=arrow)
        ax.text(0.46, y + 0.12, condition(signal, threshold, leaf_on_left), ha="center", fontsize=9.5, color=MUTED)
        leaf_text = f"{code}  {SHORT[code]}\n{samples:,} training samples  ·  test recall {recall[code]:.0%}"
        ax.text(0.76, y, leaf_text, ha="center", va="center", fontsize=9.5, color=INK, bbox=leaf_box)
        ax.annotate("", xy=(0.2, y - 0.68), xytext=(0.2, y - 0.32), arrowprops=arrow)
        ax.text(0.215, y - 0.5, condition(signal, threshold, not leaf_on_left), va="center", fontsize=9.5, color=MUTED)
    y = -n
    levels = int(depth[rest].max() - depth[node]) + 1
    ax.text(
        0.2,
        y,
        f"remaining subtree\n{levels} more levels, {len(rest)} nodes",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
        bbox=rest_box,
    )
    ax.annotate("", xy=(0.56, y), xytext=(0.36, y), arrowprops=arrow)
    scores = [f"{SHORT[code]} {recall[code]:.0%}" for code in rest_classes]
    summary = "  ·  ".join(scores[:3]) + "\n" + "  ·  ".join(scores[3:]) + "   (test recall)"
    summary += "\nsplits mostly on: " + ", ".join(rest_signals.index[:3])
    ax.text(0.76, y, summary, ha="center", va="center", fontsize=9, color=INK, bbox=rest_box)
    _title(fig, "A decision tree trained on both domains learns the safety interlocks first")
    fig.tight_layout()
    _save(fig, "decision_tree.png")


def realtime_timeline(recordings: list[Recording], detector: FaultDetector) -> None:
    """Unseen real-plant runs streamed payload by payload through the online detector."""
    names = ["rp_c03_run3.csv", "rp_c06_run3.csv", "rp_c08_run4.csv", "rp_c10_run3.csv"]
    by_name = {rec.path.name: rec for rec in recordings}
    predictor = OnlinePredictor(detector, dedupe=False)
    truth, predicted, latency, boundaries = [], [], [], []
    for name in names:
        rec = by_name[name]
        _, labels = rec.load()
        for index, (_, payload) in enumerate(iter_payloads(rec.path)):
            prediction = predictor.update(payload)
            if prediction is not None:
                truth.append(int(labels[index]))
                predicted.append(prediction.code)
                latency.append(prediction.latency_ms)
        boundaries.append(len(truth))

    codes = sorted(set(truth) | set(predicted))
    level = {code: i for i, code in enumerate(codes)}
    fig, ax = plt.subplots(figsize=(13, 3.6))
    x = np.arange(len(truth))
    ax.step(x, [level[c] for c in truth], where="post", color=INK, linewidth=2.2, label="true state")
    ax.scatter(x, [level[c] for c in predicted], s=7, color=PLANT, zorder=3, label="online prediction")
    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color="#9AA5B1", linestyle="--", linewidth=1)
    start = 0
    for name, end in zip(names, boundaries, strict=True):
        ax.text((start + end) / 2, len(codes) - 0.35, name.removesuffix(".csv"), ha="center", fontsize=9, color=INK)
        start = end
    ax.set_yticks(range(len(codes)), [class_label(code) for code in codes], fontsize=9)
    ax.set_ylim(-0.5, len(codes) - 0.1)
    ax.set_xlim(0, len(truth))
    ax.set_xlabel("input sample (recorded MQTT payload)")
    agreement = np.mean(np.array(truth) == np.array(predicted))
    ax.set_title(
        "Offline stream replay of four held-out plant runs (SVC, combined training, window 3)\n"
        f"agreement {agreement:.1%}  |  median local processing {np.median(latency):.1f} ms/payload; "
        "no broker transport",
        loc="left",
        fontsize=10,
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "realtime_timeline.png")


def main() -> None:
    recordings = load_manifest(MANIFEST)
    bench = pd.read_csv(RESULTS / "benchmark.csv")
    dataset_overview(recordings)
    press_cycle()
    domain_gap(recordings)
    results_overview(bench)
    reported_vs_reproduced(bench)
    window_size(bench)
    transfer_per_fault(per_fault_recall(recordings))
    confusion_figures(recordings)
    clipping_ablation(recordings)
    decision_tree(recordings)
    train, _ = scenario_data(recordings, "C-C")
    realtime_timeline(recordings, FaultDetector("svc", WINDOW).fit(train))


if __name__ == "__main__":
    main()
