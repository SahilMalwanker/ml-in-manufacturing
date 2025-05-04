"""Figures for evaluations (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .detector import Evaluation
from .schema import fault_name

TEAL = "#0A8276"
TEAL_CMAP = LinearSegmentedColormap.from_list("ml-manufacturing", ["#F4FAF9", "#7CC4BA", TEAL, "#06524A"])


def plot_confusion(evaluation: Evaluation, path: str | Path, title: str = "") -> Path:
    """Confusion matrix with counts and row-normalised shading."""
    matrix = evaluation.confusion()
    labels = evaluation.labels
    row_sums = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    size = 1.4 + 0.62 * len(labels)
    fig, ax = plt.subplots(figsize=(size + 3.2, size))
    image = ax.imshow(shares, cmap=TEAL_CMAP, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                color = "white" if shares[i, j] > 0.55 else "#1F2933"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9, color=color)
    ticks = [f"{code} {fault_name(code)}" for code in labels]
    ax.set_xticks(range(len(labels)), [str(code) for code in labels])
    ax.set_yticks(range(len(labels)), ticks, fontsize=8)
    ax.set_xlabel("Predicted fault code")
    ax.set_ylabel("True fault")
    ax.set_title(title or f"Accuracy {evaluation.accuracy:.1%}  |  macro F1 {evaluation.f1:.2f}", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share of true class")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
