"""Small, deterministic report-quality plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heatmap(matrix: pd.DataFrame, title: str, path: Path, annotate: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(matrix) * .55), max(7, len(matrix) * .5)))
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    if annotate and len(matrix) <= 12:
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def line_plot(x, series: dict, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in series.items(): ax.plot(x, values, marker="o", label=label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); fig.savefig(path.with_suffix(".svg"), bbox_inches="tight"); plt.close(fig)


def bar_plot(frame: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    frame.plot.bar(x=x, y=y, ax=ax, legend=False, color="#2f6f9f")
    ax.set(title=title, ylabel=ylabel); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); fig.savefig(path.with_suffix(".svg"), bbox_inches="tight"); plt.close(fig)


def metric_dashboard(frame: pd.DataFrame, title: str, path: Path) -> None:
    """Plot all model metrics in separate panels with their native units."""
    development = frame[frame["split"] == "development"].copy()
    if development.empty:
        return
    labels = development["variant"].tolist()
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = [
        ("mae", "MAE (seconds)"),
        ("mse", "MSE (seconds²)"),
        ("rmse", "RMSE (seconds)"),
        ("event_mae_median", "Event median MAE (seconds)"),
        ("event_mae_iqr", "Event MAE IQR (seconds)"),
        ("degradation_slope_error", "Degradation-slope error (seconds/lap)"),
    ]
    for axis, (column, ylabel) in zip(axes.flat, panels):
        values = development[column].to_numpy(dtype=float)
        axis.bar(positions, values, color="#2f6f9f")
        axis.set_title(column.replace("_", " ").title())
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels, rotation=65, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=.25)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
