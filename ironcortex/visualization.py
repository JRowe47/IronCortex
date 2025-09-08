"""Realtime training visualization utilities.

TODO: read AGENTS.md completely
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt


@dataclass
class TrainVisualizer:
    """Plot training metrics with a dark themed realtime chart.

    Parameters
    ----------
    axes_map:
        Mapping of axis name to iterable of metric keys to plot on that axis. This
        allows grouping metrics with similar scales on separate subplots.
    """

    axes_map: Mapping[str, Iterable[str]] | None = None

    def __post_init__(self) -> None:
        plt.style.use("dark_background")
        if self.axes_map is None:
            self.axes_map = {
                "loss": ["ff", "rtd", "denoise", "critic", "verify", "total"],
                "energy": ["E_pos", "E_neg"],
                "eval": [
                    "cross_entropy",
                    "perplexity",
                    "gain_mean",
                    "tau_mean",
                ],
            }
        n_axes = len(self.axes_map)
        self.fig, axes = plt.subplots(n_axes, 1, figsize=(8, 2.5 * n_axes))
        if not isinstance(axes, Iterable):  # pragma: no cover
            axes = [axes]
        self.axes: Dict[str, plt.Axes] = {}
        self.lines: Dict[str, plt.Line2D] = {}
        self.data: Dict[str, list[tuple[int, float]]] = defaultdict(list)
        for ax, (axis_name, metrics) in zip(axes, self.axes_map.items()):
            ax.set_title(axis_name)
            for m in metrics:
                (line,) = ax.plot([], [], label=m)
                self.lines[m] = line
            ax.legend(loc="upper right")
            self.axes[axis_name] = ax
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

    def update(
        self, step: int, metrics: Mapping[str, float], eval_metrics: Mapping[str, float]
    ) -> None:
        """Append new metrics and refresh the chart."""

        combined: Dict[str, float] = {**metrics, **eval_metrics}
        for key, val in combined.items():
            if key in self.lines:
                self.data[key].append((step, float(val)))
                xs, ys = zip(*self.data[key])
                line = self.lines[key]
                line.set_data(xs, ys)
                ax = line.axes
                ax.relim()
                ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
