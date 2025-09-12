"""3D hex-region state visualization.

TODO: read AGENTS.md completely
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
from ironcortex.visualization import _is_interactive_backend

# Matplotlib is an optional dependency for hex state visualization. Import it
# lazily so the rest of the package can be used in environments without the
# library (e.g. during headless testing).
try:  # pragma: no cover - availability depends on runtime environment
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception:  # pragma: no cover - allows import without matplotlib
    plt = None
    Poly3DCollection = None

from ironcortex.wiring import hex_axial_coords


@dataclass
class HexStateVisualizer:
    """Interactive 3D plot of region states on a hex grid.

    Parameters
    ----------
    R:
        Number of regions to display. Regions are laid out using
        ``hex_axial_coords`` and drawn in the ``z=0`` plane.
    size:
        Radius of each outer hexagon.
    """

    R: int
    size: float = 1.0

    def __post_init__(self) -> None:
        if plt is None or Poly3DCollection is None:
            raise RuntimeError("matplotlib is required for HexStateVisualizer")
        self.coords = hex_axial_coords(self.R).tolist()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_axis_off()
        # Flat grid visualized in 3D but with minimal z-scale
        self.ax.set_box_aspect((1, 1, 0.1))
        self.interactive = _is_interactive_backend()
        if self.interactive:
            plt.ion()
            try:  # pragma: no cover - headless environments
                plt.show(block=False)
            except Exception:
                self.interactive = False
        else:  # pragma: no cover - used for user feedback in headless setups
            warnings.warn(
                "Matplotlib is using a non-interactive backend; hex state"
                " plots will not be displayed.",
                RuntimeWarning,
            )
        self._collections: list[Poly3DCollection] = []

    def _axial_to_cart(self, q: float, r: float) -> tuple[float, float]:
        x = self.size * np.sqrt(3) * (q + r / 2)
        y = self.size * 1.5 * r
        return x, y

    def _hex_vertices(
        self, x: float, y: float, scale: float
    ) -> list[tuple[float, float, float]]:
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        xs = x + scale * self.size * np.cos(angles)
        ys = y + scale * self.size * np.sin(angles)
        zs = np.zeros_like(xs)
        return list(zip(xs, ys, zs))

    def update(self, states: Sequence[float]) -> None:
        """Render new region states.

        ``states`` should be an iterable of floats in ``[0, 1]``. Values are
        mapped from black (0) to matrix green (1).
        """

        if not hasattr(self, "_collections"):
            self._collections = []
        for coll in self._collections:
            try:
                coll.remove()
            except Exception:
                pass
        self._collections.clear()
        for (q, r), s in zip(self.coords, states):
            x, y = self._axial_to_cart(q, r)
            outer = Poly3DCollection(
                [self._hex_vertices(x, y, 1.0)],
                facecolors="none",
                edgecolors="white",
            )
            self.ax.add_collection3d(outer)
            self._collections.append(outer)
            s_clamped = float(min(1.0, max(0.0, s)))
            color = (0.0, s_clamped, 0.0)
            inner = Poly3DCollection(
                [self._hex_vertices(x, y, 0.5)], facecolors=[color], edgecolors="none"
            )
            self.ax.add_collection3d(inner)
            self._collections.append(inner)
        try:
            self.fig.canvas.draw()
            if self.interactive:
                if hasattr(self.fig.canvas, "flush_events"):
                    self.fig.canvas.flush_events()
                plt.pause(0.001)
        except Exception:
            pass
