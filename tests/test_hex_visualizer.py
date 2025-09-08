import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ironcortex import HexStateVisualizer


def test_hex_state_visualizer_update():
    vis = HexStateVisualizer(R=3)
    vis.update([0.0, 1.0, 0.5])
    # Each region adds an outer and inner hexagon collection
    assert len(vis.ax.collections) == 6
    plt.close(vis.fig)
