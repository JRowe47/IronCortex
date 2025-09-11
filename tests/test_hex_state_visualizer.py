import importlib.util
import sys
from pathlib import Path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hv_mod = load_module(
    "hex_visualizer",
    Path(__file__).resolve().parent.parent / "ironcortex" / "hex_visualizer.py",
)


class DummyArtistList(list):
    def clear(self) -> None:  # pragma: no cover - exercised via hex_visualizer
        raise TypeError("ArtistList object does not support item deletion")


class DummyAx:
    def __init__(self) -> None:
        self.collections = DummyArtistList()

    def add_collection3d(self, coll) -> None:
        self.collections.append(coll)


class DummyFig:
    class Canvas:
        def draw(self) -> None:
            pass

    def __init__(self) -> None:
        self.canvas = self.Canvas()


def test_update_handles_artistlist(monkeypatch):
    hv = hv_mod.HexStateVisualizer.__new__(hv_mod.HexStateVisualizer)
    hv.R = 1
    hv.size = 1.0
    hv.coords = [(0, 0)]
    hv.ax = DummyAx()
    hv.fig = DummyFig()
    hv.interactive = False
    monkeypatch.setattr(hv_mod, "Poly3DCollection", lambda *args, **kwargs: object())
    hv.update([0.5])
    assert len(hv.ax.collections) == 2
