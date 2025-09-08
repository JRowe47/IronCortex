import matplotlib

matplotlib.use("Agg")

from ironcortex.visualization import TrainVisualizer, _is_interactive_backend


def test_visualizer_update():
    viz = TrainVisualizer()
    assert "perplexity" not in viz.axes_map["eval"]
    assert viz.axes_map["perplexity"] == ["perplexity"]
    metrics = {
        "ff": 0.1,
        "rtd": 0.2,
        "denoise": 0.3,
        "critic": 0.4,
        "verify": 0.5,
        "E_pos": 0.6,
        "E_neg": 0.7,
        "total": 2.8,
    }
    eval_metrics = {
        "cross_entropy": 0.2,
        "perplexity": 1.5,
        "gain_mean": 0.1,
        "tau_mean": 0.2,
    }
    viz.update(1, metrics, eval_metrics)


def test_inline_backend_detection(monkeypatch):
    monkeypatch.setattr(
        matplotlib, "get_backend", lambda: "module://matplotlib_inline.backend_inline"
    )
    assert not _is_interactive_backend()
