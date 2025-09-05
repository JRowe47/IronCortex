import os
import urllib.request

import torch

from ironcortex.data import download_tiny_shakespeare, TextDiffusionDataset


def test_download_tiny_shakespeare(tmp_path, monkeypatch):
    fake_content = b"hello"

    def fake_retrieve(url, filename):
        with open(filename, "wb") as f:
            f.write(fake_content)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    path = download_tiny_shakespeare(tmp_path)
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == fake_content


def test_text_diffusion_dataset():
    tokens = torch.arange(0, 100, dtype=torch.long)
    ds = TextDiffusionDataset(tokens, seq_len=10, vocab_size=256)
    sample = ds[0]
    assert sample.noisy.shape == (10,)
    assert sample.clean.shape == (10,)
    assert 0.0 <= sample.t <= 1.0
