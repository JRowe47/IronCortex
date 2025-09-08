import torch

from ironcortex import CortexConfig, CortexReasoner, hex_axial_coords, hex_neighbors


def test_sparse_token_encoder_k_active():
    cfg = CortexConfig(R=2, d=16, V=10, K_inner=1, B_br=1, k_active=1, max_T=8, sdr_k=4)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg)
    tokens = torch.randint(0, cfg.V, (5,))
    emb = model.embed(tokens)
    assert (emb.sum(-1) == cfg.sdr_k).all()
    assert ((emb == 0) | (emb == 1)).all()
