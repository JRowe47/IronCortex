import torch

from ironcortex.gate import Router
from ironcortex.wiring import hex_neighbors, hex_axial_coords
from ironcortex.iron_rope import relative_fourier_bias


def _manual_messages(
    router: Router, H: torch.Tensor, reg_mask: torch.Tensor, reg_coords: torch.Tensor
) -> torch.Tensor:
    device = H.device
    M = torch.zeros(router.R, router.d, device=device)
    P = reg_coords.to(H.dtype).to(device).unsqueeze(0)
    for r in range(router.R):
        acc = torch.zeros(router.d, device=device)
        for s in router.neighbors[r]:
            if not bool(reg_mask[s]):
                continue
            msg = router.W_edge[f"{s}->{r}"](H[s])
            b = relative_fourier_bias(
                P[:, r : r + 1, :],
                P[:, s : s + 1, :],
                router.W_reg,
                router.beta_cos,
                router.beta_sin,
                router.fb_scale,
            )[0, 0, 0, 0]
            acc = acc + (1.0 + router.fb_alpha * b) * msg
        M[r] = acc
    return M


def test_router_parity_flag_off():
    torch.manual_seed(0)
    R, d = 3, 8
    neighbors = hex_neighbors(R)
    reg_coords = hex_axial_coords(R)
    router = Router(neighbors, d, R, enable_precision_routed_messages=False)
    H = torch.randn(R, d)
    reg_mask = torch.ones(R, dtype=torch.bool)
    M = router.messages(H, reg_mask, reg_coords)
    M_manual = _manual_messages(router, H, reg_mask, reg_coords)
    assert torch.allclose(M, M_manual, atol=1e-6)


def test_router_precision_downweights_inconsistent_neighbors():
    torch.manual_seed(0)
    R, d = 3, 4
    neighbors = [[1], [0, 2], [1]]
    reg_coords = torch.zeros(R, 2)
    router = Router(neighbors, d, R, enable_precision_routed_messages=True)
    router.fb_alpha = 0.0
    for r in range(R):
        router.query_lin[r].weight.data = torch.eye(d)
    for lin in router.key_lin.values():
        lin.weight.data = torch.eye(d)
    for p in router.raw_P_edge.values():
        p.data.fill_(1.0)
    H = torch.zeros(R, d)
    H[2] = 5.0
    reg_mask = torch.ones(R, dtype=torch.bool)
    router.messages(H, reg_mask, reg_coords)
    w_consistent = router.last_weights["0->1"]
    w_inconsistent = router.last_weights["2->1"]
    assert w_consistent > w_inconsistent
