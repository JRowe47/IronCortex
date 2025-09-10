import torch

from ironcortex.region import RWKVRegionCell
from ironcortex.utils import KWTA
from ironcortex.constants import EPS_DIV


def _manual_step(cell: RWKVRegionCell, x: torch.Tensor) -> torch.Tensor:
    x_norm = cell.norm(x + cell.pred)
    cell.pred.zero_()
    lam = cell.decay()
    w = torch.exp(cell.k_lin(x_norm))
    v = cell.v_lin(x_norm)
    state_num = cell.state_num * lam + w * v
    state_den = cell.state_den * lam + w
    r = torch.sigmoid(cell.r_lin(x_norm))
    y = r * (state_num / (state_den + EPS_DIV))
    h = KWTA(x_norm + cell.o_lin(y), k=max(1, cell.d // 8))
    return h, state_num, state_den


def test_forward_parity_flag_off():
    torch.manual_seed(0)
    d = 8
    cell = RWKVRegionCell(d, m_time_pairs=0, enable_adaptive_filter_dynamics=False)
    x = torch.randn(d)
    h_expected, num_expected, den_expected = _manual_step(cell, x)
    h = cell.step(x, 0.0)
    assert torch.allclose(h, h_expected, atol=1e-6)
    assert torch.allclose(cell.state_num, num_expected, atol=1e-6)
    assert torch.allclose(cell.state_den, den_expected, atol=1e-6)
    assert torch.all(cell.state_var == 0)


def test_state_var_non_negative_decreases_with_small_gain():
    torch.manual_seed(0)
    d = 8
    cell = RWKVRegionCell(d, m_time_pairs=0, enable_adaptive_filter_dynamics=True)
    cell.obs_noise_param.data.fill_(4.0)
    cell.state_var += 0.1
    prior_var = cell.state_var.clone()
    x = torch.randn(d)
    cell.step(x, 0.0)
    assert torch.all(cell.state_var >= 0)
    assert torch.all(cell.state_var < prior_var)


def test_gradients_stable_random_run():
    torch.manual_seed(0)
    d = 8
    cell = RWKVRegionCell(d, m_time_pairs=0, enable_adaptive_filter_dynamics=True)
    xs = torch.randn(4, d, requires_grad=True)
    outs = []
    for i in range(4):
        outs.append(cell.step(xs[i], 0.0))
    loss = sum(o.sum() for o in outs)
    loss.backward()
    for p in cell.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))


def test_direction_norm_unit_and_gradients():
    torch.manual_seed(0)
    d = 8
    cell = RWKVRegionCell(d, m_time_pairs=0, enable_radial_tangential_updates=True)
    x = torch.randn(d, requires_grad=True)
    h = cell.step(x, 0.0)
    assert torch.allclose(cell.last_dir.norm(), torch.tensor(1.0), atol=1e-5)
    h.sum().backward()
    for p in cell.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))


def test_radial_update_reduces_spikes():
    torch.manual_seed(0)
    d = 8
    cell = RWKVRegionCell(d, m_time_pairs=0, enable_radial_tangential_updates=True)
    cell.step(torch.ones(d), 0.0)
    spike = 10 * torch.ones(d)
    cell.step(spike, 0.0)
    assert cell.radius.item() < cell.last_norm.item()


def test_scalar_noise_mode_broadcasts():
    d = 4
    cell = RWKVRegionCell(d, m_time_pairs=0, enable_adaptive_filter_dynamics=True)
    assert cell.process_noise_param.shape == ()
    assert cell.process_noise().shape == (d,)
    cell_vec = RWKVRegionCell(
        d,
        m_time_pairs=0,
        enable_adaptive_filter_dynamics=True,
        afd_noise_mode="vector",
    )
    assert cell_vec.process_noise_param.shape == (d,)
    assert cell_vec.process_noise().shape == (d,)


def test_fast_forward_dt_updates_buffers():
    torch.manual_seed(0)
    d = 4
    cell = RWKVRegionCell(d, m_time_pairs=0, enable_adaptive_filter_dynamics=True)
    cell.state_num.fill_(1.0)
    cell.state_den.fill_(1.0)
    cell.state_var.fill_(0.5)
    pn = cell.process_noise()
    dt = 3
    lam = cell.decay(dt=dt)
    cell.fast_forward(dt)
    assert torch.allclose(cell.state_num, lam)
    assert torch.allclose(cell.state_den, lam)
    expected_var = 0.5 * lam.pow(2) + pn * dt
    assert torch.allclose(cell.state_var, expected_var)


def test_predictive_trace_toggle():
    d = 4
    cell = RWKVRegionCell(
        d,
        m_time_pairs=0,
        enable_adaptive_filter_dynamics=False,
        use_predictive_trace=False,
    )
    msg = torch.ones(d)
    cell.predict(msg)
    assert torch.all(cell.pred == 0)
    x = torch.randn(d)
    cell.step(x, 0.0)
    assert torch.all(cell.pred == 0)
