import torch
from ironcortex.utils import KWTA
from ironcortex.region import RWKVRegionCell


def test_kwta_vectorized_soft():
    x = torch.tensor([[3.0, 2.0, 1.0], [1.0, -2.0, 0.5]])
    y_hard = KWTA(x, k=1)
    assert y_hard.nonzero().shape[0] == 2
    y_soft = KWTA(x, k=1, soft=True, temp=1.0)
    assert y_soft.shape == x.shape
    assert y_soft.abs().sum() > y_hard.abs().sum()


def test_region_kwta_schedule_and_disable():
    torch.manual_seed(0)
    d = 4
    cell = RWKVRegionCell(
        d,
        m_time_pairs=0,
        kwta_k=1,
        kwta_k_start=4,
        kwta_k_schedule=10,
        kwta_soft_mode=True,
        kwta_soft_temp=5.0,
    )
    x = torch.randn(d)
    cell.global_step = 0
    h0 = cell.step(x, 0.0)
    assert h0.nonzero().numel() > 1
    cell.global_step = 10
    h1 = cell.step(x, 0.0)
    assert h1.nonzero().shape[0] == 1
    cell_disable = RWKVRegionCell(d, m_time_pairs=0, disable_kwta=True)
    x2 = torch.randn(d)
    cell_disable.global_step = 10
    h_disable = cell_disable.step(x2, 0.0)
    assert h_disable.nonzero().shape[0] == d
