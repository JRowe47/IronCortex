import torch

from ironcortex.wiring import hex_axial_coords, hex_neighbors


def test_hex_axial_coords_clockwise_spiral():
    coords = hex_axial_coords(7)
    expected = torch.tensor(
        [[0, 0], [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]],
        dtype=torch.float32,
    )
    assert torch.equal(coords, expected)


def test_hex_neighbors_basic():
    neighbors = hex_neighbors(7)
    # center connects to all six surrounding regions
    assert set(neighbors[0]) == set(range(1, 7))
    # first region on outer ring connects to center and its two adjacent cells
    assert set(neighbors[1]) == {0, 2, 6}
