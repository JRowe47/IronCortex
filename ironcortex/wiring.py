from typing import List

import torch

# 10) Minimal wiring helpers (hex-like neighbor graph)
# ==========================================================

def hex_neighbors_grid(R: int, side: int) -> List[List[int]]:
    """Build a simple 2D grid neighborhood (4-neighbors) as a hex proxy.

    For research convenience; replace with true hex axial coords if you have them.
    Assumes R == side * side. Returns adjacency list.
    """
    assert R == side * side
    neighbors = [[] for _ in range(R)]
    def idx(x, y): return x * side + y
    for i in range(side):
        for j in range(side):
            r = idx(i, j)
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neigh
                ni, nj = i + di, j + dj
                if 0 <= ni < side and 0 <= nj < side:
                    neighbors[r].append(idx(ni, nj))
    return neighbors


def hex_axial_coords_from_grid(R: int, side: int) -> torch.Tensor:
    """Produce 2-D coordinates for regions laid out on a square grid (proxy for hex)."""
    coords = []
    for i in range(side):
        for j in range(side):
            coords.append([float(i), float(j)])
    return torch.tensor(coords, dtype=torch.float32)


