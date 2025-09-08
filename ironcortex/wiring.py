"""Hex-grid wiring helpers using axial coordinates.

The previous implementation approximated hex wiring with a square grid. This
module now provides true axial hex layouts that can handle arbitrary numbers of
regions. Regions are assigned in a clockwise spiral starting from the origin
and adjacency lists are constructed using the six axial directions.
"""

from typing import List, Dict, Tuple

import torch


def hex_axial_coords(R: int) -> torch.Tensor:
    """Return axial ``(q, r)`` coordinates for ``R`` regions.

    Regions are arranged on a hex grid in a clockwise spiral starting at the
    origin. The output is a ``(R, 2)`` float tensor where each row contains the
    ``q`` and ``r`` coordinates of a region.
    """

    if R <= 0:
        return torch.zeros((0, 2), dtype=torch.float32)

    coords: List[Tuple[int, int]] = [(0, 0)]
    if R == 1:
        return torch.tensor(coords, dtype=torch.float32)

    k = 1  # current ring radius
    while len(coords) < R:
        q, r = k, 0  # start at eastern side of the ring
        # Directions for clockwise traversal starting from east → southeast
        dirs = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]
        for dq, dr in dirs:
            for _ in range(k):
                coords.append((q, r))
                if len(coords) == R:
                    return torch.tensor(coords, dtype=torch.float32)
                q += dq
                r += dr
        k += 1

    return torch.tensor(coords, dtype=torch.float32)


def hex_neighbors(R: int) -> List[List[int]]:
    """Build clockwise neighbor lists for ``R`` hex regions.

    The neighbor ordering follows the six axial directions starting from east
    and moving clockwise. Regions on the boundary simply omit missing
    neighbors.
    """

    coords = hex_axial_coords(R)
    coord_map: Dict[Tuple[int, int], int] = {
        (int(q), int(r)): i for i, (q, r) in enumerate(coords.tolist())
    }

    neighbors: List[List[int]] = [[] for _ in range(R)]
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for i, (q, r) in enumerate(coords.tolist()):
        for dq, dr in directions:
            j = coord_map.get((int(q + dq), int(r + dr)))
            if j is not None:
                neighbors[i].append(j)

    return neighbors
