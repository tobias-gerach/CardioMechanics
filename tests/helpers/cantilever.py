"""Small cantilever meshes: a 4 x 1 x 1 block of 8 x 2 x 2 cubes, each split into six tetrahedra.

The block is clamped at x = 0. Its top face z = 1 is a loaded T6 surface and its free end x = 4 a
second, T3 surface, for boundary conditions other than the clamp.

A Kuhn split of identically oriented cubes is conforming, and all edge midpoints of its
tetrahedra fall on the half-spacing grid, so every half-grid point is a node.
"""
import itertools

import numpy as np

from helpers.box import T10_EDGES

CELLS = (8, 2, 2)
CELL_SIZE = 0.5
MATERIAL, SURFACE, END_SURFACE = 30, 130, 131
SHAPE = tuple(2 * c + 1 for c in CELLS)           # half-grid points per axis


def node(p):
    """One-based tetgen node index of half-grid point p."""
    i, j, k = p
    return 1 + i + SHAPE[0] * (j + SHAPE[1] * k)


def write_mesh(tetgen_dir, scale=(1, 1, 1)):
    """Write the cantilever as tetgen .node/.ele/.sur files, T10 elements, T6 top and T3 end faces,
    with the block stretched by scale along each axis."""
    def mid(a, b):
        return tuple((x + y) // 2 for x, y in zip(a, b))

    elements = []
    for cell in itertools.product(*(range(c) for c in CELLS)):
        origin = np.array(cell) * 2
        for perm in itertools.permutations(range(3)):
            v = [origin.copy()]
            for axis in perm:
                step = np.zeros(3, dtype=int)
                step[axis] = 2
                v.append(v[-1] + step)
            if np.linalg.det(np.array([v[1] - v[0], v[2] - v[0], v[3] - v[0]])) < 0:
                v[1], v[2] = v[2], v[1]
            v = [tuple(x) for x in v]
            elements.append([node(x) for x in v] + [node(mid(v[a], v[b])) for a, b in T10_EDGES])

    surfaces = []

    def add_square(corner, index, quadratic):
        """Two outward-facing triangles, T6 or T3, on the square spanned by corner(0..1, 0..1)."""
        for tri in ((corner(0, 0), corner(1, 0), corner(1, 1)),
                    (corner(0, 0), corner(1, 1), corner(0, 1))):
            mids = [node(mid(tri[a], tri[b])) for a, b in ((0, 1), (1, 2), (2, 0))] if quadratic else []
            surfaces.append((index, [node(p) for p in tri] + mids))

    top, end = 2 * CELLS[2], 2 * CELLS[0]
    for ci, cj in itertools.product(range(CELLS[0]), range(CELLS[1])):
        add_square(lambda dx, dy: (2 * (ci + dx), 2 * (cj + dy), top), SURFACE, quadratic=True)
    # Nodes are numbered with z slowest, so the free end spans every rank's node block, whereas
    # the bottom face would lie on the first rank alone. It is linear because the Robin boundary
    # elements are three-node triangles.
    for cj, ck in itertools.product(range(CELLS[1]), range(CELLS[2])):
        add_square(lambda dy, dz: (end, 2 * (cj + dy), 2 * (ck + dz)), END_SURFACE, quadratic=False)

    points = list(itertools.product(*(range(n) for n in SHAPE)))
    points.sort(key=node)
    with open(tetgen_dir / "cantilever.node", "w") as f:
        f.write(f"{len(points)} 3 1 0\n")
        for p in points:
            x, y, z = (CELL_SIZE / 2 * q * s for q, s in zip(p, scale))
            f.write(f"{node(p)} {x} {y} {z} {7 if p[0] == 0 else 0}\n")
    with open(tetgen_dir / "cantilever.ele", "w") as f:
        f.write(f"{len(elements)} 10 1\n")
        for n, e in enumerate(elements, 1):
            f.write(f"{n} {' '.join(map(str, e))} {MATERIAL}\n")
    with open(tetgen_dir / "cantilever.sur", "w") as f:
        f.write(f"{len(surfaces)} 6 2\n")
        for n, (index, s) in enumerate(surfaces, 1):
            f.write(f"{n} {' '.join(map(str, s))} {index} {index}\n")
