"""Small cantilever meshes: a 4 x 1 x 1 block of 8 x 2 x 2 cubes, each split into six tetrahedra.

The block is clamped at x = 0. Its top face z = 1 is a loaded T6 surface, T3 on the T4 mesh, and its
free end x = 4 a second, T3 surface, for boundary conditions other than the clamp.

A Kuhn split of identically oriented cubes is conforming, and all edge midpoints of its
tetrahedra fall on the half-spacing grid, so every half-grid point is a node of the T10 mesh. The
T4 mesh has the cube corners alone.
"""
import itertools

import numpy as np

from helpers.box import T10_EDGES

CELLS = (8, 2, 2)
CELL_SIZE = 0.5
MATERIAL, SURFACE, END_SURFACE = 30, 130, 131
SHAPE = tuple(2 * c + 1 for c in CELLS)           # half-grid points per axis


def node(p):
    """One-based tetgen node index of half-grid point p on the T10 mesh."""
    i, j, k = p
    return 1 + i + SHAPE[0] * (j + SHAPE[1] * k)


def write_mesh(tetgen_dir, scale=(1, 1, 1), linear=False, quadratic_end=False, closed=False):
    """Write the cantilever as tetgen .node/.ele/.sur files, T10 elements with T6 top faces, or T4
    elements with T3 top faces if linear, and T3 end faces, T6 if quadratic_end, with the block
    stretched by scale along each axis. If closed, the loaded surface is the whole boundary instead and
    there is no end surface, for plugins that refuse a cavity that does not enclose a volume. Nodes
    are numbered in the order of node(), which they match on the T10 mesh."""
    def mid(a, b):
        return tuple((x + y) // 2 for x, y in zip(a, b))

    elements = []       # of half-grid points
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
            elements.append(v + ([] if linear else [mid(v[a], v[b]) for a, b in T10_EDGES]))

    surfaces = []

    def add_square(corner, index, quadratic):
        """Two outward-facing triangles, T6 or T3, on the square spanned by corner(0..1, 0..1)."""
        for tri in ((corner(0, 0), corner(1, 0), corner(1, 1)),
                    (corner(0, 0), corner(1, 1), corner(0, 1))):
            mids = [mid(tri[a], tri[b]) for a, b in ((0, 1), (1, 2), (2, 0))] if quadratic else []
            surfaces.append((index, list(tri) + mids))

    def add_face(axis, at, index, quadratic):
        """The squares of the block face normal to axis at half-grid coordinate at, facing outward."""
        # Cyclic in-plane axes make u x v point along +axis, so the face at the lower end swaps them.
        u, v = (axis + 1) % 3, (axis + 2) % 3
        if at == 0:
            u, v = v, u
        for cu, cv in itertools.product(range(CELLS[u]), range(CELLS[v])):
            def corner(du, dv):
                p = [at] * 3
                p[u], p[v] = 2 * (cu + du), 2 * (cv + dv)
                return tuple(p)
            add_square(corner, index, quadratic)

    if closed:
        for axis in range(3):
            for at in (0, 2 * CELLS[axis]):
                add_face(axis, at, SURFACE, quadratic=not linear)
    else:
        add_face(2, 2 * CELLS[2], SURFACE, quadratic=not linear)
        # Nodes are numbered with z slowest, so the free end spans every rank's node block, whereas
        # the bottom face would lie on the first rank alone. It is linear unless asked otherwise, since
        # the Robin boundary elements are three-node triangles and the loader has to refine six-node
        # faces into four of them.
        add_face(0, 2 * CELLS[0], END_SURFACE, quadratic=quadratic_end and not linear)

    # A point in no element would be a node without stiffness.
    points = sorted({p for e in elements for p in e}, key=node)
    number = {p: n for n, p in enumerate(points, 1)}
    with open(tetgen_dir / "cantilever.node", "w") as f:
        f.write(f"{len(points)} 3 1 0\n")
        for p in points:
            x, y, z = (CELL_SIZE / 2 * q * s for q, s in zip(p, scale))
            f.write(f"{number[p]} {x} {y} {z} {7 if p[0] == 0 else 0}\n")
    with open(tetgen_dir / "cantilever.ele", "w") as f:
        f.write(f"{len(elements)} {len(elements[0])} 1\n")
        for n, e in enumerate(elements, 1):
            f.write(f"{n} {' '.join(str(number[p]) for p in e)} {MATERIAL}\n")
    with open(tetgen_dir / "cantilever.sur", "w") as f:
        f.write(f"{len(surfaces)} {3 if linear else 6} 2\n")
        for n, (index, s) in enumerate(surfaces, 1):
            f.write(f"{n} {' '.join(str(number[p]) for p in s)} {index} {index}\n")
