"""Distorted box meshes for verification against closed-form solutions.

The box is a grid of cells, each split into six tetrahedra around its main diagonal (Kuhn split),
which is conforming because every cell has the same orientation. Vertex nodes are jittered so that
no element is regular, boundary nodes included, since on a coarse grid some elements have all four
vertices on the boundary. A node on a face of the box moves only within that face, so every face
stays planar and a homogeneous deformation of the box stays exactly representable. Mid-edge nodes sit at
the midpoints of the jittered edges, so every element is affine and the element quadrature
integrates a uniform stress exactly.

Nodes are keyed by their point on the half-cell grid: the vertex at cell corner v has key 2v, the
midpoint of the edge from v to v + e has key 2v + e. Every Kuhn edge runs from some v to v + e with
e in {0, 1}^3, so no two edges share a midpoint key.
"""
import itertools

import numpy as np

MATERIAL, LOADED_FACE = 30, 130
# T10 local nodes 5-10 sit on the edges (1,2), (2,3), (1,3), (1,4), (2,4), (3,4).
T10_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
# T6 local nodes 4-6 sit on the edges (1,2), (2,3), (3,1).
T6_EDGES = ((0, 1), (1, 2), (2, 0))


def _kuhn_tetrahedra(cells):
    """Vertex keys of the six tetrahedra of every cell, ordered to a positive volume."""
    for cell in itertools.product(*(range(c) for c in cells)):
        for perm in itertools.permutations(range(3)):
            v = [2 * np.array(cell)]
            for axis in perm:
                v.append(v[-1] + 2 * np.eye(3, dtype=int)[axis])
            if np.linalg.det(np.array(v[1:]) - v[0]) < 0:
                v[1], v[2] = v[2], v[1]
            yield [tuple(int(x) for x in p) for p in v]


def _mid(a, b):
    return tuple((x + y) // 2 for x, y in zip(a, b))


CELLS = 3                   # per axis
JITTER = 0.2                # largest vertex displacement along each axis, as a fraction of the cell size


def write_box(directory, quadratic=True, basis=None, symmetry_planes=True):
    """Write the unit cube as tetgen box.node, box.ele and box.sur into directory.

    T10 elements with a T6 loaded face if quadratic, otherwise T4 elements with a T3 loaded face
    on the same vertices. If symmetry_planes, every node on the plane x_i = 0 has component i
    fixed, so the three faces through the origin are symmetry planes; otherwise every node is
    free. The loaded face is x_0 = 1, its triangles ordered so that their normals point out of
    the box. If basis is given, its rows are the fibre, sheet and sheet-normal directions of
    every element, written to box.bases.
    """
    cells = np.full(3, CELLS)
    h, upper = 1 / cells, 2 * cells       # cell size, largest half-grid key
    rng = np.random.default_rng(0)

    vertices = {}
    for key in itertools.product(*(range(0, u + 1, 2) for u in upper)):
        k = np.array(key)
        vertices[key] = k / 2 * h + ((k > 0) & (k < upper)) * rng.uniform(-JITTER, JITTER, 3) * h

    def position(key):
        e = np.array(key) % 2
        return (vertices[tuple(np.array(key) - e)] + vertices[tuple(np.array(key) + e)]) / 2

    tetrahedra = list(_kuhn_tetrahedra(cells))
    volumes = [np.linalg.det(np.array([vertices[v] - vertices[t[0]] for v in t[1:]])) for t in tetrahedra]
    assert min(volumes) > 0, f"jitter {JITTER} inverts an element"

    elements, faces = [], []
    for t in tetrahedra:
        elements.append(t + [_mid(t[i], t[j]) for i, j in T10_EDGES] if quadratic else t)
        tri = [v for v in t if v[0] == upper[0]]
        if len(tri) == 3:
            a, b, c = (vertices[v] for v in tri)
            if np.cross(b - a, c - a)[0] < 0:
                tri[1], tri[2] = tri[2], tri[1]
            faces.append(tri + [_mid(tri[i], tri[j]) for i, j in T6_EDGES] if quadratic else tri)

    keys = sorted({k for e in elements for k in e})
    index = {k: n for n, k in enumerate(keys, 1)}
    with open(directory / "box.node", "w") as f:
        f.write(f"{len(keys)} 3 1 0\n")
        for k in keys:
            x, y, z = position(k)
            fixed = sum(1 << i for i in range(3) if k[i] == 0) if symmetry_planes else 0
            f.write(f"{index[k]} {x:.17g} {y:.17g} {z:.17g} {fixed}\n")
    with open(directory / "box.ele", "w") as f:
        f.write(f"{len(elements)} {len(elements[0])} 1\n")
        for n, e in enumerate(elements, 1):
            f.write(f"{n} {' '.join(str(index[k]) for k in e)} {MATERIAL}\n")
    with open(directory / "box.sur", "w") as f:
        f.write(f"{len(faces)} {len(faces[0])} 2\n")
        for n, s in enumerate(faces, 1):
            f.write(f"{n} {' '.join(str(index[k]) for k in s)} {LOADED_FACE} {LOADED_FACE}\n")
    if basis is not None:
        with open(directory / "box.bases", "w") as f:
            f.write(f"{len(elements)} 1\n")          # one basis per element, shared by its quadrature points
            for n in range(1, len(elements) + 1):
                f.write(f"{n} {' '.join(f'{x:.17g}' for x in np.ravel(basis))}\n")
