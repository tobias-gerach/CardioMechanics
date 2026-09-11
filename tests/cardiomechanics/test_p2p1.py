"""P2P1 Taylor-Hood element on a small Neo-Hooke cantilever, static solver, one rank.

The mesh is generated here rather than shipped: a 4 x 1 x 1 block of 8 x 2 x 2 cubes,
each split into six tetrahedra, clamped at x = 0 and bent by a pressure on its top face.
Every length, the shear modulus and the bulk modulus are O(1), so the displacement,
coupling and constraint blocks of the Jacobian have comparable magnitude. On a
millimetre mesh in SI units those blocks differ by many decades, and a relative
Jacobian check would see nothing but the displacement block.

A Kuhn split of identically oriented cubes is conforming, and all edge midpoints of its
tetrahedra fall on the half-spacing grid, so every half-grid point is a node.
"""
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from helpers.compare import read_vtu_points
from helpers.run import run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "p2p1_cantilever.xml"
CELLS = (8, 2, 2)
CELL_SIZE = 0.5
MATERIAL, SURFACE = 30, 130

# T10 local nodes 5-10 sit on the edges (1,2), (2,3), (1,3), (1,4), (2,4), (3,4).
T10_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))


SHAPE = tuple(2 * c + 1 for c in CELLS)           # half-grid points per axis


def _node(p):
    """One-based tetgen node index of half-grid point p."""
    i, j, k = p
    return 1 + i + SHAPE[0] * (j + SHAPE[1] * k)


def _clamped_dofs():
    """Displacement unknowns of the clamped face x = 0; on one rank, 3 * node + component."""
    return {3 * (_node((0, j, k)) - 1) + c
            for j in range(SHAPE[1]) for k in range(SHAPE[2]) for c in range(3)}


def _write_mesh(tetgen_dir):
    """Write the cantilever as tetgen .node/.ele/.sur files, T10 elements, T6 top faces."""
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
            elements.append([_node(x) for x in v] + [_node(mid(v[a], v[b])) for a, b in T10_EDGES])

    top = 2 * CELLS[2]
    surfaces = []
    for ci, cj in itertools.product(range(CELLS[0]), range(CELLS[1])):
        c = (2 * ci, 2 * cj, top)
        corner = lambda dx, dy: (c[0] + 2 * dx, c[1] + 2 * dy, top)
        for tri in ((corner(0, 0), corner(1, 0), corner(1, 1)),
                    (corner(0, 0), corner(1, 1), corner(0, 1))):
            surfaces.append([_node(p) for p in tri]
                            + [_node(mid(tri[a], tri[b])) for a, b in ((0, 1), (1, 2), (2, 0))])

    points = list(itertools.product(*(range(n) for n in SHAPE)))
    points.sort(key=_node)
    with open(tetgen_dir / "cantilever.node", "w") as f:
        f.write(f"{len(points)} 3 1 0\n")
        for p in points:
            x, y, z = (CELL_SIZE / 2 * q for q in p)
            f.write(f"{_node(p)} {x} {y} {z} {7 if p[0] == 0 else 0}\n")
    with open(tetgen_dir / "cantilever.ele", "w") as f:
        f.write(f"{len(elements)} 10 1\n")
        for n, e in enumerate(elements, 1):
            f.write(f"{n} {' '.join(map(str, e))} {MATERIAL}\n")
    with open(tetgen_dir / "cantilever.sur", "w") as f:
        f.write(f"{len(surfaces)} 6 2\n")
        for n, s in enumerate(surfaces, 1):
            f.write(f"{n} {' '.join(map(str, s))} {SURFACE} {SURFACE}\n")


def _run(binary, cm_env, wd, element_type="T10P1", kappa=100, env=None, check=True):
    """Stage mesh and settings into wd, run serially, return (process, vtu directory)."""
    (wd / "tetgen").mkdir()
    _write_mesh(wd / "tetgen")
    (wd / "Results").mkdir()
    text = FIXTURE.read_text()
    for old, new in (("<Type>T10P1</Type>", f"<Type>{element_type}</Type>"),
                     ("<k>100</k>", f"<k>{kappa}</k>")):
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / FIXTURE.name).write_text(text)
    proc = run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name],
                      cwd=wd, env=env or cm_env, timeout=600, check=check)
    if check:
        assert "SIMULATION FAILED" not in proc.stdout, f"{element_type} kappa={kappa}\n{proc.stdout[-2000:]}"
    return proc, wd / "Results" / "cantilever_vtu"


def _final_points(vtu_dir):
    last = max(vtu_dir.glob("cantilever.*.vtu"), key=lambda p: int(p.stem.split(".")[1]))
    return read_vtu_points(last)


def test_unknown_element_type_lists_p2p1(binary, cm_env, tmp_path):
    proc, _ = _run(binary, cm_env, tmp_path, element_type="T10Q1", check=False)
    assert proc.returncode != 0, f"expected a non-zero exit\n{proc.stdout[-2000:]}"
    assert "T10P1" in proc.stdout + proc.stderr, (proc.stdout + proc.stderr)[-2000:]


def test_p2p1_static_run_converges(binary, cm_env, tmp_path):
    pytest.importorskip("meshio")
    _, vtu_dir = _run(binary, cm_env, tmp_path)
    _, pts = _final_points(vtu_dir)
    _, ref = read_vtu_points(vtu_dir / "cantilever.0.vtu")
    deflection = np.abs(pts - ref).max()
    assert deflection > 1e-2, f"cantilever barely moved: max displacement {deflection:.3e}"


JACOBIAN_THRESHOLD = 1e-6


def _jacobian_differences(view_file):
    """Non-zero (row, column, value) entries of the hand-coded minus finite-difference Jacobian.

    PETSc writes the hand-coded, the finite-difference and the thresholded difference matrix,
    in that order, and rewrites the file at every Jacobian evaluation, so it holds the last one.
    """
    sections = view_file.read_text().split("Mat Object:")[1:]
    assert len(sections) == 3, f"{view_file.name}: expected 3 matrices, found {len(sections)}"
    entries = []
    for line in sections[2].splitlines():
        m = re.match(r"\s*row (\d+):(.*)", line)
        if m:
            entries += [(int(m.group(1)), int(c), float(v))
                        for c, v in re.findall(r"\((\d+), ([^)]+)\)", m.group(2)) if float(v) != 0]
    return entries


def test_p2p1_jacobian_matches_finite_differences(binary, cm_env, tmp_path):
    """Every Jacobian block, coupling and constraint included, against PETSc's finite differences.

    kappa = 1 keeps the -1/kappa constraint block well above the threshold. Entries in clamped
    rows and columns are excluded: the hand-coded Jacobian replaces those rows by the identity and
    drops those columns, because a clamped increment is zero, whereas finite differences perturb
    clamped nodes like any other.
    """
    view = tmp_path / "jacobian.txt"
    env = dict(cm_env, PETSC_OPTIONS=f"-snes_test_jacobian {JACOBIAN_THRESHOLD} "
                                     f"-snes_test_jacobian_view ascii:{view}")
    _run(binary, cm_env, tmp_path, kappa=1, env=env)
    clamped = _clamped_dofs()
    wrong = [e for e in _jacobian_differences(view) if e[0] not in clamped and e[1] not in clamped]
    if wrong:
        row, col, value = max(wrong, key=lambda e: abs(e[2]))
        raise AssertionError(f"{len(wrong)} Jacobian entries differ beyond {JACOBIAN_THRESHOLD}, "
                             f"worst at row {row} column {col}: {value:.3e}")


# Eliminating the pressure pointwise from the perturbed constraint gives back Neo-Hooke's
# volumetric energy kappa/2 (J-1)^2, so at any kappa both elements model the same material and
# differ only in how the discretization carries the volumetric response. As kappa -> 0 that
# response vanishes and the two coincide; as kappa grows T10 locks and the gap opens up to its
# locking error. Observed gap over peak displacement: 6e-4, 2e-2, 6e-2, 8e-2 at kappa = 1, 10,
# 100, 1000.
KAPPA_SWEEP = (1, 10, 100, 1000)
MAX_GAP_AT_SMALLEST_KAPPA = 2e-3     # relative to the peak displacement


def test_p2p1_approaches_t10_as_kappa_decreases(binary, cm_env, tmp_path):
    pytest.importorskip("meshio")
    gaps = []
    for kappa in KAPPA_SWEEP:
        vtu_dirs = {}
        for element_type in ("T10", "T10P1"):
            wd = tmp_path / f"{element_type}_{kappa}"
            wd.mkdir()
            _, vtu_dirs[element_type] = _run(binary, cm_env, wd, element_type=element_type, kappa=kappa)
        pid, t10 = _final_points(vtu_dirs["T10"])
        _, p2p1 = _final_points(vtu_dirs["T10P1"])
        _, ref = read_vtu_points(vtu_dirs["T10"] / "cantilever.0.vtu")
        d = np.linalg.norm(p2p1 - t10, axis=1)
        i = int(np.argmax(d))
        peak = np.linalg.norm(t10 - ref, axis=1).max()
        gaps.append((kappa, d[i] / peak, int(pid[i])))

    report = ", ".join(f"kappa={k}: {g:.2e} at PointID={n}" for k, g, n in gaps)
    assert all(a[1] < b[1] for a, b in zip(gaps, gaps[1:])), f"gap does not shrink as kappa decreases: {report}"
    assert gaps[0][1] < MAX_GAP_AT_SMALLEST_KAPPA, f"gap at the smallest kappa too large: {report}"
