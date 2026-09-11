"""P2P1 Taylor-Hood element on a small Neo-Hooke cantilever, static solver, one rank.

The mesh is generated here rather than shipped: a 4 x 1 x 1 block of 8 x 2 x 2 cubes,
each split into six tetrahedra, clamped at x = 0 and bent by a pressure on its top face.
Its free end is a second surface, for boundary conditions other than the clamp.
Every length, the shear modulus and the bulk modulus are O(1), so the displacement,
coupling and constraint blocks of the Jacobian have comparable magnitude. On a
millimetre mesh in SI units those blocks differ by many decades, and a relative
Jacobian check would see nothing but the displacement block.

A Kuhn split of identically oriented cubes is conforming, and all edge midpoints of its
tetrahedra fall on the half-spacing grid, so every half-grid point is a node.
"""
import itertools
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers.compare import read_vtu_points
from helpers.run import run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "p2p1_cantilever.xml"
ROBIN_FIXTURE = Path(__file__).parent / "fixtures" / "p2p1_cantilever_robin.xml"
CELLS = (8, 2, 2)
CELL_SIZE = 0.5
MATERIAL, SURFACE, END_SURFACE = 30, 130, 131

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
    """Write the cantilever as tetgen .node/.ele/.sur files, T10 elements, T6 top and T3 end faces."""
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

    surfaces = []

    def add_square(corner, index, quadratic):
        """Two outward-facing triangles, T6 or T3, on the square spanned by corner(0..1, 0..1)."""
        for tri in ((corner(0, 0), corner(1, 0), corner(1, 1)),
                    (corner(0, 0), corner(1, 1), corner(0, 1))):
            mids = [_node(mid(tri[a], tri[b])) for a, b in ((0, 1), (1, 2), (2, 0))] if quadratic else []
            surfaces.append((index, [_node(p) for p in tri] + mids))

    top, end = 2 * CELLS[2], 2 * CELLS[0]
    for ci, cj in itertools.product(range(CELLS[0]), range(CELLS[1])):
        add_square(lambda dx, dy: (2 * (ci + dx), 2 * (cj + dy), top), SURFACE, quadratic=True)
    # Nodes are numbered with z slowest, so the free end spans every rank's node block, whereas
    # the bottom face would lie on the first rank alone. It is linear because the Robin boundary
    # elements are three-node triangles.
    for cj, ck in itertools.product(range(CELLS[1]), range(CELLS[2])):
        add_square(lambda dy, dz: (end, 2 * (cj + dy), 2 * (ck + dz)), END_SURFACE, quadratic=False)

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
        for n, (index, s) in enumerate(surfaces, 1):
            f.write(f"{n} {' '.join(map(str, s))} {index} {index}\n")


# Parameters of each law with every modulus O(1), for the reasons given above. The fixtures have
# no bases file, so the fibre direction is x, along the cantilever, which bending stretches and
# compresses. Holzapfel keeps its default smoothed Heavyside switch.
MATERIALS = {
    "NeoHooke": "<NeoHooke><a>1</a><k>{kappa}</k></NeoHooke>",
    "Holzapfel": "<Holzapfel><a>1</a><b>1</b><af>1</af><bf>1</bf><as>0.5</as><bs>1</bs>"
                 "<afs>0.3</afs><bfs>1</bfs><kappa>{kappa}</kappa></Holzapfel>",
    "Guccione": "<Guccione><C>1</C><bf>8</bf><bt>2</bt><bfs>4</bfs><K>{kappa}</K></Guccione>",
    "MooneyRivlin": "<MooneyRivlin><C10>1</C10><B>{kappa}</B></MooneyRivlin>",
    "Usyk": "<Usyk><a>1</a><bff>8</bff><bss>2</bss><bnn>2</bnn><bfs>4</bfs><bfn>4</bfn><bns>2</bns>"
            "<k>{kappa}</k></Usyk>",
}


def _run(binary, cm_env, wd, element_type="T10P1", kappa=100, material="NeoHooke", solver="Static",
         env=None, check=True, ranks=None, fixture=FIXTURE):
    """Stage mesh and settings into wd, run on ranks MPI ranks (serially if None), return (process, vtu directory)."""
    (wd / "tetgen").mkdir()
    _write_mesh(wd / "tetgen")
    (wd / "Results").mkdir()
    text, n = re.subn(r"<NeoHooke>.*?</NeoHooke>", MATERIALS[material].format(kappa=kappa),
                      fixture.read_text(), flags=re.DOTALL)
    assert n == 1, f"{fixture.name}: cannot substitute the NeoHooke parameters"
    for old, new in (("<Type>T10P1</Type>", f"<Type>{element_type}</Type>"),
                     ("<Type>NeoHooke</Type>", f"<Type>{material}</Type>"),
                     ("<Type>Static</Type>", f"<Type>{solver}</Type>")):
        assert text.count(old) == 1, f"{fixture.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / fixture.name).write_text(text)
    proc = run_binary(binary("CardioMechanics"), ["-settings", fixture.name],
                      cwd=wd, env=env or cm_env, timeout=600, check=check, np=ranks)
    if check:
        assert "SIMULATION FAILED" not in proc.stdout, f"{element_type} kappa={kappa}\n{proc.stdout[-2000:]}"
    return proc, wd / "Results" / "cantilever_vtu"


def _final_points(vtu_dir):
    last = max(vtu_dir.glob("cantilever.*.vtu"), key=lambda p: int(p.stem.split(".")[1]))
    return read_vtu_points(last)


def _peak_displacement(vtu_dir):
    _, pts = _final_points(vtu_dir)
    _, ref = read_vtu_points(vtu_dir / "cantilever.0.vtu")
    return np.linalg.norm(pts - ref, axis=1).max()


def test_unknown_element_type_lists_p2p1(binary, cm_env, tmp_path):
    proc, _ = _run(binary, cm_env, tmp_path, element_type="T10Q1", check=False)
    assert proc.returncode != 0, f"expected a non-zero exit\n{proc.stdout[-2000:]}"
    assert "T10P1" in proc.stdout + proc.stderr, (proc.stdout + proc.stderr)[-2000:]


def _assert_refused(proc, *names):
    """The run aborted with an error message naming every one of names on one line."""
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"expected a non-zero exit\n{out[-2000:]}"
    assert any("error" in line.lower() and all(n in line for n in names) for line in out.splitlines()), \
        f"no error names {names}\n{out[-2000:]}"


@pytest.mark.parametrize("material", ["MooneyRivlin", "Usyk"])
def test_p2p1_refuses_material_without_mixed_formulation(binary, cm_env, tmp_path, material):
    proc, _ = _run(binary, cm_env, tmp_path, material=material, check=False)
    _assert_refused(proc, material, "T10P1")


def test_active_stress_estimator_refuses_p2p1(binary, cm_env, tmp_path):
    """The estimator differentiates nodal forces with respect to element tension, which has no
    counterpart for the pressure field."""
    proc, _ = _run(binary, cm_env, tmp_path, solver="ActiveStressEstimator", check=False)
    _assert_refused(proc, "Active Stress Estimator", "T10P1")


def test_p2p1_static_run_converges(binary, cm_env, tmp_path):
    pytest.importorskip("meshio")
    _, vtu_dir = _run(binary, cm_env, tmp_path)
    _, pts = _final_points(vtu_dir)
    _, ref = read_vtu_points(vtu_dir / "cantilever.0.vtu")
    deflection = np.abs(pts - ref).max()
    assert deflection > 1e-2, f"cantilever barely moved: max displacement {deflection:.3e}"


NP = 4
DEFORM_RTOL, DEFORM_ATOL = 1e-4, 1e-8     # as for the benchmark goldens


def _assert_parallel_matches_serial(binary, cm_env, tmp_path, fixture):
    """Run fixture serially and at NP ranks, assert the final shapes agree, return the serial vtu directory."""
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    points, vtu_dirs = {}, {}
    for ranks in (None, NP):
        wd = tmp_path / f"np{ranks or 1}"
        wd.mkdir(parents=True)
        _, vtu_dirs[ranks] = _run(binary, cm_env, wd, ranks=ranks, fixture=fixture)
        points[ranks] = _final_points(vtu_dirs[ranks])

    (pid, serial), (pid_parallel, parallel) = points[None], points[NP]
    assert np.array_equal(pid, pid_parallel), "point ordering differs between serial and parallel"
    if not np.allclose(parallel, serial, rtol=DEFORM_RTOL, atol=DEFORM_ATOL):
        d = np.linalg.norm(parallel - serial, axis=1)
        i = int(np.argmax(d))
        raise AssertionError(
            f"{fixture.name}: np={NP} differs from serial beyond rtol={DEFORM_RTOL} atol={DEFORM_ATOL}: "
            f"node PointID={int(pid[i])} parallel={parallel[i]} serial={serial[i]} |delta|={d[i]:.3e}")
    return vtu_dirs[None]


@pytest.mark.mpi
def test_p2p1_parallel_matches_serial(binary, cm_env, tmp_path):
    """Every rank but the first shifts its displacement unknowns by the pressure unknowns of the
    ranks before it, and needs the pressures of vertices it ghosts, so a parallel run that writes
    a global index without the offset, or misses a ghost, departs from the serial one."""
    _assert_parallel_matches_serial(binary, cm_env, tmp_path, FIXTURE)


MIN_ROBIN_EFFECT = 0.05     # relative change of the peak displacement


@pytest.mark.mpi
def test_p2p1_robin_boundary_parallel_matches_serial(binary, cm_env, tmp_path):
    """A Robin boundary adds its forces by global node rather than through the element mapping,
    from surface elements on every rank."""
    robin = _peak_displacement(_assert_parallel_matches_serial(binary, cm_env, tmp_path / "robin", ROBIN_FIXTURE))
    (tmp_path / "free").mkdir()
    _, vtu_dir = _run(binary, cm_env, tmp_path / "free")
    free = _peak_displacement(vtu_dir)
    assert abs(robin - free) > MIN_ROBIN_EFFECT * free, \
        f"Robin boundary barely acts: peak displacement {robin:.4e} with, {free:.4e} without"


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


def _gap_to_t10(binary, cm_env, tmp_path, kappa, material="NeoHooke"):
    """Largest P2P1-T10 node distance over the T10 peak displacement, and the PointID it occurs at."""
    vtu_dirs = {}
    for element_type in ("T10", "T10P1"):
        wd = tmp_path / f"{material}_{element_type}_{kappa}"
        wd.mkdir()
        _, vtu_dirs[element_type] = _run(binary, cm_env, wd, element_type=element_type, kappa=kappa,
                                         material=material)
    pid, t10 = _final_points(vtu_dirs["T10"])
    _, p2p1 = _final_points(vtu_dirs["T10P1"])
    _, ref = read_vtu_points(vtu_dirs["T10"] / "cantilever.0.vtu")
    d = np.linalg.norm(p2p1 - t10, axis=1)
    i = int(np.argmax(d))
    return d[i] / np.linalg.norm(t10 - ref, axis=1).max(), int(pid[i])


def test_p2p1_approaches_t10_as_kappa_decreases(binary, cm_env, tmp_path):
    pytest.importorskip("meshio")
    gaps = [(kappa, *_gap_to_t10(binary, cm_env, tmp_path, kappa)) for kappa in KAPPA_SWEEP]

    report = ", ".join(f"kappa={k}: {g:.2e} at PointID={n}" for k, g, n in gaps)
    assert all(a[1] < b[1] for a, b in zip(gaps, gaps[1:])), f"gap does not shrink as kappa decreases: {report}"
    assert gaps[0][1] < MAX_GAP_AT_SMALLEST_KAPPA, f"gap at the smallest kappa too large: {report}"


@pytest.mark.parametrize("material", ["Holzapfel", "Guccione"])
def test_p2p1_matches_t10_at_small_kappa(binary, cm_env, tmp_path, material):
    """T10 uses the law's full stress, P2P1 its isochoric stress plus the pressure field, so the two
    agree at small kappa only if the isochoric stress is the law's stress minus its volumetric part.
    Holzapfel's volumetric energy is kappa/4 (J^2 - 1 - 2 ln J) rather than the kappa/2 (J - 1)^2 of
    the perturbed constraint; the two differ at third order in J - 1, far below this tolerance."""
    pytest.importorskip("meshio")
    gap, pid = _gap_to_t10(binary, cm_env, tmp_path, KAPPA_SWEEP[0], material)
    assert gap < MAX_GAP_AT_SMALLEST_KAPPA, f"{material}: P2P1 differs from T10 by {gap:.2e} at PointID={pid}"
