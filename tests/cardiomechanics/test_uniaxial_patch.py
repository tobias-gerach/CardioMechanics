"""Homogeneous uniaxial tension of a distorted box against its closed-form solution.

The other element tests compare the code with itself. Here the box of helpers.box, held by
symmetry planes on its three faces through the origin and pulled by a follower pressure on the
opposite face, deforms homogeneously, x = diag(l1, l2, l3) X. That field lies in the span of T4,
T10 and P2P1 on any mesh of affine elements, the stress it produces is uniform, and the loaded face
stays planar and normal to the load, so the follower pressure is exactly the Cauchy traction t. A
consistent element therefore reproduces it at every node to solver precision however distorted
the mesh, whereas any error in the element or the law shows up as a discretization-sized misfit.

Neo-Hooke. With F = diag(lam, mu, mu), J = lam mu^2 and I1 = lam^2 + 2 mu^2, the PK2 stress
a J^(-2/3) (I - I1/3 C^-1) + kappa (J - 1) J C^-1 pushes forward to the Cauchy stress

    sigma_11 = a J^(-5/3) (lam^2 - I1/3) + kappa (J - 1)
    sigma_22 = a J^(-5/3) (mu^2  - I1/3) + kappa (J - 1).

The isochoric part is traceless, so sigma_11 + 2 sigma_22 = 3 kappa (J - 1). The lateral faces
are free, sigma_22 = 0, and sigma_11 = t, hence J = 1 + t / (3 kappa). With mu^2 = J / lam, the
difference sigma_11 - sigma_22 = t becomes the cubic

    lam^3 - (t J^(5/3) / a) lam - J = 0,

whose coefficients change sign exactly once, so it has exactly one positive root. The P2P1
pressure kappa (J - 1) equals t / 3, whatever kappa.

Holzapfel and Guccione are checked at the constitutive-law level, by the GoogleTests of
mechanics/tests/ConstitutiveModelTest.cpp.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from helpers.box import write_box
from helpers.compare import read_vtu_point_field, read_vtu_points
from helpers.materials import LAWS, material_block
from helpers.run import assert_refused, run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "uniaxial_box.xml"
TRACTION = float(re.search(r"<Amplitude>(.*?)</Amplitude>", FIXTURE.read_text()).group(1))
STEPS = 4                                  # StopTime / TimeStep of the fixture, one export per step
ELEMENTS = {"T4": "T3", "T10": "T6", "T10P1": "T6"}     # element type: loaded face type
# Nearly incompressible, Poisson's ratio 0.4995. J - 1 is 3e-4, so the kappa term moves the lateral
# stretch far beyond the tolerance.
KAPPA = 1000
# Element type and Mesh.QuadratureDegree of each run, None for the default. Degree 5 selects the
# 14-point rule of P2P1.
RUNS = [("T4", None), ("T10", None), ("T10P1", None), ("T10P1", 5)]

# The VTU points are single precision: rounding moves a coordinate below 2 by at most 1.2e-7, in
# the reference and the deformed frame alike, and the expected position inherits the reference
# error scaled by a stretch below 1.5. The solver stops at Precision 1e-12, far below that.
# Observed at most 1.1e-7; dropping the deviatoric projection of Neo-Hooke misses by 4e-2.
POSITION_TOL = 1e-6
# The pressure is written in double precision, so only the fixture's solver Precision of 1e-12
# limits it. Observed 1.2e-13.
PRESSURE_RTOL = 1e-10


def analytic_stretches(traction, a, kappa):
    """Axial and lateral stretch of the Neo-Hooke bar under Cauchy traction, from the cubic above."""
    J = 1 + traction / (3 * kappa)
    roots = np.roots([1, 0, -traction * J ** (5 / 3) / a, -J])
    lam = roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)].real
    assert len(lam) == 1, f"expected one positive root, found {roots}"
    return lam[0], np.sqrt(J / lam[0])


def _run_box(binary, cm_env, wd, element_type, law, basis=None, degree=None, check=True):
    """Stage the box in wd with the fibre basis of write_box, if any, and run it under one element
    type, law and Mesh.QuadratureDegree (the default if None). Returns the process."""
    (wd / "tetgen").mkdir()
    (wd / "Results").mkdir()
    write_box(wd / "tetgen", quadratic=element_type != "T4", basis=basis)
    text, n = re.subn(r"<NeoHooke>.*?</NeoHooke>", material_block(law, KAPPA), FIXTURE.read_text(), flags=re.DOTALL)
    assert n == 1, f"{FIXTURE.name}: cannot substitute the NeoHooke parameters"
    substitutions = [("<Type>T10P1</Type>", f"<Type>{element_type}</Type>"),
                     ("<Type>T6</Type>", f"<Type>{ELEMENTS[element_type]}</Type>"),
                     ("<Type>NeoHooke</Type>", f"<Type>{law}</Type>")]
    if basis is not None:
        substitutions.append(("<Surfaces>./tetgen/box.sur</Surfaces>",
                              "<Surfaces>./tetgen/box.sur</Surfaces><Bases>./tetgen/box.bases</Bases>"))
    if degree is not None:
        substitutions.append(("<Format>Tetgen</Format>",
                              f"<Format>Tetgen</Format><QuadratureDegree>{degree}</QuadratureDegree>"))
    for old, new in substitutions:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / FIXTURE.name).write_text(text)
    proc = run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd, env=cm_env, timeout=600,
                      check=check)
    if check:
        assert "SIMULATION FAILED" not in proc.stdout, proc.stdout[-2000:]
    return proc


@pytest.fixture(scope="module", params=RUNS, ids=lambda p: p[0] + (f"-degree{p[1]}" if p[1] else ""))
def solution(request, binary, cm_env, tmp_path_factory):
    """Run the Neo-Hooke box under one element type and quadrature degree. Returns (label, PointID,
    reference points, final points, final pressure or None, expected stretches)."""
    pytest.importorskip("meshio")
    element_type, degree = request.param
    label = f"{element_type}{' degree ' + str(degree) if degree else ''}"
    wd = tmp_path_factory.mktemp(f"{element_type}_{degree}")
    _run_box(binary, cm_env, wd, element_type, "NeoHooke", degree=degree)

    vtu_dir = wd / "Results" / "box_vtu"
    last = vtu_dir / f"box.{STEPS}.vtu"
    assert last.exists(), f"{label}: run stopped before its stop time"
    pid, ref = read_vtu_points(vtu_dir / "box.0.vtu")
    _, cur = read_vtu_points(last)
    pressure = read_vtu_point_field(last, "Pressure")[1] if element_type == "T10P1" else None
    lam, mu = analytic_stretches(TRACTION, LAWS["NeoHooke"][1]["a"], KAPPA)
    return label, pid, ref.astype(float), cur.astype(float), pressure, np.array([lam, mu, mu])


def _affine_fit(ref, cur):
    """Least-squares affine map from reference to current points, and the points it maps to."""
    A = np.hstack([ref, np.ones((len(ref), 1))])
    coef = np.linalg.lstsq(A, cur, rcond=None)[0]
    return coef[:3].T, A @ coef


def _assert_nodes_match(label, pid, cur, expected, tol, what):
    d = np.linalg.norm(cur - expected, axis=1)
    i = int(np.argmax(d))
    assert d[i] <= tol, (f"{label}: {what} beyond {tol:.1e}, worst node PointID={int(pid[i])} "
                         f"at {cur[i]}, expected {expected[i]}, |delta|={d[i]:.3e}")


def test_displacement_is_linear(solution):
    """Separates an inconsistent element, whose field bends between nodes, from a wrong but
    consistently applied law, which only gets the stretches wrong."""
    label, pid, ref, cur, _, _ = solution
    _, fit = _affine_fit(ref, cur)
    _assert_nodes_match(label, pid, cur, fit, POSITION_TOL, "displacement not linear")


def test_reproduces_analytic_stretches(solution):
    label, pid, ref, cur, _, stretches = solution
    F, _ = _affine_fit(ref, cur)
    _assert_nodes_match(label, pid, cur, ref * stretches, POSITION_TOL,
                        f"fitted F = {np.array2string(F, precision=7)} misses stretches "
                        f"{np.array2string(stretches, precision=7)}")


def test_p2p1_pressure_is_uniform_kappa_j_minus_one(solution):
    """Checked against the reference J: the single-precision points would leave kappa (J - 1)
    only about 1e-7 kappa accurate."""
    label, pid, _, _, pressure, stretches = solution
    if pressure is None:
        pytest.skip("displacement-only element, no pressure field")
    expected = KAPPA * (np.prod(stretches) - 1)
    _assert_nodes_match(label, pid, pressure[:, None], np.full((len(pressure), 1), expected),
                        PRESSURE_RTOL * abs(expected), "pressure not kappa (J - 1)")


def test_unsupported_quadrature_degree_is_refused(binary, cm_env, tmp_path):
    proc = _run_box(binary, cm_env, tmp_path, "T10P1", "NeoHooke", degree=3, check=False)
    assert_refused(proc, "QuadratureDegree", "3")


def test_degree_5_refuses_bases_that_differ_between_quadrature_points(binary, cm_env, tmp_path):
    """The bases file holds frames at the centroid and the four points of the default rule, which
    are not points of the 14-point rule."""
    frames = [np.eye(3)] + [np.eye(3)[[1, 2, 0]]] * 4
    proc = _run_box(binary, cm_env, tmp_path, "T10P1", "Holzapfel", basis=frames, degree=5, check=False)
    assert_refused(proc, "QuadratureDegree", "bases")
