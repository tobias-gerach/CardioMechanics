"""Inflation of a thick-walled sphere: convergence of P2P1 to the exact radial solution.

The patch tests and the rigid rotation are reproduced exactly by any consistent element, so they
cannot show whether P2P1 converges at the rate of the pairing. A wrong Jacobian block, a mis-signed
coupling or an under-integrated constraint can pass all of them and still cost accuracy here.

The octant INNER <= |X| <= OUTER of helpers.sphere is held by symmetry planes on its three plane
faces, and a follower pressure P inflates it from the inner surface. The exact deformation is
radial, x = r(R) X / R, which the symmetry planes leave free. With the principal stretches
l_r = r' and l_t = r / R, the Neo-Hooke law of the code (isochoric/volumetric split, shear modulus
a, bulk modulus kappa) gives the Cauchy stresses

    sigma_r = a J^(-5/3) (l_r^2 - I1/3) + kappa (J - 1),   J = l_r l_t^2,  I1 = l_r^2 + 2 l_t^2,

and sigma_t the same with l_t^2 in the bracket. Radial equilibrium, d sigma_r/dr + 2 (sigma_r -
sigma_t) / r = 0, becomes d sigma_r/dR = -2 r' (sigma_r - sigma_t) / r in the reference radius,
and the chain rule through l_r and l_t turns it into a second-order equation for r(R). The follower
pressure is the Cauchy traction on the deformed surface, so sigma_r = -P at INNER and 0 at OUTER.
radial_solution shoots from OUTER, where sigma_r = 0 fixes l_r for a trial r(OUTER), and solves
for the r(OUTER) that meets sigma_r = -P at INNER; r(INNER) is the inner radius under P. The partial
derivatives of sigma_r are taken by complex step, which is exact to round-off.

The reference is compressible on purpose. Eliminating the P2P1 pressure pointwise from its
perturbed constraint gives back the volumetric energy kappa/2 (J - 1)^2 (ADR-0002), so P2P1
converges to exactly this solution at any kappa, pressure p = kappa (J - 1) included, and there is
no compressibility error to bound. T10 converges to the same solution, but locks at large kappa.
As kappa grows the reference approaches the closed form of the incompressible sphere (Green and
Zerna; Rivlin): r^3 = R^3 + r_i^3 - INNER^3, sigma_t - sigma_r = a (l^2 - l^-4) with l = r / R, and
since dr / r = -dl / (l (l^3 - 1)),

    P = 2 a [1/l + 1/(4 l^4)] evaluated from l_i = r_i / INNER down to l_o = r_o / OUTER.

The errors are integrated over the curved elements, against the reference at the image of each
quadrature point, so they measure the solution on the mesh's own P2 approximation of the shell,
whose O(h^3) geometric error is at or below every expected order. The deformed points are single
precision, which moves them by at most 1.2e-7, far below the finest error.

The rates are asserted between the two finest levels of each family, because the coarsest mesh has
only two elements across the wall.

P2P1 runs under both quadrature rules of Mesh.QuadratureDegree: the default 4-point rule of degree
2 and the 14-point rule of degree 5. J is cubic on an affine element, so the constraint integrand
N_a (J - 1 - p / kappa) is of degree 4, which only the second integrates exactly. The rules also
differ in their weights: the default takes the determinant of the element map at the centroid, the
14-point rule at each point. Only the determinant matters here. On the curved elements at the
spheres the centroid determinant nearly doubles the finest displacement L2 and pressure errors, whereas the 4-point
rule with pointwise determinants matches the 14-point rule to 0.6%.

Serial runtime, P2P1: 0.3 s, 1.6 s and 27 s at sizes 0.5, 0.25 and 0.125; the fourth level, in the
slow family, takes about 20 min. T10 needs 87 s at size 0.125 and is slow as well, and so is every
run under the 14-point rule.
"""
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from helpers.box import T6_EDGES, T10_EDGES
from helpers.compare import read_vtu_point_field, read_vtu_points
from helpers.gmsh_tetgen import t10_quadrature
from helpers.run import run_binary
from helpers.sphere import CAVITY, INNER, OUTER, write_sphere_octant

FIXTURE = Path(__file__).parent / "fixtures" / "sphere_octant.xml"
PRESSURE = -float(re.search(r"<Amplitude>(.*?)</Amplitude>", FIXTURE.read_text()).group(1))
SHEAR = float(re.search(r"<a>(.*?)</a>", FIXTURE.read_text()).group(1))
KAPPA = float(re.search(r"<k>(.*?)</k>", FIXTURE.read_text()).group(1))
SIZES = (0.5, 0.25, 0.125, 0.0625)         # gmsh element sizes of the mesh family
DEGREES = (2, 5)                           # Mesh.QuadratureDegree
# P2P1 orders: displacement O(h^3) in L2 and O(h^2) in H1, pressure O(h^2) in L2.
ORDERS = {"displacement L2": 3, "displacement H1": 2, "pressure L2": 2}
# Before the asymptotic range the rates fall short of the orders by up to 0.24 on the coarse family
# and 0.20 on the fine one, both in the pressure. The tolerance still fails the loss of half an order, as from a boundary,
# constraint or load integration error.
RATE_TOL = 0.3
# The reference is not polynomial and the elements are curved, so no rule is exact. Against a
# degree 10 rule the norms move by at most 2e-5 relative, far below the change between levels.
QUADRATURE = "Gauss6"
COMPLEX_STEP = 1e-30
# Faces of a T10: three vertices, then the mid-edge nodes of their edges in T6 order.
T10_FACES = ((0, 1, 2, 4, 5, 6), (0, 1, 3, 4, 8, 7), (0, 2, 3, 6, 9, 7), (1, 2, 3, 5, 9, 8))


def cauchy_stress(l_r, l_t, a, kappa):
    """Radial and hoop Cauchy stress of the Neo-Hooke law under the principal stretches (l_r, l_t, l_t)."""
    J = l_r * l_t ** 2
    iso = a * J ** (-5 / 3)
    I1 = l_r ** 2 + 2 * l_t ** 2
    return iso * (l_r ** 2 - I1 / 3) + kappa * (J - 1), iso * (l_t ** 2 - I1 / 3) + kappa * (J - 1)


def radial_solution(pressure, a, kappa):
    """The exact deformation of the shell under the inner pressure, as a function of R returning the
    rows r(R) and r'(R)."""
    def radial_stretch(l_t, sigma_r):
        return brentq(lambda l_r: cauchy_stress(l_r, l_t, a, kappa)[0] - sigma_r, 0.2, 5, xtol=1e-15)

    def rhs(R, y):
        r, l_r = y
        l_t = r / R
        sigma_r, sigma_t = cauchy_stress(l_r, l_t, a, kappa)
        ds_dlr, ds_dlt = (cauchy_stress(l_r + 1j * COMPLEX_STEP * e[0], l_t + 1j * COMPLEX_STEP * e[1], a, kappa)[0].imag
                          / COMPLEX_STEP for e in np.eye(2))
        return [l_r, (-2 * l_r * (sigma_r - sigma_t) / r - ds_dlt * (l_r - l_t) / R) / ds_dlr]

    def shoot(r_outer):
        return solve_ivp(rhs, (OUTER, INNER), [r_outer, radial_stretch(r_outer / OUTER, 0)],
                         method="DOP853", rtol=1e-13, atol=1e-14, dense_output=True)

    def misfit(r_outer):
        r, l_r = shoot(r_outer).y[:, -1]
        return cauchy_stress(l_r, r / INNER, a, kappa)[0] + pressure

    # An undeformed outer surface leaves the whole shell undeformed, below any positive load; a 20%
    # hoop stretch there carries 0.8 a in the incompressible limit, above the loads used here.
    return shoot(brentq(misfit, OUTER, 1.2 * OUTER, xtol=1e-15)).sol


def rivlin_inner_radius(pressure, a):
    """Deformed inner radius of the incompressible Neo-Hooke shell, from the closed form above."""
    def load(l_inner):
        l_outer = np.cbrt(OUTER ** 3 + INNER ** 3 * (l_inner ** 3 - 1)) / OUTER
        return 2 * a * sum(s * (1 / l + 1 / (4 * l ** 4)) for s, l in ((1, l_outer), (-1, l_inner)))
    return INNER * brentq(lambda l: load(l) - pressure, 1, 2, xtol=1e-15)


def _read_mesh(directory):
    """Node coordinates and fixation flags, T10 elements and the rows of the surface file, 0-based."""
    nodes = np.loadtxt(directory / "sphere.node", skiprows=1)
    elements = np.loadtxt(directory / "sphere.ele", skiprows=1, dtype=int)[:, 1:11] - 1
    rows = np.loadtxt(directory / "sphere.sur", skiprows=1, dtype=int)
    return nodes[:, 1:4], nodes[:, 4].astype(int), elements, rows


def discretization_errors(directory, element_type, reference):
    """Errors of the final VTU of the run in directory against the reference, integrated over the
    curved elements by QUADRATURE: displacement in L2 and H1 seminorm, and for P2P1 the pressure in
    L2. Returns them as a dict, with the mean element size (volume / elements)^(1/3) as "h"."""
    X, _, elements, _ = _read_mesh(directory / "tetgen")
    vtu_dir = directory / "Results" / "sphere_vtu"
    last = vtu_dir / "sphere.1.vtu"            # one export at the fixture's stop time
    assert last.exists(), f"{element_type}: run stopped before its stop time"
    pid, x0 = read_vtu_points(vtu_dir / "sphere.0.vtu")
    # PointID is the 0-based tetgen index; the single-precision reference points confirm it.
    assert np.array_equal(pid, np.arange(len(X))) and np.abs(x0 - X).max() < 2e-7
    u = read_vtu_points(last)[1].astype(float) - X

    weights, N, dN, L = t10_quadrature(QUADRATURE)
    Xq = np.einsum("qn,end->eqd", N, X[elements])
    dX = np.einsum("qnk,end->eqdk", dN, X[elements])
    det = np.linalg.det(dX)
    assert det.min() > 0, "an element is inverted at a quadrature point"
    dV = weights * det
    du = np.einsum("qnk,end->eqdk", dN, u[elements]) @ np.linalg.inv(dX)

    # Points of elements on the spheres can lie O(h^3) outside the shell, where the dense output of
    # the reference extends smoothly.
    R = np.linalg.norm(Xq, axis=-1)
    r, l_r = reference(R.ravel()).reshape(2, *R.shape)
    e = Xq / R[..., None]
    radial = e[..., :, None] * e[..., None, :]
    F = l_r[..., None, None] * radial + (r / R)[..., None, None] * (np.eye(3) - radial)

    def norm(misfit):
        return np.sqrt(np.sum(dV * (misfit.reshape(*dV.shape, -1) ** 2).sum(axis=-1)))

    errors = {"h": (dV.sum() / len(elements)) ** (1 / 3),
              "displacement L2": norm(np.einsum("qn,end->eqd", N, u[elements]) - (r / R - 1)[..., None] * Xq),
              "displacement H1": norm(du - (F - np.eye(3)))}
    if element_type == "T10P1":
        pressure = read_vtu_point_field(last, "Pressure")[1]
        errors["pressure L2"] = norm(np.einsum("qa,ea->eq", L, pressure[elements[:, :4]])
                                     - KAPPA * (l_r * (r / R) ** 2 - 1))
    return errors


def rates(errors):
    """Observed order of each error between consecutive levels."""
    log_h = np.log([e["h"] for e in errors])
    return {k: np.diff(np.log([e[k] for e in errors])) / np.diff(log_h) for k in errors[0] if k != "h"}


def _table(label, errors):
    lines = [f"{label}: " + "  ".join(f"{k:>16}" for k in errors[0])]
    lines += ["  " + "  ".join(f"{v:16.4e}" for v in e.values()) for e in errors]
    lines += ["  rates " + "  ".join(f"{k}: {np.array2string(v, precision=2)}" for k, v in rates(errors).items())]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def errors_at(binary, cm_env, tmp_path_factory):
    """Discretization errors of one element type at one mesh size, each run once per module."""
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")
    reference = radial_solution(PRESSURE, SHEAR, KAPPA)
    cache = {}

    def run(element_type, size, degree=DEGREES[0]):
        if (element_type, size, degree) not in cache:
            wd = tmp_path_factory.mktemp(f"{element_type}_{size}_{degree}")
            (wd / "tetgen").mkdir()
            (wd / "Results").mkdir()
            write_sphere_octant(wd / "tetgen", size)
            text = FIXTURE.read_text()
            for old, new in [("<Type>T10P1</Type>", f"<Type>{element_type}</Type>"),
                             ("<Format>Tetgen</Format>",
                              f"<Format>Tetgen</Format><QuadratureDegree>{degree}</QuadratureDegree>")]:
                assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
                text = text.replace(old, new)
            (wd / FIXTURE.name).write_text(text)
            proc = run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd, env=cm_env, timeout=7200)
            assert "SIMULATION FAILED" not in proc.stdout, proc.stdout[-2000:]
            cache[element_type, size, degree] = discretization_errors(wd, element_type, reference)
        return cache[element_type, size, degree]
    return run


def _assert_mid_nodes_near_midpoints(X, cells, edges):
    """Checks the node order: the mid-edge nodes, after the vertices, each lie near their own edge."""
    first = cells.shape[1] - len(edges)
    for k, (i, j) in enumerate(edges):
        a, b, m = X[cells[:, i]], X[cells[:, j]], X[cells[:, first + k]]
        assert np.all(np.linalg.norm(m - (a + b) / 2, axis=1) < 0.1 * np.linalg.norm(b - a, axis=1))


def test_octant_mesh_is_curved_and_constrained(tmp_path):
    pytest.importorskip("gmsh")
    write_sphere_octant(tmp_path, 0.5)
    X, fixed, elements, rows = _read_mesh(tmp_path)
    assert np.all(rows[:, 7:] == CAVITY)
    loaded = rows[:, 1:7] - 1
    _assert_mid_nodes_near_midpoints(X, elements, T10_EDGES)
    _assert_mid_nodes_near_midpoints(X, loaded, T6_EDGES)
    radius = np.linalg.norm(X, axis=1)
    assert radius.min() > INNER - 1e-12 and radius.max() < OUTER + 1e-12

    # Boundary faces belong to one element only. Every node of those on either sphere, mid-edge
    # nodes included, lies on it rather than on a chord, and the loaded faces are those on the inner one.
    faces = elements[:, T10_FACES].reshape(-1, 6)
    _, first, count = np.unique(np.sort(faces[:, :3], axis=1), axis=0, return_index=True, return_counts=True)
    boundary = faces[first[count == 1]]
    for sphere in (INNER, OUTER):
        on = np.isclose(radius[boundary[:, :3]], sphere, atol=1e-12).all(axis=1)
        assert on.any()
        np.testing.assert_allclose(radius[boundary[on]], sphere, atol=1e-12)
    inner = boundary[np.isclose(radius[boundary[:, :3]], INNER, atol=1e-12).all(axis=1)]
    np.testing.assert_array_equal(np.unique(np.sort(inner, axis=1), axis=0), np.unique(np.sort(loaded, axis=1), axis=0))

    # Positive vertex volumes, and loaded faces whose normals point out of the body, into the cavity.
    v = X[elements[:, :4]]
    assert np.all(np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0]) > 0)
    t = X[loaded[:, :3]]
    assert np.all(np.einsum("ij,ij->i", np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), t.mean(axis=1)) < 0)

    # Component i of every node on the plane X_i = 0 is fixed, and no other.
    for i in range(3):
        np.testing.assert_array_equal((fixed >> i) & 1, np.abs(X[:, i]) < 1e-12)


def test_reference_approaches_rivlin_closed_form():
    """At kappa = 1e6 the compressible reference is 8e-8 from the incompressible limit, with round-off
    amplified by kappa, where a wrong factor in the law would show up at 1e-2."""
    reference = radial_solution(PRESSURE, SHEAR, 1e6)
    R = np.linspace(INNER, OUTER, 11)
    r_inner = rivlin_inner_radius(PRESSURE, SHEAR)
    np.testing.assert_allclose(reference(R)[0], np.cbrt(R ** 3 + r_inner ** 3 - INNER ** 3), atol=1e-6)


def test_reference_approaches_lame_solution():
    """Under a small load the reference must reduce to linear elasticity with shear modulus a and
    bulk modulus kappa, u = A R + B / R^2 with sigma_rr = 3 kappa A - 4 a B / R^3. At kappa = a the
    bulk term is as large as the shear term, which checks the compressible part the incompressible
    limit misses. The nonlinear terms are O(pressure) relative, so 1e-6 here."""
    pressure, kappa = 1e-6, SHEAR
    A, B = np.linalg.solve([[3 * kappa, -4 * SHEAR / INNER ** 3], [3 * kappa, -4 * SHEAR / OUTER ** 3]], [-pressure, 0])
    R = np.linspace(INNER, OUTER, 11)
    np.testing.assert_allclose(radial_solution(pressure, SHEAR, kappa)(R)[0] - R, A * R + B / R ** 2, rtol=1e-5)


@pytest.mark.parametrize("norm", ["displacement L2", "displacement H1", "pressure L2"])
@pytest.mark.parametrize("sizes", [SIZES[:3], pytest.param(SIZES[1:], marks=pytest.mark.slow)], ids=["coarse", "fine"])
@pytest.mark.parametrize("degree", [DEGREES[0], pytest.param(DEGREES[1], marks=pytest.mark.slow)],
                         ids=[f"degree{d}" for d in DEGREES])
def test_p2p1_converges_at_expected_rate(errors_at, degree, sizes, norm):
    """Asserted on the finest pair of levels; every rate is shown with -rP."""
    errors = [errors_at("T10P1", s, degree) for s in sizes]
    label = f"P2P1 degree {degree}"
    print(_table(label, errors))
    rate = rates(errors)[norm][-1]
    assert rate >= ORDERS[norm] - RATE_TOL, (f"{norm} converges at {rate:.2f}, below the P2P1 order "
                                             f"{ORDERS[norm]} less {RATE_TOL}\n{_table(label, errors)}")


@pytest.mark.slow
def test_quadrature_degrees_are_compared(errors_at):
    """The P2P1 errors under both rules, and their ratio at each level, shown with -rP."""
    errors = {d: [errors_at("T10P1", s, d) for s in SIZES] for d in DEGREES}
    for d in DEGREES:
        print(_table(f"P2P1 degree {d}", errors[d]))
    norms = [k for k in errors[DEGREES[0]][0] if k != "h"]
    print(f"degree {DEGREES[1]} / degree {DEGREES[0]}: " + "  ".join(f"{k:>16}" for k in norms))
    for low, high in zip(errors[DEGREES[0]], errors[DEGREES[1]]):
        print("  " + "  ".join(f"{high[k] / low[k]:16.4f}" for k in norms))


@pytest.mark.slow
def test_t10_rates_are_reported(errors_at):
    """T10 converges to the same solution but locks at this kappa, which also slows its Newton
    iteration; its rates are shown with -rP, not asserted."""
    print(_table("T10", [errors_at("T10", s) for s in SIZES[:3]]))
