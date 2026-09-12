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

Holzapfel and Guccione. With the fibre, sheet and sheet-normal directions along the box axes, the
laws are orthotropic in the principal axes of F, so the field is still homogeneous with zero shear
stress, but the two lateral stretches differ in general. Each principal Cauchy stress follows from the strain
energy as sigma_i = (l_i / J) dW/dl_i. The reference therefore needs only W, not the stress
derivation the code implements, and exposes an error there, such as a dropped derivative of
Holzapfel's fibre switch. With W = W_0 + U(J), sigma_i = (l_i / J) dW_0/dl_i + p where p = U'(J),
and sigma = (t, 0, 0) is solved for l1, l2 and p, with l3 = J(p) / (l1 l2). U is kappa/2 (J - 1)^2,
so p = kappa (J - 1), for Guccione and for every law under P2P1, whose perturbed constraint
replaces the law's own U. Holzapfel's own U = kappa/4 (J^2 - 1 - 2 ln J), used by T4 and T10, gives
p = kappa/2 (J - 1/J) instead; at kappa = 10 the two differ by 3e-4 in the stretches. Fibres along
the load are stretched and fibres across it compressed, so both flanks of Holzapfel's switch are
exercised. The fibre-sheet coupling terms vanish under this load.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from helpers.box import write_box
from helpers.compare import read_vtu_point_field, read_vtu_points
from helpers.run import run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "uniaxial_box.xml"
TRACTION = float(re.search(r"<Amplitude>(.*?)</Amplitude>", FIXTURE.read_text()).group(1))
STEPS = 4                                  # StopTime / TimeStep of the fixture, one export per step
ELEMENTS = {"T4": "T3", "T10": "T6", "T10P1": "T6"}     # element type: loaded face type
# Compressible and nearly incompressible, Poisson's ratio 0.45 and 0.4995. J - 1 is 3e-2 and 3e-4,
# so the kappa term moves the lateral stretch far beyond the tolerance at either.
KAPPAS = (10, 1000)
# Settings of each law, with the tag of its bulk modulus; every modulus is O(1). Holzapfel's k is
# the slope of its smoothed Heavyside switch. At 10 rather than the default 100, the fibre I4 of
# 1.25 along the load and 0.80 across it lies on the flank of the switch, where its derivative adds
# about 2e-2 to the fibre and sheet stress; at 100 it would add at most 7e-5.
LAWS = {
    "NeoHooke": ("k", {"a": 1}),
    "Holzapfel": ("kappa", {"a": 1, "b": 1, "af": 1, "bf": 1, "as": 0.5, "bs": 1, "afs": 0.3, "bfs": 1,
                            "k": 10}),
    "Guccione": ("K", {"C": 1, "bf": 8, "bt": 2, "bfs": 4}),
}
FIBRES = {"along": (0, 1, 2), "across": (1, 2, 0)}      # box axis of the fibre, sheet and sheet normal
CASES = [("NeoHooke", None)] + [(law, fibres) for law in ("Holzapfel", "Guccione") for fibres in FIBRES]

# The VTU points are single precision: rounding moves a coordinate below 2 by at most 1.2e-7, in
# the reference and the deformed frame alike, and the expected position inherits the reference
# error scaled by a stretch below 1.5. The solver stops at Precision 1e-12, far below that.
# Observed at most 1.1e-7; dropping the deviatoric projection of Neo-Hooke misses by 4e-2, and
# dropping the derivative of Holzapfel's switch fails every Holzapfel case.
POSITION_TOL = 1e-6
# The pressure is written in double precision, so only the solver Precision limits it. Under
# Holzapfel the non-deviatoric fibre term couples the pressure to the stretches, so it carries the
# stopping error of the iteration, which the fixture's Precision of 1e-12 keeps far below this
# tolerance. Observed 2.6e-13.
PRESSURE_RTOL = 1e-10
COMPLEX_STEP = 1e-30


def analytic_stretches(traction, a, kappa):
    """Axial and lateral stretch of the Neo-Hooke bar under Cauchy traction, from the cubic above."""
    J = 1 + traction / (3 * kappa)
    roots = np.roots([1, 0, -traction * J ** (5 / 3) / a, -J])
    lam = roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)].real
    assert len(lam) == 1, f"expected one positive root, found {roots}"
    return lam[0], np.sqrt(J / lam[0])


def neo_hooke_energy(l1, l2, l3, p):
    return p["a"] / 2 * ((l1 * l2 * l3) ** (-2 / 3) * (l1 ** 2 + l2 ** 2 + l3 ** 2) - 3)


def holzapfel_energy(lf, ls, ln, p):
    """W_0 over the stretches along fibre, sheet and sheet normal. The fibre and sheet terms are
    in the full I4, not its isochoric part, as the code implements them."""
    J = lf * ls * ln
    I1 = J ** (-2 / 3) * (lf ** 2 + ls ** 2 + ln ** 2)

    def switched(a, b, I4):
        return a / (2 * b) * (np.exp(b * (I4 - 1) ** 2) - 1) / (1 + np.exp(-p["k"] * (I4 - 1)))

    return (p["a"] / (2 * p["b"]) * (np.exp(p["b"] * (I1 - 3)) - 1)
            + switched(p["af"], p["bf"], lf ** 2) + switched(p["as"], p["bs"], ls ** 2))


def guccione_energy(lf, ls, ln, p):
    """W_0 over the stretches along fibre, sheet and sheet normal, in the isochoric Green strain."""
    J = lf * ls * ln
    ef, es, en = (0.5 * (J ** (-2 / 3) * l ** 2 - 1) for l in (lf, ls, ln))
    return p["C"] / 2 * (np.exp(p["bf"] * ef ** 2 + p["bt"] * (es ** 2 + en ** 2)) - 1)


ENERGIES = {"Holzapfel": holzapfel_energy, "Guccione": guccione_energy}


def j_of_pressure(law, element_type, kappa):
    """Inverse of p = U'(J), the volumetric Cauchy pressure of law under element_type."""
    if law == "Holzapfel" and element_type != "T10P1":
        return lambda p: p / kappa + np.hypot(p / kappa, 1)
    return lambda p: 1 + p / kappa


def uniaxial_stretches(energy, j_of_p, traction):
    """Stretches along x, y and z of the bar under Cauchy traction along x, for W_0 = energy(l1, l2,
    l3) and J = j_of_p(p). Newton on l1, l2 and p: with J as the unknown instead of p, the kappa
    stiffness of U would amplify the round-off of the residual into the step."""
    def stretches(x):
        return np.array([x[0], x[1], j_of_p(x[2]) / (x[0] * x[1])])

    def residual(x):
        l = stretches(x)
        # The complex-step derivative has no subtractive cancellation, so it is exact to round-off.
        dW = np.array([energy(*(l + 1j * COMPLEX_STEP * e)).imag / COMPLEX_STEP for e in np.eye(3)])
        return l * dW / np.prod(l) + x[2] - [traction, 0, 0]

    # The Jacobian only sets the convergence rate, so central differences at the square root of
    # machine precision suffice; the root is as exact as the residual.
    step = 1e-7
    x = np.array([1.0, 1.0, 0.0])
    for _ in range(50):
        jacobian = np.column_stack([(residual(x + h) - residual(x - h)) / (2 * step) for h in step * np.eye(3)])
        dx = np.linalg.solve(jacobian, residual(x))
        x -= dx
        if np.abs(dx).max() < 1e-14:
            return stretches(x)
    raise AssertionError(f"Newton did not converge, last step {dx}")


def reference_stretches(element_type, law, fibres, kappa):
    params = LAWS[law][1]
    if law == "NeoHooke":
        lam, mu = analytic_stretches(TRACTION, params["a"], kappa)
        return np.array([lam, mu, mu])
    axes = FIBRES[fibres]
    return uniaxial_stretches(lambda *l: ENERGIES[law](*(l[i] for i in axes), params),
                              j_of_pressure(law, element_type, kappa), TRACTION)


def material_block(law, kappa):
    tag, params = LAWS[law]
    return f"<{law}>" + "".join(f"<{k}>{v}</{k}>" for k, v in {**params, tag: kappa}.items()) + f"</{law}>"


@pytest.fixture(scope="module", params=[(e, law, f, k) for e in ELEMENTS for law, f in CASES for k in KAPPAS],
                ids=lambda p: f"{p[0]}-{p[1]}{'-fibres-' + p[2] if p[2] else ''}-kappa{p[3]}")
def solution(request, binary, cm_env, tmp_path_factory):
    """Run the box under one element type, law, fibre direction and kappa. Returns (label, kappa,
    PointID, reference points, final points, final pressure or None, expected stretches)."""
    pytest.importorskip("meshio")
    element_type, law, fibres, kappa = request.param
    label = f"{element_type} {law}{' fibres ' + fibres if fibres else ''} kappa={kappa}"
    wd = tmp_path_factory.mktemp(f"{element_type}_{law}_{fibres}_{kappa}")
    (wd / "tetgen").mkdir()
    (wd / "Results").mkdir()
    write_box(wd / "tetgen", quadratic=element_type != "T4",
              basis=np.eye(3)[list(FIBRES[fibres])] if fibres else None)
    text, n = re.subn(r"<NeoHooke>.*?</NeoHooke>", material_block(law, kappa), FIXTURE.read_text(), flags=re.DOTALL)
    assert n == 1, f"{FIXTURE.name}: cannot substitute the NeoHooke parameters"
    substitutions = [("<Type>T10P1</Type>", f"<Type>{element_type}</Type>"),
                     ("<Type>T6</Type>", f"<Type>{ELEMENTS[element_type]}</Type>"),
                     ("<Type>NeoHooke</Type>", f"<Type>{law}</Type>")]
    if fibres:
        substitutions.append(("<Surfaces>./tetgen/box.sur</Surfaces>",
                              "<Surfaces>./tetgen/box.sur</Surfaces><Bases>./tetgen/box.bases</Bases>"))
    for old, new in substitutions:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / FIXTURE.name).write_text(text)
    proc = run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd, env=cm_env, timeout=600)
    assert "SIMULATION FAILED" not in proc.stdout, proc.stdout[-2000:]

    vtu_dir = wd / "Results" / "box_vtu"
    last = vtu_dir / f"box.{STEPS}.vtu"
    assert last.exists(), f"{label}: run stopped before its stop time"
    pid, ref = read_vtu_points(vtu_dir / "box.0.vtu")
    _, cur = read_vtu_points(last)
    pressure = read_vtu_point_field(last, "Pressure")[1] if element_type == "T10P1" else None
    return (label, kappa, pid, ref.astype(float), cur.astype(float), pressure,
            reference_stretches(element_type, law, fibres, kappa))


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


@pytest.mark.parametrize("kappa", KAPPAS)
def test_energy_reference_matches_neo_hooke_closed_form(kappa):
    """Checks the stress-from-energy reference of Holzapfel and Guccione where a closed form exists."""
    params = LAWS["NeoHooke"][1]
    lam, mu = analytic_stretches(TRACTION, params["a"], kappa)
    stretches = uniaxial_stretches(lambda *l: neo_hooke_energy(*l, params),
                                   j_of_pressure("NeoHooke", "T4", kappa), TRACTION)
    np.testing.assert_allclose(stretches, [lam, mu, mu], rtol=1e-13)


def test_displacement_is_linear(solution):
    """Separates an inconsistent element, whose field bends between nodes, from a wrong but
    consistently applied law, which only gets the stretches wrong."""
    label, _, pid, ref, cur, _, _ = solution
    _, fit = _affine_fit(ref, cur)
    _assert_nodes_match(label, pid, cur, fit, POSITION_TOL, "displacement not linear")


def test_reproduces_analytic_stretches(solution):
    label, _, pid, ref, cur, _, stretches = solution
    F, _ = _affine_fit(ref, cur)
    _assert_nodes_match(label, pid, cur, ref * stretches, POSITION_TOL,
                        f"fitted F = {np.array2string(F, precision=7)} misses stretches "
                        f"{np.array2string(stretches, precision=7)}")


def test_p2p1_pressure_is_uniform_kappa_j_minus_one(solution):
    """Checked against the reference J: the single-precision points would leave kappa (J - 1)
    only about 1e-7 kappa accurate."""
    label, kappa, pid, _, _, pressure, stretches = solution
    if pressure is None:
        pytest.skip("displacement-only element, no pressure field")
    expected = kappa * (np.prod(stretches) - 1)
    _assert_nodes_match(label, pid, pressure[:, None], np.full((len(pressure), 1), expected),
                        PRESSURE_RTOL * abs(expected), "pressure not kappa (J - 1)")
