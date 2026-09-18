"""Time integrator accuracy on a single-mode free vibration.

A cantilever is bent by a smooth pressure pulse and then rings freely, without Rayleigh damping,
at an amplitude small enough for its response to be linear. Each mode of a linear system then
obeys the scalar oscillator x'' + omega^2 x = 0, and a scheme advances it by its amplification
matrix, whose principal root z fixes the discrete period and decay of the mode. Those are known in
closed form for the Newmark and Chung-Hulbert family at every omega dt, so the measured period and
logarithmic decrement of the first bending mode check the scheme itself, not merely its agreement
with another scheme. The frequency omega is measured on a reference run at a 32 times smaller
step, whose own period error is about 2e-5.

The tip trace is sampled at the coarsest step and its discrete poles are found by the matrix
pencil method. A linear free vibration sampled at a fixed step is exactly a sum of z^n terms, one
per mode and root of the amplification matrix, so the pole of the first bending mode is z itself,
whatever the other modes and the spurious root of generalized-alpha contribute.

Every run rebuilds the Jacobian at every Newton iteration. By default the solvers keep it across
time steps, and on a slender beam that is a poor linearization: rotating the beam by 5e-3 rad
changes the stiff volumetric part of the tangent by 5e-3 relative, a large error against the soft
bending mode. Newton then takes up to 50 iterations per step and at RhoInf <= 0.5 stalls in its
line search, whereas a fresh Jacobian converges in 2 or 3.

Newmark-beta supports no pressure field, so P2P1 runs under generalized-alpha only.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from helpers.cantilever import CELLS, node, write_mesh
from helpers.compare import read_vtu_point_field
from helpers.run import run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "free_vibration_cantilever.xml"
# Half the height of the square section of helpers.cantilever. Both bending modes of a square
# section share a frequency, and the mesh, which is not symmetric, couples them, so the tip would
# beat between the two.
SCALE = (1, 1, 0.5)
TIP = node((2 * CELLS[0], CELLS[1], CELLS[2]))      # centre of the free end

DT = 0.8                      # coarsest step and export interval of the fixture, omega dt = 0.48
PULSE = 9.6                   # the load is off from the 13th sample on
STOP_TIME = 105.6             # about ten periods of ring-down
REFERENCE_DT = DT / 32
SAMPLES = int(round(STOP_TIME / DT)) + 1
RING_DOWN = int(round(PULSE / DT)) + 1
# The displacement error is taken over the load and the first period after it. Later, the phase
# error of the ring-down, second order but growing with time, outgrows a first-order error in the
# forcing, such as loads evaluated at the end of the step instead of at the intermediate level.
CONVERGENCE_SAMPLES = 2 * RING_DOWN
FRESH_JACOBIAN = "-mech_snes_lag_jacobian 1 -mech_snes_lag_preconditioner 1"

ELEMENTS = ("T10", "T10P1")
RHO_INF = (1.0, 0.8, 0.5, 0.2, 0.0)
# None is trapezoidal Newmark-beta, a number generalized-alpha at that RhoInf.
INTEGRATORS = [("T10", None)] + [(e, r) for e in ELEMENTS for r in RHO_INF]

# Observed: the first bending mode leaves 3.2e-4 (T10) and 6.6e-5 (P2P1) of the ring-down to the others.
MAX_OTHER_MODES = 3e-3
# Observed misfits against the closed form, at omega dt = 0.48 (T10) and 0.46 (P2P1): period error
# -1.6e-5 to -1.8e-5 for every integrator, the reference's own period error, and decrement 4e-8 to
# 8.5e-6. Neighbouring RhoInf differ by at least 1e-3 in period error, and the smallest non-zero
# decrement, 3.9e-4 at RhoInf = 0.8, is 8 times the tolerance.
PERIOD_ERROR_TOL = 1e-4
DECREMENT_TOL = 5e-5
# Observed rates on the finest pair: 1.98 Newmark, 1.94 T10 and 1.98 P2P1 at RhoInf = 0.5, and
# 1.79 to 1.96 on the coarse pair, which is not yet asymptotic. Evaluating the loads at the end of
# the step instead of at the intermediate level gives 0.94.
RATE_TOL = 0.2

# About 3.5 min serially, half of it the two reference runs.
pytestmark = pytest.mark.slow


def _id(value):
    if value is None:
        return "newmark"
    return f"rho{value}" if isinstance(value, float) else value


def _tip_trace(binary, cm_env, wd, element_type, rho_inf, dt=DT):
    """Run the fixture in wd, return the vertical tip displacement at every multiple of DT up to STOP_TIME."""
    (wd / "tetgen").mkdir(parents=True)
    write_mesh(wd / "tetgen", SCALE)
    (wd / "Results").mkdir()
    step = "<TimeStep>{0}</TimeStep><MinTimeStep>{0}</MinTimeStep>"
    substitutions = [("<Type>T10</Type>", f"<Type>{element_type}</Type>"), (step.format(DT), step.format(dt))]
    if rho_inf is None:
        substitutions.append(("<Type>GeneralizedAlpha</Type>", "<Type>NewmarkBeta</Type>"))
    else:
        substitutions.append(("<RhoInf>1.0</RhoInf>", f"<RhoInf>{rho_inf}</RhoInf>"))
    text = FIXTURE.read_text()
    for old, new in substitutions:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / FIXTURE.name).write_text(text)
    proc = run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd,
                      env=dict(cm_env, PETSC_OPTIONS=FRESH_JACOBIAN), timeout=1800)

    results = wd / "Results"
    # The time loop sums its steps, so it can take one more step just past StopTime.
    frames = [(float(t), f) for t, f in re.findall(r'timestep="([^"]+)".*?file="([^"]+)"',
                                                   (results / "cantilever.pvd").read_text())
              if float(t) < STOP_TIME + DT / 2]
    times = np.array([t for t, _ in frames])
    assert len(times) == SAMPLES and np.allclose(times, DT * np.arange(SAMPLES)), f"exported at {times}"
    trace = []
    for _, f in frames:
        pid, u = read_vtu_point_field(results / f, "AbsDisplacement")
        trace.append(u[pid == TIP - 1][0, 2])
    return np.array(trace)


def _poles(trace, order):
    """Discrete poles of a trace by the matrix pencil method (Hua & Sarkar 1990), largest share of the trace first."""
    hankel = np.lib.stride_tricks.sliding_window_view(trace, len(trace) // 3 + 1)
    v = np.linalg.svd(hankel, full_matrices=False)[2][:order].T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    amplitudes = np.linalg.lstsq(np.vander(z, len(trace), increasing=True).T, trace.astype(complex), rcond=None)[0]
    largest_first = np.argsort(-np.abs(amplitudes))
    return z[largest_first], amplitudes[largest_first]


def _first_mode(trace):
    """Pole of the first bending mode in the ring-down, modelled together with the next mode."""
    z, _ = _poles(trace[RING_DOWN:], 4)
    return z[z.imag > 0][0]


def _principal_root(omega_dt, rho_inf):
    """Principal root of the amplification matrix of x'' + omega^2 x = 0 under generalized-alpha (Chung & Hulbert
    1993), or under trapezoidal Newmark-beta, its member with alphaM = alphaF = 0, if rho_inf is None."""
    am, af = (0.0, 0.0) if rho_inf is None else ((2 * rho_inf - 1) / (rho_inf + 1), rho_inf / (rho_inf + 1))
    beta, gamma = 0.25 * (1 - am + af) ** 2, 0.5 - am + af
    # (d, dt v, dt^2 a) of the next step, from the balance at the intermediate level and the Newmark updates
    new = np.array([[omega_dt ** 2 * (1 - af), 0, 1 - am], [1, 0, -beta], [0, 1, -gamma]])
    old = np.array([[-omega_dt ** 2 * af, 0, -am], [1, 1, 0.5 - beta], [0, 1, 1 - gamma]])
    roots = np.linalg.eigvals(np.linalg.solve(new, old))
    return roots[np.argmax(roots.imag)]


def _period_error_and_decrement(z, omega_dt):
    """Relative period error T_h / T - 1 and logarithmic decrement per period of pole z."""
    return omega_dt / np.angle(z) - 1, -2 * np.pi * np.log(abs(z)) / np.angle(z)


@pytest.fixture(scope="module")
def reference(binary, cm_env, tmp_path_factory):
    """Returns a function of the element type giving (trace, omega) of its reference run, generalized-alpha at
    RhoInf = 1 and step REFERENCE_DT, run once per element type."""
    pytest.importorskip("meshio")
    cache = {}

    def get(element_type):
        if element_type not in cache:
            wd = tmp_path_factory.mktemp(f"{element_type}_reference")
            trace = _tip_trace(binary, cm_env, wd, element_type, 1.0, REFERENCE_DT)
            cache[element_type] = trace, np.angle(_first_mode(trace)) / DT
        return cache[element_type]
    return get


@pytest.mark.parametrize("element_type", ELEMENTS)
def test_ring_down_is_first_bending_mode(reference, element_type):
    """The first bending mode alone reproduces the ring-down, so its pole is well defined."""
    trace, _ = reference(element_type)
    ring_down = trace[RING_DOWN:]
    z, amplitudes = _poles(ring_down, 2)
    fit = np.real(np.vander(z, len(ring_down), increasing=True).T @ amplitudes)
    misfit = np.abs(fit - ring_down).max() / np.abs(ring_down).max()
    assert misfit <= MAX_OTHER_MODES, f"the first bending mode leaves {misfit:.2e} of the ring-down unexplained"


@pytest.mark.parametrize("element_type, rho_inf", INTEGRATORS, ids=_id)
def test_period_and_decrement_match_closed_form(binary, cm_env, tmp_path, reference, element_type, rho_inf):
    _, omega = reference(element_type)
    trace = _tip_trace(binary, cm_env, tmp_path, element_type, rho_inf)
    period_error, decrement = _period_error_and_decrement(_first_mode(trace), omega * DT)
    expected_period_error, expected_decrement = _period_error_and_decrement(_principal_root(omega * DT, rho_inf),
                                                                            omega * DT)
    assert abs(period_error - expected_period_error) <= PERIOD_ERROR_TOL, (
        f"period error {period_error:.6e}, closed form {expected_period_error:.6e} at omega dt = {omega * DT:.4f}")
    assert abs(decrement - expected_decrement) <= DECREMENT_TOL, (
        f"logarithmic decrement {decrement:.6e}, closed form {expected_decrement:.6e} at omega dt = {omega * DT:.4f}")


@pytest.mark.parametrize("element_type, rho_inf", [("T10", None), ("T10", 0.5), ("T10P1", 0.5)], ids=_id)
def test_displacement_converges_at_second_order(binary, cm_env, tmp_path, reference, element_type, rho_inf):
    trace, _ = reference(element_type)
    errors = []
    for n in (1, 2, 4):
        u = _tip_trace(binary, cm_env, tmp_path / str(n), element_type, rho_inf, DT / n)
        errors.append(np.abs(u - trace)[:CONVERGENCE_SAMPLES].max() / np.abs(trace).max())
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    assert rates[-1] >= 2 - RATE_TOL, f"errors {errors} at dt = {DT}, {DT / 2}, {DT / 4}, rates {rates}"
