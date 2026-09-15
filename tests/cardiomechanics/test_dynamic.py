"""Mechanics-only dynamic regression test.

Benchmark 2015 Problem 3 — a T10 truncated-ellipsoid ventricle inflated by a
linear endocardial pressure ramp while contracting under a linear active tension
ramp — integrated with a dynamic solver instead of the benchmark's Static solver, with
Rayleigh damping added. There is no electrophysiology, no cell model and no
tension model in the loop, so a deviation here points at the mechanics time
integration itself rather than at anything upstream of it.

The settings file is our own rather than the benchmark's Problem3.xml, so that
the published-benchmark fixture and this dynamic one can move independently.
Everything but the solver, the damping and the output paths is Problem 3 as
published.

The Newmark-beta fixture runs at Beta=0.25 / Gamma=0.5, the trapezoidal rule.
That is the scheme the Chung-Hulbert family collapses onto at a spectral radius
of one, which is what makes the generalized-alpha equivalence check below a
comparison between two integrators of the same problem rather than between two
different amounts of numerical damping.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers.compare import read_table, read_vtu_points
from helpers.run import assert_no_petsc_error, run_binary

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "examples" / "benchmark2015"
FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = FIXTURES / "dynamic_ellipsoid.xml"
GENALPHA_SETTINGS = FIXTURES / "dynamic_ellipsoid_genalpha.xml"
LUMPED_SETTINGS = FIXTURES / "dynamic_ellipsoid_lumped.xml"
GENALPHA_LUMPED_SETTINGS = FIXTURES / "dynamic_ellipsoid_genalpha_lumped.xml"
CREEP_SETTINGS = FIXTURES / "dynamic_ellipsoid_creep.xml"
GENALPHA_CREEP_SETTINGS = FIXTURES / "dynamic_ellipsoid_genalpha_creep.xml"
GOLDEN_DIR = Path(__file__).parent / "golden"

NP = 4                     # CardioMechanics is tested only in parallel, as in test_benchmark
LAST = 20                  # dynamic.<LAST>.vtu at StopTime=0.2, export dt 1e-2
DEFORM_RTOL, DEFORM_ATOL = 1e-4, 1e-8      # coordinates in m

# Trapezoidal Newmark and generalized-alpha at RhoInf=1 are distinct second-order
# schemes, so they agree only to their own truncation error, not to round-off.
# 1e-6 m is 0.016% of the 6.1 mm peak displacement this fixture reaches.
EQUIV_RTOL, EQUIV_ATOL = 1e-4, 1e-6

CREEP_BETA = 0.3           # s; Rayleigh Beta of the creep fixtures, which have no Alpha
CREEP_DT = 1e-2            # Solver.TimeStep of the creep fixtures
CREEP_STOP_TIME = 0.9      # Solver.StopTime of the creep fixtures
CREEP_FIT_FROM = 0.1       # s; skips the start-up transient
CREEP_RTOL = 0.05          # observed 0.6%; a damping matrix that accumulates gives 43%

pytestmark = [pytest.mark.mpi, pytest.mark.slow]


def _stage(wd, settings):
    """Copy mesh and settings into wd.

    The fixtures use paths relative to the working dir, so the mesh and the
    ./Results output folder have to live next to the settings file.
    """
    (wd / "tetgen").mkdir()
    for f in SRC.glob("tetgen/ellipsoid.*"):
        shutil.copy(f, wd / "tetgen")
    shutil.copy(settings, wd)
    (wd / "Results").mkdir()
    return wd


def _run(binary, cm_env, wd, settings, **kwargs):
    proc = run_binary(binary("CardioMechanics"), ["-settings", settings.name],
                      cwd=wd, env=cm_env, **kwargs)
    (wd / "run.log").write_text(proc.stdout + proc.stderr)
    return proc


def _require_mpi():
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")


@pytest.fixture(scope="module")
def dynamic_vtu_dir(binary, cm_env, tmp_path_factory):
    """Stage the Newmark-beta fixture into an isolated tree and run it once at np=NP."""
    _require_mpi()
    wd = _stage(tmp_path_factory.mktemp("dynamic"), SETTINGS)
    _run(binary, cm_env, wd, SETTINGS, np=NP, timeout=1800)
    return wd / "Results" / "dynamic_vtu"


@pytest.fixture(scope="module")
def genalpha_vtu_dir(binary, cm_env, tmp_path_factory):
    """The same fixture under generalized-alpha at RhoInf=1."""
    _require_mpi()
    wd = _stage(tmp_path_factory.mktemp("dynamic_genalpha"), GENALPHA_SETTINGS)
    _run(binary, cm_env, wd, GENALPHA_SETTINGS, np=NP, timeout=1800)
    return wd / "Results" / "dynamic_vtu"


@pytest.fixture(scope="module")
def lumped_vtu_dirs(binary, cm_env, tmp_path_factory):
    """Both integrators on the lumped mass matrix, which no golden covers.

    A lumped run deforms differently from a consistent one, so there is nothing
    to compare either of these against except each other.
    """
    _require_mpi()
    out = []
    for name, settings in (("dynamic_lumped", LUMPED_SETTINGS),
                           ("dynamic_genalpha_lumped", GENALPHA_LUMPED_SETTINGS)):
        wd = _stage(tmp_path_factory.mktemp(name), settings)
        _run(binary, cm_env, wd, settings, np=NP, timeout=1800)
        out.append(wd / "Results" / "dynamic_vtu")
    return out


def _assert_matches_golden(vtu_dir, rtol, atol, update_golden=False):
    pid, pts = read_vtu_points(vtu_dir / f"dynamic.{LAST}.vtu")
    golden_path = GOLDEN_DIR / "dynamic_deformation.npz"
    if update_golden:
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(golden_path, pointid=pid, points=pts)
        pytest.skip(f"updated golden {golden_path.name}")
    assert golden_path.is_file(), f"missing golden {golden_path}; run with --update-golden"
    g = np.load(golden_path)
    assert np.array_equal(pid, g["pointid"]), "point ordering / mesh identity changed"
    if not np.allclose(pts, g["points"], rtol=rtol, atol=atol):
        d = np.linalg.norm(pts - g["points"], axis=1)
        i = int(np.argmax(d))
        raise AssertionError(
            f"deformed coordinates differ beyond rtol={rtol} atol={atol}: "
            f"node PointID={int(pid[i])} actual={pts[i]} golden={g['points'][i]} "
            f"|delta|={d[i]:.3e} m")


def test_dynamic_deformation(dynamic_vtu_dir, update_golden):
    _assert_matches_golden(dynamic_vtu_dir, DEFORM_RTOL, DEFORM_ATOL, update_golden)


def test_generalized_alpha_reproduces_newmark_at_rhoinf_one(genalpha_vtu_dir):
    """RhoInf=1 is the non-dissipative end of the Chung-Hulbert family.

    The golden is never regenerated from here: it is the Newmark-beta reference
    this run is judged against, so writing to it would erase the comparison.
    """
    _assert_matches_golden(genalpha_vtu_dir, EQUIV_RTOL, EQUIV_ATOL)


@pytest.mark.parametrize("settings_name, expected", [
    ("dynamic_ellipsoid_genalpha_no_rhoinf.xml", "Solver.GeneralizedAlpha.RhoInf"),
    ("dynamic_ellipsoid_genalpha_bad_rhoinf.xml", "outside the valid range [0, 1]"),
])
def test_generalized_alpha_rejects_bad_rhoinf(binary, cm_env, tmp_path, settings_name, expected):
    """A missing or out-of-range spectral radius has to abort initialisation.

    Run serially, since these never reach the solver and the parallel path would
    add nothing but startup cost.
    """
    settings = FIXTURES / settings_name
    wd = _stage(tmp_path, settings)
    proc = _run(binary, cm_env, wd, settings, timeout=300, check=False)
    assert proc.returncode != 0, f"expected a non-zero exit\n{proc.stdout[-2000:]}"
    assert expected in proc.stdout + proc.stderr, (
        f"error message did not mention {expected!r}\n{(proc.stdout + proc.stderr)[-2000:]}")


@pytest.mark.parametrize("settings", [CREEP_SETTINGS, GENALPHA_CREEP_SETTINGS],
                         ids=["newmark", "genalpha"])
def test_rayleigh_damping_creeps_with_time_constant_beta(binary, cm_env, tmp_path, settings):
    """The damping matrix has to be Beta K of the current state at every step.

    Beta = 0.3 s overdamps every mode of the ellipsoid many times over (zeta = Beta omega / 2,
    with omega above 2 pi 100 Hz), so inertia drops out and each mode obeys Beta K v + K u = f:
    under a held step load the cavity volume creeps towards equilibrium as exp(-t / Beta),
    whatever the mode. A damping matrix that carries a share of its previous value into each
    update settles at Beta K / (1 - Beta) instead, which stretches the time constant to 0.43 s.
    The load is small because its follower stiffness is part of the relaxation but not of C.
    """
    _require_mpi()
    wd = _stage(tmp_path, settings)
    _run(binary, cm_env, wd, settings, np=NP, timeout=1800)
    names, data = read_table(wd / "Results" / "Pressure.dat")
    t, v = data[:, names.index("time")], data[:, names.index("volume1")]
    # CardioMechanics exits 0 even when the solve gives up, so a short trace is how that shows,
    # and the fit below needs every step exactly once.
    assert len(t) == int(round(CREEP_STOP_TIME / CREEP_DT)) + 1, f"run stopped at t={t[-1]} s"
    assert np.allclose(np.diff(t), CREEP_DT), "trace is not sampled uniformly"
    # The increments of exp(-t / tau) on a uniform grid decay like the function itself.
    fit = t[1:] >= CREEP_FIT_FROM
    tau = -1 / np.polyfit(t[1:][fit], np.log(np.abs(np.diff(v)[fit])), 1)[0]
    assert abs(tau / CREEP_BETA - 1) <= CREEP_RTOL, (
        f"volume creeps with time constant {tau:.4f} s, expected Beta = {CREEP_BETA} s "
        f"within {CREEP_RTOL:.0%}")


def test_generalized_alpha_jacobian_fits_its_preallocation(genalpha_vtu_dir, lumped_vtu_dirs):
    """The node-neighbour preallocation holds every entry of M and C, clamped couplings included.
    An entry outside it would reallocate the Jacobian row block at every build, which costs more
    than the rest of the build together, so PETSc's default for a preallocated matrix refuses it."""
    for vtu_dir in (genalpha_vtu_dir, lumped_vtu_dirs[1]):
        assert_no_petsc_error((vtu_dir.parents[1] / "run.log").read_text())


def test_generalized_alpha_matches_newmark_with_lumped_mass(lumped_vtu_dirs):
    """The lumped mass path has to carry the RhoInf=1 equivalence too."""
    newmark_dir, genalpha_dir = lumped_vtu_dirs
    npid, npts = read_vtu_points(newmark_dir / f"dynamic.{LAST}.vtu")
    gpid, gpts = read_vtu_points(genalpha_dir / f"dynamic.{LAST}.vtu")
    assert np.array_equal(npid, gpid), "point ordering / mesh identity differs between runs"
    if not np.allclose(gpts, npts, rtol=EQUIV_RTOL, atol=EQUIV_ATOL):
        d = np.linalg.norm(gpts - npts, axis=1)
        i = int(np.argmax(d))
        raise AssertionError(
            f"lumped-mass coordinates differ beyond rtol={EQUIV_RTOL} atol={EQUIV_ATOL}: "
            f"node PointID={int(gpid[i])} generalized-alpha={gpts[i]} newmark={npts[i]} "
            f"|delta|={d[i]:.3e} m")
