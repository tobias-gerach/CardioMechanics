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
GENALPHA_LUMPED_SETTINGS = FIXTURES / "dynamic_ellipsoid_genalpha_lumped.xml"
CREEP_SETTINGS = FIXTURES / "dynamic_ellipsoid_creep.xml"
GENALPHA_CREEP_SETTINGS = FIXTURES / "dynamic_ellipsoid_genalpha_creep.xml"
GOLDEN_DIR = Path(__file__).parent / "golden"

NP = 4                     # CardioMechanics is tested only in parallel, as in test_benchmark
LAST = 20                  # dynamic.<LAST>.vtu at StopTime=0.2, export dt 1e-2
DEFORM_RTOL, DEFORM_ATOL = 1e-4, 1e-8      # coordinates in m

CREEP_BETA = 0.3           # s; Rayleigh Beta of the creep fixtures, which have no Alpha
CREEP_DT = 1e-2            # Solver.TimeStep of the creep fixtures
CREEP_STOP_TIME = 0.9      # Solver.StopTime of the creep fixtures
CREEP_FIT_FROM = 0.1       # s; skips the start-up transient
CREEP_RTOL = 0.05          # observed 0.6%; a damping matrix that accumulates gives 43%


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
def genalpha_lumped_vtu_dir(binary, cm_env, tmp_path_factory):
    """Generalized-alpha on the lumped mass matrix, the second mass layout the preallocation must hold."""
    _require_mpi()
    wd = _stage(tmp_path_factory.mktemp("dynamic_genalpha_lumped"), GENALPHA_LUMPED_SETTINGS)
    _run(binary, cm_env, wd, GENALPHA_LUMPED_SETTINGS, np=NP, timeout=1800)
    return wd / "Results" / "dynamic_vtu"


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


@pytest.mark.mpi
@pytest.mark.slow
def test_dynamic_deformation(dynamic_vtu_dir, update_golden):
    _assert_matches_golden(dynamic_vtu_dir, DEFORM_RTOL, DEFORM_ATOL, update_golden)


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


@pytest.mark.parametrize("settings", [SETTINGS, GENALPHA_SETTINGS], ids=["newmark", "genalpha"])
def test_unknown_damping_type_is_rejected(binary, cm_env, tmp_path, settings):
    """A damping type other than Rayleigh, the only one implemented, has to abort initialisation.

    Run serially for the same reason as the RhoInf check.
    """
    wd = _stage(tmp_path, settings)
    staged = wd / settings.name
    xml = staged.read_text()
    assert "<Type>Rayleigh</Type>" in xml
    staged.write_text(xml.replace("<Type>Rayleigh</Type>", "<Type>Viscous</Type>"))
    proc = _run(binary, cm_env, wd, staged, timeout=300, check=False)
    assert proc.returncode != 0, f"expected a non-zero exit\n{proc.stdout[-2000:]}"
    expected = "Global damping type Viscous"
    assert expected in proc.stdout + proc.stderr, (
        f"error message did not mention {expected!r}\n{(proc.stdout + proc.stderr)[-2000:]}")


@pytest.mark.mpi
@pytest.mark.slow
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


@pytest.mark.mpi
@pytest.mark.slow
def test_generalized_alpha_jacobian_fits_its_preallocation(genalpha_vtu_dir, genalpha_lumped_vtu_dir):
    """The node-neighbour preallocation holds every entry of M and C, clamped couplings included.
    An entry outside it would reallocate the Jacobian row block at every build, which costs more
    than the rest of the build together, so PETSc's default for a preallocated matrix refuses it."""
    for vtu_dir in (genalpha_vtu_dir, genalpha_lumped_vtu_dir):
        assert_no_petsc_error((vtu_dir.parents[1] / "run.log").read_text())


def _log_view_count(log_view, event):
    """The number of calls of a PETSc event in a -log_view report, 0 if it never ran."""
    for line in log_view.splitlines():
        fields = line.split()
        if fields and fields[0] == event:
            return int(fields[1])
    return 0


def _log_view_of_short_run(binary, cm_env, tmp_path, settings, stop_time, export_step=1e-2):
    """The -log_view report of the first stop_time seconds of the ellipsoid on 2 ranks."""
    _require_mpi()
    wd = _stage(tmp_path, settings)
    staged = wd / settings.name
    xml = staged.read_text()
    stop = "<StopTime>0.2</StopTime>\n    <TimeStep>1e-3</TimeStep>"
    export = "<Prefix>Results/dynamic</Prefix>\n    <TimeStep>0.01</TimeStep>"
    assert stop in xml and export in xml
    xml = xml.replace(stop, f"<StopTime>{stop_time}</StopTime>\n    <TimeStep>1e-3</TimeStep>")
    xml = xml.replace(export, f"<Prefix>Results/dynamic</Prefix>\n    <TimeStep>{export_step}</TimeStep>")
    staged.write_text(xml)
    env = dict(cm_env, PETSC_OPTIONS="-log_view :log_view.txt")
    proc = _run(binary, env, wd, staged, np=2, timeout=600)
    assert_no_petsc_error(proc.stdout + proc.stderr)
    return (wd / "log_view.txt").read_text()


@pytest.mark.mpi
@pytest.mark.parametrize("settings", [SETTINGS, GENALPHA_SETTINGS], ids=["newmark", "genalpha"])
def test_jacobian_refresh_reuses_the_symbolic_factorisation(binary, cm_env, tmp_path, settings):
    """Without contact the nonzero pattern of the Jacobian is that of the mesh for the whole run,
    so the direct solver orders and factorises it symbolically once and only refactorises it
    numerically at every later refresh. The first 10 ms of the ellipsoid refresh the Jacobian
    four times by themselves, as their Newton solves slow down."""
    log_view = _log_view_of_short_run(binary, cm_env, tmp_path, settings, stop_time=0.01)
    numeric = _log_view_count(log_view, "MatLUFactorNum")
    assert numeric > 1, f"the run refreshed the Jacobian {numeric} times, the test needs at least 2"
    assert _log_view_count(log_view, "MatLUFactorSym") == 1


@pytest.mark.mpi
def test_residual_assembly_reuses_its_off_process_pattern(binary, cm_env, tmp_path):
    """Without contact the entries the residual sends to other ranks are fixed by the partition,
    so its two vectors find their partners in a rendezvous at their first assembly only. The
    rendezvous left are set-up and per step. The run exports its initial state only, so that the
    output adds none."""
    log_view = _log_view_of_short_run(binary, cm_env, tmp_path, GENALPHA_SETTINGS, stop_time=0.02,
                                      export_step=1)
    evaluations = _log_view_count(log_view, "SNESFunctionEval")
    rendezvous = _log_view_count(log_view, "BuildTwoSided")
    assert evaluations > 0
    assert rendezvous < evaluations, f"{rendezvous} rendezvous for {evaluations} residual evaluations"


@pytest.mark.mpi
@pytest.mark.parametrize("settings", [SETTINGS, GENALPHA_SETTINGS], ids=["newmark", "genalpha"])
def test_displacement_only_damping_update_extracts_no_block(binary, cm_env, tmp_path, settings):
    """Rayleigh Beta rebuilds the damping matrix at every step from the element tangent. The
    ellipsoid's T10 elements have no pressure unknowns, so the tangent is the displacement block
    itself and goes straight into the damping matrix, with no block to extract."""
    log_view = _log_view_of_short_run(binary, cm_env, tmp_path, settings, stop_time=0.01)
    updates = _log_view_count(log_view, "CBDampingUpdate")
    assert updates >= 10, f"{updates} damping updates in 10 steps"
    assert _log_view_count(log_view, "CBDampingElems") >= updates
    assert _log_view_count(log_view, "CBDampingSubMat") == 0
