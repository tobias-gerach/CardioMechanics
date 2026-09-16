"""MINI element on the linear cantilever of test_p2p1: the T4 displacement field enriched by a
condensed bubble, with a linear pressure field on the same four vertices.
"""
import numpy as np
import pytest

from helpers.compare import read_vtu_points
from helpers.run import assert_no_petsc_error, assert_refused
from test_p2p1 import DYNAMIC_FIXTURE, DYNAMIC_STEPS, FIXTURE, _assert_parallel_matches_serial, _last_vtu, _run

STEPS = 4       # StopTime / TimeStep of the static fixture, one export per step


def _degree(degree):
    """Settings substitution selecting the quadrature rule of the given degree."""
    return [("<Sorting>None</Sorting>", f"<Sorting>None</Sorting><QuadratureDegree>{degree}</QuadratureDegree>")]


def test_unknown_element_type_lists_t4mini(binary, cm_env, tmp_path):
    proc, _ = _run(binary, cm_env, tmp_path, element_type="T4Q1", check=False)
    assert proc.returncode != 0, f"expected a non-zero exit\n{proc.stdout[-2000:]}"
    assert "T4MINI" in proc.stdout + proc.stderr, (proc.stdout + proc.stderr)[-2000:]


@pytest.mark.parametrize("material", ["MooneyRivlin", "Usyk"])
def test_t4mini_refuses_material_without_mixed_formulation(binary, cm_env, tmp_path, material):
    proc, _ = _run(binary, cm_env, tmp_path, element_type="T4MINI", material=material, check=False)
    assert_refused(proc, material, "T4MINI")


def test_active_stress_estimator_refuses_t4mini(binary, cm_env, tmp_path):
    proc, _ = _run(binary, cm_env, tmp_path, element_type="T4MINI", solver="ActiveStressEstimator", check=False)
    assert_refused(proc, "Active Stress Estimator", "T4MINI")


def test_newmark_beta_refuses_t4mini(binary, cm_env, tmp_path):
    proc, _ = _run(binary, cm_env, tmp_path, element_type="T4MINI", solver="NewmarkBeta", check=False)
    assert_refused(proc, "Newmark Beta Solver", "T4MINI")


@pytest.mark.parametrize("degree", [1, 3])
def test_t4mini_refuses_quadrature_degree(binary, cm_env, tmp_path, degree):
    """Under the single-point rule the bubble's gradient, zero at the centroid, leaves its block singular."""
    proc, _ = _run(binary, cm_env, tmp_path, element_type="T4MINI", replace=_degree(degree), check=False)
    assert_refused(proc, f"QuadratureDegree {degree}", "T4MINI")


@pytest.mark.parametrize("degree", [2, 5])
def test_t4mini_static_run_converges_and_exports_pressure(binary, cm_env, tmp_path, degree):
    meshio = pytest.importorskip("meshio")
    proc, vtu_dir = _run(binary, cm_env, tmp_path, element_type="T4MINI", replace=_degree(degree))
    assert_no_petsc_error(proc.stdout + proc.stderr)
    last = _last_vtu(vtu_dir)
    assert last.name == f"cantilever.{STEPS}.vtu", "run stopped before its stop time"
    _, pts = read_vtu_points(last)
    _, ref = read_vtu_points(vtu_dir / "cantilever.0.vtu")
    deflection = np.abs(pts - ref).max()
    assert deflection > 1e-2, f"cantilever barely moved: max displacement {deflection:.3e}"
    p = np.asarray(meshio.read(str(last)).point_data["Pressure"]).ravel()
    assert np.abs(p).max() > 0, "exported pressure is zero everywhere"


@pytest.mark.mpi
@pytest.mark.parametrize("fixture", [
    pytest.param(FIXTURE, id="static"),
    pytest.param(DYNAMIC_FIXTURE, id="generalized_alpha", marks=pytest.mark.slow),
])
def test_t4mini_parallel_matches_serial(binary, cm_env, tmp_path, fixture):
    """Every node carries a pressure, unlike on P2P1, so each rank's pressure block spans all its
    nodes and a rank shifts its displacement unknowns by every node of the ranks before it. An
    offset, or a translation of the node-wise mass and damping matrices, that counts only some
    nodes as carrying a pressure departs from the serial run."""
    _assert_parallel_matches_serial(binary, cm_env, tmp_path, fixture, element_type="T4MINI")


@pytest.mark.parametrize("consistent_mass", ["true", "false"], ids=["consistent_mass", "lumped_mass"])
def test_t4mini_generalized_alpha_run_completes(binary, cm_env, tmp_path, consistent_mass):
    """The bubble carries no mass, so both of T4's mass matrices apply unchanged."""
    meshio = pytest.importorskip("meshio")
    proc, vtu_dir = _run(binary, cm_env, tmp_path, element_type="T4MINI", fixture=DYNAMIC_FIXTURE, replace=[
        ("<ConsistentMassMatrix>true</ConsistentMassMatrix>",
         f"<ConsistentMassMatrix>{consistent_mass}</ConsistentMassMatrix>")])
    assert_no_petsc_error(proc.stdout + proc.stderr)
    last = _last_vtu(vtu_dir)
    assert last.name == f"cantilever.{DYNAMIC_STEPS}.vtu", "run stopped before its stop time"
    _, pts = read_vtu_points(last)
    _, ref = read_vtu_points(vtu_dir / "cantilever.0.vtu")
    peak = np.linalg.norm(pts - ref, axis=1).max()
    assert peak > 1e-2, f"cantilever barely moved: peak displacement {peak:.3e}"
    p = np.asarray(meshio.read(str(last)).point_data["Pressure"]).ravel()
    assert np.abs(p).max() > 0, "exported pressure is zero everywhere"
