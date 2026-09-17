"""Iterative linear solver presets on the cantilever of test_p2p1.

Each Newton step of an iterative run solves its linear system only to a relative residual of 1e-8, so its
converged shape agrees with that of the direct solve to well below the resolution of the exported points.
The amg presets take the displacement-only element types, the fieldsplit preset the mixed ones.
"""
import re
import shutil

import numpy as np
import pytest

from helpers.cantilever import CELLS
from helpers.run import assert_no_petsc_error, assert_refused
from test_p2p1 import DYNAMIC_FIXTURE, FIXTURE, NP, _final_points, _run

ATOL = 1e-6     # the exported points are float32, which resolves 2.4e-7 at the coordinates of up to 4
SOLVER_TYPE = {FIXTURE: "Static", DYNAMIC_FIXTURE: "GeneralizedAlpha"}
# Both mixed element types carry one pressure unknown per cube corner of the mesh.
PRESSURE_DOFS = (CELLS[0] + 1) * (CELLS[1] + 1) * (CELLS[2] + 1)


def _linear_solver(fixture, preset, options=""):
    """Settings substitution selecting the linear solver of fixture's Solver."""
    solver = f"<Type>{SOLVER_TYPE[fixture]}</Type>"
    return [(solver, f"{solver}<LinearSolver><Preset>{preset}</Preset><Options>{options}</Options></LinearSolver>")]


def _assert_same_shape(actual, expected, what):
    (pid, points), (pid_expected, points_expected) = actual, expected
    assert np.array_equal(pid, pid_expected), f"{what}: point ordering differs"
    d = np.linalg.norm(points - points_expected, axis=1)
    assert d.max() < ATOL, f"{what}: shape differs from the direct solve by {d.max():.3e} at PointID={pid[d.argmax()]}"


@pytest.mark.parametrize("fixture", [FIXTURE, DYNAMIC_FIXTURE], ids=["Static", "GeneralizedAlpha"])
@pytest.mark.parametrize("element_type", ["T4", "T10"])
def test_amg_presets_match_direct_solve(binary, cm_env, tmp_path, element_type, fixture):
    pytest.importorskip("meshio")
    shapes = {}
    for preset in ("direct", "amg", "amg-hypre"):
        wd = tmp_path / preset
        wd.mkdir()
        # The view shows whether the rigid-body modes survive until the preconditioner is built.
        options = "-mech_ksp_view" if preset == "amg" else ""
        proc, vtu_dir = _run(binary, cm_env, wd, element_type=element_type, fixture=fixture,
                             replace=_linear_solver(fixture, preset, options))
        assert_no_petsc_error(proc.stdout + proc.stderr)
        shapes[preset] = _final_points(vtu_dir)
        if preset == "amg":
            assert "type: gamg" in proc.stdout and "bs=3" in proc.stdout, proc.stdout[-2000:]
            assert "has attached near null space" in proc.stdout, proc.stdout[-2000:]
    for preset in ("amg", "amg-hypre"):
        _assert_same_shape(shapes[preset], shapes["direct"], preset)


@pytest.mark.mpi
def test_amg_parallel_matches_serial_direct_solve(binary, cm_env, tmp_path):
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    shapes = {}
    for preset, ranks in (("direct", None), ("amg", NP)):
        wd = tmp_path / preset
        wd.mkdir()
        proc, vtu_dir = _run(binary, cm_env, wd, element_type="T10", ranks=ranks,
                             replace=_linear_solver(FIXTURE, preset))
        assert_no_petsc_error(proc.stdout + proc.stderr)
        shapes[preset] = _final_points(vtu_dir)
    _assert_same_shape(shapes["amg"], shapes["direct"], f"amg at np={NP}")


@pytest.mark.parametrize("preset, pc", [("amg", "gamg"), ("amg-hypre", "hypre")])
@pytest.mark.parametrize("element_type", ["T10P1", "T4MINI"])
def test_pressure_field_refuses_amg_presets(binary, cm_env, tmp_path, element_type, preset, pc):
    proc, _ = _run(binary, cm_env, tmp_path, element_type=element_type, check=False,
                   replace=_linear_solver(FIXTURE, preset))
    assert_refused(proc, pc, "lu", "fieldsplit")


def _split_matrix(view, field):
    """What -mech_ksp_view says about the matrix of the `field` block of the fieldsplit: its size and
    the properties, the near-null space among them, listed under it.

    Matched on the matrix itself rather than on the first one under the block, which for multigrid is
    its coarsest level.
    """
    match = re.search(rf"Mat Object: \(mech_fieldsplit_{field}_\)[^\n]*\n\s*type: \w+\n"
                      rf"(?P<body>(?:[ \t]+\S[^\n]*\n)+?)(?=[ \t]*(?:Up solver|KSP|PC|Mat|linear system))", view)
    assert match, f"the view shows no {field} block\n{view[-2000:]}"
    return match.group("body")


def _split_pc(view, field):
    """The preconditioner type of the `field` block of the fieldsplit."""
    match = re.search(rf"PC Object: \(mech_fieldsplit_{field}_\)[^\n]*\n\s*type: (\w+)", view)
    assert match, f"the view shows no preconditioner for the {field} block\n{view[-2000:]}"
    return match.group(1)


def _split_rows(view, field):
    """Rows of the matrix of the `field` block of the fieldsplit."""
    return int(re.search(r"rows=(\d+)", _split_matrix(view, field)).group(1))


def _displacement_dofs(wd):
    """Three per node of the mesh staged in wd, as the tetgen header counts them."""
    return 3 * int((wd / "tetgen" / "cantilever.node").read_text().split(maxsplit=1)[0])


@pytest.mark.parametrize("fixture", [FIXTURE, DYNAMIC_FIXTURE], ids=["Static", "GeneralizedAlpha"])
@pytest.mark.parametrize("element_type", ["T10P1", "T4MINI"])
def test_fieldsplit_matches_direct_solve(binary, cm_env, tmp_path, element_type, fixture):
    pytest.importorskip("meshio")
    shapes = {}
    for preset in ("direct", "fieldsplit"):
        wd = tmp_path / preset
        wd.mkdir()
        # The view shows which system the split is actually over.
        options = "-mech_ksp_view" if preset == "fieldsplit" else ""
        proc, vtu_dir = _run(binary, cm_env, wd, element_type=element_type, fixture=fixture,
                             replace=_linear_solver(fixture, preset, options))
        assert_no_petsc_error(proc.stdout + proc.stderr)
        shapes[preset] = _final_points(vtu_dir)
        if preset == "fieldsplit":
            assert "FieldSplit with Schur preconditioner, factorization FULL" in proc.stdout, proc.stdout[-2000:]
            assert "Schur complement formed from A11" in proc.stdout, proc.stdout[-2000:]
            assert _split_rows(proc.stdout, "u") == _displacement_dofs(wd)
            assert _split_rows(proc.stdout, "p") == PRESSURE_DOFS
            # The displacement block is the one multigrid needs a node's three components and the
            # rigid-body modes on, neither of which the whole Jacobian of a mixed model carries.
            block = _split_matrix(proc.stdout, "u")
            assert _split_pc(proc.stdout, "u") == "gamg", proc.stdout[-2000:]
            assert "bs=3" in block, block
            assert "has attached near null space" in block, block
    _assert_same_shape(shapes["fieldsplit"], shapes["direct"], "fieldsplit")


@pytest.mark.mpi
def test_fieldsplit_parallel_matches_serial_direct_solve(binary, cm_env, tmp_path):
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    shapes = {}
    for preset, ranks in (("direct", None), ("fieldsplit", NP)):
        wd = tmp_path / preset
        wd.mkdir()
        proc, vtu_dir = _run(binary, cm_env, wd, element_type="T10P1", ranks=ranks,
                             replace=_linear_solver(FIXTURE, preset))
        assert_no_petsc_error(proc.stdout + proc.stderr)
        shapes[preset] = _final_points(vtu_dir)
    _assert_same_shape(shapes["fieldsplit"], shapes["direct"], f"fieldsplit at np={NP}")


@pytest.mark.parametrize("element_type", ["T4", "T10"])
def test_displacement_only_model_refuses_fieldsplit(binary, cm_env, tmp_path, element_type):
    proc, _ = _run(binary, cm_env, tmp_path, element_type=element_type, check=False,
                   replace=_linear_solver(FIXTURE, "fieldsplit"))
    assert_refused(proc, "fieldsplit", "T10P1", "T4MINI")
