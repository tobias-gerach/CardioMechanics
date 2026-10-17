"""A rank that owns no element, on the one-element-thick bar of the Land benchmark.

The bar's nodes are numbered vertices first, and a rank owns a contiguous range of them, so the last
ranks hold mid-edge nodes alone. Each element goes to the rank that holds most of its nodes, which
leaves rank 1 of 2 and rank 2 of 4 without elements. Such a rank still owns nodes, and for T10P1
no pressure unknowns, and has to take part in every residual and Jacobian assembly.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers.compare import read_vtu_point_field, read_vtu_points
from helpers.run import assert_no_petsc_error, run_binary

FIXTURES = Path(__file__).parent / "fixtures"
DEFORM_RTOL, DEFORM_ATOL = 1e-4, 1e-8     # as for the benchmark goldens
PRESSURE_RTOL = 1e-4                      # absolute tolerance scaled by the peak pressure


def _run(binary, cm_env, wd, element_type, ranks):
    """Run the bar in wd on ranks MPI ranks (serially if None), return its last vtu file."""
    wd.mkdir()
    shutil.copytree(FIXTURES / "thin_bar", wd / "thin_bar")
    (wd / "Results").mkdir()
    text = (FIXTURES / "thin_bar.xml").read_text()
    assert text.count("<Type>T10</Type>") == 1
    (wd / "thin_bar.xml").write_text(text.replace("<Type>T10</Type>", f"<Type>{element_type}</Type>"))
    proc = run_binary(binary("CardioMechanics"), ["-settings", "thin_bar.xml"], cwd=wd, env=cm_env, np=ranks)
    assert_no_petsc_error(proc.stdout + proc.stderr)
    return max((wd / "Results" / "bar_vtu").glob("bar.*.vtu"), key=lambda p: int(p.stem.split(".")[1]))


@pytest.mark.mpi
@pytest.mark.parametrize("element_type", ["T10", "T10P1"])
def test_rank_without_elements_matches_serial(binary, cm_env, tmp_path, element_type):
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    serial_vtu = _run(binary, cm_env, tmp_path / "np1", element_type, None)
    pid, serial = read_vtu_points(serial_vtu)
    for ranks in (2, 4):
        vtu = _run(binary, cm_env, tmp_path / f"np{ranks}", element_type, ranks)
        pid_parallel, parallel = read_vtu_points(vtu)
        assert np.array_equal(pid, pid_parallel), "point ordering differs between serial and parallel"
        if not np.allclose(parallel, serial, rtol=DEFORM_RTOL, atol=DEFORM_ATOL):
            d = np.linalg.norm(parallel - serial, axis=1)
            i = int(np.argmax(d))
            raise AssertionError(
                f"np={ranks} differs from serial beyond rtol={DEFORM_RTOL} atol={DEFORM_ATOL}: "
                f"node PointID={int(pid[i])} parallel={parallel[i]} serial={serial[i]} |delta|={d[i]:.3e}")
        if element_type != "T10P1":
            continue
        # The empty rank holds mid-edge nodes alone, so it owns no pressure unknown, yet takes part
        # in every collective call on the pressure vectors.
        _, p_serial = read_vtu_point_field(serial_vtu, "Pressure")
        _, p_parallel = read_vtu_point_field(vtu, "Pressure")
        atol = PRESSURE_RTOL * np.abs(p_serial).max()
        if not np.allclose(p_parallel, p_serial, rtol=PRESSURE_RTOL, atol=atol):
            i = int(np.argmax(np.abs(p_parallel - p_serial)))
            raise AssertionError(
                f"np={ranks} pressure differs from serial beyond rtol={PRESSURE_RTOL} atol={atol:.3e}: "
                f"node PointID={int(pid[i])} parallel={p_parallel[i]:.6e} serial={p_serial[i]:.6e}")
