"""Solver.DomainDecomposition renumbers the nodes by a graph partitioning, on the bar of the Land benchmark.

Input given per node in the original numbering is placed through the model's forward mapping, which the
renumbering has to keep up to date. LoadUnloadedState reads its unloaded and inflated configurations
that way; here both are the mesh's own node file, so a correct mapping leaves the geometry unchanged
and a wrong one moves nodes onto each other's positions. The export writes the point of each original
node through the same mapping.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers.compare import read_vtu_points
from helpers.run import assert_no_petsc_error, run_binary

FIXTURES = Path(__file__).parent / "fixtures"
DEFORM_RTOL, DEFORM_ATOL = 1e-4, 1e-8     # as for the benchmark goldens


def _run(binary, cm_env, wd, ranks, decomposition):
    """Run the bar in wd on ranks MPI ranks (serially if None), return its last vtu file."""
    wd.mkdir()
    shutil.copytree(FIXTURES / "thin_bar", wd / "thin_bar")
    (wd / "Results").mkdir()
    text = (FIXTURES / "thin_bar.xml").read_text()
    states = ("<LoadUnloadedState>"
              "<unloadedNodes>./thin_bar/bar.node</unloadedNodes>"
              "<inflatedNodes>./thin_bar/bar.node</inflatedNodes>"
              "</LoadUnloadedState>")
    substitutions = [
        ("<DomainDecomposition>false</DomainDecomposition>",
         f"<DomainDecomposition>{str(decomposition).lower()}</DomainDecomposition>"),
        ("<ApplyPressureFromFunction>true</ApplyPressureFromFunction>",
         "<LoadUnloadedState>true</LoadUnloadedState><ApplyPressureFromFunction>true</ApplyPressureFromFunction>"),
        ("<Plugins>\n    <ApplyPressureFromFunction>", f"<Plugins>\n    {states}\n    <ApplyPressureFromFunction>"),
    ]
    for old, new in substitutions:
        assert text.count(old) == 1, f"thin_bar.xml: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / "thin_bar.xml").write_text(text)
    proc = run_binary(binary("CardioMechanics"), ["-settings", "thin_bar.xml"], cwd=wd, env=cm_env, np=ranks)
    assert_no_petsc_error(proc.stdout + proc.stderr)
    return max((wd / "Results" / "bar_vtu").glob("bar.*.vtu"), key=lambda p: int(p.stem.split(".")[1]))


@pytest.mark.mpi
def test_decomposed_run_with_per_node_input_matches_serial(binary, cm_env, tmp_path):
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    pid, serial = read_vtu_points(_run(binary, cm_env, tmp_path / "np1", None, False))
    for ranks in (2, 4):
        pid_parallel, parallel = read_vtu_points(_run(binary, cm_env, tmp_path / f"np{ranks}", ranks, True))
        assert np.array_equal(pid, pid_parallel), "point ordering differs between serial and parallel"
        if not np.allclose(parallel, serial, rtol=DEFORM_RTOL, atol=DEFORM_ATOL):
            d = np.linalg.norm(parallel - serial, axis=1)
            i = int(np.argmax(d))
            raise AssertionError(
                f"np={ranks} with domain decomposition differs from serial beyond rtol={DEFORM_RTOL} "
                f"atol={DEFORM_ATOL}: node PointID={int(pid[i])} parallel={parallel[i]} serial={serial[i]} "
                f"|delta|={d[i]:.3e}")
