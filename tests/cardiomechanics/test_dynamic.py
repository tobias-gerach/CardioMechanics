"""Mechanics-only dynamic regression test.

Benchmark 2015 Problem 3 — a T10 truncated-ellipsoid ventricle inflated by a
linear endocardial pressure ramp while contracting under a linear active tension
ramp — integrated with NewmarkBeta instead of the benchmark's Static solver, with
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

from helpers.compare import read_vtu_points
from helpers.run import run_binary

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "examples" / "benchmark2015"
SETTINGS = Path(__file__).parent / "fixtures" / "dynamic_ellipsoid.xml"
GOLDEN_DIR = Path(__file__).parent / "golden"

NP = 4                     # CardioMechanics is tested only in parallel, as in test_benchmark
LAST = 20                  # dynamic.<LAST>.vtu at StopTime=0.2, export dt 1e-2
DEFORM_RTOL, DEFORM_ATOL = 1e-4, 1e-8      # coordinates in m

pytestmark = [pytest.mark.mpi, pytest.mark.slow]


@pytest.fixture(scope="module")
def dynamic_vtu_dir(binary, cm_env, tmp_path_factory):
    """Stage the fixture into an isolated tree and run it once at np=NP.

    dynamic_ellipsoid.xml uses paths relative to the working dir, so the mesh
    and the ./Results output folder have to live next to it.
    """
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    wd = tmp_path_factory.mktemp("dynamic")
    (wd / "tetgen").mkdir()
    for f in SRC.glob("tetgen/ellipsoid.*"):
        shutil.copy(f, wd / "tetgen")
    shutil.copy(SETTINGS, wd)
    (wd / "Results").mkdir()
    run_binary(binary("CardioMechanics"), ["-settings", SETTINGS.name],
               cwd=wd, env=cm_env, np=NP, timeout=1800)
    return wd / "Results" / "dynamic_vtu"


def test_dynamic_deformation(dynamic_vtu_dir, update_golden):
    pid, pts = read_vtu_points(dynamic_vtu_dir / f"dynamic.{LAST}.vtu")
    golden_path = GOLDEN_DIR / "dynamic_deformation.npz"
    if update_golden:
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(golden_path, pointid=pid, points=pts)
        pytest.skip(f"updated golden {golden_path.name}")
    assert golden_path.is_file(), f"missing golden {golden_path}; run with --update-golden"
    g = np.load(golden_path)
    assert np.array_equal(pid, g["pointid"]), "point ordering / mesh identity changed"
    if not np.allclose(pts, g["points"], rtol=DEFORM_RTOL, atol=DEFORM_ATOL):
        d = np.linalg.norm(pts - g["points"], axis=1)
        i = int(np.argmax(d))
        raise AssertionError(
            f"deformed coordinates differ beyond rtol={DEFORM_RTOL} atol={DEFORM_ATOL}: "
            f"node PointID={int(pid[i])} actual={pts[i]} golden={g['points'][i]} "
            f"|delta|={d[i]:.3e} m")
