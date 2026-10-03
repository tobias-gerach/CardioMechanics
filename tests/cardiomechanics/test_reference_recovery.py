"""ReferenceRecovery with a Robin boundary, on the cantilever of test_p2p1 with its whole boundary as
the cavity and its free end held by springs.

The recovery relocates the mesh to each new unloaded guess and makes it the reference. The springs
have to follow: anchored at the reference, they are relaxed in the unloaded state, as in a forward
run that starts from it, and a dashpot does not read the relocation as a velocity.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from helpers.cantilever import END_SURFACE, write_mesh
from helpers.compare import read_vtu_points
from helpers.run import run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "p2p1_cantilever_recovery.xml"
TOLERANCE = 1e-4        # the recovery's, an infinity norm in the mesh unit
TIME_STEP = 0.01        # solver and export
# The fixture holds the end by the general plugin; this swaps in the one that projects on the normal.
ROBIN_NORMAL = (("<RobinBoundaryGeneral>true</RobinBoundaryGeneral>", "<RobinBoundary>true</RobinBoundary>"),
                ("<RobinBoundaryGeneral>\n", "<RobinBoundary>\n"),
                ("</RobinBoundaryGeneral>\n", "</RobinBoundary>\n"))


def _run(binary, cm_env, wd, replace=()):
    """Stage the closed cantilever with its Robin end into wd, apply the (old, new) substitutions to the
    fixture and run it serially. Returns the vtu directory."""
    (wd / "tetgen").mkdir(parents=True)
    write_mesh(wd / "tetgen", closed=True, end=True)
    (wd / "Results").mkdir()
    text = FIXTURE.read_text()
    for old, new in replace:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / FIXTURE.name).write_text(text)
    run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd, env=cm_env)
    return wd / "Results" / "cantilever_vtu"


def _vtu(vtu_dir, index):
    return vtu_dir / f"cantilever.{index}.vtu"


@pytest.mark.parametrize("robin", [(), ROBIN_NORMAL], ids=["general", "normal"])
def test_recovered_state_inflates_onto_the_target_with_the_same_robin_boundary(binary, cm_env, tmp_path, robin):
    """A forward run from the recovered state, with the springs relaxed there, reaches the mesh the
    recovery started from. Springs anchored at that mesh instead pull every unloaded guess back
    towards it and carry part of the load the forward run puts on the wall."""
    pytest.importorskip("meshio")
    recovery = _run(binary, cm_env, tmp_path / "recovery", replace=robin)
    target = read_vtu_points(_vtu(recovery, 0))[1]

    forward_dir = tmp_path / "forward"
    forward = _run(binary, cm_env, forward_dir, replace=(
        ("<ReferenceRecovery>true</ReferenceRecovery>", "<ApplyPressureFromFunction>true</ApplyPressureFromFunction>"),
        ("<Nodes>./tetgen/cantilever.node</Nodes>", "<Nodes>../recovery/Results/UnloadedState_Incr1.node</Nodes>"),
        ("<Unit>1</Unit>", "<Unit>1e-3</Unit>"),
        ("<StopTime>2</StopTime>", "<StopTime>0.1</StopTime>")) + robin)
    inflated = read_vtu_points(_vtu(forward, round(0.1 / TIME_STEP)))[1]
    miss = np.abs(inflated - target).max()
    assert miss < TOLERANCE, f"the recovered state inflates to {miss:.3e} from the target"


def test_dashpot_does_not_read_the_relocation_to_a_new_guess_as_a_velocity(binary, cm_env, tmp_path):
    """A dashpot alone, so the Robin traction is beta times the velocity of each triangle. Every
    inflation starts from rest in a relocated reference, so no step of a later one pushes back harder
    than the steepest step of the first; a jump to the next guess read as a velocity spans a whole
    inflation in one step, several times the steepest of the ten ramp steps."""
    meshio = pytest.importorskip("meshio")
    vtu_dir = _run(binary, cm_env, tmp_path, replace=(
        ("<Alpha>5</Alpha>", "<Alpha>0</Alpha>"),
        ("<Beta>0</Beta>", "<Beta>0.05</Beta><Export>true</Export>")))
    cycles = (tmp_path / "Results" / "CycleInfo.dat").read_text().splitlines()[1:]
    assert len(cycles) > 1, "the recovery ended after one inflation"
    first_end = round(float(cycles[0].split()[2]) / TIME_STEP)
    last = max(int(re.search(r"\.(\d+)\.vtu$", p.name).group(1)) for p in vtu_dir.glob("cantilever.*.vtu"))

    def peak(index):
        data = meshio.read(str(_vtu(vtu_dir, index))).cell_data
        return max(p[m == END_SURFACE].max(initial=0) for m, p in zip(data["Material"], data["ContactPressure"]))

    first = max(peak(i) for i in range(1, first_end + 1))
    later = max(peak(i) for i in range(first_end + 1, last + 1))
    assert first > 0, "the dashpot does not act"
    assert later < 2 * first, f"a later inflation peaks at {later:.3e}, the first at {first:.3e}"
