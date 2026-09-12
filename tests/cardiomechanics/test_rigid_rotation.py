"""Large rigid rotation of a distorted box, which must leave it free of strain and stress.

The points-control plugin drives every node of the face x_0 = 0 of the box of helpers.box along
x = c + R(t) (X - c), a rotation by up to 90 degrees about a skew axis through the box centre c.
Every other node is free, so the only equilibrium is the rigidly rotated body, F = R. The motion
lies in the span of T4, T10 and P2P1 on affine elements, so every node must follow it to solver
precision however large the angle. The Green-Lagrange strain (F^T F - I) / 2 then vanishes, and so
must the stress of any law that depends on F only through C = F^T F, as a frame-indifferent law
does. The solver computes the exported strain from the element's F, so the strain exposes an
element whose F is not the rotation; the stress also exposes a law or element that uses F where it
should use C, or linearised kinematics. Either error is of the order of the rotation. The skew axis
leaves no component of the motion trivially zero, and the centre makes it a rotation plus a
translation.

Holzapfel and Guccione run with their fibre, sheet and sheet-normal directions along the box axes;
under C = I every fibre invariant sits at its stress-free value whatever the basis.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from helpers.box import write_box
from helpers.compare import read_vtu_cell_field, read_vtu_point_field, read_vtu_points
from helpers.materials import LAWS, material_block
from helpers.run import run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "rigid_rotation_box.xml"
STOP_TIME = float(re.search(r"<StopTime>(.*?)</StopTime>", FIXTURE.read_text()).group(1))
STEPS = 4                                  # StopTime / TimeStep of the fixture, one export per step
ELEMENTS = ("T4", "T10", "T10P1")
# Nearly incompressible, where the kappa term is stiffest and a spurious volume change costs most.
KAPPA = 1000
ANGLE = np.pi / 2                          # at the stop time
AXIS = np.array([1, 2, 3]) / np.sqrt(14)
CENTRE = np.full(3, 0.5)

# The VTU points are single precision, and every coordinate stays below 2 through the rotation, so
# rounding moves each by at most 1.2e-7 in the reference and the rotated frame alike. Observed 8e-8.
POSITION_TOL = 1e-6
# Strain, stress and pressure are written in double precision, so only the solver Precision of
# 1e-12 on the residual limits them. The strain carries the stopping error of the iteration, observed
# at most 7e-12. A rotation handled as anything but rigid would give a strain of order 1e-1 or more.
STRAIN_TOL = 1e-10
# The bulk modulus turns a volumetric strain error into a stress and pressure error KAPPA times as
# large. Observed at most 3.3e-10 in the stress, under Guccione's stiff fibre exponent, and 1.5e-11
# in the pressure.
STRESS_TOL = PRESSURE_TOL = KAPPA * STRAIN_TOL


def rotate(points, fraction):
    """points rigidly rotated by fraction of ANGLE about AXIS through CENTRE (Rodrigues' formula)."""
    K = np.cross(np.eye(3), AXIS)           # K v = AXIS x v
    theta = fraction * ANGLE
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return CENTRE + (points - CENTRE) @ R.T


def write_rotation(directory, driven):
    """Write the driven node indices and their rotated coordinates at every step, in the formats
    of the points-control plugin. driven holds rows of index, x, y, z. The plugin skips a target of
    exactly (0, 0, 0), which only the origin corner has, at t = 0, where it is at rest anyway."""
    (directory / "driven.txt").write_text("".join(f"{int(n)}\n" for n in driven[:, 0]))
    with open(directory / "rotation.list", "w") as f:
        for k in range(STEPS + 1):
            dat = directory / f"rotation.{k}.dat"
            x = rotate(driven[:, 1:], k / STEPS)
            with open(dat, "wb") as d:      # native byte order, as the plugin reads it
                np.int32(x.size).tofile(d)
                x.astype(np.float64).tofile(d)
            f.write(f"{k / STEPS * STOP_TIME} {dat}\n")


@pytest.fixture(scope="module", params=[(e, law) for e in ELEMENTS for law in LAWS], ids="-".join)
def solution(request, binary, cm_env, tmp_path_factory):
    """Run the box under one element type and law. Returns (label, fields), where fields[k] holds
    the exported PointID, points, CellID, strain, stress and pressure (or None) of step k."""
    pytest.importorskip("meshio")
    element_type, law = request.param
    label = f"{element_type} {law}"
    wd = tmp_path_factory.mktemp(f"{element_type}_{law}")
    tetgen = wd / "tetgen"
    tetgen.mkdir()
    (wd / "Results").mkdir()
    write_box(tetgen, quadratic=element_type != "T4", basis=np.eye(3), symmetry_planes=False)
    # Rows of index, x, y, z and fixation. The jitter keeps every face node within its face, so the
    # nodes of the face x_0 = 0 have x written as exactly 0.
    nodes = np.loadtxt(tetgen / "box.node", skiprows=1)
    write_rotation(tetgen, nodes[nodes[:, 1] == 0, :4])

    text, n = re.subn(r"<NeoHooke>.*?</NeoHooke>", material_block(law, KAPPA), FIXTURE.read_text(), flags=re.DOTALL)
    assert n == 1, f"{FIXTURE.name}: cannot substitute the NeoHooke parameters"
    for old, new in [("<Type>T10P1</Type>", f"<Type>{element_type}</Type>"),
                     ("<Type>NeoHooke</Type>", f"<Type>{law}</Type>")]:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / FIXTURE.name).write_text(text)
    proc = run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd, env=cm_env, timeout=600)
    assert "SIMULATION FAILED" not in proc.stdout, proc.stdout[-2000:]

    fields = []
    for k in range(STEPS + 1):
        vtu = wd / "Results" / "box_vtu" / f"box.{k}.vtu"
        assert vtu.exists(), f"{label}: run stopped before step {k}"
        pid, points = read_vtu_points(vtu)
        cid, strain = _symmetric_tensor(vtu, "E")
        _, stress = _symmetric_tensor(vtu, "S")
        pressure = read_vtu_point_field(vtu, "Pressure")[1] if element_type == "T10P1" else None
        fields.append((pid, points.astype(float), cid, strain, stress, pressure))
    return label, fields


def _symmetric_tensor(vtu, name):
    """CellID and the six components of the per-element tensor exported as name_ii and name_ij."""
    cid, diagonal = read_vtu_cell_field(vtu, f"{name}_ii")
    _, off_diagonal = read_vtu_cell_field(vtu, f"{name}_ij")
    return cid, np.hstack([diagonal, off_diagonal])


def _assert_small(label, step, what, ids, id_name, misfit, tol):
    """Fails naming the row of misfit, one per node or element, with the largest component."""
    misfit = misfit.reshape(len(ids), -1)
    worst = np.abs(misfit).max(axis=1)
    i = int(np.argmax(worst))
    assert worst[i] <= tol, (f"{label}, step {step}: {what} beyond {tol:.1e}, worst {id_name}={int(ids[i])} "
                             f"with {np.array2string(misfit[i], precision=3)}")


def test_nodes_follow_rigid_rotation(solution):
    label, fields = solution
    pid, ref = fields[0][:2]
    for k, (_, points, *_) in enumerate(fields):
        _assert_small(label, k, "distance from the rotated reference", pid, "PointID",
                      points - rotate(ref, k / STEPS), POSITION_TOL)


def test_strain_is_zero(solution):
    label, fields = solution
    for k, (_, _, cid, strain, _, _) in enumerate(fields):
        _assert_small(label, k, "Green-Lagrange strain", cid, "CellID", strain, STRAIN_TOL)


def test_stress_is_zero(solution):
    label, fields = solution
    for k, (_, _, cid, _, stress, _) in enumerate(fields):
        _assert_small(label, k, "PK2 stress", cid, "CellID", stress, STRESS_TOL)


def test_p2p1_pressure_is_zero(solution):
    label, fields = solution
    if fields[0][5] is None:
        pytest.skip("displacement-only element, no pressure field")
    for k, (pid, _, _, _, _, pressure) in enumerate(fields):
        _assert_small(label, k, "pressure", pid, "PointID", pressure, PRESSURE_TOL)
