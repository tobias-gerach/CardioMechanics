"""ApplyPressureFromFunctionNodeExport: a pressure load that stops the run and writes the deformed
nodes as a tetgen node file, once a cavity volume or a time is reached.

The cantilever of test_p2p1 is loaded through the NodeExport plugin instead of the plain one. With
the whole boundary as surface 130, a negative pressure shrinks the enclosed 4e6 mL by about 30 mL
per step of 0.25 s.
"""
from pathlib import Path

import numpy as np

from helpers.cantilever import write_mesh
from helpers.compare import read_vtu_points
from helpers.run import assert_refused, run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "p2p1_cantilever.xml"
NODE_FILE = "nodes.node"


def _run(binary, cm_env, wd, settings, closed=True, loaded="130", check=True):
    """Stage the cantilever into wd with its pressure applied by the NodeExport plugin to the surfaces
    loaded, and the plugin also reading settings, an XML fragment. Unless closed, the cantilever has the
    top surface 130 and the free end 131 instead of surface 130 all round."""
    (wd / "tetgen").mkdir()
    write_mesh(wd / "tetgen", closed=closed)
    (wd / "Results").mkdir()
    text = FIXTURE.read_text().replace("ApplyPressureFromFunction", "ApplyPressureFromFunctionNodeExport")
    substitutions = [("<Amplitude>0.003</Amplitude>", "<Amplitude>-0.003</Amplitude>"),
                     ("<Groups>", f"<NodeExportFile>./Results/{NODE_FILE}</NodeExportFile>{settings}<Groups>")]
    if loaded != "130":
        substitutions.append(("<Surfaces>130</Surfaces>", f"<Surfaces>{loaded}</Surfaces>"))
    for old, new in substitutions:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    (wd / FIXTURE.name).write_text(text)
    return run_binary(binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd, env=cm_env, check=check)


def _times(wd):
    """The times of the steps the plugin wrote to its pressure and volume table."""
    return [float(line.split()[0]) for line in (wd / "Results" / "Pressure.dat").read_text().splitlines()[1:]]


def _last_vtu(wd):
    return max((wd / "Results" / "cantilever_vtu").glob("cantilever.*.vtu"), key=lambda p: int(p.stem.split(".")[1]))


def test_stop_volume_ends_the_run_and_exports_the_final_geometry(binary, cm_env, tmp_path):
    """The volume falls below 3999950 mL in the step to 0.5 s, which is the last one computed. The
    node file holds that step's geometry in mm and the clamp of the input mesh as a bitmask."""
    _run(binary, cm_env, tmp_path, "<StopSurface>130</StopSurface><StopVolume>3999950</StopVolume>")
    assert _times(tmp_path) == [0.0, 0.25, 0.5]

    header, *rows = (tmp_path / "Results" / NODE_FILE).read_text().splitlines()
    inputs = (tmp_path / "tetgen" / "cantilever.node").read_text().splitlines()[1:]
    assert header.split() == [str(len(inputs)), "3", "1", "0"]
    assert len(rows) == len(inputs)
    table = np.array([row.split() for row in rows], dtype=float)
    assert np.array_equal(table[:, 0], np.arange(1, len(rows) + 1))
    assert np.array_equal(table[:, 4], [float(line.split()[4]) for line in inputs])
    _, points = read_vtu_points(_last_vtu(tmp_path))
    # The node file is written at the stream's default six significant digits.
    assert np.allclose(table[:, 1:4], 1e3 * points, rtol=1e-5, atol=1e-6)


def test_export_time_between_steps_stops_at_the_nearest_step(binary, cm_env, tmp_path):
    """No step lands on 0.7 s. The step to 0.75 s is the first within half a step of it."""
    _run(binary, cm_env, tmp_path, "<NodeExportTime>0.7</NodeExportTime>")
    assert _times(tmp_path) == [0.0, 0.25, 0.5, 0.75]
    assert (tmp_path / "Results" / NODE_FILE).is_file()


def test_export_time_writes_the_node_file_once_for_several_cavities(binary, cm_env, tmp_path):
    proc = _run(binary, cm_env, tmp_path, "<NodeExportTime>0.5</NodeExportTime>", closed=False, loaded="130,131")
    assert _times(tmp_path) == [0.0, 0.25, 0.5]
    assert proc.stdout.count(f"Exported nodes to ./Results/{NODE_FILE}") == 1, proc.stdout[-2000:]


def test_stop_surface_without_pressure_is_refused(binary, cm_env, tmp_path):
    """Surface 131 exists but no group loads it, so its volume is never computed."""
    proc = _run(binary, cm_env, tmp_path, "<StopSurface>131</StopSurface><StopVolume>0</StopVolume>",
                closed=False, check=False)
    assert_refused(proc, "StopSurface", "131")
