from pathlib import Path

import numpy as np
import pytest

from helpers.compare import compare_columns, read_golden, read_vtu_points, write_golden
from helpers.run import assert_no_petsc_error, run_binary

GOLDEN_DIR = Path(__file__).parent / "golden"
NP = 4
DEFORM_RTOL, DEFORM_ATOL = 1e-4, 1e-8
# Sensor traces are written with 1e-6 absolute precision and the coupled EP
# solve (with mechano-electric feedback) is reproducible only to a few 1e-6
# run-to-run — near solver tolerance. An absolute floor of 2e-5 absorbs that.
SENSOR_RTOL, SENSOR_ATOL = 1e-3, 2e-5

pytestmark = [pytest.mark.mpi, pytest.mark.slow]


@pytest.fixture(scope="module")
def em01_em_out(em01_root, em01_sim_length, cm_env, binary):
    """Run the coupled electromechanics (NewmarkBeta solver + acCELLerate
    plugin, Land17 tension) once on the EM01 cube, shortened to em01_sim_length,
    and return the Results directory."""
    settings = em01_root / "settings"
    xml = (settings / "M_1mm.xml").read_text().replace(
        "<StopTime>1.0</StopTime>", f"<StopTime>{em01_sim_length}</StopTime>"
    )
    (settings / "M_short.xml").write_text(xml)
    run_binary(
        binary("CardioMechanics"),
        ["-settings", "M_short.xml"],
        cwd=settings,
        env=cm_env,
        np=NP,
        timeout=1200,
    )
    return em01_root / "Results"


def _last_vtu(vtu_dir):
    return max(vtu_dir.glob("Cube.*.vtu"), key=lambda p: int(p.stem.split(".")[1]))


def test_em01_deformation(em01_em_out, update_golden):
    pid, pts = read_vtu_points(_last_vtu(em01_em_out / "Cube_vtu"))
    golden_path = GOLDEN_DIR / "em01_em_deformation.npz"
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
            f"node PointID={int(pid[i])} |delta|={d[i]:.3e} m"
        )


def test_em01_coupled_sensors(em01_em_out, update_golden):
    """P8 Vm/Cai from the plugin EP solve (with mechano-electric feedback), a
    distinct signal from the standalone acCELLerate run."""
    from helpers.compare import read_headerless

    ep = em01_em_out / "EP"
    t_vm, vm = read_headerless(ep / "P8_Vm.txt", ["t", "Vm"])[1].T
    t_cai, cai = read_headerless(ep / "P8_Cai.txt", ["t", "Cai"])[1].T
    assert np.allclose(t_vm, t_cai), "Vm and Cai sensor time bases differ"
    actual = (["t", "Vm", "Cai"], np.column_stack([t_vm, vm, cai]))

    golden_path = GOLDEN_DIR / "em01_em_sensors.csv"
    if update_golden:
        GOLDEN_DIR.mkdir(exist_ok=True)
        write_golden(golden_path, *actual)
        pytest.skip(f"updated golden {golden_path.name}")
    assert golden_path.is_file(), f"missing golden {golden_path}; run with --update-golden"
    compare_columns(actual, read_golden(golden_path), rtol=SENSOR_RTOL, atol=SENSOR_ATOL)


def _generalized_alpha(xml):
    """Settings text with the example's NewmarkBeta solver replaced by generalized-alpha."""
    solver = xml[xml.index("<Type>NewmarkBeta</Type>"):xml.index("</NewmarkBeta>") + len("</NewmarkBeta>")]
    return xml.replace(solver, "<Type>GeneralizedAlpha</Type>\n"
                               "    <GeneralizedAlpha>\n"
                               "        <RhoInf>0.8</RhoInf>\n"
                               "        <ConsistentMassMatrix>true</ConsistentMassMatrix>\n"
                               "    </GeneralizedAlpha>")


def test_em01_runs_under_generalized_alpha(em01_root, em01_sim_length, cm_env, binary):
    """Smoke test that the coupled path survives the new integrator.

    Generalized-alpha is a different scheme from the NewmarkBeta settings the
    example ships with, so the results legitimately differ and there is nothing
    to compare against. All this asserts is that electrophysiology coupling,
    the cell models, the stimuli and the sensors still run end to end. Output
    goes to its own folder so the golden run above is left alone.
    """
    settings = em01_root / "settings"
    xml = (settings / "M_1mm.xml").read_text()
    for old, new in (("<StopTime>1.0</StopTime>", f"<StopTime>{em01_sim_length}</StopTime>"),
                     ("../Results/", "../ResultsGenAlpha/")):
        assert old in xml, f"M_1mm.xml no longer contains {old!r}"
        xml = xml.replace(old, new)
    (settings / "M_short_genalpha.xml").write_text(_generalized_alpha(xml))
    (em01_root / "ResultsGenAlpha").mkdir(exist_ok=True)
    run_binary(
        binary("CardioMechanics"),
        ["-settings", "M_short_genalpha.xml"],
        cwd=settings,
        env=cm_env,
        np=NP,
        timeout=1200,
    )
    vtu = em01_root / "ResultsGenAlpha" / "Cube_vtu"
    assert list(vtu.glob("Cube.*.vtu")), f"no deformation output written to {vtu}"


def test_em01_runs_on_t4mini(em01_root, em01_sim_length, cm_env, binary):
    """Land17 takes the calcium of the electrophysiology through a dispatch on the element type,
    which has to know T4MINI. T4MINI does not support NewmarkBeta, so the run uses
    generalized-alpha, and there is again nothing to compare against."""
    settings = em01_root / "settings"
    xml = (settings / "M_1mm.xml").read_text()
    for old, new in (("<StopTime>1.0</StopTime>", f"<StopTime>{em01_sim_length}</StopTime>"),
                     ("../Results/", "../ResultsT4Mini/"),
                     ("<Type>T4</Type>", "<Type>T4MINI</Type>")):
        assert old in xml, f"M_1mm.xml no longer contains {old!r}"
        xml = xml.replace(old, new)
    (settings / "M_short_t4mini.xml").write_text(_generalized_alpha(xml))
    (em01_root / "ResultsT4Mini").mkdir(exist_ok=True)
    proc = run_binary(
        binary("CardioMechanics"),
        ["-settings", "M_short_t4mini.xml"],
        cwd=settings,
        env=cm_env,
        np=NP,
        timeout=1200,
    )
    assert_no_petsc_error(proc.stdout + proc.stderr)
    assert "SIMULATION FAILED" not in proc.stdout, proc.stdout[-2000:]
    vtu = em01_root / "ResultsT4Mini" / "Cube_vtu"
    assert list(vtu.glob("Cube.*.vtu")), f"no deformation output written to {vtu}"
