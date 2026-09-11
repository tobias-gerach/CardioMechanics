"""Sensitivity of the strain output to kappa, under T10 and under P2P1.

In the displacement-only formulation kappa is a penalty whose value the modeller has no
principled way to choose, and T10 strains move with it: the stiffer the penalty, the more the
element locks. In P2P1 the volumetric response is carried by the pressure field, so the strain
should be a property of the material rather than of kappa. This module measures both on the
cantilever of test_p2p1, for every law with a mixed formulation.

The sweep spans the nearly incompressible range: every modulus of the cantilever is O(1), so
kappa = 100 already puts Poisson's ratio above 0.49. It stops at 1e4 because T10 does not
converge at 1e5 on this fixture, for any of the three laws and whatever the load step, whereas
P2P1 does.

The strain is the Green-Lagrange strain CardioMechanics exports per element, at its centroid.
"""
import shutil

import numpy as np
import pytest

from test_p2p1 import _last_vtu, _run

NP = 4                              # CardioMechanics is tested only in parallel, as in test_benchmark
LAWS = ("NeoHooke", "Holzapfel", "Guccione")
KAPPA_SWEEP = (100, 1000, 10000)
COMPONENTS = ("E11", "E22", "E33", "E12", "E13", "E23")     # E_ii, then E_ij, as exported

# Spread: largest range over the sweep of any strain component in any element. Both element
# types see the same mesh and load, so their spreads compare directly. Observed for NeoHooke,
# Holzapfel and Guccione, T10 moves 4.19, 3.55 and 3.36 times as much as P2P1. What P2P1 keeps
# is genuine compressibility, which no discretization removes: it converges as 1/kappa, P2P1
# moving a tenth as much again from 1e3 to 1e5. The rest of the T10 spread is locking. Half
# leaves a margin under the worst observed ratio and still fails well before P2P1 moves as much
# as T10.
MAX_SPREAD_RATIO = 0.5              # P2P1 spread over T10 spread
# T10 spread over its peak strain at the stiffest kappa, observed 0.070, 0.068 and 0.127: about
# half the smallest, so a fixture that stops locking fails here rather than passing the comparison.
MIN_T10_SPREAD = 0.03

pytestmark = pytest.mark.mpi


def _strain(vtu):
    """(CellID, Green-Lagrange strain of shape (cell, component)) of the tetrahedra, ordered by CellID."""
    import meshio

    m = meshio.read(str(vtu))
    blocks = [i for i, b in enumerate(m.cells) if b.type == "tetra10"]
    cid = np.concatenate([np.asarray(m.cell_data["CellID"][i]).ravel() for i in blocks])
    strain = np.vstack([np.hstack([m.cell_data["E_ii"][i], m.cell_data["E_ij"][i]]) for i in blocks])
    order = np.argsort(cid, kind="stable")
    return cid[order], strain[order]


@pytest.fixture(scope="module", params=LAWS)
def sweep(request, binary, cm_env, tmp_path_factory):
    """Run the cantilever at every kappa of the sweep under both element types, for one law.
    Returns (law, {element type: (CellID, strain of shape (kappa, cell, component))})."""
    pytest.importorskip("meshio")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    law, strains = request.param, {}
    for element_type in ("T10", "T10P1"):
        runs = []
        for kappa in KAPPA_SWEEP:
            wd = tmp_path_factory.mktemp(f"{law}_{element_type}_{kappa}")
            _, vtu_dir = _run(binary, cm_env, wd, element_type=element_type, kappa=kappa, material=law, ranks=NP)
            runs.append(_strain(_last_vtu(vtu_dir)))
        cid = runs[0][0]
        assert all(np.array_equal(cid, r[0]) for r in runs), f"{law} {element_type}: cells differ between runs"
        strains[element_type] = cid, np.stack([r[1] for r in runs])
    return law, strains


def _spread(cid, strain):
    """Spread of strain over the sweep, and a description of the element and component it occurs at."""
    ranges = np.ptp(strain, axis=0)
    c, k = np.unravel_index(np.argmax(ranges), ranges.shape)
    values = ", ".join(f"{e:.4e}" for e in strain[:, c, k])
    return ranges[c, k], f"CellID={int(cid[c])} {COMPONENTS[k]} = {values}"


def _peak(strain):
    """Largest strain component of any element at the stiffest kappa."""
    return np.abs(strain[-1]).max()


def _report(law, strains):
    lines = []
    for element_type, (cid, strain) in strains.items():
        spread, where = _spread(cid, strain)
        lines.append(f"{law} {element_type}: spread {spread:.3e} at {where}, "
                     f"peak {_peak(strain):.3e} at kappa = {KAPPA_SWEEP[-1]}")
    return "; ".join(lines)


def test_t10_strain_moves_with_kappa(sweep):
    """Without a kappa-sensitive T10 there is nothing for P2P1 to improve on."""
    law, strains = sweep
    spread, _ = _spread(*strains["T10"])
    peak = _peak(strains["T10"][1])
    assert spread >= MIN_T10_SPREAD * peak, (f"T10 spread under {MIN_T10_SPREAD} of its peak over "
                                             f"kappa = {KAPPA_SWEEP}: {_report(law, strains)}")


def test_p2p1_strain_less_sensitive_to_kappa_than_t10(sweep):
    law, strains = sweep
    t10, _ = _spread(*strains["T10"])
    p2p1, _ = _spread(*strains["T10P1"])
    assert p2p1 <= MAX_SPREAD_RATIO * t10, (
        f"P2P1 spread {p2p1:.3e} is {p2p1 / t10:.2f} of the T10 spread {t10:.3e}, above {MAX_SPREAD_RATIO}, "
        f"over kappa = {KAPPA_SWEEP}: {_report(law, strains)}")
