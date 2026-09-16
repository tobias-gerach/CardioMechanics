#!/usr/bin/env python3
"""Sensitivity of the strain output to kappa, under each displacement-only element and its mixed
counterpart on the same mesh: T10 against P2P1, and T4 against T4MINI.

In the displacement-only formulation kappa is a penalty whose value the modeller has no
principled way to choose, and T10 and T4 strains move with it: the stiffer the penalty, the more
the element locks. In the mixed elements the volumetric response is carried by the pressure field,
so the strain should be a property of the material rather than of kappa. This script measures both
on the cantilever of tests/cardiomechanics/test_p2p1.py, for every law with a mixed formulation.

The sweep spans the nearly incompressible range: every modulus of the cantilever is O(1), so
kappa = 100 already puts Poisson's ratio above 0.49. It stops at 1e4 because T10 does not
converge at 1e5 on this fixture, for any of the three laws and whatever the load step, whereas
P2P1, T4 and T4MINI do.

The strain is the Green-Lagrange strain CardioMechanics exports per element, at its centroid. The
spread is the largest range over the sweep of any strain component in any element, and the peak the
largest strain component at the stiffest kappa. Both element types of a pair see the same mesh and
load, so their spreads compare directly. Observed for NeoHooke, Holzapfel and Guccione, T10 moves
4.19, 3.55 and 3.36 times as much as P2P1, and T4 4.03, 4.10 and 3.75 times as much as MINI; the
displacement-only spread is 0.070, 0.068 and 0.127 of its peak for T10 and 0.44, 0.42 and 0.63 for
T4. What the mixed elements keep is genuine compressibility, which no discretization removes: it
converges as 1/kappa, P2P1 moving a tenth as much again from 1e3 to 1e5 and MINI a hundredth as much
again from 1e4 to 1e5. The rest of the displacement-only spread is locking. T4 locks far more: at
kappa = 1e4 its peak strain is a quarter to a third of MINI's.

Run from anywhere; the binary is found as by the tests, so set CM_BIN_DIR to the Release build.
"""
import os
import sys
import tempfile
from pathlib import Path

import meshio
import numpy as np

TESTS = Path(__file__).resolve().parents[3] / "tests"
sys.path[:0] = [str(TESTS), str(TESTS / "cardiomechanics")]
from conftest import _find_binary  # noqa: E402
from test_p2p1 import _last_vtu, _run  # noqa: E402

NP = 4                              # CardioMechanics is tested only in parallel, as in test_benchmark
LAWS = ("NeoHooke", "Holzapfel", "Guccione")
PAIRS = (("T10", "T10P1"), ("T4", "T4MINI"))     # (displacement-only, mixed) on the same mesh
KAPPA_SWEEP = (100, 1000, 10000)
COMPONENTS = ("E11", "E22", "E33", "E12", "E13", "E23")     # E_ii, then E_ij, as exported


def _strain(vtu):
    """(CellID, Green-Lagrange strain of shape (cell, component)) of the tetrahedra, ordered by CellID."""
    m = meshio.read(str(vtu))
    blocks = [i for i, b in enumerate(m.cells) if b.type in ("tetra", "tetra10")]
    cid = np.concatenate([np.asarray(m.cell_data["CellID"][i]).ravel() for i in blocks])
    strain = np.vstack([np.hstack([m.cell_data["E_ii"][i], m.cell_data["E_ij"][i]]) for i in blocks])
    order = np.argsort(cid, kind="stable")
    return cid[order], strain[order]


def sweep(root, element_type, law, env):
    """(CellID, strain of shape (kappa, cell, component)) of the cantilever at every kappa of the sweep."""
    runs = []
    for kappa in KAPPA_SWEEP:
        wd = root / f"{law}_{element_type}_{kappa}"
        wd.mkdir()
        _, vtu_dir = _run(_find_binary, env, wd, element_type=element_type, kappa=kappa, material=law, ranks=NP)
        runs.append(_strain(_last_vtu(vtu_dir)))
    cid = runs[0][0]
    assert all(np.array_equal(cid, r[0]) for r in runs), f"{law} {element_type}: cells differ between runs"
    return cid, np.stack([r[1] for r in runs])


def spread(cid, strain):
    """Spread of strain over the sweep, and the element and component it occurs at."""
    ranges = np.ptp(strain, axis=0)
    c, k = np.unravel_index(np.argmax(ranges), ranges.shape)
    return ranges[c, k], f"CellID={int(cid[c])} {COMPONENTS[k]}"


if __name__ == "__main__":
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    print(f"kappa = {KAPPA_SWEEP}, peak at kappa = {KAPPA_SWEEP[-1]}")
    print(f"{'law':10} {'element':7} {'spread':>10} {'peak':>10} {'spread/peak':>11} {'disp/mixed':>10}  at")
    with tempfile.TemporaryDirectory() as tmp:
        for pair in PAIRS:
            for law in LAWS:
                results = {}
                for element_type in pair:
                    cid, strain = sweep(Path(tmp), element_type, law, env)
                    results[element_type] = *spread(cid, strain), np.abs(strain[-1]).max()
                reference = results[pair[0]][0]
                for element_type, (s, where, peak) in results.items():
                    print(f"{law:10} {element_type:7} {s:10.3e} {peak:10.3e} {s / peak:11.3f} "
                          f"{reference / s:10.2f}  {where}")
