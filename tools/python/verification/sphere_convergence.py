#!/usr/bin/env python3
"""Convergence of P2P1 and MINI on the inflated thick-walled sphere under both quadrature rules of
Mesh.QuadratureDegree, over the coarse and fine mesh families, and the rates of T10.

The problem, the reference and the error norms are those of
tests/cardiomechanics/test_sphere_convergence.py, which asserts the rates of the coarse family under
the default rule. This script prints the errors of every level under each rule with the rates
between consecutive levels, the last of which is the rate of the fine family, and the ratio of the
errors under the two rules at each level.

P2P1 runs under the default 4-point rule of degree 2 and the 14-point rule of degree 5. J is cubic
on an affine element, so the constraint integrand N_a (J - 1 - p / kappa) is of degree 4, which only
the second integrates exactly. The rules also differ in their weights: the default takes the
determinant of the element map at the centroid, the 14-point rule at each point. Only the
determinant matters here. On the curved elements at the spheres the centroid determinant nearly
doubles the finest displacement L2 and pressure errors, whereas the 4-point rule with pointwise
determinants matches the 14-point rule to 0.6%. Before the asymptotic range the P2P1 rates of the
fine family fall short of the orders by up to 0.20, in the pressure.

T4MINI runs under both rules as well. Its elements are affine, so the rules share the determinant
and differ only in the terms the bubble makes non-polynomial of high degree, which neither
integrates exactly. Both keep the MINI orders, so the 4-point rule does not destabilise the
pressure, and the 14-point rule is the less accurate: at the finest level its displacement errors
are 4% (L2) and 2% (H1) larger and its pressure error 63% larger.

T10 converges to the same solution but locks at this kappa, which also slows its Newton iteration,
so it runs on the coarse family under the default rule only.

Serial runtime, P2P1: 0.3 s, 1.6 s and 27 s at sizes 0.5, 0.25 and 0.125; the finest level takes
about 20 min. T10 needs 87 s at size 0.125. T4MINI needs 3 s, 24 s and 4.5 min at sizes 0.25, 0.125
and 0.0625 under the 4-point rule, and about 2.5 times as long under the 14-point rule.

Run from anywhere; the binary is found as by the tests, so set CM_BIN_DIR to the Release build.
"""
import os
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parents[3] / "tests"
sys.path[:0] = [str(TESTS), str(TESTS / "cardiomechanics")]
from conftest import _find_binary  # noqa: E402
from test_sphere_convergence import (KAPPA, ORDERS, PRESSURE, SHEAR, SIZES, _run_octant, _table,  # noqa: E402
                                     radial_solution)

DEGREES = (2, 5)                    # Mesh.QuadratureDegree: the default 4-point and the 14-point rule


if __name__ == "__main__":
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    reference = radial_solution(PRESSURE, SHEAR, KAPPA)
    with tempfile.TemporaryDirectory() as tmp:
        def family(element_type, sizes, degree):
            runs = []
            for size in sizes:
                wd = Path(tmp) / f"{element_type}_{size}_{degree}"
                wd.mkdir()
                runs.append(_run_octant(_find_binary, env, wd, element_type, size, reference, degree))
            return runs

        for element_type in ORDERS:
            by_degree = {d: family(element_type, SIZES, d) for d in DEGREES}
            for d, runs in by_degree.items():
                print(_table(f"{element_type} degree {d}", runs), flush=True)
            norms = [k for k in by_degree[DEGREES[0]][0] if k != "h"]
            print(f"degree {DEGREES[1]} / degree {DEGREES[0]}: " + "  ".join(f"{k:>16}" for k in norms))
            for low, high in zip(*by_degree.values()):
                print("  " + "  ".join(f"{high[k] / low[k]:16.4f}" for k in norms), flush=True)
        print(_table(f"T10 degree {DEGREES[0]}", family("T10", SIZES[:3], DEGREES[0])), flush=True)
