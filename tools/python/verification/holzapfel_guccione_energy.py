#!/usr/bin/env python3
"""Strain energy of Holzapfel and Guccione at the deformations of the law-level GoogleTest.

Prints the values that mechanics/tests/ConstitutiveModelTest.cpp hard-codes. The energy functions
are written from the published strain energy functions, independently of the C++ laws, in the
stretches along fibre, sheet and sheet normal of a diagonal F, where the fibre-sheet coupling
terms vanish. The parameters are those of the verification tests (tests/helpers/materials.py),
with bulk modulus KAPPA.

Holzapfel: W = a/(2b) (exp(b (I1_bar - 3)) - 1) + sum over f, s of H(I4) a_i/(2b_i) (exp(b_i (I4 - 1)^2) - 1)
               + kappa/4 (J^2 - 1 - 2 ln J),
with H(I4) = 1 / (1 + exp(-k (I4 - 1))) and the fibre and sheet terms in the full I4.
Guccione:  W = C/2 (exp(bf E_ff^2 + bt (E_ss^2 + E_nn^2)) - 1) + K/2 (J - 1)^2,
with E the Green strain of the isochoric part of F.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from helpers.materials import LAWS  # noqa: E402

KAPPA = 10
# Diagonal stretches along fibre, sheet and sheet normal. Holzapfel's fibre is stretched and its
# sheet compressed in the first, the reverse in the second, and both compressed in the third, so
# the switch is evaluated on both flanks for either family.
STRETCHES = ((1.12, 0.93, 0.97), (0.90, 1.08, 1.02), (0.95, 0.97, 1.10))


def holzapfel_energy(lf, ls, ln, p):
    J = lf * ls * ln
    I1 = J ** (-2 / 3) * (lf ** 2 + ls ** 2 + ln ** 2)

    def switched(a, b, I4):
        return a / (2 * b) * (np.exp(b * (I4 - 1) ** 2) - 1) / (1 + np.exp(-p["k"] * (I4 - 1)))

    return (p["a"] / (2 * p["b"]) * (np.exp(p["b"] * (I1 - 3)) - 1)
            + switched(p["af"], p["bf"], lf ** 2) + switched(p["as"], p["bs"], ls ** 2)
            + KAPPA / 4 * (J ** 2 - 1 - 2 * np.log(J)))


def guccione_energy(lf, ls, ln, p):
    J = lf * ls * ln
    ef, es, en = (0.5 * (J ** (-2 / 3) * l ** 2 - 1) for l in (lf, ls, ln))
    return p["C"] / 2 * (np.exp(p["bf"] * ef ** 2 + p["bt"] * (es ** 2 + en ** 2)) - 1) + KAPPA / 2 * (J - 1) ** 2


if __name__ == "__main__":
    for law, energy in (("Holzapfel", holzapfel_energy), ("Guccione", guccione_energy)):
        for stretches in STRETCHES:
            print(f"{law:10} {stretches}  W = {energy(*stretches, LAWS[law][1]):.17g}")
