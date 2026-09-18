#!/usr/bin/env python3
"""Strain energy of Holzapfel, Guccione and Usyk at the deformations of the law-level GoogleTest.

Prints the values that mechanics/tests/ConstitutiveModelTest.cpp hard-codes. The energy functions
are written from the published strain energy functions, independently of the C++ laws, with the
fibre, sheet and sheet normal along the x, y and z axes. The parameters are those of the
verification tests (tests/helpers/materials.py), with bulk modulus KAPPA.

Holzapfel: W = a/(2b) (exp(b (I1_bar - 3)) - 1) + sum over f, s of H(I4) a_i/(2b_i) (exp(b_i (I4 - 1)^2) - 1)
               + afs/(2bfs) (exp(bfs I8fs^2) - 1) + kappa/4 (J^2 - 1 - 2 ln J),
with H(I4) = 1 / (1 + exp(-k (I4 - 1))). The fibre and sheet terms are in the full I4, not its
isochoric part, which is the variant the code implements.
Guccione:  W = C/2 (exp(Q) - 1) + K/2 (J - 1)^2,
Q = bf E_ff^2 + bt (E_ss^2 + E_nn^2 + 2 E_sn^2) + bfs (2 E_fs^2 + 2 E_fn^2).
Usyk:      W = a/2 (exp(Q) - 1) + k/2 (ln J)^2,
Q = bff E_ff^2 + bss E_ss^2 + bnn E_nn^2 + 2 bfs E_fs^2 + 2 bfn E_fn^2 + 2 bns E_ns^2.
In Guccione and Usyk, E is the Green strain of the isochoric part of F.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from helpers.materials import LAWS  # noqa: E402

KAPPA = 10
# Rows of F. In the three diagonal ones Holzapfel's fibre is stretched in the first and compressed
# in the other two, its sheet compressed in the first and third, so each switch is evaluated below
# and above I4 = 1. The coupling terms vanish there; the last F shears every pair of axes, so the
# fibre-sheet term of Holzapfel and every shear term of Guccione and Usyk contribute.
DEFORMATIONS = (np.diag([1.12, 0.93, 0.97]), np.diag([0.90, 1.08, 1.02]), np.diag([0.95, 0.97, 1.10]),
                np.array([[1.05, 0.12, 0.04], [0.03, 0.96, 0.08], [-0.05, 0.06, 1.02]]))


def holzapfel_energy(F, p):
    C = F.T @ F
    J = np.linalg.det(F)
    I1 = J ** (-2 / 3) * np.trace(C)

    def switched(a, b, I4):
        return a / (2 * b) * (np.exp(b * (I4 - 1) ** 2) - 1) / (1 + np.exp(-p["k"] * (I4 - 1)))

    return (p["a"] / (2 * p["b"]) * (np.exp(p["b"] * (I1 - 3)) - 1)
            + switched(p["af"], p["bf"], C[0, 0]) + switched(p["as"], p["bs"], C[1, 1])
            + p["afs"] / (2 * p["bfs"]) * (np.exp(p["bfs"] * C[0, 1] ** 2) - 1)
            + KAPPA / 4 * (J ** 2 - 1 - 2 * np.log(J)))


def guccione_energy(F, p):
    J = np.linalg.det(F)
    E = 0.5 * (J ** (-2 / 3) * F.T @ F - np.eye(3))
    Q = (p["bf"] * E[0, 0] ** 2 + p["bt"] * (E[1, 1] ** 2 + E[2, 2] ** 2 + 2 * E[1, 2] ** 2)
         + p["bfs"] * (2 * E[0, 1] ** 2 + 2 * E[0, 2] ** 2))
    return p["C"] / 2 * (np.exp(Q) - 1) + KAPPA / 2 * (J - 1) ** 2


def usyk_energy(F, p):
    J = np.linalg.det(F)
    E = 0.5 * (J ** (-2 / 3) * F.T @ F - np.eye(3))
    Q = (p["bff"] * E[0, 0] ** 2 + p["bss"] * E[1, 1] ** 2 + p["bnn"] * E[2, 2] ** 2
         + 2 * p["bfs"] * E[0, 1] ** 2 + 2 * p["bfn"] * E[0, 2] ** 2 + 2 * p["bns"] * E[1, 2] ** 2)
    return p["a"] / 2 * (np.exp(Q) - 1) + KAPPA / 2 * np.log(J) ** 2


if __name__ == "__main__":
    for law, energy in (("Holzapfel", holzapfel_energy), ("Guccione", guccione_energy), ("Usyk", usyk_energy)):
        for F in DEFORMATIONS:
            print(f"{law:10} F = {F.flatten().tolist()}  W = {energy(F, LAWS[law][1]):.17g}")
