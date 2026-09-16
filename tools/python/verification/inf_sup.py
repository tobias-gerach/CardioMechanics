#!/usr/bin/env python3
"""Numerical inf-sup study of P2P1 (Chapelle and Bathe 1993) on the system matrix of the code.

The perturbed constraint J - 1 - p / kappa keeps the saddle-point system non-singular at any kappa,
so a pressure mode that the displacement cannot control does not make a solve fail. It shows up as
loose, noisy pressure and volume ratio instead. Taylor-Hood P2P1 in 3D is proven stable only on
meshes where every tetrahedron has a vertex inside the body (Boffi 1997), and the thin walls the
code is used on, two to four elements thick, are full of tetrahedra with all four vertices on the
boundary.

On a body clamped on its whole boundary, the discrete inf-sup constant is

    beta_h = inf_p sup_v (p, div v) / (|p|_L2 |grad v|_L2),

with p ranging over the pressures of zero mean, since the constant pressure does no work against a
clamped displacement. beta_h^2 is the smallest eigenvalue of G T^-1 G^T p = lambda M p on that
space, with G the coupling block, T the vector Laplacian of the free displacements and M the
pressure mass matrix. Clamping removes the most displacement freedom from the boundary elements, so
it is the hardest case for them; free and loaded faces only enlarge the sup.

G and M come from the system matrix the code assembles, exported by PETSc at the first Newton step
of an unloaded run, where the displacement and the pressure vanish. There, the coupling rows are
integral N_a J C^-1 : dE = integral N_a div v, and the pressure block is -M / kappa. T is assembled
here, from the gmsh basis functions, because the displacement block of the code is the deviatoric
Neo-Hooke stiffness alone at the reference state, not a norm. In a serial run the unknowns are the
three components of every node in node order, then the pressures of the vertex nodes in node order.

Two families are refined: the unit cube, and an octant of the shell INNER <= |X| <= OUTER whose
wall is two, three and four elements thick. A stable element keeps beta_h bounded away from zero as
the mesh is refined, an unstable one loses it at a rate in h, as Chapelle and Bathe show: Q1-P0,
the classical failure of the test, loses it at the rate h.

Observed, the rate fitted to log beta_h over log h at the three levels is below 0.01 in magnitude on
both families, while unstructured meshes scatter beta_h by up to 4% from one level to the next, a
rate of 0.1 between the two finest boxes. A spurious mode would sit at round-off: the P1P1 pair on
the same boxes gives beta_h of 4e-5 at most. The smallest beta_h is 0.12, on the shell, whose thin
wall lowers the continuous constant as well.

Run from anywhere; the binary is found as by the tests, so set CM_BIN_DIR to the Release build.
"""
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lobpcg, splu

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from conftest import _find_binary  # noqa: E402
from helpers.compare import read_petsc_csr  # noqa: E402
from helpers.gmsh_tetgen import tetrahedron_quadrature, write_tetgen  # noqa: E402
from helpers.run import run_binary  # noqa: E402

FIXTURE = Path(__file__).parent / "inf_sup.xml"
KAPPA = float(re.search(r"<k>(.*?)</k>", FIXTURE.read_text()).group(1))
MATERIAL, BOUNDARY = 30, 1                 # physical tags of the volume and of its whole boundary
CLAMPED = 7                                # fixation mask of a node with all three components fixed
INNER, OUTER = 1.0, 1.2                    # radii of the thin shell
# gmsh element sizes. gmsh edges come out about 1.2 times the size, so the shell sizes are those that
# put two, three and four elements through its wall.
SIZES = {"box": (0.25, 0.125, 0.0833), "shell": (0.1, 0.059, 0.042)}
# The unloaded residual is round-off, so SNES would stop before assembling a Jacobian unless forced
# to iterate. The right-hand side is round-off too, which one application of no preconditioner
# solves as well as any factorization, at no cost. PETSc writes the operator of that solve.
PETSC_OPTIONS = "-snes_force_iteration -ksp_type preonly -pc_type none -ksp_view_mat binary:J.bin"
# Degree 2 is exact on the affine box. The curved elements of the shell have rational integrands, but
# degrees 4 and 8 leave beta_h unchanged there to six digits.
QUADRATURE = "Gauss2"


def write_clamped(directory, geometry, size):
    """Write the geometry, meshed by gmsh at element size `size` as second-order tetrahedra, as tetgen
    mesh.node and mesh.ele into directory, with every node on the boundary fixed."""
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        if geometry == "box":
            volumes = [(3, occ.addBox(0, 0, 0, 1, 1, 1))]
        else:
            shell, _ = occ.cut([(3, occ.addSphere(0, 0, 0, OUTER))], [(3, occ.addSphere(0, 0, 0, INNER))])
            volumes, _ = occ.intersect(shell, [(3, occ.addBox(0, 0, 0, OUTER, OUTER, OUTER))])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in volumes], MATERIAL)
        gmsh.model.addPhysicalGroup(2, [tag for _, tag in gmsh.model.getEntities(2)], BOUNDARY)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        write_tetgen(directory, "mesh", [], {BOUNDARY: CLAMPED})
    finally:
        gmsh.finalize()


def system(root, geometry, size):
    """The exported system matrix of one mesh, with the node coordinates, the fixation flags and the
    T10 elements (0-based)."""
    wd = root / f"{geometry}_{size}"
    (wd / "tetgen").mkdir(parents=True)
    (wd / "Results").mkdir()
    write_clamped(wd / "tetgen", geometry, size)
    (wd / FIXTURE.name).write_text(FIXTURE.read_text())
    proc = run_binary(_find_binary("CardioMechanics"), ["-settings", FIXTURE.name], cwd=wd,
                      env={**os.environ, "OMP_NUM_THREADS": "1", "PETSC_OPTIONS": PETSC_OPTIONS})
    assert "SIMULATION FAILED" not in proc.stdout, proc.stdout[-2000:]
    nodes = np.loadtxt(wd / "tetgen" / "mesh.node", skiprows=1)
    elements = np.loadtxt(wd / "tetgen" / "mesh.ele", skiprows=1, dtype=int)[:, 1:11] - 1
    return read_petsc_csr(wd / "J.bin"), nodes[:, 1:4], nodes[:, 4].astype(int), elements


def assemble(X, elements, rule):
    """The scalar P2 Laplacian over all nodes, numbered in node order, by the quadrature rule over the
    (curved) elements. Returns it with the mesh size (volume / elements)^(1/3)."""
    weights, _, dN, _ = tetrahedron_quadrature(rule, 2)
    dX = np.einsum("qnk,end->eqdk", dN, X[elements])
    dV = weights * np.linalg.det(dX)
    grad = np.einsum("qnk,eqkd->eqnd", dN, np.linalg.inv(dX))
    local = np.einsum("eq,eqid,eqjd->eij", dV, grad, grad)
    rows = np.repeat(elements, elements.shape[1], axis=1).ravel()
    cols = np.tile(elements, elements.shape[1]).ravel()
    laplacian = csr_matrix((local.ravel(), (rows, cols)), shape=(len(X),) * 2)
    return laplacian, (dV.sum() / len(elements)) ** (1 / 3)


def inf_sup(J, X, fixed, elements):
    """beta_h from the system matrix J of the clamped mesh, and the mesh size."""
    laplacian, h = assemble(X, elements, QUADRATURE)
    u = 3 * len(X)
    # A different unknown layout would still give a plausible beta_h, so check it: the vertex
    # pressures follow the displacements, and the fixed displacements decouple with a unit row.
    vertices = len(np.unique(elements[:, :4]))
    assert J.shape == (u + vertices,) * 2, f"system of size {J.shape}, not {u} displacements and {vertices} pressures"
    clamped = np.flatnonzero(np.repeat(fixed == CLAMPED, 3))
    assert len(clamped) and np.all(abs(J[clamped]).sum(axis=1).A1 == 1), "fixed displacements are not unit rows"
    free = np.flatnonzero(fixed == 0)
    coupling = J[u:, :u].tocsc()
    G = [coupling[:, 3 * free + c] for c in range(3)]      # T is the scalar Laplacian on each component
    T = splu(laplacian[free][:, free].tocsc())
    M = -KAPPA * J[u:, u:]
    n = M.shape[0]
    schur = LinearOperator((n, n), matvec=lambda p: sum(Gc @ T.solve(Gc.T @ p) for Gc in G), dtype=float)
    # The eigenvalues lie in [0, 1], since |div v| <= |grad v| on a clamped body, so M^-1 is a good
    # preconditioner of the Schur complement, whatever the mesh. Constraining the iterates M-orthogonal
    # to the constant pressure removes its eigenvalue. That is zero on the box, and 1e-5 on the shell,
    # where the code integrates div v over the curved elements only approximately. The smallest
    # eigenvalue of the shell is a pair, split by the mesh alone, which a block of two resolves.
    M_inverse = splu(M.tocsc())
    lam, _ = lobpcg(schur, np.random.default_rng(0).standard_normal((n, 2)), B=M,
                    M=LinearOperator((n, n), matvec=M_inverse.solve, dtype=float),
                    Y=np.ones((n, 1)), largest=False, tol=1e-8, maxiter=500)
    return np.sqrt(lam.min()), h


if __name__ == "__main__":
    # An eigenvalue LOBPCG has not converged is no measurement of beta_h.
    warnings.filterwarnings("error", "Exited at iteration", UserWarning)
    with tempfile.TemporaryDirectory() as tmp:
        for geometry, sizes in SIZES.items():
            rows = []
            for size in sizes:
                J, X, fixed, elements = system(Path(tmp), geometry, size)
                beta, h = inf_sup(J, X, fixed, elements)
                # On the clamped mesh the boundary vertices are the fixed ones.
                on_boundary = int((fixed[elements[:, :4]] == CLAMPED).all(axis=1).sum())
                layers = (OUTER - INNER) / np.ptp(np.linalg.norm(X[elements[:, :4]], axis=2), axis=1).mean()
                rows.append((size, len(elements), on_boundary, layers if geometry == "shell" else np.nan, h, beta))
            rate = np.polyfit(np.log([r[4] for r in rows]), np.log([r[5] for r in rows]), 1)[0]
            print(f"{geometry}:    size  elements  all-boundary  wall layers       h  beta_h")
            for s, n, b, w, h, beta in rows:
                print(f"  {s:10.4f}  {n:8d}  {b:12d}  {w:11.2f}  {h:.4f}  {beta:.4f}")
            print(f"  fitted rate {rate:.4f}")
